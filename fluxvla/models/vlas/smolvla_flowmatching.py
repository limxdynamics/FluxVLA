# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from lerobot SmolVLA VLAFlowMatching:
# https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py

import math
from functools import partial
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _or_policy

from fluxvla.engines import (VLAS, build_llm_backbone_from_cfg,
                             build_projector_from_cfg)
from fluxvla.engines.utils.model_utils import (apply_rope,
                                               create_sinusoidal_pos_embedding,
                                               make_att_2d_masks)
from fluxvla.engines.utils.overwatch import initialize_overwatch
from .base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)

# ------------------------------------------------------------------
# SmolVLA Flow Matching Model
# ------------------------------------------------------------------


@VLAS.register_module()
class SmolVLAFlowMatching(BaseVLA):
    """SmolVLA Flow Matching model for Vision-Language-Action tasks.

    Architecture:
        vlm_backbone  -- SmolVLM (vision + connector + text)
        llm_expert    -- smaller LM expert (built from config)

        prefix = [image_embs, lang_embs, state_emb]  (fed to vlm text)
        suffix = [action_time_embs]                    (fed to llm_expert)
        VLM text layers and expert layers are interleaved with
        cross-attention or self-attention.

    Args:
        vlm_backbone: Config dict for SmolVLMBackbone.
        llm_expert: Config dict for SmolVLMExpert (action expert).
        state_proj: Config dict for state projector.
        action_in_proj: Config dict for action input projector.
        action_out_proj: Config dict for action output projector.
        action_time_mlp_in: Config dict for action-time MLP input.
        action_time_mlp_out: Config dict for action-time MLP output.
        max_action_dim: Action vector dimension (padded).
        chunk_size: Number of action steps per chunk.
        num_steps: Number of Euler ODE denoising steps at inference.
        add_image_special_tokens: Whether to wrap image embeddings
            with special start/end tokens.
    """

    def __init__(
        self,
        vlm_backbone=None,
        llm_expert: Dict = None,
        # Projection layers (built via registry)
        state_proj: Dict = None,
        action_in_proj: Dict = None,
        action_out_proj: Dict = None,
        action_time_mlp_in: Dict = None,
        action_time_mlp_out: Dict = None,
        # VLA config
        max_action_dim: int = 32,
        chunk_size: int = 50,
        num_steps: int = 10,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        add_image_special_tokens: bool = False,
        attention_implementation: Optional[str] = 'sdpa',
        # BaseVLA params
        vla_head=None,
        enable_mixed_precision_training: bool = True,
        freeze_vlm_backbone=True,
        vision_backbone_fp32: bool = False,
        unfreeze_last_layer: bool = False,
        ignore_index: int = -100,
        norm_stats: Dict = None,
        pretrained_name_or_path: Optional[str] = None,
        name_mapping: Optional[Dict] = None,
        strict_mapping: Optional[bool] = False,
        ori_action_dim: int = None,
        **kwargs,
    ):
        super().__init__(
            vision_backbone=None,
            llm_backbone=None,
            vlm_backbone=vlm_backbone,
            projector=None,
            vla_head=vla_head,
            enable_mixed_precision_training=enable_mixed_precision_training,
            freeze_vision_backbone=True,
            freeze_llm_backbone=True,
            freeze_vlm_backbone=freeze_vlm_backbone,
            freeze_projector=True,
            vision_backbone_fp32=vision_backbone_fp32,
            unfreeze_last_layer=unfreeze_last_layer,
            ignore_index=ignore_index,
            norm_stats=norm_stats,
            pretrained_name_or_path=pretrained_name_or_path,
            name_mapping=name_mapping,
            strict_mapping=strict_mapping,
        )

        # VLA config
        self.max_action_dim = max_action_dim
        self.chunk_size = chunk_size
        self.num_steps = num_steps
        self.min_period = min_period
        self.max_period = max_period
        self.add_image_special_tokens = add_image_special_tokens
        self.attention_implementation = attention_implementation
        self.attention_interface = self.get_attention_interface()
        self.ori_action_dim = ori_action_dim

        # --- Build the expert via registry ---
        self.llm_expert = build_llm_backbone_from_cfg(llm_expert)

        self.num_vlm_layers = len(self.vlm_backbone.layers)
        self.num_attention_heads = self.vlm_backbone.num_attention_heads
        self.num_key_value_heads = self.vlm_backbone.num_key_value_heads
        self.num_expert_layers = len(self.llm_expert.layers)
        self.self_attn_every_n_layers = getattr(self.llm_expert,
                                                'self_attn_every_n_layers', -1)
        self.attention_mode = getattr(self.llm_expert, 'attention_mode',
                                      'cross_attn')
        self.expert_hidden_size = self.llm_expert.hidden_size

        # --- Projection layers (built via registry) ---
        if state_proj is not None:
            self.state_proj = build_projector_from_cfg(state_proj)
        else:
            self.state_proj = None
        self.action_in_proj = build_projector_from_cfg(action_in_proj)
        self.action_out_proj = build_projector_from_cfg(action_out_proj)
        if action_time_mlp_in is not None:
            self.action_time_mlp_in = build_projector_from_cfg(
                action_time_mlp_in)
        else:
            self.action_time_mlp_in = None
        if action_time_mlp_out is not None:
            self.action_time_mlp_out = build_projector_from_cfg(
                action_time_mlp_out)
        else:
            self.action_time_mlp_out = None

        # Image special tokens (if enabled)
        if self.add_image_special_tokens:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                vlm_backbone.get('model_id', ''))
            tokenizer = processor.tokenizer
            self.fake_image_token = tokenizer.fake_image_token_id
            self.global_image_token = tokenizer.global_image_token_id
            self.register_buffer(
                'global_image_start_token',
                torch.tensor([self.fake_image_token, self.global_image_token],
                             dtype=torch.long))
            self.register_buffer(
                'image_end_token',
                torch.tensor([self.fake_image_token], dtype=torch.long))

        # --- Dimension consistency checks ---
        self._validate_dimensions()

        self.all_module_keys = [
            'vlm_backbone',
            'llm_expert',
            'state_proj',
            'action_in_proj',
            'action_out_proj',
            'action_time_mlp_in',
            'action_time_mlp_out',
        ]
        self.trainable_module_keys = []
        self._logged_forward_keys = False
        self._logged_predict_keys = False

    # ------------------------------------------------------------------
    # Dimension validation
    # ------------------------------------------------------------------

    def _validate_dimensions(self):
        """Validate VLM-expert dimension alignment."""
        vlm = self.vlm_backbone
        expert_cfg = self.llm_expert.expert.config

        assert vlm.head_dim == expert_cfg.head_dim, (
            f'head_dim: vlm={vlm.head_dim} != expert={expert_cfg.head_dim}')
        assert vlm.num_key_value_heads == expert_cfg.num_key_value_heads, (
            f'num_kv_heads: vlm={vlm.num_key_value_heads} '
            f'!= expert={expert_cfg.num_key_value_heads}')
        assert vlm.num_attention_heads == expert_cfg.num_attention_heads, (
            f'num_attn_heads: vlm={vlm.num_attention_heads} '
            f'!= expert={expert_cfg.num_attention_heads}')

        if 'cross' in self.attention_mode:
            vlm_kv_dim = vlm.num_key_value_heads * vlm.head_dim
            assert self.llm_expert.vlm_kv_dim == vlm_kv_dim, (
                f'vlm_kv_dim: expert={self.llm_expert.vlm_kv_dim} '
                f'!= vlm kv_heads*head_dim={vlm_kv_dim}')

        assert self.num_vlm_layers % self.num_expert_layers == 0, (
            f'num_vlm_layers({self.num_vlm_layers}) must be divisible by '
            f'num_expert_layers({self.num_expert_layers})')

    # ------------------------------------------------------------------
    # Noise and time sampling
    # ------------------------------------------------------------------

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(
            concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize, )).to(
            device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    # ------------------------------------------------------------------
    # Attention interface
    # ------------------------------------------------------------------

    def get_attention_interface(self):
        """Return the attention function based on attention_implementation."""
        if self.attention_implementation == 'fa2':
            raise NotImplementedError('FA2 is not implemented (yet)')
        elif self.attention_implementation == 'flex':
            raise NotImplementedError(
                'Flex attention is not implemented (yet)')
        elif self.attention_implementation == 'sdpa':
            return self._sdpa_attention_forward
        elif self.attention_implementation == 'xformer':
            raise NotImplementedError(
                'Xformer attention is not implemented (yet)')
        else:
            raise ValueError(
                f'Invalid attention implementation: '
                f'{self.attention_implementation}. '
                "Expected one of ['fa2', 'flex', 'sdpa', 'xformer'].")

    # ------------------------------------------------------------------
    # Interleaved layer helpers
    # ------------------------------------------------------------------

    def _get_model_layers(self):
        """Build aligned VLM / expert layer lists, handling asymmetric
        layer counts.

        Returns:
            Tuple of ([vlm_layers, expert_layers], is_self_attn) where
            is_self_attn[i] indicates the attention mode for VLM layer i,
            read from the expert's own is_self_attn (keyed by expert index).
        """
        vlm_layers = []
        expert_layers = []
        is_self_attn = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
                is_self = True
            else:
                expert_layer_index = (
                    i // multiple_of if multiple_of > 0 else i)
                expert_layer = self.llm_expert.layers[expert_layer_index]
                is_self = self.llm_expert.is_self_attn[expert_layer_index]
            vlm_layers.append(self.vlm_backbone.layers[i])
            expert_layers.append(expert_layer)
            is_self_attn.append(is_self)
        return [vlm_layers, expert_layers], is_self_attn

    # ------------------------------------------------------------------
    # QKV / cross-KV helpers
    # ------------------------------------------------------------------

    def _compute_qkv(self, layer, hidden_states, compute_kv=True):
        """Compute Q (and optionally K, V) from a transformer layer."""
        normed = layer.input_layernorm(hidden_states)
        hidden_shape = (*normed.shape[:-1], -1, layer.self_attn.head_dim)
        normed = normed.to(dtype=layer.self_attn.q_proj.weight.dtype)
        q = layer.self_attn.q_proj(normed).view(hidden_shape)
        if not compute_kv:
            return q
        k = layer.self_attn.k_proj(normed).view(hidden_shape)
        v = layer.self_attn.v_proj(normed).view(hidden_shape)
        return q, k, v

    def _project_cross_kv(self, layer, key_states, value_states):
        """Project VLM key/value through expert cross-attn k/v projections."""
        flat_key = key_states.to(
            dtype=layer.self_attn.k_proj.weight.dtype).reshape(
                *key_states.shape[:2], -1)
        k = layer.self_attn.k_proj(flat_key).view(*flat_key.shape[:-1], -1,
                                                  layer.self_attn.head_dim)
        flat_value = value_states.to(
            dtype=layer.self_attn.v_proj.weight.dtype).reshape(
                *value_states.shape[:2], -1)
        v = layer.self_attn.v_proj(flat_value).view(*flat_value.shape[:-1], -1,
                                                    layer.self_attn.head_dim)
        return k, v

    def _apply_residual_ffn(self, layer, att_output, hidden_states):
        """o_proj + residual + FFN + residual."""
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out = layer.self_attn.o_proj(att_output)
        out = out + hidden_states
        residual = out
        out = layer.mlp(layer.post_attention_layernorm(out))
        return out + residual

    def _sdpa_attention_forward(
        self,
        attention_mask,
        query_states,
        key_states,
        value_states,
    ) -> torch.Tensor:
        """Multi-head attention via scaled dot-product attention."""
        batch_size = query_states.shape[0]
        head_dim = query_states.shape[-1]
        num_att_heads = self.num_attention_heads
        num_kv_heads = self.num_key_value_heads
        num_kv_groups = num_att_heads // num_kv_heads
        seq_len = key_states.shape[1]

        # GQA: expand K/V heads
        if num_kv_groups > 1:
            key_states = key_states[:, :, :, None, :].expand(
                batch_size, seq_len, num_kv_heads, num_kv_groups,
                head_dim).reshape(batch_size, seq_len,
                                  num_kv_heads * num_kv_groups, head_dim)
            value_states = value_states[:, :, :, None, :].expand(
                batch_size, seq_len, num_kv_heads, num_kv_groups,
                head_dim).reshape(batch_size, seq_len,
                                  num_kv_heads * num_kv_groups, head_dim)

        # (B, L, H, D) -> (B, H, L, D)
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)

        # Upcast q/k to float32 for numerical stability (matches lerobot).
        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)

        # Boolean mask (B, L_q, L_kv) -> (B, 1, L_q, L_kv)
        attn_mask = attention_mask[:, None, :, :]

        att_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask)

        # Downcast output back to value dtype.
        att_output = att_output.to(dtype=value_states.dtype)

        # (B, H, L, D) -> (B, L, H*D)
        att_output = att_output.transpose(1,
                                          2).reshape(batch_size, -1,
                                                     num_att_heads * head_dim)

        return att_output

    # ------------------------------------------------------------------
    # Per-layer attention
    # ------------------------------------------------------------------

    def _forward_attn_layer(
        self,
        vlm_layer,
        expert_layer,
        vlm_hidden,
        expert_hidden,
        position_ids,
        attention_mask,
        use_cache=False,
        past_key_values=None,
        layer_idx=None,
    ):
        """Self-attention layer.

        Handles VLM-only, joint VLM+expert, and decode uniformly
        by iterating over present inputs.  KV cache is managed
        inline following the standard HF convention.

        Returns:
            (vlm_out, expert_out) — either may be None.
        """
        query_states = []
        key_states = []
        value_states = []

        for layer, hidden in [(vlm_layer, vlm_hidden),
                              (expert_layer, expert_hidden)]:
            if hidden is None or layer is None:
                continue
            q, k, v = self._compute_qkv(layer, hidden)
            query_states.append(q)
            key_states.append(k)
            value_states.append(v)

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)

        seq_len = query_states.shape[1]
        pos = position_ids[:, :seq_len]
        query_states = apply_rope(query_states, pos)
        key_states = apply_rope(key_states, pos)

        if use_cache:
            if past_key_values[layer_idx] is None:
                # Prefill: store prefix KV
                past_key_values[layer_idx] = (key_states, value_states)
            else:
                # Decode: prepend cached prefix KV for attention,
                # but don't update cache (ODE steps are independent)
                cached_k, cached_v = past_key_values[layer_idx]
                key_states = torch.cat([cached_k, key_states], dim=1)
                value_states = torch.cat([cached_v, value_states], dim=1)

        _mask = attention_mask[:, :seq_len, :key_states.shape[1]]

        att_output = self.attention_interface(_mask, query_states, key_states,
                                              value_states)

        # Split output back to VLM / expert
        vlm_out = None
        expert_out = None
        start = 0
        if vlm_hidden is not None:
            vlm_len = vlm_hidden.shape[1]
            vlm_out = att_output[:, start:start + vlm_len]
            start += vlm_len
        if expert_hidden is not None:
            expert_out = att_output[:, start:]

        return vlm_out, expert_out

    def _forward_cross_attn_layer(
        self,
        vlm_layer,
        expert_layer,
        vlm_hidden,
        expert_hidden,
        position_ids,
        attention_mask,
        use_cache=False,
        past_key_values=None,
        layer_idx=None,
    ):
        """Cross-attention between VLM and expert.

        VLM does self-attention; expert cross-attends on VLM KV.
        VLM KV comes from live computation or cache.

        Returns:
            (vlm_out, expert_out) — either may be None.
        """
        # VLM self-attention, or read KV from cache
        # (follows HF encoder-decoder cross-attn convention:
        #  check "encoder input" first, fall back to cache)
        vlm_out = None
        if vlm_hidden is not None:
            vlm_len = vlm_hidden.shape[1]
            vlm_pos = position_ids[:, :vlm_len]
            vlm_q, vlm_k, vlm_v = self._compute_qkv(vlm_layer, vlm_hidden)
            vlm_q = apply_rope(vlm_q, vlm_pos)
            vlm_k = apply_rope(vlm_k, vlm_pos)
            vlm_mask = attention_mask[:, :vlm_len, :vlm_len]
            vlm_out = self.attention_interface(vlm_mask, vlm_q, vlm_k, vlm_v)
            if use_cache:
                past_key_values[layer_idx] = (vlm_k, vlm_v)
        else:
            vlm_k, vlm_v = past_key_values[layer_idx]

        # Expert cross-attention on VLM KV
        expert_out = None
        if expert_hidden is not None:
            expert_len = expert_hidden.shape[1]
            expert_q = self._compute_qkv(
                expert_layer, expert_hidden, compute_kv=False)
            expert_k, expert_v = self._project_cross_kv(
                expert_layer, vlm_k, vlm_v)
            expert_pos = position_ids[:, -expert_len:]
            expert_pos = (
                expert_pos - expert_pos.min(dim=1, keepdim=True).values)
            expert_q = apply_rope(expert_q, expert_pos)
            expert_mask = attention_mask[:, -expert_len:, :expert_k.shape[1]]
            expert_out = self.attention_interface(expert_mask, expert_q,
                                                  expert_k, expert_v)

        return vlm_out, expert_out

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def forward_model(
        self,
        attention_mask,
        position_ids,
        inputs_embeds,
        past_key_values=None,
        use_cache=False,
    ):
        """Unified forward through interleaved VLM + expert layers.

        Supports three usage patterns (standard HF convention):
          Training:  use_cache=False,  past_key_values=None
          Prefill:   use_cache=True,   past_key_values=None  → creates & fills
          Decode:    use_cache=True,   past_key_values=filled → reads cache

        Args:
            attention_mask: 3-D boolean mask (B, L, L).
            position_ids: Position ids (B, L).
            inputs_embeds: List of [prefix_embeds, suffix_embeds].
            past_key_values: List of per-layer VLM KV tuples, or None.
            use_cache: Whether to use KV caching.

        Returns:
            (suffix_output, past_key_values)
        """
        model_layers, is_self_attn = self._get_model_layers()
        prefix_embeds, suffix_embeds = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = [None] * self.num_vlm_layers

        for layer_idx in range(self.num_vlm_layers):
            vlm_layer = model_layers[0][layer_idx]
            expert_layer = model_layers[1][layer_idx]
            expert_hidden = (
                suffix_embeds if expert_layer is not None else None)

            # Nothing to process (decode + no expert at this layer)
            if prefix_embeds is None and expert_hidden is None:
                continue

            if is_self_attn[layer_idx]:
                vlm_out, expert_out = self._forward_attn_layer(
                    vlm_layer,
                    expert_layer,
                    prefix_embeds,
                    expert_hidden,
                    position_ids,
                    attention_mask,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    layer_idx=layer_idx)
            else:
                vlm_out, expert_out = self._forward_cross_attn_layer(
                    vlm_layer,
                    expert_layer,
                    prefix_embeds,
                    expert_hidden,
                    position_ids,
                    attention_mask,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    layer_idx=layer_idx)

            if vlm_out is not None:
                prefix_embeds = self._apply_residual_ffn(
                    vlm_layer, vlm_out, prefix_embeds)
            if expert_out is not None:
                suffix_embeds = self._apply_residual_ffn(
                    expert_layer, expert_out, suffix_embeds)

        # Final norms
        if prefix_embeds is not None:
            prefix_embeds = self.vlm_backbone.norm(prefix_embeds)
        if suffix_embeds is not None:
            suffix_embeds = self.llm_expert.norm(suffix_embeds)

        return suffix_embeds, past_key_values

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, states):
        """Embed prefix: [image_embs, lang_embs, state_emb].

        Args:
            images: (B, N_cam*3, H, W) or list of (B, 3, H, W).
            img_masks: (B, N_cam) camera validity masks.
            lang_tokens: (B, L) token ids.
            lang_masks: (B, L) language masks.
            states: (B, state_dim).
        """
        bsize = img_masks.shape[0]
        n_cameras = img_masks.shape[1]
        device = img_masks.device
        lang_scale = math.sqrt(self.vlm_backbone.hidden_size)

        # --- Images: batched vision forward ---
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = images.unflatten(1, (n_cameras, 3)).flatten(0, 1)
        else:
            images = torch.stack(images, dim=1).flatten(0, 1)

        img_emb = self.vlm_backbone.forward_connector(
            self.vlm_backbone.forward_vision(images))
        img_emb = img_emb * torch.tensor(
            img_emb.shape[-1]**0.5, dtype=img_emb.dtype, device=img_emb.device)
        n_patches = img_emb.shape[1]
        # (B*N_cam, N_patches, D) → (B, N_cam, N_patches, D)
        img_emb = img_emb.unflatten(0, (bsize, n_cameras))

        # --- Assemble per-camera image embeddings ---
        embs = []
        pad_masks = []

        for cam_idx in range(n_cameras):
            if self.add_image_special_tokens:
                tok = self.vlm_backbone.embed_tokens(
                    self.global_image_start_token)[None].expand(bsize, -1, -1)
                embs.append(tok)
                pad_masks.append(
                    torch.ones(
                        bsize, tok.shape[1], dtype=torch.bool, device=device))

            embs.append(img_emb[:, cam_idx])
            pad_masks.append(img_masks[:, cam_idx,
                                       None].expand(bsize, n_patches))

            if self.add_image_special_tokens:
                tok = self.vlm_backbone.embed_tokens(
                    self.image_end_token)[None].expand(bsize, -1, -1)
                embs.append(tok)
                pad_masks.append(
                    torch.ones(
                        bsize, tok.shape[1], dtype=torch.bool, device=device))

        # --- Language ---
        embs.append(self.vlm_backbone.embed_tokens(lang_tokens) * lang_scale)
        pad_masks.append(lang_masks)

        # --- State ---
        state_emb = self.state_proj(states)
        if state_emb.ndim == 2:
            state_emb = state_emb[:, None, :]
        embs.append(state_emb)
        pad_masks.append(
            torch.ones(
                bsize, state_emb.shape[1], dtype=torch.bool, device=device))

        # --- Concatenate ---
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        # att_masks: False = bidirectional group (image/lang see
        # each other), True = cumsum boundary (state sees prefix,
        # but prefix can't see state).
        prefix_len = embs.shape[1] - state_emb.shape[1]
        att_masks = torch.cat([
            torch.zeros(prefix_len, dtype=torch.bool, device=device),
            torch.ones(state_emb.shape[1], dtype=torch.bool, device=device),
        ])[None].expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed suffix: noisy actions fused with timestep.

        Args:
            noisy_actions: (B, chunk_size, action_dim).
            timestep: (B,) scalar timestep.

        Returns:
            Tuple of (embs, pad_masks, att_masks).
        """
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.expert_hidden_size,
            min_period=self.min_period,
            max_period=self.max_period,
            device=device,
        )
        time_emb = time_emb.to(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)

        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs = action_time_emb
        action_time_dim = embs.shape[1]
        pad_masks = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device)
        att_masks = torch.tensor(
            [1] * self.chunk_size, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(bsize, -1)

        return embs, pad_masks, att_masks

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        img_masks: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        noise=None,
        time=None,
        **kwargs,
    ):
        """Training forward pass with flow matching loss."""
        if not self._logged_forward_keys:
            overwatch.info(
                f'[SmolVLA forward] kwargs keys: {list(kwargs.keys())}',
                ctx_level=1)
            overwatch.info(
                f'[SmolVLA forward] images={type(images)}, '
                f'lang_tokens={lang_tokens.shape}, '
                f'states={states.shape}, actions={actions.shape}, '
                f'img_masks={img_masks.shape if img_masks is not None else None}, '  # noqa: E501
                f'lang_masks={lang_masks.shape if lang_masks is not None else None}, '  # noqa: E501
                f'action_masks={action_masks.shape if action_masks is not None else None}',  # noqa: E501
                ctx_level=1)
            self._logged_forward_keys = True
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Flow matching interpolation
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix and suffix
        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.embed_prefix(images, img_masks, lang_tokens, lang_masks,
                              states))
        suffix_embs, suffix_pad_masks, suffix_att_masks = (
            self.embed_suffix(x_t, time))

        # Build full attention mask
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Forward through interleaved VLM + expert layers
        suffix_out, _ = self.forward_model(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=[prefix_embs, suffix_embs],
        )
        suffix_out = suffix_out[:, -self.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        # Trim to original action dim if needed
        if self.ori_action_dim is not None:
            v_t = v_t[:, :, :self.ori_action_dim]
            u_t = u_t[:, :, :self.ori_action_dim]

        # Compute loss
        if action_masks is not None:
            losses = F.mse_loss(u_t, v_t, reduction='none')
            losses = losses * action_masks.unsqueeze(-1)
            loss = losses.sum() / (action_masks.sum() * u_t.shape[-1] + 1e-8)
        else:
            loss = F.mse_loss(u_t, v_t)

        return dict(predictions=v_t, loss=loss)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_action(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        states: torch.Tensor,
        img_masks: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        noise=None,
        **kwargs,
    ):
        """Predict actions via iterative Euler ODE denoising."""
        if not self._logged_predict_keys:
            overwatch.info(
                f'[SmolVLA predict_action] kwargs keys: '
                f'{list(kwargs.keys())}',
                ctx_level=1)
            overwatch.info(
                f'[SmolVLA predict_action] '
                f'lang_tokens={lang_tokens.shape}, '
                f'states={states.shape}, '
                f'img_masks={img_masks.shape if img_masks is not None else None}, '  # noqa: E501
                f'lang_masks={lang_masks.shape if lang_masks is not None else None}',  # noqa: E501
                ctx_level=1)
            self._logged_predict_keys = True
        bsize = states.shape[0]
        device = states.device

        if noise is None:
            actions_shape = (bsize, self.chunk_size, self.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Embed prefix and cache KV
        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.embed_prefix(images, img_masks, lang_tokens, lang_masks,
                              states))
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks,
                                                prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.forward_model(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Euler ODE denoising loop
        dt = -1.0 / self.num_steps
        x_t = noise
        for step in range(self.num_steps):
            t = 1.0 + step * dt
            t_batch = torch.tensor(
                t, dtype=torch.float32, device=device).expand(bsize)
            v_t = self._denoise_step(x_t, prefix_pad_masks, past_key_values,
                                     t_batch)
            x_t = x_t + dt * v_t

        return x_t

    def _denoise_step(self, x_t, prefix_pad_masks, past_key_values, timestep):
        """Single denoising step using cached prefix KV."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = (
            self.embed_suffix(x_t, timestep))

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks,
                                                suffix_att_masks)
        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = (
            prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1)

        suffix_out, _ = self.forward_model(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=[None, suffix_embs],
            past_key_values=past_key_values,
            use_cache=True,
        )
        suffix_out = suffix_out[:, -self.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    # ------------------------------------------------------------------
    # Freeze / FSDP
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for the VLM backbone."""
        self.vlm_backbone.enable_gradient_checkpointing()
        overwatch.info(
            'Enabled gradient checkpointing for SmolVLM text model',
            ctx_level=1)

    def get_fsdp_wrapping_policy(self) -> Callable:
        wrapping_policies = []

        if self.vlm_backbone is not None:
            wrapping_policies.append(
                self.vlm_backbone.get_fsdp_wrapping_policy())

        if self.llm_expert is not None:
            wrapping_policies.append(
                self.llm_expert.get_fsdp_wrapping_policy())

        def match_module(module, *args, **kwargs):
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                return True
            if hasattr(module, 'is_fsdp_wrap') and module.is_fsdp_wrap:
                return True
            return False

        return partial(_or_policy, policies=[*wrapping_policies, match_module])

    # ------------------------------------------------------------------
    # GenerationMixin stubs (required by BaseVLA)
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self.vlm_backbone.config

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values
