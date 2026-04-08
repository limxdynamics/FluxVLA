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

import math
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from pytest import Cache
from torch.distributed.fsdp.wrap import _or_policy

from fluxvla.engines import (VLAS, build_llm_backbone_from_cfg,
                             build_projector_from_cfg)
from fluxvla.engines.utils.model_utils import (apply_rotary_pos_emb,
                                               create_sinusoidal_pos_embedding,
                                               eager_attention_forward,
                                               gated_residual,
                                               make_att_2d_masks, sample_beta)
from fluxvla.engines.utils.overwatch import initialize_overwatch
from .base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class PI0FlowMatching(BaseVLA):
    """PI0 Flow Matching Model for Vision-Language Alignment.
    Implemented based on https://arxiv.org/abs/2410.24164

    This model is designed to handle vision-language alignment tasks
    using flow matching techniques, leveraging a vision backbone,
    language model backbone, projector, and a VLA head.

    Args:
        state_proj (Dict): Configuration dictionary for the state
            projector.
        action_in_proj (Dict): Configuration dictionary for the action
            input projector.
        action_out_proj (Dict): Configuration dictionary for the action
            output projector.
        action_time_mlp_in (Dict): Configuration dictionary for the action
            time MLP input.
        action_time_mlp_out (Dict): Configuration dictionary for the action
            time MLP output.
        vlm_backbone (str): Identifier for the vision-language model backbone.
        vla_head (str): Identifier for the vision-language alignment head.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
        freeze_vision_backbone (bool): Whether to freeze the vision backbone.
        freeze_llm_backbone (bool): Whether to freeze the language model
            backbone.
        freeze_projector (bool): Whether to freeze the projector.
        vision_backbone_fp32 (bool): Whether to use FP32 for the vision
            backbone.
        unfreeze_last_layer (bool): Whether to unfreeze the last layer
            of the model.
        ignore_index (int): Index to ignore in loss calculations.
        norm_stats (Dict, optional): Normalization statistics for the model.
        **kwargs: Additional keyword arguments for model configuration.
    """

    def __init__(self,
                 action_in_proj: Dict,
                 action_out_proj: Dict,
                 llm_expert: Dict,
                 proj_width: int,
                 time_mlp_in: Dict = None,
                 time_mlp_out: Dict = None,
                 action_time_mlp_in: Dict = None,
                 action_time_mlp_out: Dict = None,
                 n_action_steps: int = 50,
                 state_proj: Dict = None,
                 llm_backbone: Dict = None,
                 vision_backbone: Dict = None,
                 vlm_backbone: str = None,
                 projector: Dict = None,
                 vla_head: str = None,
                 enable_mixed_precision_training: bool = True,
                 freeze_vision_backbone=True,
                 freeze_llm_backbone=True,
                 freeze_vlm_backbone=True,
                 freeze_llm_expert=False,
                 freeze_projector=False,
                 vision_backbone_fp32: bool = False,
                 unfreeze_last_layer: bool = False,
                 ignore_index: int = -100,
                 norm_stats: Dict = None,
                 pretrained_name_or_path: Optional[str] = None,
                 name_mapping: Optional[Dict[str, str]] = None,
                 strict_mapping: Optional[bool] = False,
                 attention_implementation: Optional[str] = 'eager',
                 params_to_change_dtype: Optional[List[str]] = [],
                 max_action_dim: int = 7,
                 ori_action_dim: int = None,
                 num_steps: int = 10,
                 rtc_training_config: Optional[Dict] = None,
                 **kwargs):
        super(PI0FlowMatching, self).__init__(
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            vlm_backbone=vlm_backbone,
            projector=projector,
            vla_head=vla_head,
            pretrained_name_or_path=pretrained_name_or_path,
            enable_mixed_precision_training=enable_mixed_precision_training,
            freeze_vision_backbone=freeze_vision_backbone,
            freeze_llm_backbone=freeze_llm_backbone,
            freeze_vlm_backbone=freeze_vlm_backbone,
            freeze_projector=freeze_projector,
            vision_backbone_fp32=vision_backbone_fp32,
            unfreeze_last_layer=unfreeze_last_layer,
            ignore_index=ignore_index,
            norm_stats=norm_stats)
        self.all_module_keys = [
            'vlm_backbone', 'vision_backbone', 'llm_backbone', 'projector',
            'state_proj', 'action_in_proj', 'action_out_proj',
            'action_time_mlp_in', 'action_time_mlp_out', 'head', 'llm_expert'
        ]
        if state_proj is not None:
            self.state_proj = build_projector_from_cfg(state_proj)
        else:
            self.state_proj = None
        self.freeze_llm_expert = freeze_llm_expert
        self.trainable_module_keys = []
        self.action_in_proj = build_projector_from_cfg(action_in_proj)
        self.action_out_proj = build_projector_from_cfg(action_out_proj)
        if time_mlp_in is not None:
            self.time_mlp_in = build_projector_from_cfg(time_mlp_in)
        else:
            self.time_mlp_in = None
        if time_mlp_out is not None:
            self.time_mlp_out = build_projector_from_cfg(time_mlp_out)
        else:
            self.time_mlp_out = None
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
        self.proj_width = proj_width
        self.n_action_steps = n_action_steps
        self.llm_expert = build_llm_backbone_from_cfg(llm_expert)

        self.attention_implementation = attention_implementation
        self.attention_interface = self.get_attention_interface()
        self.params_to_change_dtype = params_to_change_dtype
        self.name_mapping = name_mapping
        self.strict_mapping = strict_mapping
        self.max_action_dim = max_action_dim
        self.ori_action_dim = ori_action_dim
        self.num_steps = num_steps
        self.rtc_training_config = rtc_training_config

    def to_bfloat16(self):
        for name, param in self.named_parameters():
            if any(selector in name
                   for selector in self.params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def get_attention_interface(self):
        if self.attention_implementation == 'fa2':
            raise NotImplementedError('FA2 is not implemented (yet)')
        elif self.attention_implementation == 'flex':
            # attention_interface = flex_attention_forward
            raise NotImplementedError(
                'Flex attention is not implemented (yet)')
        elif self.attention_implementation == 'eager':
            attention_interface = eager_attention_forward
        elif self.attention_implementation == 'xformer':
            # attention_interface = xformer_attention_forward
            raise NotImplementedError(
                'Xformer attention is not implemented (yet)')
        else:
            raise ValueError(
                f'Invalid attention implementation: {self.attention_implementation}. '  # noqa: E501
                "Expected one of ['fa2', 'flex', 'eager', 'xformer'].")
        return attention_interface

    def embed_prefix(self, images, lang_tokens, img_masks, lang_masks, *args,
                     **kwargs):
        """Embed the prefix tokens for the Pi0 head.

        Args:
            images (torch.Tensor): The image tensor of shape
                (bsize, 3, 224, 224).
            lang_tokens (torch.Tensor): The language tokens tensor
                of shape (bsize, lang_token_dim).
            img_masks (torch.Tensor): The image attention masks
                tensor of shape (bsize, num_img_embs).
            lang_masks (torch.Tensor): The language attention masks
                tensor of shape (bsize, num_lang_embs).
        """
        embs = list()
        pad_masks = list()
        attn_masks = list()
        img_masks = img_masks.permute(1, 0)
        img_emb = self.projector(self.vision_backbone(images))
        embs.append(img_emb)
        bsize, num_img_embs = img_emb.shape[:2]
        for img_mask in img_masks:
            img_mask = img_mask[:, None].expand(bsize,
                                                num_img_embs // len(img_masks))

            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            attn_masks += [0] * (num_img_embs // len(img_masks))
        lang_emb = self.llm_backbone.embed_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        attn_masks += [0] * num_lang_embs
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        attn_masks = torch.tensor(
            attn_masks, dtype=torch.bool, device=pad_masks.device)
        attn_masks = attn_masks[None, :].expand(bsize, len(attn_masks))
        return embs, pad_masks, attn_masks

    def embed_suffix(self, states, noisy_actions, timestep):
        """Embed the suffix tokens for the Pi0 head.

        Args:
            state (torch.Tensor): The state tensor of shape (bsize, state_dim).
            noisy_actions (torch.Tensor): The noisy actions tensor of shape
                (bsize, n_action_steps, action_dim).
            timestep (torch.Tensor): The timestep tensor of shape (bsize,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple
                containing the embedded suffix tokens, padding masks,
                and attention masks.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        bsize = states.shape[0]
        dtype = states.dtype
        device = states.device

        if self.state_proj is not None:
            state_emb = self.state_proj(states)
            embs.append(state_emb[:, None, :])
            pad_masks.append(
                torch.ones(bsize, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        # Set attention masks so that image and language
        # inputs do not attend to state or actions

        # Embed timestep using sine-cosine positional
        # encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device)
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        if time_emb.ndim == 2:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state
        # inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, None

    def _forward_transformer_layers(
        self,
        inputs_embeds: List[torch.Tensor],
        attention_masks: torch.Tensor,
        position_ids: torch.Tensor,
        models: List,
        num_layers: int,
        adarms_cond: List[Optional[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        Forward pass through transformer layers.

        Args:
            inputs_embeds (List[torch.Tensor]): List of input embeddings.
            attention_masks (torch.Tensor): Attention masks.
            position_ids (torch.Tensor): Position IDs.
            models (List): List of model instances (VLM and LLM expert).
            num_layers (int): Number of layers to process.
            adarms_cond (List[Optional[torch.Tensor]]): ADARMS
                condition tensors.

        Returns:
            List[torch.Tensor]: Output embeddings after processing all layers.
        """
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            gates = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue

                layer = models[i].layers[layer_idx]
                hidden_states, gate = layer.input_layernorm(
                    hidden_states, cond=adarms_cond[i])
                gates.append(gate)
                hidden_shape = (*hidden_states.shape[:-1], -1,
                                layer.self_attn.head_dim)

                query_state = layer.self_attn.q_proj(hidden_states).view(
                    hidden_shape).transpose(1, 2)
                key_state = layer.self_attn.k_proj(hidden_states).view(
                    hidden_shape).transpose(1, 2)
                value_state = layer.self_attn.v_proj(hidden_states).view(
                    hidden_shape).transpose(1, 2)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            dummy_tensor = torch.zeros(
                query_states.shape[0],
                query_states.shape[2],
                query_states.shape[-1],
                device=query_states.device,
                dtype=query_states.dtype,
            )
            cos, sin = (
                self.llm_backbone.rotary_emb(dummy_tensor, position_ids))
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin)

            batch_size = query_states.shape[0]
            scaling = self.llm_backbone.layers[layer_idx].self_attn.scaling

            att_output, _ = self.attention_interface(
                self.llm_backbone.layers[layer_idx].self_attn,
                query_states,
                key_states,
                value_states,
                attention_masks,
                scaling,
            )
            # Get head_dim from the current layer, not from the model
            head_dim = self.llm_backbone.layers[layer_idx].self_attn.head_dim
            att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

            # Process layer outputs
            outputs_embeds = []
            start_pos = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = models[i].layers[layer_idx]
                end_pos = start_pos + hidden_states.shape[1]

                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(
                        layer.self_attn.o_proj.weight.dtype)
                out_emb = layer.self_attn.o_proj(att_output[:,
                                                            start_pos:end_pos])

                # first residual
                out_emb = gated_residual(hidden_states, out_emb,
                                         gates[i])  # noqa: SLF001
                after_first_residual = out_emb.clone()
                out_emb, gate = layer.post_attention_layernorm(
                    out_emb, cond=adarms_cond[i])
                # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)

                out_emb = layer.mlp(out_emb)
                # second residual
                out_emb = gated_residual(after_first_residual, out_emb,
                                         gate)  # noqa: SLF001
                outputs_embeds.append(out_emb)
                start_pos = end_pos

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds

    def forward_model(self,
                      inputs_embeds: List[torch.Tensor],
                      attention_masks: List[torch.Tensor],
                      position_ids: torch.Tensor,
                      past_key_values: Optional[Union[List[torch.FloatTensor],
                                                      Cache]] = None,
                      use_cache: Optional[bool] = None,
                      fill_kv_cache: Optional[bool] = None,
                      time=None,
                      adarms_cond=None,
                      *args,
                      **kwarg):
        """
        Forward pass through the PI0 Flow Matching model.
        Args:
            images (List[torch.Tensor]): List of input image
                tensors.
            lang_tokens (torch.Tensor): Input language
                tokens.
            states (torch.Tensor): Input states tensor.
            x_t (torch.Tensor): Input noisy actions tensor.
            actions (torch.Tensor): Input actions tensor.
            img_masks (Optional[torch.Tensor]): Image attention masks.
            lang_masks (Optional[torch.Tensor]): Language attention masks.
            past_key_values (Optional[Union[List[torch.FloatTensor],
                Cache]]): Past key values for caching.
            use_cache (Optional[bool]): Whether to use cache for
                faster inference.
            fill_kv_cache (Optional[bool]): Whether to fill the
                key-value cache.
            noise (Optional[torch.Tensor]): Noise tensor for
                flow matching.
            time (Optional[torch.Tensor]): Time tensor for
                flow matching.
            adarms_cond (Optional[torch.Tensor]): ADaRMS condition tensor for
                flow matching.

        Returns:
            torch.Tensor: Output embeddings from the VLA head.
        """
        if adarms_cond is None:
            adarms_cond = [None, None]
        num_layers = len(self.llm_expert.layers)
        assert num_layers == len(self.llm_backbone.layers), \
            f'Number of layers in llm_model ({num_layers}) does not match the expected number ({self.llm_expert.language_model.num_hidden_layers}).'  # noqa: E501
        if inputs_embeds[1] is None:
            prefix_output = self.llm_backbone(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0]
                if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None

        elif inputs_embeds[0] is None:
            suffix_output = self.llm_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1]
                if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.llm_backbone, self.llm_expert]
            outputs_embeds = self._forward_transformer_layers(
                inputs_embeds=inputs_embeds,
                attention_masks=attention_masks,
                position_ids=position_ids,
                models=models,
                num_layers=num_layers,
                adarms_cond=adarms_cond,
            )
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None
        return suffix_output, prefix_past_key_values

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def forward(self,
                images: List[torch.Tensor],
                lang_tokens: torch.Tensor,
                states: torch.Tensor,
                actions: torch.Tensor,
                action_masks: Optional[torch.Tensor] = None,
                img_masks: Optional[torch.Tensor] = None,
                lang_masks: Optional[torch.Tensor] = None,
                past_key_values: Optional[Union[List[torch.FloatTensor],
                                                Cache]] = None,
                use_cache: Optional[bool] = None,
                fill_kv_cache: Optional[bool] = None,
                noise=None,
                time=None,
                *args,
                **kwarg):
        """
        Forward pass for training the PI0 Flow Matching model.

        Args:
            inputs_embeds (List[torch.Tensor]): List of input
                embeddings, including image and language embeddings.
            attention_masks (List[torch.Tensor]): List of attention masks
                for the inputs.
            position_ids (torch.Tensor): Position IDs for the inputs.
            past_key_values (Optional[Union[List[torch.FloatTensor],
                Cache]]): Past key values for caching.
            use_cache (Optional[bool]): Whether to use cache for
                faster inference.
            fill_kv_cache (Optional[bool]): Whether to fill the
                key-value cache.
            noise (Optional[torch.Tensor]): Noise tensor for
                flow matching.
            time (Optional[torch.Tensor]): Time tensor for
                flow matching.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # `time` is the sampled scalar flow-matching time, shape (B,).
        # Below we derive `t` which is passed to embed_suffix as the timestep:
        #   - RTC:     t is (B, T), per-position time (delay positions get
        #              0.0, meaning clean in PI0 convention).
        #   - vanilla: t keeps (B,), same time for all positions. Must stay
        #              1-D because PI05 embed_suffix only accepts (B,).
        T = actions.shape[1]
        if (self.rtc_training_config
                and self.rtc_training_config.get('enabled', False)):
            from fluxvla.engines.utils.rtc_training import (
                apply_rtc_time_conditioning, sample_training_delay)
            delays = sample_training_delay(
                batch_size=actions.shape[0],
                max_delay=self.rtc_training_config.get('max_delay', 5),
                distribution=self.rtc_training_config.get(
                    'distribution', 'exponential'),
                device=actions.device)
            t, action_masks = apply_rtc_time_conditioning(
                time, action_masks, delays, T, clean_time=0.0)  # (B, T)
            x_t = t.unsqueeze(-1) * noise + (1 - t.unsqueeze(-1)) * actions
        else:
            t = time  # (B,)
            x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions

        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images,
            lang_tokens=lang_tokens,
            img_masks=img_masks,
            lang_masks=lang_masks,
            past_key_values=past_key_values)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(states, x_t, t))
        inputs_embeds = [prefix_embs, suffix_embs]
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        attention_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(attention_masks)

        suffix_out, _ = self.forward_model(
            inputs_embeds=inputs_embeds,
            attention_masks=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
            adarms_cond=[None, adarms_cond],
            time=time)
        suffix_out = suffix_out[:, -self.n_action_steps:]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        if self.ori_action_dim is not None:
            v_t = v_t[:, :, :self.ori_action_dim]
            u_t = u_t[:, :, :self.ori_action_dim]
        if action_masks is not None:
            losses = F.mse_loss(u_t, v_t, reduction='none')
            losses = losses * action_masks.unsqueeze(-1)
            loss = losses.sum() / (action_masks.sum() * u_t.shape[-1] + 1e-8)
        else:
            loss = F.mse_loss(u_t, v_t)

        return_dict = dict(
            predictions=v_t,
            loss=loss,
        )
        return return_dict

    def _predict_action_plain(self, x_t, denoise, bsize, dt, device):
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            t_batch = time.expand(bsize)
            v_t = denoise(x_t, t_batch)
            x_t += dt * v_t
            time += dt
        return x_t

    def _predict_action_prefix_rtc(self, x_t, denoise, bsize, dt, device,
                                   prev_actions, prefix_len):
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        prefix_mask = torch.zeros(
            bsize, self.n_action_steps, dtype=torch.bool, device=device)
        prefix_mask[:, :prefix_len] = True

        while time >= -dt / 2:
            t_batch = time.expand(bsize)
            x_t[:, :prefix_len] = prev_actions[:, :prefix_len]
            t_pos = t_batch.unsqueeze(1)
            t_pos = t_pos.expand(-1, self.n_action_steps).clone()
            t_pos[prefix_mask] = 0.0
            v_t = denoise(x_t, t_pos)
            x_t += dt * v_t
            time += dt

        x_t[:, :prefix_len] = prev_actions[:, :prefix_len]
        return x_t

    def _predict_action_guidance_rtc(self, x_t, denoise, bsize, dt, device,
                                     prev_actions, prefix_len, rtc_config):
        from fluxvla.engines.utils.rtc_guidance import (apply_rtc_guidance,
                                                        compute_prefix_weights)

        prefix_weights = compute_prefix_weights(
            self.n_action_steps,
            prefix_len,
            rtc_config.get('decay_end', prefix_len * 2),
            rtc_config.get('schedule', 'exp'),
            device=device)
        max_gw = rtc_config.get('max_guidance_weight', 5.0)
        use_vjp = rtc_config.get('use_vjp', False)

        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            t_batch = time.expand(bsize)
            # PI0 uses opposite velocity sign.
            # rtc_guidance expects: 0=noise, 1=clean.
            v_fm = apply_rtc_guidance(
                x_t,
                lambda x: -denoise(x, t_batch),
                prev_actions,
                prefix_weights,
                1.0 - time.item(),
                max_gw,
                use_vjp=use_vjp)
            x_t += dt * (-v_fm)
            time += dt
        return x_t

    def predict_action(self,
                       images: List[torch.Tensor],
                       lang_tokens: torch.Tensor,
                       states: torch.Tensor,
                       img_masks: Optional[torch.Tensor] = None,
                       lang_masks: Optional[torch.Tensor] = None,
                       past_key_values: Optional[Union[List[torch.FloatTensor],
                                                       Cache]] = None,
                       noise=None,
                       prev_actions=None,
                       prefix_len: int = 0,
                       rtc_config: dict = None,
                       *args,
                       **kwargs):
        device = states.device
        bsize = states.shape[0]
        if noise is None:
            actions_shape = (bsize, self.n_action_steps, self.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images,
            lang_tokens=lang_tokens,
            img_masks=img_masks,
            lang_masks=lang_masks)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks,
                                                prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
            prefix_att_2d_masks)
        inputs_embeds = [prefix_embs, None]
        position_ids = prefix_position_ids
        self.llm_backbone.config._attn_implementation = self.attention_implementation  # noqa: SLF001, E501
        _, past_key_values = self.forward_model(
            inputs_embeds=inputs_embeds,
            attention_masks=prefix_att_2d_masks_4d,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=None)

        dt = -1.0 / self.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        if (prev_actions is not None and self.ori_action_dim is not None
                and prev_actions.shape[-1] < self.max_action_dim):
            pad_size = self.max_action_dim - prev_actions.shape[-1]
            prev_actions = F.pad(prev_actions, (0, pad_size), value=0.0)

        rtc_method = None
        if prev_actions is not None and prefix_len > 0 and rtc_config:
            rtc_method = rtc_config.get('method', 'prefix')

        x_t = noise

        def denoise(x, t):
            return self.denoise_step(states, prefix_pad_masks, past_key_values,
                                     x, t)

        if rtc_method == 'prefix':
            x_t = self._predict_action_prefix_rtc(
                x_t=x_t,
                denoise=denoise,
                bsize=bsize,
                dt=dt,
                device=device,
                prev_actions=prev_actions,
                prefix_len=prefix_len)
        elif rtc_method == 'guidance':
            x_t = self._predict_action_guidance_rtc(
                x_t=x_t,
                denoise=denoise,
                bsize=bsize,
                dt=dt,
                device=device,
                prev_actions=prev_actions,
                prefix_len=prefix_len,
                rtc_config=rtc_config)
        else:
            x_t = self._predict_action_plain(
                x_t=x_t, denoise=denoise, bsize=bsize, dt=dt, device=device)

        return x_t

    def denoise_step(
        self,
        states,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(states, x_t, timestep))

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks,
                                                suffix_att_masks)

        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            full_att_2d_masks)

        self.llm_expert.config._attn_implementation = self.attention_implementation  # noqa: SLF001, E501

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(
            suffix_pad_masks, dim=1) - 1

        suffix_out, _ = self.forward_model(
            attention_masks=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond])
        suffix_out = suffix_out[:, -self.n_action_steps:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Returns the FSDP auto wrapping policy for the model.

        This policy wraps low-level modules (e.g., nn.Linear, nn.LayerNorm),
        and combines with VLM's existing policy. It explicitly avoids wrapping
        nn.Embedding to prevent errors during sharding.
        """
        wrapping_policies = []
        if self.vlm_backbone is not None:
            vlm_wrapping_policy = self.vlm_backbone.get_fsdp_wrapping_policy()
            wrapping_policies.append(vlm_wrapping_policy)
        if self.llm_backbone is not None:
            llm_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()
            wrapping_policies.append(llm_wrapping_policy)

        if self.vision_backbone is not None:
            vision_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy(  # noqa: E501
            )
            wrapping_policies.append(vision_wrapping_policy)

        def match_module(module, *args, **kwargs):
            import torch.nn as nn

            # Wrap Linear and LayerNorm
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                return True
            # Explicit opt-in
            if hasattr(module, 'is_fsdp_wrap') and module.is_fsdp_wrap:
                return True
            return False

        return partial(
            _or_policy,
            policies=[
                *wrapping_policies,
                match_module,
            ],
        )
