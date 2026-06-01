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

from functools import partial
from typing import Callable, Dict, Optional, Type, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen3_vl.modeling_qwen3_vl import \
    Qwen3VLTextDecoderLayer

from fluxvla.engines import VLM_BACKBONES
from fluxvla.engines.utils.name_map import str_to_dtype
from .hf_vlm import VLMBackbone


@VLM_BACKBONES.register_module()
class Qwen3VL(VLMBackbone):
    """
    HuggingFace-compatible wrapper for Qwen3-VL.
    Inherits from VLMBackbone and provides Qwen3-VL-specific handling
    including DeepStack visual features. Registered into
    `VLM_BACKBONES` registry.
    Args:
        vlm_backbone_id (str): Identifier string for this backbone.
        vlm_config (Dict, optional): Configuration dictionary for the VLM.
        vlm_path (Optional[str]): Path to the VLM model weights.
        use_projection (bool): If True, add a projection from backbone dim
            to projection_output_dim for action head compatibility.
            Default False.
        projection_output_dim (int, optional): Output dimension of the
            projection. Required when use_projection=True.
        projection_type (str): "linear" or "mlp". Default "linear".
        projection_mlp_hidden_dim (int, optional): Hidden dim for MLP
            projection. Used only when projection_type="mlp".
        attn_implementation (str): "flash_attention_2", "sdpa", or "eager".
            Passed to from_pretrained so the model is built with this impl.
            Default "flash_attention_2". Use "sdpa" if you see training
            regression (e.g. older checkpoints or envs used sdpa).
        torch_dtype (torch.dtype or str): dtype for loading. Defaults to
            'bf16'.
    """

    def __init__(self,
                 vlm_backbone_id: str,
                 vlm_config: Dict = None,
                 vlm_path: Optional[str] = None,
                 use_projection: bool = False,
                 projection_output_dim: Optional[int] = None,
                 projection_type: str = 'linear',
                 projection_mlp_hidden_dim: Optional[int] = None,
                 attn_implementation: str = 'flash_attention_2',
                 torch_dtype: Union[torch.dtype, str] = 'bf16') -> None:
        self._attn_implementation = attn_implementation
        assert torch_dtype is not None, 'torch_dtype must be specified'
        if isinstance(torch_dtype, str):
            torch_dtype = str_to_dtype(torch_dtype)

        super().__init__(
            vlm_backbone_id,
            vlm_config,
            vlm_path=vlm_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )

        if hasattr(self.vlm.config, 'attn_implementation'):
            self.vlm.config.attn_implementation = attn_implementation

        self._use_projection = use_projection
        self._embed_dim_override: Optional[int] = None
        if use_projection:
            if projection_output_dim is None:
                raise ValueError('projection_output_dim is required when '
                                 'use_projection=True')
            in_dim = self.vlm.model.language_model.config.hidden_size
            if projection_type == 'linear':
                self.projection = nn.Linear(in_dim, projection_output_dim)
            elif projection_type == 'mlp':
                hid = projection_mlp_hidden_dim or max(in_dim,
                                                       projection_output_dim)
                self.projection = nn.Sequential(
                    nn.Linear(in_dim, hid),
                    nn.GELU(),
                    nn.Linear(hid, projection_output_dim),
                )
            else:
                raise ValueError("projection_type must be 'linear' or 'mlp', "
                                 'got {}'.format(projection_type))
            self._embed_dim_override = projection_output_dim
        else:
            self.projection = None

    @property
    def embed_dim(self) -> int:
        """Output dim: projection_output_dim if use_projection else LLM
        hidden_size."""
        if self._embed_dim_override is not None:
            return self._embed_dim_override
        return self.vlm.model.language_model.config.hidden_size

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen3VLTextDecoderLayer

    def _compute_qwen3_grid_thw(
        self,
        pixel_values: torch.Tensor,
        batch_size: int,
        image_grid_thw: Optional[torch.LongTensor],
    ) -> torch.LongTensor:
        """Compute grid_thw in Qwen3-VL convention from pixel_values.
        Matches HF Qwen2VLImageProcessor / Qwen3-VL: grid_thw =
        (1, H//patch_size, W//patch_size); grid_h/w >= 2 and divisible
        by spatial_merge_size for fast_pos_embed_interpolate."""
        ndim = pixel_values.dim()
        if ndim == 5:
            num_imgs = pixel_values.shape[0] * pixel_values.shape[1]
            H, W = pixel_values.shape[3], pixel_values.shape[4]
        elif ndim == 4:
            num_imgs = pixel_values.shape[0]
            H, W = pixel_values.shape[2], pixel_values.shape[3]
        elif ndim == 3:
            num_imgs = pixel_values.shape[0]
            H, W = pixel_values.shape[1], pixel_values.shape[2]
        else:
            raise ValueError(
                f'qwen3_vl: pixel_values must be 3D/4D/5D, got ndim={ndim}')
        patch_size = getattr(self.vlm.model.visual.config, 'patch_size', 16)
        merge_size = getattr(self.vlm.model.visual.config,
                             'spatial_merge_size', 2)
        # Official: grid_h, grid_w = H // patch_size, W // patch_size
        grid_h = max(2, H // patch_size)
        grid_w = max(2, W // patch_size)
        # fast_pos_embed_interpolate requires h, w divisible by merge_size
        if grid_h % merge_size != 0 or grid_w % merge_size != 0:
            grid_h = max(merge_size,
                         (grid_h + merge_size - 1) // merge_size * merge_size)
            grid_w = max(merge_size,
                         (grid_w + merge_size - 1) // merge_size * merge_size)
        device = pixel_values.device
        grid_thw = torch.zeros(num_imgs, 3, dtype=torch.long, device=device)
        grid_thw[:, 0] = 1
        grid_thw[:, 1] = grid_h
        grid_thw[:, 2] = grid_w
        return grid_thw

    def forward(self,
                images: torch.Tensor,
                lang_tokens: torch.Tensor,
                img_masks: torch.Tensor,
                lang_masks: Optional[torch.Tensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                *args,
                **kwargs):
        """
        VLA path: vision + language tokens ->
        (last_hidden_state, mask, mask).
        """
        device = next(self.vlm.parameters()).device
        images = images.to(device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)
        img_masks = img_masks.to(device)
        if lang_masks is not None:
            lang_masks = lang_masks.to(device)
        lang_tokens = lang_tokens.to(device)

        batch_size = images.shape[0]
        pixel_values = images
        # In VLA mode, processor outputs may be flattened. In that case,
        # grid info must come from dataloader-provided image_grid_thw.
        # For 4D/5D pixel_values, infer grid directly from tensor shape.
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.view(-1, pixel_values.shape[2],
                                             pixel_values.shape[3],
                                             pixel_values.shape[4])
        elif pixel_values.dim() == 3:
            # (B, H, W) raw spatial vs (B, num_patches, feat_dim) flattened
            vc = self.vlm.model.visual.config
            patch_feat_dim = (
                getattr(vc, 'in_channels', 3) *
                getattr(vc, 'temporal_patch_size', 2) *
                getattr(vc, 'patch_size', 16)**2)
            if pixel_values.shape[-1] == patch_feat_dim:
                pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
            else:
                pixel_values = pixel_values.unsqueeze(1).expand(
                    -1, 3, pixel_values.shape[1], pixel_values.shape[2])
        pixel_values = pixel_values.contiguous()

        if pixel_values.dim() in (4, 5):
            image_grid_thw_flat = self._compute_qwen3_grid_thw(
                pixel_values, batch_size, image_grid_thw)
        else:
            if image_grid_thw is None:
                raise ValueError('qwen3_vl: image_grid_thw required when '
                                 'pixel_values are flattened (2D/3D).')
            # HF expects (num_total_images, 3); collator may give (B,N,3).
            image_grid_thw_flat = image_grid_thw.view(-1, 3)

        # pooler_output: after vision merger (t*h*w -> t*h*w//merge^2);
        # get_image_features splits by (grid_thw.prod(-1)//merge_size^2).
        # deepstack_features: list of (total_visual_tokens, H) per layer.
        image_outputs = self.vlm.get_image_features(
            pixel_values, image_grid_thw_flat, return_dict=True)
        image_embeds = image_outputs.pooler_output  # list of (L_i, H)
        deepstack_features = image_outputs.deepstack_features

        num_images = image_grid_thw_flat.shape[0] // batch_size
        if num_images <= 0:
            num_images = 1

        # Build image embeddings (B, L_img, H): same as Qwen2.5-VL
        img_emb = torch.cat(image_embeds, dim=0)
        hidden_size = img_emb.shape[-1]
        if all(e.shape[0] == image_embeds[0].shape[0] for e in image_embeds):
            img_emb = img_emb.reshape(batch_size, -1, hidden_size)
        else:
            per_batch = [
                sum(emb.shape[0]
                    for emb in image_embeds[i * num_images:(i + 1) *
                                            num_images])
                for i in range(batch_size)
            ]
            chunks = torch.split(img_emb, per_batch)
            max_len = max(c.shape[0] for c in chunks)
            img_emb = torch.stack([
                torch.nn.functional.pad(c, (0, 0, 0, max_len - c.shape[0]))
                for c in chunks
            ])

        text_embeds = self.vlm.get_input_embeddings()(lang_tokens)
        inputs_embeds = torch.cat([img_emb, text_embeds], dim=1)

        L_img = img_emb.shape[1]
        img_part = torch.cat([
            img_masks[:, i:i + 1].repeat(1, img_emb.shape[1] // num_images)
            for i in range(img_masks.shape[1])
        ],
                             dim=1)
        attention_mask_2d = torch.cat([img_part, lang_masks], dim=1)
        seq_len = inputs_embeds.shape[1]
        # Dataloader may pad masks to max length; trim to actual seq len.
        if attention_mask_2d.shape[1] != seq_len:
            attention_mask_2d = attention_mask_2d[:, :seq_len].contiguous()
        # Flash needs 2D padding mask (cu_seqlens shape batch_size+1).
        # SDPA/eager accept 4D causal+padding; 2D can cause SDPA error.
        attn_impl = getattr(self, '_attn_implementation', 'sdpa') or 'sdpa'
        if attn_impl == 'flash_attention_2':
            # 2D (B, seq_len): 1=valid, 0=pad; Flash uses for cu_seqlens.
            mask_for_lm = attention_mask_2d
        else:
            causal = torch.tril(
                torch.ones(
                    seq_len,
                    seq_len,
                    device=inputs_embeds.device,
                    dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
            padding_ok = attention_mask_2d.unsqueeze(1).unsqueeze(3)
            combined = causal & padding_ok
            zero = torch.zeros((),
                               device=inputs_embeds.device,
                               dtype=inputs_embeds.dtype)
            inf = torch.full((),
                             float('-inf'),
                             device=inputs_embeds.device,
                             dtype=inputs_embeds.dtype)
            mask_for_lm = torch.where(combined, zero, inf)

        # visual_pos_masks: (B, seq_len), True at image-token positions.
        visual_pos_masks = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=inputs_embeds.device)
        visual_pos_masks[:, :L_img] = True

        # deepstack_visual_embeds: list (total_visual_tokens, H) per layer
        deepstack_visual_embeds = None
        if deepstack_features is not None:
            dev, dt = inputs_embeds.device, inputs_embeds.dtype
            deepstack_visual_embeds = [
                d.reshape(-1, d.shape[-1]).to(device=dev, dtype=dt)
                for d in deepstack_features
            ]

        outputs = self.vlm.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=mask_for_lm,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=False,
        )
        last_hidden_state = outputs.last_hidden_state
        if self.projection is not None:
            last_hidden_state = self.projection(last_hidden_state)
        # Keep mask semantics aligned with attention implementation:
        # flash_attention_2 -> 2D padding mask;
        # sdpa/eager -> 4D causal+padding.
        return last_hidden_state, mask_for_lm, mask_for_lm

    def enable_gradient_checkpointing(self) -> None:
        """Enable HuggingFace gradient checkpointing on the inner VLM."""
        if hasattr(self.vlm, 'gradient_checkpointing_enable'):
            self.vlm.gradient_checkpointing_enable()

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for Qwen3VLTextDecoderLayer."""
        transformer_block_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen3VLTextDecoderLayer})
        return transformer_block_policy
