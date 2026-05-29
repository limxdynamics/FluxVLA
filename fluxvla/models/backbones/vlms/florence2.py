# Copyright 2025 2toINF (https://github.com/2toINF)
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

# Origin: Modified from
# Upstream-Repo: 2toINF/X-VLA
# Upstream-Path: models/modeling_xvla.py
# Upstream-Ref: origin/main@6bc2513f5f1cbec715cc668b414392a6cae5c671
# SPDX-License-Identifier: Apache-2.0
#
# Notes: Extracted the Florence2 encoder path into a FluxVLA backbone wrapper
# while preserving X-VLA's multimodal merge semantics.

from functools import partial
from typing import Callable, Dict, Optional, Type

import torch
from torch import nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from fluxvla.engines import VLM_BACKBONES
from fluxvla.models.third_party_models.xvla_models.modeling_florence2 import (
    Florence2EncoderLayer, Florence2ForConditionalGeneration)

from fluxvla.models.third_party_models.xvla_models.configuration_florence2 import (  # isort: skip  # noqa: E501
    Florence2Config)


@VLM_BACKBONES.register_module()
class Florence2Backbone(nn.Module):
    """Training-time Florence2 encoder backbone for XVLA."""

    def __init__(self,
                 vlm_path: str,
                 dtype: str = 'float32',
                 vlm_config: Optional[Dict] = None):
        super().__init__()
        from fluxvla.engines import str_to_dtype
        target_dtype = str_to_dtype(dtype) if dtype else None

        self.vlm = self._build_vlm(vlm_path, target_dtype, vlm_config)
        self.config = self.vlm.config
        self._remove_decoder(self.vlm)

    @classmethod
    def _remove_decoder(cls, vlm: Florence2ForConditionalGeneration) -> None:
        if hasattr(vlm, 'language_model'):
            lm = vlm.language_model
            if hasattr(lm, 'model') and hasattr(lm.model, 'decoder'):
                del lm.model.decoder
            if hasattr(lm, 'lm_head'):
                del lm.lm_head

    @staticmethod
    def _build_config(vlm_config: Optional[Dict]) -> Optional[Florence2Config]:
        if isinstance(vlm_config, Florence2Config):
            return Florence2Config(**vlm_config.to_dict())
        if isinstance(vlm_config, dict):
            return Florence2Config(**vlm_config)
        if vlm_config is None:
            return None
        raise TypeError(
            '`vlm_config` must be a dict, Florence2Config, or None.')

    @classmethod
    def _build_vlm(
        cls,
        vlm_path: str,
        target_dtype: Optional[torch.dtype],
        vlm_config: Optional[Dict],
    ) -> Florence2ForConditionalGeneration:
        config = cls._build_config(vlm_config)

        if config is not None and target_dtype is not None:
            # Keep the config dtype aligned with the requested runtime dtype.
            config.torch_dtype = target_dtype
            if getattr(config, 'text_config', None) is not None:
                config.text_config.torch_dtype = target_dtype
            if getattr(config, 'vision_config', None) is not None:
                config.vision_config.torch_dtype = target_dtype

        if config is not None:
            if target_dtype is not None:
                return Florence2ForConditionalGeneration._from_config(
                    config, torch_dtype=target_dtype)
            return Florence2ForConditionalGeneration(config)

        if vlm_path is None:
            raise ValueError(
                'Florence2Backbone requires either `vlm_config` or `vlm_path`.'
            )
        load_kwargs = {'trust_remote_code': True}
        if target_dtype is not None:
            load_kwargs['torch_dtype'] = target_dtype
        return Florence2ForConditionalGeneration.from_pretrained(
            vlm_path, **load_kwargs)

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Florence2EncoderLayer

    def _reshape_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            batch_size, views_channels, image_height, image_width = \
                images.shape
            num_views = views_channels // 3
            images = images.view(batch_size, num_views, 3, image_height,
                                 image_width)
        return images

    def _flatten_valid_view_indices(self,
                                    img_masks: torch.Tensor) -> torch.Tensor:
        valid_view_indices = img_masks.view(-1).to(
            torch.bool).nonzero(as_tuple=False).flatten()
        if valid_view_indices.numel() == 0:
            raise ValueError(
                'At least one image view must be valid per batch.')
        return valid_view_indices

    def _scatter_valid_view_features(
        self,
        valid_features: torch.Tensor,
        batch_size: int,
        num_views: int,
        valid_view_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_size = valid_features.shape[1:]
        image_features = valid_features.new_zeros(
            batch_size * num_views,
            num_tokens,
            hidden_size,
        )
        image_features.index_copy_(0, valid_view_indices, valid_features)
        return image_features.view(batch_size, num_views, num_tokens,
                                   hidden_size)

    def _encode_image_features_static(
        self,
        images: torch.Tensor,
        valid_view_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_views = images.shape[:2]
        flat_images = images.flatten(0, 1)
        valid_features = self.vlm._encode_image(
            flat_images.index_select(0, valid_view_indices))
        return self._scatter_valid_view_features(
            valid_features=valid_features,
            batch_size=batch_size,
            num_views=num_views,
            valid_view_indices=valid_view_indices,
        )

    def _encode_image_features(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode_image_features_static(
            images=images,
            valid_view_indices=self._flatten_valid_view_indices(img_masks),
        )

    def _forward_from_image_features(
        self,
        image_features: torch.Tensor,
        lang_tokens: torch.Tensor,
    ):
        batch_size, _, _, hidden_size = image_features.shape
        inputs_embeds = self.vlm.get_input_embeddings()(lang_tokens)
        merged_embeds, attention_mask = \
            self.vlm._merge_input_ids_with_image_features(
                image_features[:, 0],
                inputs_embeds,
            )
        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]
        aux_visual_inputs = image_features[:,
                                           1:].reshape(batch_size, -1,
                                                       hidden_size)
        return enc_out, attention_mask, aux_visual_inputs

    def forward(
        self,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        img_masks: torch.Tensor,
        lang_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del lang_masks, kwargs
        images = self._reshape_images(images)
        image_features = self._encode_image_features(images, img_masks)
        return self._forward_from_image_features(image_features, lang_tokens)

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.vlm, 'language_model'):
            self.vlm.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Florence2EncoderLayer},
        )


__all__ = ['Florence2Backbone']
