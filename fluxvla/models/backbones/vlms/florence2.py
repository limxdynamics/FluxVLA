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

import json
import os
from functools import partial
from typing import Callable, Dict, Optional, Type

import torch
from safetensors.torch import load_file
from torch import nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from fluxvla.engines import VLM_BACKBONES
from fluxvla.models.third_party_models.xvla_models.configuration_florence2 import (
    Florence2Config)
from fluxvla.models.third_party_models.xvla_models.modeling_florence2 import (
    Florence2EncoderLayer, Florence2ForConditionalGeneration)


@VLM_BACKBONES.register_module()
class Florence2Backbone(nn.Module):
    """
    Florence2 encoder-only backbone for X-VLA integration.

    Wraps Florence2ForConditionalGeneration (decoder removed) and exposes
    the standard FluxVLA vlm_backbone interface:
        forward(images, lang_tokens, img_masks, lang_masks)
            -> (vlm_features [B, T_enc, D],
                attention_mask [B, T_enc],
                aux_visual_inputs [B, (V-1)*N, D])

    The third return value carries auxiliary visual features from extra
    camera views (e.g. wrist camera), consumed by XVLAFlowMatchingHead.

    Args:
        vlm_path (str): Path to pretrained Florence2 weights or an XVLA HF
            checkpoint directory that stores Florence weights under `vlm.*`.
        dtype (str): Model dtype ('float32', 'bf16', 'fp16').
    """

    _TIED_ENCODER_EMBED_KEY = 'language_model.model.encoder.embed_tokens.weight'
    _SHARED_EMBED_KEY = 'language_model.model.shared.weight'

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
        # Mirror original X-VLA XVLA.__init__(): build Florence2, then keep
        # encoder-only execution by deleting decoder/lm_head.
        if hasattr(vlm, 'language_model'):
            lm = vlm.language_model
            if hasattr(lm, 'model') and hasattr(lm.model, 'decoder'):
                del lm.model.decoder
            if hasattr(lm, 'lm_head'):
                del lm.lm_head

    @staticmethod
    def _read_config_json(model_dir: str) -> Optional[Dict]:
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.isfile(config_path):
            return None
        with open(config_path, 'r') as f:
            return json.load(f)

    @classmethod
    def _is_xvla_checkpoint_dir(cls, vlm_path: str) -> bool:
        if not os.path.isdir(vlm_path):
            return False
        config = cls._read_config_json(vlm_path)
        return config is not None and config.get('model_type') == 'xvla'

    @staticmethod
    def _load_directory_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
        state_dict = {}
        safetensor_files = sorted(
            file for file in os.listdir(model_dir)
            if file.endswith('.safetensors'))
        if safetensor_files:
            for file in safetensor_files:
                state_dict.update(
                    load_file(os.path.join(model_dir, file), device='cpu'))
            return state_dict

        torch_files = sorted(
            file for file in os.listdir(model_dir)
            if file.endswith(('.bin', '.pt', '.pth')))
        if not torch_files:
            raise FileNotFoundError(
                f'No checkpoint shards found under {model_dir}.')

        for file in torch_files:
            shard = torch.load(os.path.join(model_dir, file),
                               map_location='cpu')
            if isinstance(shard, dict) and 'state_dict' in shard:
                shard = shard['state_dict']
            if not isinstance(shard, dict):
                raise ValueError(
                    f'Unsupported checkpoint shard format: {file}')
            state_dict.update(shard)
        return state_dict

    @classmethod
    def _load_xvla_vlm(cls,
                       vlm_path: str,
                       target_dtype: Optional[torch.dtype],
                       vlm_config: Optional[Dict]) -> Florence2ForConditionalGeneration:
        xvla_config = cls._read_config_json(vlm_path)
        if xvla_config is None or 'florence_config' not in xvla_config:
            raise ValueError(
                f'XVLA checkpoint at {vlm_path} does not contain '
                '`florence_config`.')

        florence_config = vlm_config or xvla_config['florence_config']
        if isinstance(florence_config, dict):
            florence_config = Florence2Config(**florence_config)

        vlm = Florence2ForConditionalGeneration(florence_config)
        cls._remove_decoder(vlm)

        raw_state_dict = cls._load_directory_state_dict(vlm_path)
        vlm_state_dict = {
            name.removeprefix('vlm.'): tensor
            for name, tensor in raw_state_dict.items()
            if name.startswith('vlm.')
        }
        if not vlm_state_dict:
            raise ValueError(
                f'No `vlm.*` weights found in XVLA checkpoint {vlm_path}.')

        # XVLA stores the shared embedding once; Florence2 state_dict also
        # exposes the tied encoder alias.
        if (cls._SHARED_EMBED_KEY in vlm_state_dict
                and cls._TIED_ENCODER_EMBED_KEY not in vlm_state_dict):
            vlm_state_dict[cls._TIED_ENCODER_EMBED_KEY] = vlm_state_dict[
                cls._SHARED_EMBED_KEY]

        missing_keys, unexpected_keys = vlm.load_state_dict(vlm_state_dict,
                                                            strict=False)
        missing_keys = [
            key for key in missing_keys
            if key != cls._TIED_ENCODER_EMBED_KEY
        ]
        if missing_keys or unexpected_keys:
            raise ValueError(
                'XVLA Florence load mismatch: '
                f'missing={missing_keys[:20]}, '
                f'unexpected={unexpected_keys[:20]}')

        if target_dtype is not None:
            vlm = vlm.to(dtype=target_dtype)
        return vlm

    @classmethod
    def _build_vlm(cls,
                   vlm_path: str,
                   target_dtype: Optional[torch.dtype],
                   vlm_config: Optional[Dict]) -> Florence2ForConditionalGeneration:
        if cls._is_xvla_checkpoint_dir(vlm_path):
            return cls._load_xvla_vlm(vlm_path, target_dtype, vlm_config)

        load_kwargs = {'trust_remote_code': True}
        if target_dtype is not None:
            load_kwargs['torch_dtype'] = target_dtype
        if vlm_config is not None:
            load_kwargs['config'] = Florence2Config(**vlm_config)
        return Florence2ForConditionalGeneration.from_pretrained(
            vlm_path, **load_kwargs)

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        """Required by base_train_runner.py for FSDP gradient checkpointing."""
        return Florence2EncoderLayer

    def forward(
        self,
        images: torch.Tensor,        # [B, V, C, H, W]
        lang_tokens: torch.Tensor,   # [B, L]
        img_masks: torch.Tensor,     # [B, V]
        lang_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Encode multi-view images + language tokens via Florence2 encoder.

        Returns:
            vlm_features      [B, T_enc, D]  — encoder output (view-0 + text)
            attention_mask    [B, T_enc]     — 1 for valid tokens
            aux_visual_inputs [B, (V-1)*N, D] — remaining view features
        """
        # Support both [B, V, C, H, W] and [B, V*C, H, W] layouts
        if images.ndim == 4:
            B, VC, H, W = images.shape
            V = VC // 3
            images = images.view(B, V, 3, H, W)
        B, V, C, H, W = images.shape
        flat_mask = img_masks.view(-1).to(torch.bool)   # [B*V]
        flat_images = images.flatten(0, 1)               # [B*V, C, H, W]

        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError('At least one image view must be valid per batch.')

        valid_feats = self.vlm._encode_image(flat_images[flat_mask])  # [#valid, N, D]
        N, D = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros(B * V, N, D)
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(B, V, N, D)  # [B, V, N, D]

        inputs_embeds = self.vlm.get_input_embeddings()(lang_tokens)  # [B, L, D]

        # Merge first-view image features with text embeddings
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],  # [B, N, D]
            inputs_embeds,         # [B, L, D]
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]  # [B, T_enc, D]

        # Remaining views become auxiliary visual inputs
        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)  # [B, (V-1)*N, D]

        return enc_out, attention_mask, aux_visual_inputs

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.vlm, 'language_model'):
            self.vlm.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Florence2EncoderLayer},
        )
