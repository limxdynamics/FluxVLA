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

from __future__ import annotations
import copy
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from fluxvla.engines import (build_head_from_cfg, build_llm_backbone_from_cfg,
                             build_projector_from_cfg,
                             build_vision_backbone_from_cfg,
                             build_vlm_backbone_from_cfg, initialize_overwatch)

overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for arbitrary Vision-Language Models ===
class BaseVLA(nn.Module, GenerationMixin, ABC):

    def __init__(self,
                 vision_backbone: str = None,
                 llm_backbone: str = None,
                 vlm_backbone: str = None,
                 projector: str = None,
                 vla_head: str = None,
                 enable_mixed_precision_training: bool = True,
                 freeze_vision_backbone=True,
                 freeze_llm_backbone=True,
                 freeze_vlm_backbone=True,
                 freeze_projector=False,
                 vision_backbone_fp32: bool = False,
                 unfreeze_last_layer: bool = False,
                 ignore_index: int = -100,
                 norm_stats: Dict = None,
                 pretrained_name_or_path: str = None,
                 name_mapping: Dict = None,
                 strict_mapping: bool = False) -> None:
        super().__init__()
        if vision_backbone is not None:
            self.vision_backbone = build_vision_backbone_from_cfg(
                copy.deepcopy(vision_backbone))
        else:
            self.vision_backbone = None
        if llm_backbone is not None:
            self.llm_backbone = build_llm_backbone_from_cfg(llm_backbone)
        else:
            self.llm_backbone = None
        if vlm_backbone is not None:
            self.vlm_backbone = build_vlm_backbone_from_cfg(vlm_backbone)
        else:
            self.vlm_backbone = None
        if projector is not None:
            self.projector = build_projector_from_cfg(projector)
        else:
            self.projector = None
        if vla_head is not None:
            self.vla_head = build_head_from_cfg(vla_head)
        else:
            self.vla_head = None

        self.freeze_vision_backbone = freeze_vision_backbone
        self.freeze_llm_backbone = freeze_llm_backbone
        self.freeze_vlm_backbone = freeze_vlm_backbone
        self.freeze_projector = freeze_projector
        self.vision_backbone_fp32 = vision_backbone_fp32
        self.unfreeze_last_layer = unfreeze_last_layer
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.ignore_index = ignore_index
        self.norm_stats = norm_stats
        self.pretrained_name_or_path = pretrained_name_or_path
        self.name_mapping = name_mapping
        self.strict_mapping = strict_mapping
        # Instance Attributes for a generic VLM
        self.all_module_keys, self.trainable_module_keys = None, None

    @property
    def device(self) -> torch.device:
        """Borrowed from `transformers.modeling_utils.py` -- checks
        parameter device; assumes model on *ONE* device!"""
        return next(self.parameters()).device

    def freeze_backbones(self) -> None:
        """
        Freeze the designated modules of the model.
        """
        if self.vision_backbone is not None:
            self.vision_backbone.requires_grad_(
                not self.freeze_vision_backbone)
        if self.llm_backbone is not None:
            self.llm_backbone.requires_grad_(not self.freeze_llm_backbone)
        if self.vlm_backbone is not None:
            self.vlm_backbone.requires_grad_(not self.freeze_vlm_backbone)
        if self.projector is not None:
            self.projector.requires_grad_(not self.freeze_projector)

        # Add to `self.trainable_module_keys`
        self.trainable_module_keys = []
        if not self.freeze_vision_backbone:
            self.trainable_module_keys.append('vision_backbone')
        if not self.freeze_llm_backbone:
            self.trainable_module_keys.append('llm_backbone')
        if not self.freeze_projector:
            self.trainable_module_keys.append('projector')
        if not self.freeze_vlm_backbone:
            self.trainable_module_keys.append('vlm_backbone')

        # Update Trackers
        self.vision_backbone_requires_grad = not self.freeze_vision_backbone

        # Explicitly Log Frozen / Trainable Components
        if self.vision_backbone is not None:
            if self.freeze_vision_backbone:
                overwatch.info(
                    '[Frozen]    🥶 =>> Vision Backbone', ctx_level=1)
            else:
                overwatch.info(
                    '[TRAINABLE] 🔥 =>> Vision Backbone', ctx_level=1)
        if self.llm_backbone is not None:
            if self.freeze_llm_backbone:
                overwatch.info('[Frozen]    🥶 =>> LLM Backbone', ctx_level=1)
            else:
                overwatch.info('[TRAINABLE] 🔥 =>> LLM Backbone', ctx_level=1)
        if self.vlm_backbone is not None:
            if self.freeze_vlm_backbone:
                overwatch.info('[Frozen]    🥶 =>> VLM Backbone', ctx_level=1)
            else:
                overwatch.info('[TRAINABLE] 🔥 =>> VLM Backbone', ctx_level=1)
        if self.projector is not None:
            if self.freeze_projector:
                overwatch.info('[Frozen]    🥶 =>> Projector', ctx_level=1)
            else:
                overwatch.info('[TRAINABLE] 🔥 =>> Projector', ctx_level=1)

        if self.vision_backbone_fp32:
            self.vision_backbone.dtype = torch.float32

        if self.unfreeze_last_layer:
            for module in self.llm_backbone.last_layer_finetune_modules:
                module.requires_grad_(True)

        overwatch.debug('##################################################')
        overwatch.debug('#####      Trainable Network Parameters:     #####')
        overwatch.debug('##################################################')
        for name, param in self.named_parameters():
            if param.requires_grad:
                overwatch.debug(name)

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        ...

    # === GenerationMixin Expected Properties & Methods (DO NOT MODIFY) ===
    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def config(self) -> PretrainedConfig:
        return self.llm_backbone.llm.config

    # => Beam Search Utility
    def _reorder_cache(self, past_key_values, beam_idx):
        return self.llm_backbone.llm._reorder_cache(past_key_values, beam_idx)

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        ...

    def from_pretrained(self):
        # Load weights based on file format
        if self.pretrained_name_or_path is None:
            return
        if self.pretrained_name_or_path.endswith(
                '.safetensors') or os.path.isdir(self.pretrained_name_or_path):
            # Handle safetensors format
            if self.pretrained_name_or_path.endswith('.safetensors'):
                pretrained_weights = load_file(
                    self.pretrained_name_or_path, device='cpu')
            else:
                # Load from directory containing safetensors files
                pretrained_weights = dict()
                for file in os.listdir(self.pretrained_name_or_path):
                    if file.endswith('.safetensors'):
                        file_path = os.path.join(self.pretrained_name_or_path,
                                                 file)
                        pretrained_weights.update(
                            load_file(file_path, device='cpu'))
        elif self.pretrained_name_or_path.endswith(
                '.pt') or self.pretrained_name_or_path.endswith('.pth'):
            # Handle pt/pth format using torch.load
            checkpoint = torch.load(
                self.pretrained_name_or_path, map_location='cpu')
            # Handle both dict format {'model': state_dict}
            # and direct state_dict
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                pretrained_weights = checkpoint['model']
            else:
                pretrained_weights = checkpoint
        else:
            raise ValueError(f'Unsupported checkpoint format: '
                             f'{self.pretrained_name_or_path}')

        # Load weights with name_mapping handling
        if not self.name_mapping:
            self.load_state_dict(
                pretrained_weights, strict=self.strict_mapping)
        else:
            for name, param in self.named_parameters():
                if self.name_mapping is None:
                    if self.strict_mapping and name not in pretrained_weights:
                        raise ValueError(
                            f"Parameter '{name}' not found in pretrained weights."  # noqa: E501
                        )
                    if name in pretrained_weights and param.size(
                    ) == pretrained_weights[name].size():
                        # Copy the parameter withname the correct dtype
                        with torch.no_grad():
                            param.copy_(pretrained_weights[name].to(
                                param.dtype))
                    else:
                        overwatch.info(
                            f"Parameter '{name}' not found in pretrained weights, skipping."  # noqa: E501
                        )
                else:
                    matched = False
                    for key, val in self.name_mapping.items():
                        if key in name:
                            mapped_name = name.replace(key, val)
                            if mapped_name not in pretrained_weights:
                                continue
                            if matched:
                                raise ValueError(
                                    f"Parameter '{name}' matched multiple times in name_mapping."  # noqa: E501
                                )
                            with torch.no_grad():
                                if param.size(
                                ) == pretrained_weights[mapped_name].size():
                                    param.copy_(
                                        pretrained_weights[mapped_name].to(
                                            param.dtype))
                                else:
                                    continue
                            matched = True
                    if not matched:
                        if self.strict_mapping:
                            raise ValueError(
                                f"Parameter '{name}' not found in pretrained weights with mapping."  # noqa: E501
                            )
                        else:
                            overwatch.info(
                                f"Parameter '{name}' not found in pretrained weights, skipping."  # noqa: E501
                            )  # noqa: E501
