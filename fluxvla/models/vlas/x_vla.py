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
# Notes: Refactored from X-VLA's monolithic model into FluxVLA's backbone /
# head / wrapper split while preserving the external XVLA data flow.

from functools import partial
from typing import Callable, Dict, List, Optional

import torch
from torch.distributed.fsdp.wrap import _or_policy
from transformers import PretrainedConfig

from fluxvla.engines import VLAS
from .base_vla import BaseVLA


@VLAS.register_module()
class X_VLA(BaseVLA):
    """Training-time X_VLA wrapper.

    This keeps the XVLA-specific multi-view backbone/head data plumbing in a
    dedicated VLA class.
    """

    def __init__(self,
                 vlm_backbone: Dict,
                 vla_head: Dict,
                 enable_mixed_precision_training: bool = True,
                 freeze_vlm_backbone: bool = False,
                 vision_backbone_fp32: bool = False,
                 unfreeze_last_layer: bool = False,
                 ignore_index: int = -100,
                 norm_stats: Dict = None,
                 pretrained_name_or_path: str = None,
                 name_mapping: Dict = None,
                 strict_mapping: bool = False,
                 *args,
                 **kwargs):
        del args, kwargs
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
        self.all_module_keys = ['vlm_backbone', 'vla_head']

    @property
    def config(self) -> PretrainedConfig:
        return self.vlm_backbone.config

    def _reorder_cache(self, past_key_values, beam_idx):
        del beam_idx
        return past_key_values

    def _forward_backbone(
        self,
        images: Optional[torch.Tensor],
        lang_tokens: Optional[torch.Tensor],
        img_masks: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
    ):
        return self.vlm_backbone(
            images=images,
            lang_tokens=lang_tokens,
            img_masks=img_masks,
            lang_masks=lang_masks,
        )

    def forward(
        self,
        lang_tokens: Optional[torch.LongTensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        img_masks: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.LongTensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict:
        del kwargs
        last_hidden_state, fused_attention_mask, aux_visual_inputs = \
            self._forward_backbone(
                images=images,
                lang_tokens=lang_tokens,
                img_masks=img_masks,
                lang_masks=lang_masks,
            )
        return self.vla_head(
            input_features=last_hidden_state,
            states=states,
            attention_mask=fused_attention_mask,
            actions=actions,
            action_masks=action_masks,
            embodiment_ids=embodiment_ids,
            aux_visual_inputs=aux_visual_inputs,
        )

    def predict_action(
        self,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        states: torch.Tensor,
        img_masks: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        last_hidden_state, fused_attention_mask, aux_visual_inputs = \
            self._forward_backbone(
                images=images,
                lang_tokens=lang_tokens,
                img_masks=img_masks,
                lang_masks=lang_masks,
            )
        pred_actions = self.vla_head.predict_action(
            input_features=last_hidden_state,
            states=states,
            attention_mask=fused_attention_mask,
            embodiment_ids=embodiment_ids,
            aux_visual_inputs=aux_visual_inputs,
            **kwargs,
        )
        return pred_actions.float()

    def get_fsdp_wrapping_policy(self) -> Callable:
        wrapping_policies = []
        if (self.vlm_backbone is not None
                and hasattr(self.vlm_backbone, 'get_fsdp_wrapping_policy')):
            wrapping_policies.append(
                self.vlm_backbone.get_fsdp_wrapping_policy())
        if self.vla_head is not None and hasattr(self.vla_head,
                                                 'get_fsdp_wrapping_policy'):
            wrapping_policies.append(self.vla_head.get_fsdp_wrapping_policy())
        if not wrapping_policies:
            raise ValueError('X_VLA could not build any FSDP wrapping policy.')
        return partial(_or_policy, policies=wrapping_policies)

    def get_lr_param_group_strategy(self, learning_rate: float, lr_coef: float,
                                    weight_decay: Optional[float],
                                    canonicalize_param_name) -> List[Dict]:
        """Provide XVLA's canonical parameter grouping for groupwise LR."""
        wd = weight_decay if weight_decay is not None else 0.0
        vlm_ids, soft_ids, action_ids, core_ids = set(), set(), set(), set()
        vlm_params, soft_params, action_params, core_params = [], [], [], []
        soft_prompt_prefix = 'vla_head.transformer.soft_prompt_hub.'
        action_prefixes = (
            'vla_head.transformer.action_encoder.',
            'vla_head.transformer.action_decoder.',
        )

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            canonical_name = canonicalize_param_name(name)
            pid = id(param)

            if canonical_name.startswith('vlm_backbone.'):
                if pid not in vlm_ids:
                    vlm_ids.add(pid)
                    vlm_params.append(param)
            elif canonical_name.startswith(soft_prompt_prefix):
                if pid not in soft_ids:
                    soft_ids.add(pid)
                    soft_params.append(param)
            elif canonical_name.startswith(action_prefixes):
                if pid not in action_ids:
                    action_ids.add(pid)
                    action_params.append(param)
            else:
                if pid not in core_ids:
                    core_ids.add(pid)
                    core_params.append(param)

        return [
            {
                'name': 'vlm',
                'params': vlm_params,
                'lr': 0.0,
                'weight_decay': wd
            },
            {
                'name': 'transformer_core',
                'params': core_params,
                'lr': 0.0,
                'weight_decay': wd
            },
            {
                'name': 'soft_prompts',
                'params': soft_params,
                'lr': learning_rate * lr_coef,
                'weight_decay': wd
            },
            {
                'name': 'action_heads',
                'params': action_params,
                'lr': learning_rate,
                'weight_decay': wd
            },
        ]


__all__ = [
    'X_VLA',
]
