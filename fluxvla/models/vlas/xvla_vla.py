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

from typing import Dict, List, Optional

import torch

from fluxvla.engines import VLAS
from .llava_vla import LlavaVLA


@VLAS.register_module()
class XVLAVla(LlavaVLA):
    """
    X-VLA integration for FluxVLA benchmark.

    Extends LlavaVLA with two changes:
    1. Passes aux_visual_inputs (third return of Florence2Backbone) to the head.
       LlavaVLA discards it; XVLAFlowMatchingHead needs it for multi-view encoding.
    2. Overrides all_module_keys to ['vlm_backbone', 'vla_head'] so the runner
       saves the correct modules (OpenVLA defaults to vision/llm/projector/head).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # base_train_runner.py:118 reads this for checkpoint saving
        self.all_module_keys = ['vlm_backbone', 'vla_head']

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
        last_hidden_state, fused_attention_mask, aux_visual_inputs = self.vlm_backbone(
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
        last_hidden_state, fused_attention_mask, aux_visual_inputs = self.vlm_backbone(
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
        )
        return pred_actions.float()
