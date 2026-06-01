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
"""Utilities for handling shared tensors in model checkpoints."""

from typing import Dict

import torch


def handle_shared_tensors(state_dict: Dict[str, torch.Tensor],
                          model_state_dict: Dict[str, torch.Tensor],
                          overwatch=None) -> Dict[str, torch.Tensor]:
    """Restore omitted shared tensors before strict checkpoint loading.

    Some models have parameters that share storage. Safetensors checkpoints
    often contain only one copy of such tied weights; this helper restores the
    missing alias expected by the current model state dict.
    """
    model_keys = set(model_state_dict.keys())
    loaded_keys = set(state_dict.keys())
    missing_keys = model_keys - loaded_keys

    if not missing_keys:
        return state_dict

    shared_pairs = [
        ('vlm_backbone.vlm.language_model.model.embed_tokens.weight',
         'vlm_backbone.vlm.language_model.lm_head.weight'),
    ]

    for key1, key2 in shared_pairs:
        if key1 in missing_keys and key2 in loaded_keys:
            if overwatch:
                overwatch.info(f'Restoring shared tensor: {key1} <- {key2}')
            state_dict[key1] = state_dict[key2]
        elif key2 in missing_keys and key1 in loaded_keys:
            if overwatch:
                overwatch.info(f'Restoring shared tensor: {key2} <- {key1}')
            state_dict[key2] = state_dict[key1]

    return state_dict
