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

import torch
from torch.distributed.fsdp import StateDictType


def str_to_dtype(s: str):
    mapping = {
        'fp32': torch.float32,
        'float32': torch.float32,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'uint8': torch.uint8,
        'bool': torch.bool
    }
    s = s.lower()
    if s in mapping:
        return mapping[s]
    else:
        raise ValueError(f'Unsupported dtype string: {s}')


def state_dict_type_map(s: str):
    if s == 'full_state_dict':
        return StateDictType.FULL_STATE_DICT
    if s == 'local_state_dict':
        return StateDictType.LOCAL_STATE_DICT
    if s == 'sharded_state_dict':
        return StateDictType.SHARDED_STATE_DICT
    assert False, f'Unsupported state dict type: {s}'
