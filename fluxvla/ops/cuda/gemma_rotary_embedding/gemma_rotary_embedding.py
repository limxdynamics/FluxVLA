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
from torch.autograd import Function

from . import gemma_rotary_embedding_ext


class GemmaRotaryEmbedding(Function):

    @staticmethod
    def forward(ctx, position_ids: torch.Tensor, inv_freq: torch.Tensor,
                attention_scaling: float) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure position_ids is int32 type as expected by CUDA kernel
        position_ids = position_ids.to(dtype=torch.int32).contiguous()
        assert inv_freq.is_contiguous()
        batch_size = position_ids.shape[0]
        num_pos = position_ids.shape[1]
        num_channels = inv_freq.shape[0]
        cos_output_features = inv_freq.new_zeros(batch_size, num_pos,
                                                 num_channels * 2)
        sin_output_features = inv_freq.new_zeros(batch_size, num_pos,
                                                 num_channels * 2)
        gemma_rotary_embedding_ext.gemma_rotary_embedding_forward_wrapper(
            batch_size,
            num_pos,
            num_channels,
            attention_scaling,
            position_ids,
            inv_freq,
            cos_output_features,
            sin_output_features,
        )
        return cos_output_features, sin_output_features

    @staticmethod
    def backward(ctx, grad_output_features):
        raise NotImplementedError('Backward pass is not implemented yet')


gemma_rotary_embedding_cuda = GemmaRotaryEmbedding.apply
