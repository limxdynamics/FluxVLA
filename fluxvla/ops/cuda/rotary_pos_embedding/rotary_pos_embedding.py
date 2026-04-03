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

from . import rotary_pos_embedding_ext


class RotaryPosEmbedding(Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure position_ids is int32 type as expected by CUDA kernel
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        num_pos = q.shape[2]
        num_channels = q.shape[3]
        q_embed_output = torch.zeros_like(q)
        k_embed_output = torch.zeros_like(k)
        rotary_pos_embedding_ext.rotary_pos_embedding_forward_wrapper(
            batch_size, num_heads, num_pos, num_channels, q, k, cos, sin,
            q_embed_output, k_embed_output)
        return q_embed_output, k_embed_output

    @staticmethod
    def backward(ctx, grad_output_features):
        raise NotImplementedError('Backward pass is not implemented yet')


rotary_pos_embedding_cuda = RotaryPosEmbedding.apply
