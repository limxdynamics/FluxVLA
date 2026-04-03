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

from . import matmul_bias_ext


def matmul_bias_cuda(inp: torch.Tensor,
                     weight: torch.Tensor,
                     bias: torch.Tensor,
                     out: torch.Tensor | None = None,
                     res: torch.Tensor | None = None) -> torch.Tensor:
    """cublasLt fused GEMM + bias [+ residual] (single kernel launch).

    Without res:  out(M,N) = inp(M,K) @ weight(K,N) + bias(N)
    With    res:  out(M,N) = inp(M,K) @ weight(K,N) + bias(N) + res(M,N)

    All tensors must be contiguous bf16 on the same CUDA device.
    Accepts 2D or 3D inp/res/out; 3D tensors are flattened to 2D internally.

    Args:
        out: optional pre-allocated (M, N) bf16 tensor (for CUDA Graph).
        res: optional residual (M, N) bf16 tensor added via beta=1.
    """
    orig_shape = inp.shape
    if inp.dim() == 3:
        inp = inp.reshape(-1, orig_shape[-1])
    assert inp.dim() == 2 and inp.is_contiguous()

    M, K = inp.shape
    N = weight.size(1)

    if res is not None:
        res_2d = res.reshape(M, N) if res.dim() != 2 else res

        if out is not None:
            out_2d = out.reshape(M, N) if out.dim() != 2 else out
            matmul_bias_ext.matmul_bias_res_forward_out_wrapper(
                inp, weight, bias, res_2d, out_2d)
            return out
        return matmul_bias_ext.matmul_bias_res_forward_wrapper(
            inp, weight, bias, res_2d)

    if out is not None:
        out_2d = out.reshape(M, N) if out.dim() != 2 else out
        matmul_bias_ext.matmul_bias_forward_out_wrapper(
            inp, weight, bias, out_2d)
        return out
    return matmul_bias_ext.matmul_bias_forward_wrapper(inp, weight, bias)
