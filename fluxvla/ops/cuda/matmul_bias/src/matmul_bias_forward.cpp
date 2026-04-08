// Copyright 2026 Limx Dynamics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime_api.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void matmul_bias_forward_kernel_launcher(at::Tensor inp, at::Tensor weight, at::Tensor bias,
                                         at::Tensor out, cudaStream_t stream);

void matmul_bias_res_forward_kernel_launcher(at::Tensor inp, at::Tensor weight, at::Tensor bias,
                                             at::Tensor res, at::Tensor out, cudaStream_t stream);

static void check_inputs(at::Tensor inp, at::Tensor weight, at::Tensor bias) {
  CHECK_INPUT(inp);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);
  TORCH_CHECK(inp.scalar_type() == at::kBFloat16, "inp must be bf16");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bf16");
  TORCH_CHECK(bias.scalar_type() == at::kBFloat16, "bias must be bf16");
  TORCH_CHECK(inp.dim() == 2, "inp must be 2D (M, K)");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D (K, N)");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D (N,)");
  TORCH_CHECK(inp.size(1) == weight.size(0), "inp K-dim must match weight K-dim");
  TORCH_CHECK(weight.size(1) == bias.size(0), "weight N-dim must match bias dim");
}

static void check_out(at::Tensor out, int64_t M, int64_t N) {
  CHECK_INPUT(out);
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bf16");
  TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape must be (M, N)");
}

// ---------------------------------------------------------------------------
// matmul_bias:  out = inp @ weight + bias
// ---------------------------------------------------------------------------

at::Tensor matmul_bias_forward_wrapper(at::Tensor inp, at::Tensor weight, at::Tensor bias) {
  check_inputs(inp, weight, bias);
  auto out = at::empty({inp.size(0), weight.size(1)}, inp.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  matmul_bias_forward_kernel_launcher(inp, weight, bias, out, stream);
  return out;
}

void matmul_bias_forward_out_wrapper(at::Tensor inp, at::Tensor weight, at::Tensor bias,
                                     at::Tensor out) {
  check_inputs(inp, weight, bias);
  check_out(out, inp.size(0), weight.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  matmul_bias_forward_kernel_launcher(inp, weight, bias, out, stream);
}

// ---------------------------------------------------------------------------
// matmul_bias_res:  out = inp @ weight + bias + res
// ---------------------------------------------------------------------------

at::Tensor matmul_bias_res_forward_wrapper(at::Tensor inp, at::Tensor weight, at::Tensor bias,
                                           at::Tensor res) {
  check_inputs(inp, weight, bias);
  CHECK_INPUT(res);
  TORCH_CHECK(res.scalar_type() == at::kBFloat16, "res must be bf16");
  TORCH_CHECK(res.size(0) == inp.size(0) && res.size(1) == weight.size(1),
              "res shape must be (M, N)");
  auto out = at::empty_like(res);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  matmul_bias_res_forward_kernel_launcher(inp, weight, bias, res, out, stream);
  return out;
}

void matmul_bias_res_forward_out_wrapper(at::Tensor inp, at::Tensor weight, at::Tensor bias,
                                         at::Tensor res, at::Tensor out) {
  check_inputs(inp, weight, bias);
  CHECK_INPUT(res);
  TORCH_CHECK(res.scalar_type() == at::kBFloat16, "res must be bf16");
  TORCH_CHECK(res.size(0) == inp.size(0) && res.size(1) == weight.size(1),
              "res shape must be (M, N)");
  check_out(out, inp.size(0), weight.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  matmul_bias_res_forward_kernel_launcher(inp, weight, bias, res, out, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_bias_forward_wrapper", &matmul_bias_forward_wrapper,
        "D = inp @ weight + bias  (allocates output)");
  m.def("matmul_bias_forward_out_wrapper", &matmul_bias_forward_out_wrapper,
        "D = inp @ weight + bias  (pre-allocated output)");
  m.def("matmul_bias_res_forward_wrapper", &matmul_bias_res_forward_wrapper,
        "D = inp @ weight + bias + res  (allocates output)");
  m.def("matmul_bias_res_forward_out_wrapper", &matmul_bias_res_forward_out_wrapper,
        "D = inp @ weight + bias + res  (pre-allocated output)");
}
