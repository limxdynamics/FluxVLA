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
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <vector>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int rotary_pos_embedding_forward_wrapper(int batch_size, int num_heads, int num_pos,
                                         int num_channels, at::Tensor q_tensor, at::Tensor k_tensor,
                                         at::Tensor cos_tensor, at::Tensor sin_tensor,
                                         at::Tensor q_embed_output_tensor,
                                         at::Tensor k_embed_output_tensor);

void rotary_pos_embedding_forward_kernel_launcher(
    int batch_size, int num_heads, int num_pos, int num_channels, at::Tensor q_tensor,
    at::Tensor k_tensor, at::Tensor cos_tensor, at::Tensor sin_tensor,
    at::Tensor q_embed_output_tensor, at::Tensor k_embed_output_tensor, cudaStream_t stream);

int rotary_pos_embedding_forward_wrapper(int batch_size, int num_heads, int num_pos,
                                         int num_channels, at::Tensor q_tensor, at::Tensor k_tensor,
                                         at::Tensor cos_tensor, at::Tensor sin_tensor,
                                         at::Tensor q_embed_output_tensor,
                                         at::Tensor k_embed_output_tensor) {
  CHECK_INPUT(q_tensor);
  CHECK_INPUT(k_tensor);
  CHECK_INPUT(cos_tensor);
  CHECK_INPUT(sin_tensor);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  rotary_pos_embedding_forward_kernel_launcher(
      batch_size, num_heads, num_pos, num_channels, q_tensor, k_tensor, cos_tensor, sin_tensor,
      q_embed_output_tensor, k_embed_output_tensor, stream);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rotary_pos_embedding_forward_wrapper", &rotary_pos_embedding_forward_wrapper,
        "rotary_pos_embedding_forward_wrapper");
}
