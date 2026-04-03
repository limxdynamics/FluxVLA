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

int gemma_rotary_embedding_forward_wrapper(int batch_size, int num_positions, int num_channels,
                                           float attention_scaling, at::Tensor position_ids_tensor,
                                           at::Tensor inv_freq_tensor,
                                           at::Tensor cos_output_features_tensor,
                                           at::Tensor sin_output_features_tensor);

void gemma_rotary_embedding_forward_kernel_launcher(int batch_size, int num_pos, int num_channels,
                                                    float attention_scaling, const float *inv_freq,
                                                    const int *position_ids,
                                                    float *cos_output_features,
                                                    float *sin_output_features,
                                                    cudaStream_t stream);

int gemma_rotary_embedding_forward_wrapper(int batch_size, int num_pos, int num_channels,
                                           float attention_scaling, at::Tensor position_ids_tensor,
                                           at::Tensor inv_freq_tensor,
                                           at::Tensor cos_output_features_tensor,
                                           at::Tensor sin_output_features_tensor) {
  CHECK_INPUT(position_ids_tensor);
  CHECK_INPUT(inv_freq_tensor);
  CHECK_INPUT(cos_output_features_tensor);
  CHECK_INPUT(sin_output_features_tensor);
  const int *position_ids = position_ids_tensor.data_ptr<int>();
  const float *inv_freq = inv_freq_tensor.data_ptr<float>();
  float *cos_output_features = cos_output_features_tensor.data_ptr<float>();
  float *sin_output_features = sin_output_features_tensor.data_ptr<float>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  gemma_rotary_embedding_forward_kernel_launcher(batch_size, num_pos, num_channels,
                                                 attention_scaling, inv_freq, position_ids,
                                                 cos_output_features, sin_output_features, stream);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemma_rotary_embedding_forward_wrapper", &gemma_rotary_embedding_forward_wrapper,
        "gemma_rotary_embedding_forward_wrapper");
}
