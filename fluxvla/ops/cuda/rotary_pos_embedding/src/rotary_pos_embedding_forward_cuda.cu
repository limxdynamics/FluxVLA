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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#define THREADS_PER_BLOCK 128
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

template <typename T>
__global__ void rotary_pos_embedding_forward_kernel(int batch_size, int num_heads, int num_pos,
                                                    int num_channels, T *q_tensor, T *k_tensor,
                                                    T *cos_tensor, T *sin_tensor,
                                                    T *q_embed_output_tensor,
                                                    T *k_embed_output_tensor) {
  int block_idx = blockIdx.x;
  int thread_idx = threadIdx.x;

  // Each block handles one (batch, head, pos) combination
  int batch_id = block_idx / (num_heads * num_pos);
  int remainder = block_idx % (num_heads * num_pos);
  int head_id = remainder / num_pos;
  int pos_id = remainder % num_pos;

  // Boundary check
  if (batch_id >= batch_size)
    return;

  int half_channels = num_channels / 2;

  // Each thread may process multiple channels (when num_channels > THREADS_PER_BLOCK)
  for (int channel_id = thread_idx; channel_id < num_channels; channel_id += blockDim.x) {
    // Compute q/k tensor index: [batch, head, pos, channel]
    int q_idx = ((batch_id * num_heads + head_id) * num_pos + pos_id) * num_channels + channel_id;
    int k_idx = (batch_id * num_pos + pos_id) * num_channels + channel_id;
    // cos/sin tensor index: [batch, 1, pos, channel] - head dimension is broadcast
    int cs_idx = (batch_id * num_pos + pos_id) * num_channels + channel_id;

    T cos_val = cos_tensor[cs_idx];
    T sin_val = sin_tensor[cs_idx];

    T q_val = q_tensor[q_idx];
    T k_val = k_tensor[k_idx];

    // Get the value at the corresponding rotate_half position
    // rotate_half(x)[i] = -x[i + half] if i < half
    // rotate_half(x)[i] = x[i - half]  if i >= half
    int paired_channel_id;
    T q_rotated, k_rotated;

    if (channel_id < half_channels) {
      // First half: take values from the second half and negate
      paired_channel_id = channel_id + half_channels;
      int paired_q_idx =
          ((batch_id * num_heads + head_id) * num_pos + pos_id) * num_channels + paired_channel_id;
      int paired_k_idx = (batch_id * num_pos + pos_id) * num_channels + paired_channel_id;
      q_rotated = -q_tensor[paired_q_idx];
      k_rotated = -k_tensor[paired_k_idx];
    } else {
      // Second half: take values from the first half
      paired_channel_id = channel_id - half_channels;
      int paired_q_idx =
          ((batch_id * num_heads + head_id) * num_pos + pos_id) * num_channels + paired_channel_id;
      int paired_k_idx = (batch_id * num_pos + pos_id) * num_channels + paired_channel_id;
      q_rotated = q_tensor[paired_q_idx];
      k_rotated = k_tensor[paired_k_idx];
    }

    // q_embed = q * cos + rotate_half(q) * sin
    // k_embed = k * cos + rotate_half(k) * sin

    q_embed_output_tensor[q_idx] = q_val * cos_val + q_rotated * sin_val;
    k_embed_output_tensor[k_idx] = k_val * cos_val + k_rotated * sin_val;
  }
}

void rotary_pos_embedding_forward_kernel_launcher(
    int batch_size, int num_heads, int num_pos, int num_channels, at::Tensor q_tensor,
    at::Tensor k_tensor, at::Tensor cos_tensor, at::Tensor sin_tensor,
    at::Tensor q_embed_output_tensor, at::Tensor k_embed_output_tensor, cudaStream_t stream) {
  cudaError_t err;

  // Each block handles one (batch, head, pos) combination
  int total_blocks = batch_size * num_heads * num_pos;

  // Thread count: process num_channels elements, capped at THREADS_PER_BLOCK
  int threads_per_block = (num_channels < THREADS_PER_BLOCK) ? num_channels : THREADS_PER_BLOCK;

  dim3 blocks(total_blocks);
  dim3 threads(threads_per_block);

  // Select kernel based on the input tensor's dtype
  if (q_tensor.scalar_type() == at::ScalarType::BFloat16) {
    // bf16 version
    __nv_bfloat16 *q_ptr = reinterpret_cast<__nv_bfloat16 *>(q_tensor.data_ptr<at::BFloat16>());
    __nv_bfloat16 *k_ptr = reinterpret_cast<__nv_bfloat16 *>(k_tensor.data_ptr<at::BFloat16>());
    __nv_bfloat16 *cos_ptr = reinterpret_cast<__nv_bfloat16 *>(cos_tensor.data_ptr<at::BFloat16>());
    __nv_bfloat16 *sin_ptr = reinterpret_cast<__nv_bfloat16 *>(sin_tensor.data_ptr<at::BFloat16>());
    __nv_bfloat16 *q_embed_ptr =
        reinterpret_cast<__nv_bfloat16 *>(q_embed_output_tensor.data_ptr<at::BFloat16>());
    __nv_bfloat16 *k_embed_ptr =
        reinterpret_cast<__nv_bfloat16 *>(k_embed_output_tensor.data_ptr<at::BFloat16>());
    rotary_pos_embedding_forward_kernel<__nv_bfloat16>
        <<<blocks, threads, 0, stream>>>(batch_size, num_heads, num_pos, num_channels, q_ptr, k_ptr,
                                         cos_ptr, sin_ptr, q_embed_ptr, k_embed_ptr);
  } else {
    float *q_ptr = q_tensor.data_ptr<float>();
    float *k_ptr = k_tensor.data_ptr<float>();
    float *cos_ptr = cos_tensor.data_ptr<float>();
    float *sin_ptr = sin_tensor.data_ptr<float>();
    float *q_embed_ptr = q_embed_output_tensor.data_ptr<float>();
    float *k_embed_ptr = k_embed_output_tensor.data_ptr<float>();

    rotary_pos_embedding_forward_kernel<float>
        <<<blocks, threads, 0, stream>>>(batch_size, num_heads, num_pos, num_channels, q_ptr, k_ptr,
                                         cos_ptr, sin_ptr, q_embed_ptr, k_embed_ptr);
  }

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
