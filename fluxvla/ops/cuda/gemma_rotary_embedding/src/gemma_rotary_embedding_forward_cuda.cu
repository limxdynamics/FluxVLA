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
#include <cuda_fp16.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 128
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

template <typename T>
__global__ void gemma_rotary_embedding_forward_kernel(int batch_size, int num_pos, int num_channels,
                                                      float attention_scaling, const T *inv_freq,
                                                      const int *position_ids,
                                                      T *cos_output_features,
                                                      T *sin_output_features) {
  // Cache position_ids in shared memory
  extern __shared__ int shared_position_ids[];

  int thread_idx = threadIdx.x;
  int block_idx = blockIdx.x;
  int total_positions = num_pos * batch_size;
  int global_pos_idx = block_idx * blockDim.x + thread_idx;

  // Cooperatively load position_ids into shared memory (coalesced access)
  if (global_pos_idx < total_positions) {
    shared_position_ids[thread_idx] = position_ids[global_pos_idx];
  }
  __syncthreads();

  // Compute the number of valid positions for the current block
  int block_start = block_idx * blockDim.x;
  int valid_positions = min(static_cast<int>(blockDim.x), total_positions - block_start);

  // Iterate over all valid positions within the block
  for (int i = 0; i < valid_positions; i++) {
    int position_id = shared_position_ids[i];
    int curr_global_idx = block_start + i;
    int batch_idx = curr_global_idx / num_pos;
    int pos_idx = curr_global_idx % num_pos;

    // Each thread processes a subset of channels
    for (int j = 0; j < DIVUP(num_channels, THREADS_PER_BLOCK); j++) {
      int channel_idx = j * blockDim.x + thread_idx;
      if (channel_idx >= num_channels)
        break;

      T freq = inv_freq[channel_idx];
      T angle = freq * position_id;
      T cos_val = cos(angle) * attention_scaling;
      T sin_val = sin(angle) * attention_scaling;

      int output_offset =
          batch_idx * num_pos * num_channels * 2 + pos_idx * num_channels * 2 + channel_idx;

      // Write the same value to both the first and second halves
      cos_output_features[output_offset] = cos_val;
      cos_output_features[output_offset + num_channels] = cos_val;
      sin_output_features[output_offset] = sin_val;
      sin_output_features[output_offset + num_channels] = sin_val;
    }
  }
}

void gemma_rotary_embedding_forward_kernel_launcher(int batch_size, int num_pos, int num_channels,
                                                    float attention_scaling, const float *inv_freq,
                                                    const int *position_ids,
                                                    float *cos_output_features,
                                                    float *sin_output_features,
                                                    cudaStream_t stream) {
  cudaError_t err;

  int total_positions = batch_size * num_pos;
  // Use DIVUP to round up, avoiding zero blocks
  dim3 blocks(DIVUP(total_positions, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  // Shared memory size: store THREADS_PER_BLOCK position_ids
  size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(int);

  gemma_rotary_embedding_forward_kernel<float><<<blocks, threads, shared_mem_size, stream>>>(
      batch_size, num_pos, num_channels, attention_scaling, inv_freq, position_ids,
      cos_output_features, sin_output_features);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
