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

// cublasLt-based fused GEMM kernels (single kernel launch each):
//
//   matmul_bias:      D(M,N) = inp(M,K) @ weight(K,N) + bias(N)
//   matmul_bias_res:  D(M,N) = inp(M,K) @ weight(K,N) + bias(N) + res(M,N)
//
// Row-major → cublasLt col-major reinterpretation:
//
//   D'(N,M) = weight'(N,K) @ inp'(K,M) + bias(N)  [+ res'(N,M)]
//
//   cublasLt:  m=N, n=M, k=K
//     A_lt = weight  (col-major N×K, lda=N)
//     B_lt = inp     (col-major K×M, ldb=K)
//     C_lt = res|D   (col-major N×M, ldc=N)    beta=1 when res, beta=0 otherwise
//     D_lt = out     (col-major N×M, ldd=N)
//
// CUBLASLT_EPILOGUE_BIAS adds bias(m=N) to each column, equivalent to
// broadcasting bias[j] across all rows of the row-major output.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <torch/extension.h>

namespace {

cublasLtHandle_t getCublasLtHandle() {
  static cublasLtHandle_t handle = nullptr;
  if (!handle) {
    cublasStatus_t st = cublasLtCreate(&handle);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasLtCreate failed: ", static_cast<int>(st));
  }
  return handle;
}

// Unified cublasLt GEMM + fused bias epilogue.
//
//   D = alpha * (inp @ weight) + beta * C + bias
//
// For matmul_bias:     beta=0, C is ignored (pass D).
// For matmul_bias_res: beta=1, C=res.
void matmul_bias_gemm(at::Tensor inp,     // (M, K)  bf16 contiguous
                      at::Tensor weight,  // (K, N)  bf16 contiguous
                      at::Tensor bias,    // (N,)    bf16 contiguous
                      at::Tensor C,       // (M, N)  bf16 contiguous  — residual or same as D
                      at::Tensor D,       // (M, N)  bf16 contiguous  — output
                      float beta, cudaStream_t stream) {
  const int64_t M = inp.size(0);
  const int64_t K = inp.size(1);
  const int64_t N = weight.size(1);

  cublasLtHandle_t ltHandle = getCublasLtHandle();
  float alpha = 1.0f;

  cublasLtMatmulDesc_t opDesc;
  cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                 sizeof(epilogue));

  const void* bias_ptr = bias.data_ptr();
  cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr,
                                 sizeof(bias_ptr));

  cudaDataType_t biasType = CUDA_R_16BF;
  cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType,
                                 sizeof(biasType));

  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, N, K, N);
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, K, M, K);
  cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, N, M, N);
  cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, N, M, N);

  cublasStatus_t status =
      cublasLtMatmul(ltHandle, opDesc, &alpha, weight.data_ptr(), Adesc, inp.data_ptr(), Bdesc,
                     &beta, C.data_ptr(), Cdesc, D.data_ptr(), Ddesc, nullptr, nullptr, 0, stream);

  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed with status ",
              static_cast<int>(status));

  cublasLtMatrixLayoutDestroy(Ddesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatmulDescDestroy(opDesc);
}

}  // namespace

// Public entry points called from the C++ wrapper (.cpp)

void matmul_bias_forward_kernel_launcher(at::Tensor inp, at::Tensor weight, at::Tensor bias,
                                         at::Tensor out, cudaStream_t stream) {
  matmul_bias_gemm(inp, weight, bias, /*C=*/out, /*D=*/out, 0.0f, stream);
}

void matmul_bias_res_forward_kernel_launcher(at::Tensor inp, at::Tensor weight, at::Tensor bias,
                                             at::Tensor res, at::Tensor out, cudaStream_t stream) {
  matmul_bias_gemm(inp, weight, bias, /*C=*/res, /*D=*/out, 1.0f, stream);
}
