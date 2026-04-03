# Origin: Modified from
# Upstream-URL: https://github.com/dexmal/realtime-vla/blob/main/pi0_infer.py
# Upstream-Ref: main
# SPDX-License-Identifier: MIT
# Notes: Attribution normalized; no functional change.

import triton
import triton.language as tl


@triton.jit
def matmul_small_bias(inp_ptr, weight_ptr, out_ptr, bias_ptr,
                      seq_len: tl.constexpr, features: tl.constexpr,
                      hidden: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


@triton.jit
def matmul_small_bias_res(inp_ptr, weight_ptr, out_ptr, bias_ptr, res_ptr,
                          seq_len: tl.constexpr, features: tl.constexpr,
                          hidden: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0).to(tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0).to(tl.bfloat16)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0).to(tl.bfloat16)
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


@triton.jit
def matmul_small_bias_res_mod(inp_ptr, weight_ptr, out_ptr, bias_ptr, res_ptr,
                              seq_len: tl.constexpr, features: tl.constexpr,
                              hidden: tl.constexpr, i_mod: tl.constexpr,
                              BLOCK_SIZE_N: tl.constexpr,
                              BLOCK_SIZE_M: tl.constexpr,
                              BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr +
            ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] % i_mod) * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0).to(tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


@triton.jit
def matmul_512x1152x1152_twopart_bias_res(old_ptr, inp_ptr, weight_ptr,
                                          bias_ptr, out_ptr, out2_ptr,
                                          seq_len: tl.constexpr,
                                          features: tl.constexpr,
                                          hidden: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    hidden1 = 1024

    BLOCK_SIZE_N1: tl.constexpr = 64
    BLOCK_SIZE_M1: tl.constexpr = 64
    BLOCK_SIZE_K1: tl.constexpr = 32
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N1)
    grid_j = tl.cdiv(hidden1, BLOCK_SIZE_M1)

    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N1
        j = (p % grid_j) * BLOCK_SIZE_M1
        acc = tl.load(old_ptr +
                      (i + tl.arange(0, BLOCK_SIZE_N1))[:, None] * hidden +
                      (j + tl.arange(0, BLOCK_SIZE_M1))[None, :]).to(
                          tl.float32)
        acc += tl.load(bias_ptr +
                       (j + tl.arange(0, BLOCK_SIZE_M1))[None, :]).to(
                           tl.float32)
        for k in range(0, features, BLOCK_SIZE_K1):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N1))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K1))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N1))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K1))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K1))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M1))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K1))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M1))[None, :] < hidden),
                other=0.0)
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N1))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M1))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N1))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M1))[None, :] < hidden))

    BLOCK_SIZE_N2: tl.constexpr = 32
    BLOCK_SIZE_M2: tl.constexpr = 32
    BLOCK_SIZE_K2: tl.constexpr = 32
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N2)
    grid_j = tl.cdiv(hidden - hidden1, BLOCK_SIZE_M2)
    grid_k = 4
    for p in range(pid, grid_i * grid_j * grid_k, psize):
        i = (p // (grid_j * grid_k)) * BLOCK_SIZE_N2
        j = ((p // grid_k) % grid_j) * BLOCK_SIZE_M2 + hidden1
        k0 = p % grid_k
        acc = tl.zeros((BLOCK_SIZE_N2, BLOCK_SIZE_M2), dtype=tl.float32)
        if k0 == 0:
            acc += tl.load(old_ptr +
                           (i + tl.arange(0, BLOCK_SIZE_N2))[:, None] *
                           hidden +
                           (j + tl.arange(0, BLOCK_SIZE_M2))[None, :]).to(
                               tl.float32)
            acc += tl.load(bias_ptr +
                           (j + tl.arange(0, BLOCK_SIZE_M2))[None, :]).to(
                               tl.float32)
        for k in range(k0 * BLOCK_SIZE_K2, features, BLOCK_SIZE_K2 * grid_k):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N2))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K2))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N2))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K2))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K2))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M2))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K2))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M2))[None, :] < hidden),
                other=0.0)
            acc = tl.dot(x, w, acc)
        tl.store(
            out2_ptr + (i + tl.arange(0, BLOCK_SIZE_N2))[:, None] *
            ((hidden - hidden1) * grid_k) +
            (j - hidden1 + tl.arange(0, BLOCK_SIZE_M2))[None, :] + k0 *
            (hidden - hidden1),
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N2))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M2))[None, :] < hidden))


@triton.jit
def matmul_small_bias_gelu(inp_ptr, weight_ptr, out_ptr, bias_ptr,
                           seq_len: tl.constexpr, features: tl.constexpr,
                           hidden: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                           BLOCK_SIZE_M: tl.constexpr,
                           BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0).to(tl.bfloat16)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0).to(tl.bfloat16)
            acc = tl.dot(x, w, acc)
        acc = acc * tl.sigmoid(1.5957691216057308 * acc *
                               (1 + 0.044715 * acc * acc))
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


@triton.jit
def matmul_split_k(inp_ptr, weight_ptr, out_ptr, seq_len: tl.constexpr,
                   features: tl.constexpr, hidden: tl.constexpr,
                   BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                   BLOCK_SIZE_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    grid_k = SPLIT_K
    for p in range(pid, grid_i * grid_j * grid_k, psize):
        i = (p // (grid_j * grid_k)) * BLOCK_SIZE_N
        j = (p // grid_k % grid_j) * BLOCK_SIZE_M
        k_s = p % grid_k
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        k_beg = k_s * tl.cdiv(features, BLOCK_SIZE_K) // SPLIT_K * BLOCK_SIZE_K
        k_end = (k_s + 1) * tl.cdiv(features,
                                    BLOCK_SIZE_K) // SPLIT_K * BLOCK_SIZE_K
        for k in range(k_beg, k_end, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :] + k_s * seq_len * hidden,
            acc,
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


@triton.jit
def merge_split_k_bias_res(out_ptr,
                           bias_ptr,
                           res_ptr,
                           final_out_ptr,
                           seq_len: tl.constexpr,
                           hidden: tl.constexpr,
                           SPLIT_K: tl.constexpr,
                           BLOCK: tl.constexpr = 512):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid * BLOCK, seq_len * hidden, psize * BLOCK):
        acc = tl.load(
            res_ptr + i + tl.arange(0, BLOCK),
            mask=(i + tl.arange(0, BLOCK)) < (seq_len * hidden),
            other=0.0).to(tl.float32)
        acc += tl.load(
            bias_ptr + (i + tl.arange(0, BLOCK)) % hidden,
            mask=(i + tl.arange(0, BLOCK)) < (seq_len * hidden),
            other=0.0).to(tl.float32)
        for k in range(SPLIT_K):
            offset = k * seq_len * hidden
            mask = (i + tl.arange(0, BLOCK)) < (seq_len * hidden)
            vals = tl.load(
                out_ptr + offset + i + tl.arange(0, BLOCK),
                mask=mask,
                other=0.0)
            acc += vals.to(tl.float32)
        mask = (i + tl.arange(0, BLOCK)) < (seq_len * hidden)
        tl.store(
            final_out_ptr + i + tl.arange(0, BLOCK),
            acc.to(tl.bfloat16),
            mask=mask)


@triton.jit
def combine_1536_1152_twopart(out_ptr, inp_ptr, seq_len: tl.constexpr,
                              hidden: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid * 2, seq_len, psize * 2):
        inp = tl.load(inp_ptr + (i + tl.arange(0, 2)[:, None]) * 128 * 4 +
                      tl.arange(0, 128)).to(tl.float32)
        for j in range(1, 4):
            inp += tl.load(inp_ptr + (i + tl.arange(0, 2)[:, None]) * 128 * 4 +
                           tl.arange(0, 128) + 128 * j).to(tl.float32)
        tl.store(
            out_ptr + (i + tl.arange(0, 2)[:, None]) * hidden + 1024 +
            tl.arange(0, 128), inp.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_N': 16,
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_K': 64
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_N': 16,
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_N': 16,
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_K': 64
            },
            num_stages=2,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_K': 64
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_K': 32
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_K': 32
            },
            num_stages=2,
            num_warps=8),
    ],
    key=['seq_len', 'features', 'hidden'],
)
@triton.jit
def qwen3_mlp_gate_up_silu_kernel(inp_ptr, gate_w_ptr, up_w_ptr, out_ptr,
                                  seq_len, features, hidden,
                                  BLOCK_SIZE_N: tl.constexpr,
                                  BLOCK_SIZE_M: tl.constexpr,
                                  BLOCK_SIZE_K: tl.constexpr):
    """Fused gate+up projection with SiLU and element-wise multiply.

    out = silu(inp @ gate_w.T) * (inp @ up_w.T)
    Weight layout: (out_features, in_features), PyTorch nn.Linear native.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(seq_len, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(hidden, BLOCK_SIZE_M)

    GROUP_SIZE: tl.constexpr = 8
    num_pid_in_group = GROUP_SIZE * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE)
    pid_n = first_pid_n + (pid % group_size_n)
    pid_m = (pid % num_pid_in_group) // group_size_n

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_n = offs_n[:, None] < seq_len
    mask_m = offs_m[:, None] < hidden

    gate_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    inp_base = inp_ptr + offs_n[:, None] * features
    gw_base = gate_w_ptr + offs_m[:, None] * features
    uw_base = up_w_ptr + offs_m[:, None] * features

    for k in range(0, features, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k[None, :] < features
        x = tl.load(
            inp_base + offs_k[None, :], mask=mask_n & mask_k, other=0.0)
        gw = tl.load(
            gw_base + offs_k[None, :], mask=mask_m & mask_k, other=0.0)
        uw = tl.load(
            uw_base + offs_k[None, :], mask=mask_m & mask_k, other=0.0)
        gate_acc = tl.dot(x, tl.trans(gw), gate_acc)
        up_acc = tl.dot(x, tl.trans(uw), up_acc)

    result = (gate_acc * tl.sigmoid(gate_acc)) * up_acc
    out_ptrs = out_ptr + offs_n[:, None] * hidden + offs_m[None, :]
    out_mask = (offs_n[:, None] < seq_len) & (offs_m[None, :] < hidden)
    tl.store(out_ptrs, result.to(tl.bfloat16), mask=out_mask)


@triton.jit
def matmul_small(inp_ptr, weight_ptr, out_ptr, seq_len: tl.constexpr,
                 features: tl.constexpr, hidden: tl.constexpr,
                 BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                 BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


@triton.jit
def matmul_small_res(inp_ptr, weight_ptr, out_ptr, res_ptr,
                     seq_len: tl.constexpr, features: tl.constexpr,
                     hidden: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0).to(tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


@triton.jit
def matmul_small_bias_silu(inp_ptr, weight_ptr, out_ptr, bias_ptr,
                           seq_len: tl.constexpr, features: tl.constexpr,
                           hidden: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                           BLOCK_SIZE_M: tl.constexpr,
                           BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            acc = tl.dot(x, w, acc)
        acc = acc * tl.sigmoid(acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


@triton.jit
def scaled_matmul_small_bias_res(
        inp_ptr, inp_norm_factor_ptr, weight_ptr, out_ptr, bias_ptr, res_ptr,
        seq_len: tl.constexpr, features: tl.constexpr, hidden: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        norm_factor = tl.load(
            inp_norm_factor_ptr + i + tl.arange(0, BLOCK_SIZE_N),
            mask=i + tl.arange(0, BLOCK_SIZE_N) < seq_len,
            other=0)
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0).to(tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0)
            x = x * norm_factor[:, None]
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


@triton.jit
def matvec_bias_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr,
                       features: tl.constexpr, hidden: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    BLOCK_SIZE_N: tl.constexpr = 32
    BLOCK_SIZE_M: tl.constexpr = 8
    for j in range(pid * BLOCK_SIZE_M, hidden, psize * BLOCK_SIZE_M):
        acc = tl.load(bias_ptr + j + tl.arange(0, BLOCK_SIZE_M)).to(tl.float32)
        for k in range(0, features, BLOCK_SIZE_N):
            x = tl.load(
                x_ptr + k + tl.arange(0, BLOCK_SIZE_N),
                mask=(k + tl.arange(0, BLOCK_SIZE_N) < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=(k + tl.arange(0, BLOCK_SIZE_N)[:, None] < features) &
                (j + tl.arange(0, BLOCK_SIZE_M)[None, :] < hidden),
                other=0.0)
            acc += tl.sum(x[:, None] * w, axis=0)
        tl.store(
            out_ptr + j + tl.arange(0, BLOCK_SIZE_M),
            acc.to(tl.bfloat16),
            mask=j + tl.arange(0, BLOCK_SIZE_M) < hidden)


@triton.jit
def matmul_small_res_gate(inp_ptr, weight_ptr, out_ptr, res_ptr, gate_ptr,
                          seq_len: tl.constexpr, features: tl.constexpr,
                          hidden: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0).to(tl.float32)
        matmul_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            matmul_acc = tl.dot(x, w, matmul_acc)
        gate = tl.load(
            gate_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0).to(tl.float32)
        acc += matmul_acc * gate
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


@triton.jit
def matmul_small_gate(inp_ptr,
                      weight1_ptr,
                      weight2_ptr,
                      out_ptr,
                      seq_len: tl.constexpr,
                      features: tl.constexpr,
                      hidden: tl.constexpr,
                      BLOCK_SIZE_N: tl.constexpr = 128,
                      BLOCK_SIZE_M: tl.constexpr = 64,
                      BLOCK_SIZE_K: tl.constexpr = 32):
    pid1 = tl.program_id(axis=0)
    psize1 = tl.num_programs(axis=0)
    pid2 = tl.program_id(axis=1)
    psize2 = tl.num_programs(axis=1)
    for i in range(pid1 * BLOCK_SIZE_N, seq_len, psize1 * BLOCK_SIZE_N):
        for j in range(pid2 * BLOCK_SIZE_M, hidden, psize2 * BLOCK_SIZE_M):
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            for k in range(0, features, BLOCK_SIZE_K):
                x = tl.load(
                    inp_ptr +
                    (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k +
                    tl.arange(0, BLOCK_SIZE_K),
                    mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                    other=0.0)
                w = tl.load(weight1_ptr +
                            (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) *
                            hidden + j + tl.arange(0, BLOCK_SIZE_M))
                acc = tl.dot(x, w, acc)
                w2 = tl.load(weight2_ptr +
                             (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) *
                             hidden + j + tl.arange(0, BLOCK_SIZE_M))
                acc2 = tl.dot(x, w2, acc2)
            acc = acc * tl.sigmoid(1.5957691216057308 * acc *
                                   (1 + 0.044715 * acc * acc))
            acc = (acc * acc2).to(tl.bfloat16)
            tl.store(
                out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden +
                j + tl.arange(0, BLOCK_SIZE_M),
                acc,
                mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len)


@triton.jit
def scaled_matmul_small_gate(inp_ptr,
                             inp_norm_factor_ptr,
                             weight1_ptr,
                             weight2_ptr,
                             out_ptr,
                             seq_len: tl.constexpr,
                             features: tl.constexpr,
                             hidden: tl.constexpr,
                             BLOCK_SIZE_N: tl.constexpr = 64,
                             BLOCK_SIZE_M: tl.constexpr = 64,
                             BLOCK_SIZE_K: tl.constexpr = 64):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        norm_factor = tl.load(
            inp_norm_factor_ptr + i + tl.arange(0, BLOCK_SIZE_N),
            mask=i + tl.arange(0, BLOCK_SIZE_N) < seq_len,
            other=0)
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k +
                tl.arange(0, BLOCK_SIZE_K),
                mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                other=0.0)
            x = x * norm_factor[:, None]
            w = tl.load(weight1_ptr +
                        (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden +
                        j + tl.arange(0, BLOCK_SIZE_M))
            acc = tl.dot(x, w, acc)
            w2 = tl.load(weight2_ptr +
                         (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden +
                         j + tl.arange(0, BLOCK_SIZE_M))
            acc2 = tl.dot(x, w2, acc2)
        acc = acc * tl.sigmoid(1.5957691216057308 * acc *
                               (1 + 0.044715 * acc * acc))
        acc = (acc * acc2).to(tl.bfloat16)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j +
            tl.arange(0, BLOCK_SIZE_M),
            acc,
            mask=i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len)
