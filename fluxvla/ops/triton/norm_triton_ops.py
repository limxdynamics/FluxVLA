# Origin: Modified from
# Upstream-URL: https://github.com/dexmal/realtime-vla/blob/main/pi0_infer.py
# Upstream-Ref: main
# SPDX-License-Identifier: MIT
# Notes: Attribution normalized; no functional change.

import triton
import triton.language as tl


@triton.jit
def add_residual_layer_norm_kernel(a_ptr,
                                   b_ptr,
                                   out_sum_ptr,
                                   out_norm_ptr,
                                   norm_w_ptr,
                                   norm_b_ptr,
                                   seq_len: tl.constexpr,
                                   features: tl.constexpr,
                                   HAS_AFFINE: tl.constexpr = False,
                                   eps: tl.constexpr = 1e-5):
    """Fused: s = a + b; n = layer_norm(s, [w, b]). Writes both s and n."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)

    MAX_LEN: tl.constexpr = 2048
    mask = tl.arange(0, MAX_LEN) < features

    if HAS_AFFINE:
        w = tl.load(norm_w_ptr + tl.arange(0, MAX_LEN), mask=mask, other=0.0)
        nb = tl.load(norm_b_ptr + tl.arange(0, MAX_LEN), mask=mask, other=0.0)

    for i in range(pid, seq_len, psize):
        offs = i * features + tl.arange(0, MAX_LEN)
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        bv = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = a + bv

        tl.store(out_sum_ptr + offs, s.to(tl.bfloat16), mask=mask)

        mean = tl.sum(s * mask) * (1.0 / features)
        s_c = s - mean
        var = tl.sum(s_c * s_c * mask) * (1.0 / features)
        normed = s_c * (1.0 / tl.sqrt(var + eps))

        if HAS_AFFINE:
            normed = normed * w + nb

        tl.store(out_norm_ptr + offs, normed.to(tl.bfloat16), mask=mask)


@triton.jit
def ada_layer_norm_kernel(x_ptr,
                          out_ptr,
                          scale_ptr,
                          shift_ptr,
                          seq_len: tl.constexpr,
                          features: tl.constexpr,
                          eps: tl.constexpr = 1e-5):
    """Fused AdaLayerNorm: layer_norm(x) * (1 + scale) + shift.

    x:     (seq_len, features)
    scale: (features,)  broadcast over seq dim
    shift: (features,)  broadcast over seq dim
    """
    pid = tl.program_id(0)
    psize = tl.num_programs(0)

    MAX_LEN: tl.constexpr = 2048
    mask = tl.arange(0, MAX_LEN) < features

    scale = tl.load(scale_ptr + tl.arange(0, MAX_LEN), mask=mask, other=0.0)
    shift = tl.load(shift_ptr + tl.arange(0, MAX_LEN), mask=mask, other=0.0)

    for i in range(pid, seq_len, psize):
        x = tl.load(
            x_ptr + i * features + tl.arange(0, MAX_LEN), mask=mask,
            other=0.0).to(tl.float32)
        mean = tl.sum(x * mask) * (1.0 / features)
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered * mask) * (1.0 / features)
        x_norm = x_centered * (1.0 / tl.sqrt(var + eps))
        result = x_norm * (1.0 + scale) + shift
        tl.store(
            out_ptr + i * features + tl.arange(0, MAX_LEN),
            result.to(tl.bfloat16),
            mask=mask)


@triton.jit
def layer_norm_small_kernel(x_ptr,
                            out_ptr,
                            norm_w_ptr,
                            norm_b_ptr,
                            seq_len: tl.constexpr,
                            features: tl.constexpr,
                            eps: tl.constexpr = 1e-5):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)

    MAX_LEN: tl.constexpr = 2048

    for i in range(pid, seq_len, psize):
        x = tl.load(
            x_ptr + i * features + tl.arange(0, MAX_LEN),
            mask=tl.arange(0, MAX_LEN) < features,
            other=0.0)
        mean = tl.sum(x) * (1.0 / features)
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered *
                     (tl.arange(0, MAX_LEN) < features)) * (1.0 / features)
        inv_std = 1.0 / tl.sqrt(var + eps)
        x = x_centered * inv_std
        x = x * tl.load(
            norm_w_ptr + tl.arange(0, MAX_LEN),
            mask=tl.arange(0, MAX_LEN) < features,
            other=0.0)
        x = x + tl.load(
            norm_b_ptr + tl.arange(0, MAX_LEN),
            mask=tl.arange(0, MAX_LEN) < features,
            other=0.0)
        tl.store(
            out_ptr + i * features + tl.arange(0, MAX_LEN),
            x.to(tl.bfloat16),
            mask=tl.arange(0, MAX_LEN) < features)


@triton.jit
def qwen3_rmsnorm_kernel(x_ptr,
                         out_ptr,
                         norm_w_ptr,
                         seq_len: tl.constexpr,
                         feat_dim: tl.constexpr,
                         eps: tl.constexpr = 1e-5):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)

    MAX_LEN: tl.constexpr = 2048
    mask = tl.arange(0, MAX_LEN) < feat_dim

    for i in range(pid, seq_len, psize):
        x = tl.load(
            x_ptr + i * feat_dim + tl.arange(0, MAX_LEN), mask=mask,
            other=0.0).to(tl.float32)
        var = tl.sum(x * x * mask) * (1.0 / feat_dim)
        x = x * (1.0 / tl.sqrt(var + eps))
        w = tl.load(
            norm_w_ptr + tl.arange(0, MAX_LEN), mask=mask,
            other=0.0).to(tl.float32)
        tl.store(
            out_ptr + i * feat_dim + tl.arange(0, MAX_LEN),
            (x * w).to(tl.bfloat16),
            mask=mask)


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_N': 16,
            'BLOCK_K': 64
        },
                      num_stages=3,
                      num_warps=4),
        triton.Config({
            'BLOCK_N': 32,
            'BLOCK_K': 64
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_N': 32,
            'BLOCK_K': 64
        },
                      num_stages=3,
                      num_warps=4),
        triton.Config({
            'BLOCK_N': 64,
            'BLOCK_K': 64
        },
                      num_stages=3,
                      num_warps=4),
        triton.Config({
            'BLOCK_N': 64,
            'BLOCK_K': 32
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_N': 128,
            'BLOCK_K': 32
        },
                      num_stages=3,
                      num_warps=8),
        triton.Config({
            'BLOCK_N': 128,
            'BLOCK_K': 32
        },
                      num_stages=2,
                      num_warps=8),
    ],
    key=['seq_len', 'x_num_features', 'out_num_features'],
)
@triton.jit
def linear_split_qkv_kernel(x_ptr,
                            w_ptr,
                            q_norm_w_ptr,
                            k_norm_w_ptr,
                            out_q_ptr,
                            out_k_ptr,
                            out_v_ptr,
                            q_dim,
                            k_dim,
                            seq_len,
                            x_num_features,
                            out_num_features,
                            norm_num_features,
                            BLOCK_N: tl.constexpr = 64,
                            BLOCK_M: tl.constexpr = 128,
                            BLOCK_K: tl.constexpr = 64,
                            eps: tl.constexpr = 1e-5):
    """2D tiled matmul + split: out = x @ w, write to Q/K/V buffers.

    x: (seq_len, x_num_features)            e.g. (600, 2048)
    w: (x_num_features, out_num_features)    e.g. (2048, 4096)  pre-transposed
    out_q: (seq_len, q_dim)                  e.g. (600, 2048)
    out_k: (seq_len, k_dim)                  e.g. (600, 1024)
    out_v: (seq_len, v_dim)                  e.g. (600, 1024)

    Grid: 1D with swizzle. Each program computes (BLOCK_N, BLOCK_M) output tile.  # noqa: E501
    BLOCK_N tiles the seq (row) dimension.
    BLOCK_M tiles the output (col) dimension.
    BLOCK_K tiles the reduction dimension.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(seq_len, BLOCK_N)
    num_pid_m = tl.cdiv(out_num_features, BLOCK_M)
    assert BLOCK_M == norm_num_features

    GROUP_SIZE: tl.constexpr = 8
    num_pid_in_group = GROUP_SIZE * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE)
    pid_n = first_pid_n + (pid % group_size_n)
    pid_m = (pid % num_pid_in_group) // group_size_n
    m_start = pid_m * BLOCK_M

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    x_base = x_ptr + n_offs[:, None] * x_num_features
    w_base = w_ptr + m_offs[None, :]

    for k_start in range(0, x_num_features, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        x_tile = tl.load(
            x_base + k_offs[None, :],
            mask=(n_offs[:, None] < seq_len) &
            (k_offs[None, :] < x_num_features),
            other=0.0)

        w_tile = tl.load(
            w_base + k_offs[:, None] * out_num_features,
            mask=(k_offs[:, None] < x_num_features) &
            (m_offs[None, :] < out_num_features),
            other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    n_mask = n_offs[:, None] < seq_len
    m_mask = m_offs[None, :] < out_num_features
    out_mask = n_mask & m_mask
    m_start = pid_m * BLOCK_M
    qk_boundary = q_dim + k_dim
    norm_offs = tl.arange(0, BLOCK_M)

    if m_start < q_dim:
        # Q: apply RMSNorm + weight
        var = tl.sum(acc * acc, axis=1) * (1.0 / norm_num_features)
        result = (acc * (1.0 / tl.sqrt(var[:, None] + eps))).to(tl.bfloat16)
        q_w = tl.load(
            q_norm_w_ptr + norm_offs,
            mask=norm_offs < norm_num_features,
            other=0.0)
        ptrs = out_q_ptr + n_offs[:, None] * q_dim + m_offs[None, :]
        tl.store(
            ptrs,
            result * q_w[None, :],
            mask=out_mask & (m_offs[None, :] < q_dim))
    elif m_start < qk_boundary:
        # K: apply RMSNorm + weight
        var = tl.sum(acc * acc, axis=1) * (1.0 / norm_num_features)
        result = (acc * (1.0 / tl.sqrt(var[:, None] + eps))).to(tl.bfloat16)
        k_w = tl.load(
            k_norm_w_ptr + norm_offs,
            mask=norm_offs < norm_num_features,
            other=0.0)
        ptrs = out_k_ptr + n_offs[:, None] * k_dim + (m_offs[None, :] - q_dim)
        tl.store(
            ptrs,
            result * k_w[None, :],
            mask=out_mask & (m_offs[None, :] < qk_boundary))
    else:
        # V: no RMSNorm, just cast to bf16
        result = acc.to(tl.bfloat16)
        v_dim = out_num_features - qk_boundary
        ptrs = (
            out_v_ptr + n_offs[:, None] * v_dim +
            (m_offs[None, :] - qk_boundary))
        tl.store(
            ptrs, result, mask=out_mask & (m_offs[None, :] < out_num_features))


@triton.jit
def rms_norm_kernel(inp_ptr, out_ptr, seq_len: tl.constexpr,
                    features: tl.constexpr):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    BLOCK_SIZE: tl.constexpr = 512
    for i in range(pid, seq_len, psize):
        sum_x = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            x = tl.load(inp_ptr + i * features + j + tl.arange(0, BLOCK_SIZE))
            sum_x += x * x
        factor = tl.rsqrt(tl.sum(sum_x) / features + 1e-6)
        for j in range(0, features, BLOCK_SIZE):
            x = tl.load(inp_ptr + i * features + j + tl.arange(0, BLOCK_SIZE))
            x = x * factor
            tl.store(out_ptr + i * features + j + tl.arange(0, BLOCK_SIZE), x)


@triton.jit
def rmsnorm_factor_kernel(inp_ptr,
                          factor_ptr,
                          rows: tl.constexpr,
                          features: tl.constexpr,
                          eps: tl.constexpr = 1e-6,
                          BLOCK_SIZE: tl.constexpr = 1024):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    for i in range(pid, rows, psize):
        sum_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            offs = j + tl.arange(0, BLOCK_SIZE)
            x = tl.load(
                inp_ptr + i * features + offs, mask=offs < features, other=0.0)
            sum_x += x * x
        factor = tl.rsqrt(tl.sum(sum_x) / features + eps)
        tl.store(factor_ptr + i, factor)


@triton.jit
def adarms_norm_kernel(x_ptr, style_ptr, normed_x_ptr, gate_ptr,
                       seq_len: tl.constexpr, features: tl.constexpr,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid, seq_len, psize):
        row_x_offset = i * features
        sum_sq = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(
                x_ptr + row_x_offset + cols, mask=mask,
                other=0.0).to(tl.float32)
            sum_sq += x_val * x_val
        rms_factor = tl.rsqrt(tl.sum(sum_sq) / features + 1e-6)
        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(
                x_ptr + row_x_offset + cols, mask=mask,
                other=0.0).to(tl.float32)
            x_norm = x_val * rms_factor
            s_scale = tl.load(
                style_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            s_shift = tl.load(
                style_ptr + features + cols, mask=mask,
                other=0.0).to(tl.float32)
            s_gate = tl.load(
                style_ptr + 2 * features + cols, mask=mask,
                other=0.0).to(tl.float32)
            output_val = x_norm * (1.0 + s_scale) + s_shift
            tl.store(
                normed_x_ptr + row_x_offset + cols,
                output_val.to(tl.bfloat16),
                mask=mask)
            tl.store(
                gate_ptr + row_x_offset + cols,
                s_gate.to(tl.bfloat16),
                mask=mask)
