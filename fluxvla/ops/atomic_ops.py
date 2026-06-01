"""Atomic ops — parameterised building blocks that compose Triton/CUDA kernels.

Each function directly calls low-level Triton and CUDA kernels from
``fluxvla.ops``.  They are shared across model inference files (eagle2,
pi05, …) to eliminate duplicated dimension-bound wrappers.
"""
import torch
import torch.nn.functional as F

# yapf: disable
from fluxvla.ops import (layer_norm_small_kernel, matmul_small_bias_gelu,
                         matmul_small_bias_res_mod)
from fluxvla.ops.cuda.matmul_bias import matmul_bias_cuda
from fluxvla.ops.triton.attention_triton_ops import (
    matmul_n_2048_2560_qkv_rope, matmul_rope_qkv, scaled_matmul_rope_qkv)
from fluxvla.ops.triton.matmul_triton_ops import (matmul_small,
                                                  matmul_small_bias,
                                                  matmul_small_bias_silu,
                                                  matmul_small_gate,
                                                  matmul_small_res,
                                                  matmul_small_res_gate)
from fluxvla.ops.triton.norm_triton_ops import (ada_layer_norm_kernel,
                                                adarms_norm_kernel,
                                                rms_norm_kernel,
                                                rmsnorm_factor_kernel)

# yapf: enable

# ---------------------------------------------------------------------------
# Vision encoder ops (shared by eagle2 & pi05)
# ---------------------------------------------------------------------------


def conv2d_embed_res(images, patch_w, patch_b, pos_emb, out, grid_size,
                     patch_size, num_patches, vit_hidden):
    nviews = images.shape[0]
    img_input = images.view(nviews, grid_size, patch_size, grid_size,
                            patch_size, 3).permute(0, 1, 3, 2, 4,
                                                   5).contiguous()
    seq_len = num_patches * nviews
    matmul_small_bias_res_mod[(seq_len // 64) * (vit_hidden // 64), ](
        img_input,
        patch_w,
        out,
        patch_b,
        pos_emb,
        seq_len=seq_len,
        features=3 * patch_size * patch_size,
        hidden=vit_hidden,
        i_mod=num_patches,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=32)


def layer_norm_QKV_matmul_bias(x, norm_w, norm_b, qkv_w, qkv_b, out, x_norm,
                               num_patches, vit_hidden, vit_qkv_hidden):
    num_views = x.shape[0]
    seq_len = num_patches * num_views
    layer_norm_small_kernel[seq_len, ](
        x, x_norm, norm_w, norm_b, seq_len=seq_len, features=vit_hidden)
    matmul_bias_cuda(x_norm, qkv_w, qkv_b, out=out)


def AttnMultiKey(QKV, num_patches, vit_num_heads, vit_head_dim, vit_hidden):
    input_dtype = QKV.dtype
    QKV = QKV.view(-1, num_patches, 3, vit_num_heads,
                   vit_head_dim).permute(0, 2, 3, 1, 4)
    Q = QKV[:, 0]
    K = QKV[:, 1]
    V = QKV[:, 2]
    attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(1, 2).reshape(Q.shape[0], num_patches, vit_hidden)
    if attn.dtype != input_dtype:
        attn = attn.to(input_dtype)
    return attn


def matmul_bias_res(x, weight, bias, res, out, buf, num_patches, vit_hidden):
    matmul_bias_cuda(x, weight, bias, out=out, res=res)


def layer_norm_matmul_bias_gelu(x, norm_w, norm_b, weight, bias, out, x_norm,
                                num_patches, vit_hidden, vit_intermediate):
    num_views = x.shape[0]
    seq_len = num_patches * num_views
    layer_norm_small_kernel[seq_len, ](
        x, x_norm, norm_w, norm_b, seq_len=seq_len, features=vit_hidden)
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 64
    matmul_small_bias_gelu[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) *
                           ((vit_intermediate +
                             (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M), ](
                                 x_norm,
                                 weight,
                                 out,
                                 bias,
                                 seq_len=seq_len,
                                 features=vit_hidden,
                                 hidden=vit_intermediate,
                                 BLOCK_SIZE_N=BLOCK_SIZE_N,
                                 BLOCK_SIZE_M=BLOCK_SIZE_M,
                                 BLOCK_SIZE_K=BLOCK_SIZE_K)


def matmul_split_k_bias_res(x, weight, bias, res, out, buf, num_patches,
                            vit_intermediate, vit_hidden):
    matmul_bias_cuda(x, weight, bias, out=out, res=res)


# ---------------------------------------------------------------------------
# LayerNorm + matmul + bias (e.g. vision→encoder projection)
# ---------------------------------------------------------------------------


def layer_norm_matmul_bias(x,
                           norm_w,
                           norm_b,
                           proj_w,
                           proj_b,
                           out,
                           x_norm,
                           num_patches,
                           in_features,
                           out_features,
                           eps=1e-5):
    seq_len = x.shape[0] * num_patches
    layer_norm_small_kernel[seq_len, ](
        x,
        x_norm,
        norm_w,
        norm_b,
        seq_len=seq_len,
        features=in_features,
        eps=eps)
    matmul_small_bias[((seq_len + 63) // 64) * (out_features // 64), ](
        x_norm,
        proj_w,
        out,
        proj_b,
        seq_len=seq_len,
        features=in_features,
        hidden=out_features,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=64)


# ---------------------------------------------------------------------------
# RMSNorm + matmul → QKV + RoPE
# ---------------------------------------------------------------------------


def rms_matmul_qkv_rope(x, weight_qkv, rope_weight, Q, K, V, x_norm,
                        hidden_dim, head_dim, num_kv_heads):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len, )](x, x_norm, seq_len, hidden_dim)
    qkv_dim = weight_qkv.shape[1]
    matmul_n_2048_2560_qkv_rope[((seq_len + 63) // 64,
                                 qkv_dim // 64)](x_norm, weight_qkv,
                                                 rope_weight, Q, K, V, seq_len,
                                                 hidden_dim, head_dim,
                                                 num_kv_heads)


# ---------------------------------------------------------------------------
# Matmul + residual
# ---------------------------------------------------------------------------


def matmul_res(x, weight, out, in_features, out_features):
    seq_len = x.shape[0]
    BLOCK_SIZE_N = 128
    if seq_len < 512:
        BLOCK_SIZE_N = 64
    matmul_small_res[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) *
                     (out_features // 64), ](
                         x,
                         weight,
                         out,
                         out,
                         seq_len=seq_len,
                         features=in_features,
                         hidden=out_features,
                         BLOCK_SIZE_N=BLOCK_SIZE_N,
                         BLOCK_SIZE_M=64,
                         BLOCK_SIZE_K=64)


# ---------------------------------------------------------------------------
# RMSNorm + gated MLP (gate_proj * silu ⊙ up_proj)
# ---------------------------------------------------------------------------


def rms_matmul_gate(x, gate_w, up_w, out, x_norm, hidden_dim,
                    intermediate_dim):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len, )](x, x_norm, seq_len, hidden_dim)
    matmul_small_gate[((seq_len + 127) // 128, (intermediate_dim + 63) // 64)](
        x_norm, gate_w, up_w, out, seq_len, hidden_dim, intermediate_dim)


# ---------------------------------------------------------------------------
# Attention: softmax(Q·Kᵀ/√d)·V  — the V matmul part
# ---------------------------------------------------------------------------


def matmul_attn_v(x, V, out, head_dim):
    total_queries = x.shape[0]
    total_keys = V.shape[0]
    matmul_small[((total_keys + 31) // 32) * (head_dim // 32), ](
        x,
        V,
        out,
        seq_len=total_queries,
        features=total_keys,
        hidden=head_dim,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=64)


# ---------------------------------------------------------------------------
# Matmul + bias + SiLU
# ---------------------------------------------------------------------------


def matmul_bias_silu(x, weight, bias, out, in_features, out_features):
    seq_len = x.shape[0]
    matmul_small_bias_silu[((seq_len + 31) // 32) * (out_features // 32), ](
        x,
        weight,
        out,
        bias,
        seq_len=seq_len,
        features=in_features,
        hidden=out_features,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=64)


# ---------------------------------------------------------------------------
# Matmul + bias (generic small)
# ---------------------------------------------------------------------------


def matmul_bias_small(x,
                      weight,
                      bias,
                      out,
                      in_features,
                      out_features,
                      BLOCK_SIZE_N=32,
                      BLOCK_SIZE_M=32,
                      BLOCK_SIZE_K=32):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) *
                      ((out_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M), ](
                          x,
                          weight,
                          out,
                          bias,
                          seq_len=seq_len,
                          features=in_features,
                          hidden=out_features,
                          BLOCK_SIZE_N=BLOCK_SIZE_N,
                          BLOCK_SIZE_M=BLOCK_SIZE_M,
                          BLOCK_SIZE_K=BLOCK_SIZE_K)


# ---------------------------------------------------------------------------
# AdaRMSNorm + style projection
# ---------------------------------------------------------------------------


def adarms_norm_style_proj(x, time_emb, mod_w, mod_b, x_normed, gate, style,
                           hidden_dim, style_dim):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (style_dim // 32), ](
        time_emb,
        mod_w,
        style,
        mod_b,
        seq_len=seq_len,
        features=hidden_dim,
        hidden=style_dim,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_K=32)
    adarms_norm_kernel[(seq_len, )](
        x,
        style,
        x_normed,
        gate,
        seq_len=seq_len,
        features=hidden_dim,
        BLOCK_SIZE=512)


# ---------------------------------------------------------------------------
# Matmul → QKV + RoPE (no norm)
# ---------------------------------------------------------------------------


def matmul_qkv_rope(x_normed, weight_qkv, rope_weight, Q, K, V, hidden_dim,
                    head_dim, num_kv_heads):
    seq_len = x_normed.shape[0]
    matmul_rope_qkv[(128, )](x_normed, seq_len, hidden_dim, head_dim,
                             num_kv_heads, weight_qkv, rope_weight, Q, K, V)


# ---------------------------------------------------------------------------
# RMSNorm-factor + scaled matmul → QKV + RoPE
# ---------------------------------------------------------------------------


def rms_matmul_scaled_qkv_rope(x,
                               weight_qkv,
                               rope_weight,
                               Q,
                               K,
                               V,
                               x_norm_factor,
                               hidden_dim,
                               head_dim,
                               num_kv_heads,
                               eps=1e-6):
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(128, )](
        x, x_norm_factor, seq_len, hidden_dim, eps=eps, BLOCK_SIZE=1024)
    scaled_matmul_rope_qkv[(128, )](x, x_norm_factor, seq_len, hidden_dim,
                                    head_dim, num_kv_heads, weight_qkv,
                                    rope_weight, Q, K, V)


# ---------------------------------------------------------------------------
# Matmul + residual + gate
# ---------------------------------------------------------------------------


def matmul_res_gate(x,
                    weight,
                    out,
                    gate,
                    in_features,
                    out_features,
                    BLOCK_SIZE_N=32,
                    BLOCK_SIZE_M=32,
                    BLOCK_SIZE_K=128):
    seq_len = x.shape[0]
    matmul_small_res_gate[(((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * (
        (out_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M), )](
            x,
            weight,
            out,
            out,
            gate,
            seq_len=seq_len,
            features=in_features,
            hidden=out_features,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K=BLOCK_SIZE_K)


# ---------------------------------------------------------------------------
# Gated MLP (gate_proj * silu ⊙ up_proj, no norm)
# ---------------------------------------------------------------------------


def matmul_gate(x, gate_w, up_w, out, in_features, intermediate_dim):
    seq_len = x.shape[0]
    matmul_small_gate[((seq_len + 127) // 128,
                       (intermediate_dim + 63) // 64)](x, gate_w, up_w, out,
                                                       seq_len, in_features,
                                                       intermediate_dim)


# ---------------------------------------------------------------------------
# MHA + FFN blocks (FlowMatching inference)
# ---------------------------------------------------------------------------


def mha_self(x, qkv_w, qkv_b, out_w, out_b, num_heads, head_dim):
    """Self-attention with merged QKV projection (one matmul)."""
    B, S, _ = x.shape
    hdim = num_heads * head_dim
    qkv = F.linear(x, qkv_w, qkv_b)
    q, k, v = qkv.split([hdim, hdim, hdim], dim=-1)
    q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(B, S, -1).contiguous()
    return F.linear(out, out_w, out_b)


def mha_cross(x, enc, q_w, q_b, kv_w, kv_b, out_w, out_b, num_heads, head_dim):
    """Cross-attention with merged KV projection (one matmul for K+V)."""
    B, S, _ = x.shape
    S_enc = enc.shape[1]
    hdim = num_heads * head_dim
    q = F.linear(x, q_w, q_b).view(B, S, num_heads, head_dim).transpose(1, 2)
    kv = F.linear(enc, kv_w, kv_b)
    k, v = kv.split([hdim, hdim], dim=-1)
    k = k.view(B, S_enc, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, S_enc, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(B, S, -1).contiguous()
    return F.linear(out, out_w, out_b)


def ff_gelu(x, up_w_T, up_b, down_w, down_b, up_features, up_hidden):
    """FeedForward with fused linear+GELU (Triton) + linear (cuBLAS)."""
    orig_shape = x.shape
    seq_len = x.numel() // up_features
    inp = x.reshape(seq_len, up_features).contiguous()
    intermediate = torch.empty(
        seq_len, up_hidden, dtype=inp.dtype, device=inp.device)

    BN, BM, BK = 64, 64, 64
    grid_size = (((seq_len + BN - 1) // BN) * ((up_hidden + BM - 1) // BM))
    matmul_small_bias_gelu[grid_size, ](
        inp,
        up_w_T,
        intermediate,
        up_b,
        seq_len=seq_len,
        features=up_features,
        hidden=up_hidden,
        BLOCK_SIZE_N=BN,
        BLOCK_SIZE_M=BM,
        BLOCK_SIZE_K=BK)

    return F.linear(
        intermediate.view(*orig_shape[:-1], up_hidden), down_w, down_b)


def layer_norm_triton(x, w, b):
    """LayerNorm via Triton kernel."""
    features = x.shape[-1]
    seq_len = x.numel() // features
    out = torch.empty_like(x)
    layer_norm_small_kernel[seq_len, ](
        x, out, w, b, seq_len=seq_len, features=features)
    return out


def ada_layer_norm(x, scale, shift):
    """Fused AdaLayerNorm via Triton: layer_norm(x) * (1 + scale) + shift."""
    features = x.shape[-1]
    seq_len = x.numel() // features
    out = torch.empty_like(x)
    ada_layer_norm_kernel[seq_len, ](
        x.view(-1, features),
        out.view(-1, features),
        scale.view(-1),
        shift.view(-1),
        seq_len=seq_len,
        features=features)
    return out


def vl_sa_block(x, n1_w, n1_b, qkv_w, qkv_b, o_w, o_b, n3_w, n3_b, ff_up_w_T,
                ff_up_b, ff_dn_w, ff_dn_b, nh, hd, ff_features, ff_hidden):
    """VL self-attention block with merged QKV."""
    h = layer_norm_triton(x, n1_w, n1_b)
    x = mha_self(h, qkv_w, qkv_b, o_w, o_b, nh, hd) + x
    h = layer_norm_triton(x, n3_w, n3_b)
    x = ff_gelu(h, ff_up_w_T, ff_up_b, ff_dn_w, ff_dn_b, ff_features,
                ff_hidden) + x
    return x


def dit_block_self(x, temb, n1_w, n1_b, qkv_w, qkv_b, o_w, o_b, ff_up_w_T,
                   ff_up_b, ff_dn_w, ff_dn_b, nh, hd, dim, ff_features,
                   ff_hidden):
    """DiT self-attention block with merged QKV."""
    ada = F.linear(F.silu(temb), n1_w, n1_b)
    scale, shift = ada.chunk(2, dim=1)
    h = ada_layer_norm(x, scale, shift)
    x = mha_self(h, qkv_w, qkv_b, o_w, o_b, nh, hd) + x
    h = F.layer_norm(x, [dim])
    x = ff_gelu(h, ff_up_w_T, ff_up_b, ff_dn_w, ff_dn_b, ff_features,
                ff_hidden) + x
    return x


def dit_block_cross(x, enc, temb, n1_w, n1_b, q_w, q_b, kv_w, kv_b, o_w, o_b,
                    ff_up_w_T, ff_up_b, ff_dn_w, ff_dn_b, nh, hd, dim,
                    ff_features, ff_hidden):
    """DiT cross-attention block with merged KV."""
    ada = F.linear(F.silu(temb), n1_w, n1_b)
    scale, shift = ada.chunk(2, dim=1)
    h = ada_layer_norm(x, scale, shift)
    x = mha_cross(h, enc, q_w, q_b, kv_w, kv_b, o_w, o_b, nh, hd) + x
    h = F.layer_norm(x, [dim])
    x = ff_gelu(h, ff_up_w_T, ff_up_b, ff_dn_w, ff_dn_b, ff_features,
                ff_hidden) + x
    return x
