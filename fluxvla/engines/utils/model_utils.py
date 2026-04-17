# Origin: Modified from
# Upstream-URL: https://github.com/ZibinDong/openpi_pytorch/blob/main/pi0/utils.py  # noqa: E501
# Upstream-Ref: main
# SPDX-License-Identifier: Apache-2.0
# Notes: Attribution normalized and reused internal equivalents.

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from xformers.ops import memory_efficient_attention


def find_next_divisible_by_8_numpy(n: np.ndarray) -> np.ndarray:
    """
    Finds the smallest integers greater than each element in a
    NumPy array 'n'
    that are divisible by 8. Assumes non-negative integers.

    Args:
        n: A NumPy array of integers.

    Returns:
        A NumPy array containing the smallest integers
            greater than each input element that are divisible by 8.
    """
    remainder = n % 8
    # Calculate the amount to add: 0 if already divisible,
    # otherwise 8 - remainder
    # np.where is efficient for conditional operations on arrays
    amount_to_add = np.where(remainder == 0, 8, 8 - remainder)
    return n + amount_to_add


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device='cpu',
) -> Tensor:
    """Create sinusoidal positional embeddings.

    Args:
        time (torch.Tensor): Time values of shape ``(B,)`` or ``(B, T)``.
        dimension (int): Embedding dimension (must be divisible by 2).
        min_period (float): Minimum period for the sinusoidal function.
        max_period (float): Maximum period for the sinusoidal function.

    Returns:
        torch.Tensor: Shape ``(*time.shape, dimension)``."""
    if dimension % 2 != 0:
        raise ValueError(f'dimension ({dimension}) must be divisible by 2')

    fraction = torch.linspace(
        0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period)**fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = time[..., None] * scaling_factor
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.rand((bsize, ), device=device).pow(1 / alpha)
    gamma2 = torch.rand((bsize, ), device=device).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Create 2D attention masks from 1D attention and padding masks.

    Args:
        pad_masks (torch.Tensor): 1D padding masks of shape (bsize, seq_len).
        att_masks (torch.Tensor): 1D attention masks of shape (bsize, seq_len).

    Returns:
        torch.Tensor: 2D attention masks of shape (bsize, seq_len, seq_len)."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f'(b, c, h, w) expected, but {img.shape}')

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img,
        size=(resized_height, resized_width),
        mode='bilinear',
        align_corners=False)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(
        resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def gated_residual(x, y, gate):
    """
    Applies gated residual connection with optional gate parameter.

    Args:
        x: Input tensor (residual)
        y: Output tensor to be added
        gate: Optional gate tensor to modulate the addition

    Returns:
        x + y if gate is None, otherwise x + y * gate
    """
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Manual scaled dot-product attention (reference implementation).

    Explicitly materializes the full attention-weights matrix, so peak
    memory is O(B * H * L_q * L_k). Kept as a correctness/debug baseline
    and for environments where SDPA / Flash-Attn are unavailable. For
    training, prefer `sdpa_attention_forward`.

    Args:
        module: The parent attention module. Only
            `module.num_key_value_groups` and `module.training` are read.
        query: Tensor of shape (B, H, L_q, D).
        key: Tensor of shape (B, H_kv, L_k, D).
        value: Tensor of shape (B, H_kv, L_k, D).
        attention_mask: Optional additive mask of shape
            (B, 1, L_q, L_k_padded). Values should be 0 (attend) or
            -inf / a large negative number (mask). If provided, only the
            first `L_k` columns are used.
        scaling: Multiplier applied to QK^T before softmax (typically
            `1 / sqrt(head_dim)`).
        dropout: Attention dropout probability; only applied when
            `module.training` is True.
        **kwargs: Unused, kept for interface compatibility with other
            attention backends.

    Returns:
        A tuple `(attn_output, attn_weights)`:
            attn_output: Tensor of shape (B, L_q, H, D) — note the final
                head/seq transpose to match the caller's reshape contract.
            attn_weights: Full attention matrix of shape
                (B, H, L_q, L_k), useful for debugging / visualization.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Attention via `torch.nn.functional.scaled_dot_product_attention`.

    PyTorch's SDPA dispatches at runtime to the fastest available backend
    (Flash-Attn 2 when the inputs satisfy its constraints, otherwise
    memory-efficient / xformers-style attention, otherwise the math
    fallback). Peak memory is O(N) instead of O(N^2), so this is the
    recommended backend for both training and inference of PI0/PI0.5.

    Notes:
        - For the Flash-Attn backend, `query`/`key`/`value` must be fp16
          or bf16 and `attn_mask` must be None or a pure causal mask.
        - When `attention_mask` is an arbitrary additive 4D mask (as in
          PI0's block attention), SDPA falls back to the memory-efficient
          backend, which is still O(N) and well-optimized.
        - `attention_mask` should match the compute dtype (e.g. bf16) to
          avoid forcing a fallback to the slower math backend.

    Args:
        module: Parent attention module (reads `num_key_value_groups`
            and `training`).
        query: Tensor of shape (B, H, L_q, D).
        key: Tensor of shape (B, H_kv, L_k, D).
        value: Tensor of shape (B, H_kv, L_k, D).
        attention_mask: Optional additive mask of shape
            (B, 1, L_q, L_k_padded). Only the first `L_k` columns are
            used. Pass None for causal or unmasked attention.
        scaling: Softmax scale (typically `1 / sqrt(head_dim)`), passed
            through as SDPA's `scale` argument.
        dropout: Dropout probability; only applied when `module.training`
            is True.
        **kwargs: Unused, kept for interface compatibility.

    Returns:
        A tuple `(attn_output, None)`:
            attn_output: Tensor of shape (B, L_q, H, D). The second
                element is always None because SDPA does not expose the
                attention weights.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    causal_mask = None
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]

    attn_output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(
        torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which
            to unsqueeze cos[position_ids] and sin[position_ids] so that
            they can be properly broadcasted to the dimensions of q and k.
            For example, note that cos[position_ids] and sin[position_ids]
            have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then
            setting unsqueeze_dim=1 makes cos[position_ids] and
            sin[position_ids] broadcastable to the shapes of q and k.
            Similarly, if q and k have the shape [batch_size, seq_len,
            heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors
        rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
