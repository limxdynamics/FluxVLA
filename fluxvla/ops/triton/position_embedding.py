import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_position_embedding_add_kernel(
    Input,  # input: (B, T, D)
    EmbWeight,  # embedding weight: (max_seq_len, D)
    Output,  # output: (B, T, D)
    stride_in_b,
    stride_in_t,
    stride_in_d,
    stride_emb_t,
    stride_emb_d,
    stride_out_b,
    stride_out_t,
    stride_out_d,
    B,
    T,
    D,
    num_blocks_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused Position Embedding Lookup + Add kernel.

    Each program processes one row (batch_idx, token_idx), directly reads
    the embedding corresponding to pos_idx = token_idx from the embedding
    table, and adds it to the input.
    """
    # Get the current row index
    row_idx = tl.program_id(0)
    batch_idx = row_idx // (T * num_blocks_per_row)
    remainder = row_idx % (T * num_blocks_per_row)
    token_idx = remainder // num_blocks_per_row
    block_idx = remainder % num_blocks_per_row

    # Compute pointers
    input_row_ptr = Input + batch_idx * stride_in_b + token_idx * stride_in_t
    emb_row_ptr = EmbWeight + token_idx * stride_emb_t  # pos_id = token_idx
    output_row_ptr = Output + batch_idx * stride_out_b + token_idx * stride_out_t  # noqa: E501

    # Process D dimension in blocks (tl.arange must start
    # at 0, so use addition for offset)
    cols = tl.arange(0, BLOCK_SIZE) + block_idx * BLOCK_SIZE
    mask = cols < D

    # Load input and position embedding
    x = tl.load(input_row_ptr + cols * stride_in_d, mask=mask, other=0.0)
    pos_emb = tl.load(emb_row_ptr + cols * stride_emb_d, mask=mask, other=0.0)

    # Fused addition
    output = x + pos_emb

    # Store result
    tl.store(output_row_ptr + cols * stride_out_d, output, mask=mask)


def fused_position_embedding_add(
        input: torch.Tensor,  # (B, T, D)
        embedding_weight: torch.Tensor,  # (max_seq_len, D)
) -> torch.Tensor:
    """
    Fused Position Embedding Lookup + Add operation.

    Equivalent to:
        pos_ids = torch.arange(T, device=input.device)
        pos_embs = F.embedding(pos_ids, embedding_weight).unsqueeze(0)
        output = input + pos_embs

    Args:
        input: Input tensor, shape (B, T, D)
        embedding_weight: Position embedding weight, shape (max_seq_len, D)

    Returns:
        output: Output tensor, shape (B, T, D)
    """
    B, T, D = input.shape

    # Ensure contiguous memory
    input = input.contiguous()
    embedding_weight = embedding_weight.contiguous()

    # Allocate output
    output = torch.empty_like(input)

    # Compute block size
    BLOCK_SIZE = triton.next_power_of_2(D)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)

    # Launch kernel
    num_rows = B * T
    _fused_position_embedding_add_kernel[(num_rows, )](
        input,
        embedding_weight,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        embedding_weight.stride(0),
        embedding_weight.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        B,
        T,
        D,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def fused_position_embedding_add_inplace(
        input: torch.Tensor,  # (B, T, D)
        embedding_weight: torch.Tensor,  # (max_seq_len, D)
) -> torch.Tensor:
    """
    In-place version of fused Position Embedding Add.
    Directly modifies the input tensor, saving memory allocation.

    Args:
        input: Input tensor, shape (B, T, D), will be modified in-place
        embedding_weight: Position embedding weight, shape (max_seq_len, D)

    Returns:
        input: Modified input tensor
    """
    B, T, D = input.shape

    # Ensure contiguous memory
    input = input.contiguous()
    embedding_weight = embedding_weight.contiguous()

    # Compute block size
    BLOCK_SIZE = 256
    num_blocks_per_row = math.ceil(D / BLOCK_SIZE)

    # Launch kernel (output points to input for in-place operation)
    num_rows = B * T * num_blocks_per_row
    _fused_position_embedding_add_kernel[(num_rows, )](
        input,
        embedding_weight,
        input,  # output = input for in-place
        input.stride(0),
        input.stride(1),
        input.stride(2),
        embedding_weight.stride(0),
        embedding_weight.stride(1),
        input.stride(0),
        input.stride(1),
        input.stride(2),
        B,
        T,
        D,
        num_blocks_per_row,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return input


@triton.jit
def _fused_concat_pos_emb_kernel(
    StateFeatures,  # (B, T_state, D)
    FutureTokens,  # (T_future, D) - shared across batch
    ActionFeatures,  # (B, T_action, D)
    PosEmbWeight,  # (max_seq_len, D) - position embedding
    Output,  # (B, T_total, D)
    stride_sf_b,
    stride_sf_t,
    stride_sf_d,
    stride_ft_t,
    stride_ft_d,
    stride_af_b,
    stride_af_t,
    stride_af_d,
    stride_pe_t,
    stride_pe_d,
    stride_out_b,
    stride_out_t,
    stride_out_d,
    B,
    T_state,
    T_future,
    T_action,
    D,
    add_pos_emb: tl.constexpr,  # whether to add position embedding
    num_blocks_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute current processing position
    row_idx = tl.program_id(0)
    T_total = T_state + T_future + T_action
    batch_idx = row_idx // (T_total * num_blocks_per_row)
    remainder = row_idx % (T_total * num_blocks_per_row)
    token_idx = remainder // num_blocks_per_row
    block_idx = remainder % num_blocks_per_row

    # Compute output pointer
    output_ptr = Output + batch_idx * stride_out_b + token_idx * stride_out_t

    # Process D dimension in blocks
    cols = tl.arange(0, BLOCK_SIZE) + block_idx * BLOCK_SIZE
    mask = cols < D

    # Determine which source to read from based on token_idx
    if token_idx < T_state:
        # Region 1: StateFeatures
        src_ptr = StateFeatures + batch_idx * stride_sf_b + token_idx * stride_sf_t  # noqa: E501
        data = tl.load(src_ptr + cols * stride_sf_d, mask=mask, other=0.0)
    elif token_idx < T_state + T_future:
        # Region 2: FutureTokens (broadcast across batch)
        future_idx = token_idx - T_state
        src_ptr = FutureTokens + future_idx * stride_ft_t  # noqa: E501
        data = tl.load(
            src_ptr + cols * stride_ft_d, mask=mask, other=0.0)  # noqa: E501
    else:
        # Region 3: ActionFeatures + Position Embedding
        action_idx = token_idx - T_state - T_future
        src_ptr = ActionFeatures + batch_idx * stride_af_b + action_idx * stride_af_t  # noqa: E501
        data = tl.load(
            src_ptr + cols * stride_af_d, mask=mask, other=0.0)  # noqa: E501

        if add_pos_emb:
            # Add position embedding (pos_id = action_idx)
            pe_ptr = PosEmbWeight + action_idx * stride_pe_t  # noqa: E501
            pos_emb = tl.load(
                pe_ptr + cols * stride_pe_d, mask=mask, other=0.0)
            data = data + pos_emb

    # Store result
    tl.store(output_ptr + cols * stride_out_d, data, mask=mask)


def fused_concat_with_pos_emb(
        state_features: torch.Tensor,  # (B, T_state, D)
        future_tokens: torch.Tensor,  # (T_future, D)
        action_features: torch.Tensor,  # (B, T_action, D)
        position_embedding_weight: torch.Tensor = None,  # (max_seq_len, D)
) -> torch.Tensor:
    """
    Fused Concat + Position Embedding operation.

    Equivalent to:
        future_tokens_expanded = future_tokens.unsqueeze(0).expand(B, -1, -1)
        if position_embedding_weight is not None:
            pos_ids = torch.arange(T_action, device=action_features.device)
            pos_embs = F.embedding(pos_ids, position_embedding_weight).unsqueeze(0)  # noqa: E501
            action_features = action_features + pos_embs
        output = torch.cat([state_features, future_tokens_expanded, action_features], dim=1)  # noqa: E501

    Args:
        state_features: (B, T_state, D) - state encoder output
        future_tokens: (T_future, D) - future tokens embedding weight
        action_features: (B, T_action, D) - action encoder output
        position_embedding_weight: (max_seq_len, D) - position embedding weight, optional

    Returns:
        output: (B, T_state + T_future + T_action, D)
    """
    B, T_state, D = state_features.shape
    T_future = future_tokens.shape[0]
    T_action = action_features.shape[1]
    T_total = T_state + T_future + T_action

    # Unify dtypes (use state_features as reference)
    target_dtype = state_features.dtype
    if future_tokens.dtype != target_dtype:
        future_tokens = future_tokens.to(target_dtype)

    # Ensure contiguous memory
    state_features = state_features.contiguous()
    future_tokens = future_tokens.contiguous()
    action_features = action_features.contiguous()

    add_pos_emb = position_embedding_weight is not None
    if add_pos_emb:
        if position_embedding_weight.dtype != target_dtype:
            position_embedding_weight = position_embedding_weight.to(
                target_dtype)
        position_embedding_weight = position_embedding_weight.contiguous()
    else:
        # Create a dummy tensor for kernel argument (will not be used)
        position_embedding_weight = future_tokens

    # Allocate output
    output = torch.empty((B, T_total, D),
                         dtype=state_features.dtype,
                         device=state_features.device)

    # Compute block size
    BLOCK_SIZE = 256
    num_blocks_per_row = math.ceil(D / BLOCK_SIZE)

    # Launch kernel
    num_rows = B * T_total * num_blocks_per_row
    _fused_concat_pos_emb_kernel[(num_rows, )](
        state_features,
        future_tokens,
        action_features,
        position_embedding_weight,
        output,
        state_features.stride(0),
        state_features.stride(1),
        state_features.stride(2),
        future_tokens.stride(0),
        future_tokens.stride(1),
        action_features.stride(0),
        action_features.stride(1),
        action_features.stride(2),
        position_embedding_weight.stride(0),
        position_embedding_weight.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        B,
        T_state,
        T_future,
        T_action,
        D,
        add_pos_emb,
        num_blocks_per_row,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
