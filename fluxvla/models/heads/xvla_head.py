# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy

from fluxvla.engines import HEADS
from fluxvla.models.third_party_models.xvla_models.action_hub import build_action_space
from fluxvla.models.third_party_models.xvla_models.transformer import (
    SoftPromptedTransformer, TransformerBlock)


@HEADS.register_module()
class XVLAFlowMatchingHead(nn.Module):
    """
    Action head wrapping X-VLA's SoftPromptedTransformer + ActionSpace.

    Implements the standard FluxVLA head interface:
        forward(input_features, states, attention_mask, actions,
                action_masks, embodiment_ids, aux_visual_inputs=None)
        predict_action(input_features, states, attention_mask,
                       embodiment_ids, aux_visual_inputs=None)

    Flow matching schedule mirrors modeling_xvla.py:165-168:
        x_t = t * noise + (1-t) * gt

    Args:
        hidden_size: Transformer hidden dim.
        multi_modal_input_size: Florence2 projection_dim (input to vlm_proj).
        depth: Number of TransformerBlocks.
        num_heads: Attention heads.
        mlp_ratio: MLP expansion ratio.
        num_domains: Max domain/embodiment IDs (must cover embodiment_id values).
        dim_action: Action vector dimension (20 for EE6D).
        dim_propio: Proprio vector dimension (20 for EE6D).
        len_soft_prompts: Soft prompt tokens per domain.
        dim_time: Sinusoidal time embedding dim.
        max_len_seq: Max sequence length for positional embedding.
        use_hetero_proj: Use DomainAwareLinear projections.
        num_actions: Number of action timesteps (chunk size).
        num_inference_steps: Denoising steps at inference.
        action_mode: 'ee6d' | 'joint' | 'agibot_ee6d' | 'auto'.
        real_action_dim: Used only when action_mode='auto'.
        max_action_dim: Used only when action_mode='auto'.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        multi_modal_input_size: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 20,
        dim_action: int = 20,
        dim_propio: int = 20,
        len_soft_prompts: int = 32,
        dim_time: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        num_actions: int = 10,
        num_inference_steps: int = 10,
        action_mode: str = 'ee6d',
        real_action_dim: Optional[int] = None,
        max_action_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_inference_steps = num_inference_steps

        # Build action space (handles loss, pre/post-process)
        if action_mode == 'auto':
            assert real_action_dim is not None, 'real_action_dim required for auto mode'
            self.action_space = build_action_space(
                'auto',
                real_dim=real_action_dim,
                max_dim=max_action_dim or dim_action,
            )
        else:
            self.action_space = build_action_space(action_mode)

        # SoftPromptedTransformer — identical to X-VLA's transformer.py
        self.transformer = SoftPromptedTransformer(
            hidden_size=hidden_size,
            multi_modal_input_size=multi_modal_input_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_domains=num_domains,
            dim_action=self.action_space.dim_action,
            dim_propio=dim_propio,
            len_soft_prompts=len_soft_prompts,
            dim_time=dim_time,
            max_len_seq=max_len_seq,
            use_hetero_proj=use_hetero_proj,
        )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_features: torch.Tensor,   # [B, T_enc, D]
        states: torch.Tensor,            # [B, dim_propio]
        attention_mask: torch.Tensor,    # [B, T_enc]
        actions: torch.Tensor,           # [B, num_actions, dim_action]
        action_masks: torch.Tensor,      # [B, num_actions]
        embodiment_ids: torch.Tensor,    # [B]
        aux_visual_inputs: Optional[torch.Tensor] = None,  # [B, (V-1)*N, D]
        **kwargs,
    ):
        B = input_features.shape[0]
        device = input_features.device

        # Flow matching noise schedule (X-VLA modeling_xvla.py:165-168)
        t = (torch.rand(1, device=device)
             + torch.arange(B, device=device) / B) % (1 - 1e-5)

        action_noisy = (torch.randn_like(actions) * t.view(-1, 1, 1)
                        + actions * (1 - t).view(-1, 1, 1))
        proprio_m, action_noisy_m = self.action_space.preprocess(states, action_noisy)

        # When only 1 camera view, aux_visual_inputs is [B, 0, D] (empty).
        # Pass it through as-is: aux_visual_proj([B,0,D]) → [B,0,H], cat adds 0 tokens.
        # Do NOT replace with zeros — that would inject a spurious token into the sequence.
        if aux_visual_inputs is None:
            aux_visual_inputs = input_features.new_zeros(
                B, 0, input_features.shape[-1])

        pred_action = self.transformer(
            domain_id=embodiment_ids,
            vlm_features=input_features,
            aux_visual_inputs=aux_visual_inputs,
            action_with_noise=action_noisy_m,
            proprio=proprio_m,
            t=t,
        )
        loss_dict = self.action_space.compute_loss(pred_action, actions)
        total_loss = sum(loss_dict.values())
        return dict(pred_actions=pred_action, loss=total_loss, **loss_dict)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_action(
        self,
        input_features: torch.Tensor,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        embodiment_ids: torch.Tensor,
        aux_visual_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        B = input_features.shape[0]
        D = self.action_space.dim_action
        device = input_features.device
        dtype = input_features.dtype

        if aux_visual_inputs is None:
            aux_visual_inputs = input_features.new_zeros(B, 0, input_features.shape[-1])

        # Iterative denoising (X-VLA generate_actions loop)
        x1 = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
        action = torch.zeros_like(x1)

        steps = max(1, self.num_inference_steps)
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=device, dtype=dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(states, x_t)
            action = self.transformer(
                domain_id=embodiment_ids,
                vlm_features=input_features,
                aux_visual_inputs=aux_visual_inputs,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
            )
        return self.action_space.postprocess(action)

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(_module_wrap_policy, module_classes={TransformerBlock})
