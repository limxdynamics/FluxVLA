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

from typing import Callable, Dict, List, Optional

import torch
from einops import rearrange

from fluxvla.engines import VLAS, initialize_overwatch
from .base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class DreamZeroVLA(BaseVLA):
    """DreamZero World-Action Model.

    Uses ``WanBackbone`` (wan_backbone) for encoding (T5, CLIP, VAE) and
    ``DreamZeroHead`` (vla_head) for the DiT diffusion model and flow-matching.

    Data contract
    -------------
    The forward method expects the following keys from the dataloader:

    * ``images``  – ``[B, V*C, H, W]`` (concatenated multi-view) **or**
      ``[B, V, C, H, W]`` **or** ``[B, C, H, W]`` (single view).
    * ``task_description`` – ``list[str]`` of length *B* (raw text).
    * ``states``  – ``[B, state_dim]`` or ``[B, num_tokens, state_dim]``.
    * ``actions`` – ``[B, action_horizon, action_dim]``.
    * ``action_masks`` – ``[B, action_horizon]`` or
      ``[B, action_horizon, action_dim]`` boolean.
    * ``embodiment_ids`` – ``[B]`` integer (optional, defaults to 0).

    Encoding (T5, CLIP, VAE) is done by ``WanBackbone`` (wan_backbone),
    then encoded tensors are passed to ``DreamZeroHead`` (vla_head).
    """

    def __init__(
        self,
        wan_backbone: Dict = None,
        vla_head: Dict = None,
        num_views: int = 2,
        frame_window_size: int = 1,
        pretrained_name_or_path: str = None,
        name_mapping: Dict = None,
        strict_mapping: bool = False,
        freeze_wan_backbone: bool = True,
        freeze_llm_backbone: bool = True,
        freeze_vlm_backbone: bool = True,
        freeze_projector: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            wan_backbone=wan_backbone,
            vla_head=vla_head,
            pretrained_name_or_path=pretrained_name_or_path,
            name_mapping=name_mapping,
            strict_mapping=strict_mapping,
            freeze_wan_backbone=freeze_wan_backbone,
            freeze_llm_backbone=freeze_llm_backbone,
            freeze_vlm_backbone=freeze_vlm_backbone,
            freeze_projector=freeze_projector,
        )
        self.num_views = num_views
        self.frame_window_size = frame_window_size
        self.all_module_keys = ['wan_backbone', 'vla_head']

    # ------------------------------------------------------------------
    # Data format conversion
    # ------------------------------------------------------------------
    def _prepare_video(self, images: torch.Tensor) -> torch.Tensor:
        """Convert fluxvla image tensors to DreamZero video format.

        Accepts several layouts and returns ``[B, 3, T, H_tiled, W]``.

        Camera views are tiled *vertically*:
            view-0 on top, view-1 on bottom -> ``H_out = num_views * H``.

        Supported input formats:
            * ``[B, C, H, W]``         single view, single timestep
            * ``[B, V*C, H, W]``       multi-view concatenated channels
            * ``[B, V*T*C, H, W]``     multi-view + temporal concatenated
            * ``[B, V, C, H, W]``      multi-view, single timestep
            * ``[B, V, T, C, H, W]``   multi-view, temporal
        """
        V = self.num_views
        T = self.frame_window_size

        if images.ndim == 4:
            b, channels, h, w = images.shape
            if channels > 3 and channels % 3 == 0:
                n_items = channels // 3
                if T > 1 and n_items == V * T:
                    # [B, V*T*C, H, W] → [B, V, T, 3, H, W]
                    images = images.view(b, V, T, 3, h, w)
                    imgs = rearrange(images, 'b v t c h w -> b t c (v h) w')
                    return imgs.transpose(1, 2)
                # [B, V*C, H, W] single timestep multi-view
                images = images.view(b, n_items, 3, h, w)
                tiles = [images[:, i] for i in range(n_items)]
                tiled = torch.cat(tiles, dim=2)  # [B, 3, n*H, W]
                return tiled.unsqueeze(2)  # [B, 3, 1, n*H, W]
            return images.unsqueeze(2)  # [B, C, 1, H, W]
        if images.ndim == 5:
            b, v, c, h, w = images.shape
            if v <= 4 and c == 3:
                tiles = [images[:, i] for i in range(v)]
                tiled = torch.cat(tiles, dim=2)
                return tiled.unsqueeze(2)
            return images.transpose(1, 2)
        if images.ndim == 6:
            imgs = rearrange(images, 'b v t c h w -> b t c (v h) w')
            return imgs.transpose(1, 2)
        raise ValueError(f'Unsupported image shape: {images.shape}')

    def _prepare_states(self, states: torch.Tensor,
                        num_tokens: int) -> torch.Tensor:
        """Ensure states have shape ``[B, num_tokens, D]``."""
        if states.ndim == 2:
            states = states.unsqueeze(1)
        if states.shape[1] < num_tokens:
            repeats = (num_tokens + states.shape[1] - 1) // states.shape[1]
            states = states.repeat(1, repeats, 1)[:, :num_tokens]
        return states

    def _build_action_masks(self, action_masks: torch.Tensor,
                            actual_action_dim: int,
                            max_action_dim: int) -> torch.Tensor:
        """Build per-dimension action masks that mark padded dims as False.

        Returns ``[B, T, max_action_dim]`` boolean tensor.
        """
        if action_masks.ndim == 2:
            # [B, T] temporal-only mask -> expand with dimension awareness
            b, t = action_masks.shape
            dim_mask = torch.zeros(
                max_action_dim, dtype=torch.bool, device=action_masks.device)
            dim_mask[:actual_action_dim] = True
            # outer product: valid timestep AND valid dimension
            full_mask = action_masks.unsqueeze(-1) * dim_mask.unsqueeze(
                0).unsqueeze(0)
            return full_mask
        # [B, T, D] already per-dimension
        if action_masks.shape[-1] < max_action_dim:
            pad_size = max_action_dim - action_masks.shape[-1]
            action_masks = torch.nn.functional.pad(
                action_masks, (0, pad_size), value=False)
        return action_masks

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        lang_tokens: Optional[torch.Tensor] = None,
        lang_masks: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        frame_masks: Optional[torch.Tensor] = None,
        embodiment_ids: Optional[torch.Tensor] = None,
        img_masks: Optional[torch.Tensor] = None,
        # accepted but unused
        task_description: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        if lang_tokens is None:
            raise ValueError(
                'DreamZeroVLA requires `lang_tokens` and `lang_masks` '
                'in the batch. Add ProcessPrompts to the transform pipeline.')

        device = actions.device
        max_action_dim = self.vla_head.max_action_dim
        max_state_dim = self.vla_head.max_state_dim
        actual_action_dim = self.vla_head.action_dim

        # Prepare video tensor
        video = self._prepare_video(images)  # [B, C, T, H, W]
        b, c, t, h, w = video.shape

        # --- Encode with WanBackbone ---
        self.wan_backbone.set_frozen_modules_to_eval_mode()

        prompt_embs = self.wan_backbone.encode_prompt(
            lang_tokens.long().to(device),
            lang_masks.long().to(device))

        latents = self.wan_backbone.encode_video(video)

        first_frame = video[:, :, :1].transpose(1, 2)  # [B, 1, C, H, W]
        clip_feas, ys, _ = self.wan_backbone.encode_image(first_frame, t, h, w)

        latents = latents.to(device)
        clip_feas = clip_feas.to(device)
        ys = ys.to(device)
        prompt_embs = prompt_embs.to(device)

        # Prepare states [B, num_state_tokens, D]
        t_video = video.shape[2]
        latent_frames = 1 + (t_video - 1) // 4
        num_blocks = max(1, (latent_frames - 1) //
                         self.vla_head.num_frame_per_block)
        num_state_tokens = num_blocks * self.vla_head.num_state_per_block
        states = self._prepare_states(states, num_state_tokens)

        # Pad actions to max_action_dim
        if actions.shape[-1] < max_action_dim:
            pad_size = max_action_dim - actions.shape[-1]
            actions = torch.nn.functional.pad(actions, (0, pad_size))

        # Build proper per-dimension action masks
        action_masks = self._build_action_masks(action_masks,
                                                actual_action_dim,
                                                max_action_dim)

        # Pad states to max_state_dim
        if states.shape[-1] < max_state_dim:
            pad_size = max_state_dim - states.shape[-1]
            states = torch.nn.functional.pad(states, (0, pad_size))

        # Default embodiment_ids to 0
        if embodiment_ids is None:
            embodiment_ids = torch.zeros(
                actions.shape[0], dtype=torch.long, device=device)

        return self.vla_head(
            prompt_embs=prompt_embs,
            latents=latents,
            clip_feas=clip_feas,
            ys=ys,
            states=states,
            actions=actions,
            action_masks=action_masks,
            embodiment_ids=embodiment_ids,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_action(
        self,
        images: torch.Tensor,
        task_description: List[str],
        states: torch.Tensor,
        embodiment_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        video = self._prepare_video(images)
        if embodiment_ids is None:
            embodiment_ids = torch.zeros(
                images.shape[0], dtype=torch.long, device=images.device)
        return self.vla_head.predict_action(
            images=video,
            task_description=task_description,
            states=states,
            embodiment_ids=embodiment_ids,
        )

    # ------------------------------------------------------------------
    # BaseVLA abstract method implementations
    # ------------------------------------------------------------------
    def get_fsdp_wrapping_policy(self) -> Callable:
        return self.vla_head.get_fsdp_wrapping_policy()

    @property
    def config(self):
        """DreamZero has no generative LLM, so we return a minimal config."""
        from transformers import PretrainedConfig
        cfg = PretrainedConfig()
        cfg.is_encoder_decoder = False
        return cfg

    def freeze_backbones(self) -> None:
        """Freeze WanBackbone (encoders), keep DreamZeroHead trainable."""
        if self.wan_backbone is not None:
            self.wan_backbone.requires_grad_(False)
        self.trainable_module_keys = ['vla_head']
        overwatch.info(
            '[Frozen]    =>> WanBackbone (T5, CLIP, VAE)', ctx_level=1)
        overwatch.info('[TRAINABLE] =>> DreamZero Head (DiT)', ctx_level=1)
