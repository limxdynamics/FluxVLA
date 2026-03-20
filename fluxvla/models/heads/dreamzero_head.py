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

import logging
import os
from functools import partial
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torch.distributions import Beta

from fluxvla.engines import HEADS

logger = logging.getLogger(__name__)


def _import_dreamzero_modules():
    """Lazily import DreamZero modules so the rest of fluxvla still works
    even when optional dependencies (flash-attn, etc.) are missing."""
    from fluxvla.models.third_party_models.dreamzero.modules.flow_match_scheduler import \
        FlowMatchScheduler  # noqa: E501
    from fluxvla.models.third_party_models.dreamzero.modules.wan_video_dit_action_casual_chunk import \
        CausalWanModel  # noqa: E501
    return CausalWanModel, FlowMatchScheduler


def _ensure_file(path, hf_filename):
    """Return a valid local path for pretrained weights.

    Uses *path* directly when it exists on disk, otherwise downloads
    from the Wan-AI/Wan2.1-I2V-14B-480P HuggingFace repo.
    """
    if path is not None and os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id='Wan-AI/Wan2.1-I2V-14B-480P', filename=hf_filename)


@HEADS.register_module()
class DreamZeroHead(nn.Module):
    """DreamZero action head – joint video + action flow matching on the
    Wan 2.1 DiT backbone.

    This head contains the DiT diffusion model and flow-matching scheduler.
    Encoding (T5, CLIP, VAE) is handled by ``WanBackbone`` and the encoded
    tensors are passed in by ``DreamZeroVLA``.

    Args:
        action_dim: Actual robot action dimension (e.g. 7 for libero).
        max_action_dim: Padded action dim used inside the DiT.
        action_horizon: Number of action steps per generation block.
        max_state_dim: Padded state dimension.
        num_frames: Number of video frames (including conditioning frame).
        num_frame_per_block: Number of latent-time frames per DiT block.
        num_action_per_block: Number of action steps per DiT block.
        num_state_per_block: Number of state tokens per block.
        hidden_size: Hidden size for action encoder / state encoder.
        input_embedding_dim: Embedding dim inside the DiT.
        dit_dim: DiT hidden dimension (5120 for Wan 14B).
        dit_ffn_dim: DiT FFN dimension.
        dit_num_heads: Number of DiT attention heads.
        dit_num_layers: Number of DiT transformer blocks.
        max_num_embodiments: Max number of embodiment categories.
        frame_seqlen: Spatial sequence length per latent frame.
        noise_beta_alpha / noise_beta_beta / noise_s: Flow matching noise
            distribution parameters.
        train_architecture: ``"full"`` or ``"lora"``.
        lora_rank / lora_alpha / lora_target_modules: LoRA hyper-params.
        skip_pretrained_loading: If True, skip loading DiT pretrained
            weights – useful for unit testing.
        wan_model_path: Path to Wan 2.1 checkpoint directory.
    """

    def __init__(
        self,
        action_dim: int = 7,
        max_action_dim: int = 32,
        action_horizon: int = 10,
        max_state_dim: int = 64,
        num_frames: int = 9,
        num_frame_per_block: int = 2,
        num_action_per_block: int = 10,
        num_state_per_block: int = 1,
        hidden_size: int = 64,
        input_embedding_dim: int = 1536,
        dit_dim: int = 5120,
        dit_ffn_dim: int = 13824,
        dit_num_heads: int = 40,
        dit_num_layers: int = 40,
        dit_freq_dim: int = 256,
        dit_in_dim: int = 36,
        dit_out_dim: int = 16,
        max_num_embodiments: int = 32,
        frame_seqlen: int = 880,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        train_architecture: str = 'full',
        lora_rank: int = 4,
        lora_alpha: int = 4,
        lora_target_modules: str = 'q,k,v,o,ffn.0,ffn.2',
        skip_pretrained_loading: bool = False,
        wan_model_path: Optional[str] = None,
        use_gradient_checkpointing: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        CausalWanModel, FlowMatchScheduler = _import_dreamzero_modules()

        self.action_dim = action_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.max_state_dim = max_state_dim
        self.num_frames = num_frames
        self.num_frame_per_block = num_frame_per_block
        self.noise_s = noise_s
        self.train_architecture = train_architecture
        self.skip_pretrained_loading = skip_pretrained_loading
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block

        # ----- build DiT model -----
        self.model = CausalWanModel(
            diffusion_model_pretrained_path=wan_model_path,
            model_type='i2v',
            frame_seqlen=frame_seqlen,
            dim=dit_dim,
            in_dim=dit_in_dim,
            ffn_dim=dit_ffn_dim,
            out_dim=dit_out_dim,
            freq_dim=dit_freq_dim,
            num_heads=dit_num_heads,
            num_layers=dit_num_layers,
            max_chunk_size=-1,
            num_frame_per_block=num_frame_per_block,
            action_dim=max_action_dim,
            max_state_dim=max_state_dim,
            max_num_embodiments=max_num_embodiments,
            hidden_size=hidden_size,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        self.scheduler = FlowMatchScheduler(
            shift=5, sigma_min=0.0, extra_one_step=True)

        # ----- noise distributions -----
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)

        # ----- load pretrained weights -----
        if not skip_pretrained_loading:
            self._load_pretrained_weights(wan_model_path)

        # ----- set trainable -----
        self._setup_trainable(train_architecture, lora_rank, lora_alpha,
                              lora_target_modules)

        if use_gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        self.scheduler.set_timesteps(1000, training=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_pretrained_weights(self, dit_path):
        """Load pretrained weights for DiT."""
        if dit_path is not None and os.path.isdir(dit_path):
            self._load_dit_weights(dit_path)

    def _load_dit_weights(self, dit_dir):
        import json

        from safetensors.torch import load_file
        index_path = os.path.join(
            dit_dir, 'diffusion_pytorch_model.safetensors.index.json')
        single_path = os.path.join(dit_dir,
                                   'diffusion_pytorch_model.safetensors')
        state_dict = {}
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
            for shard_file in set(index['weight_map'].values()):
                shard_path = os.path.join(dit_dir, shard_file)
                state_dict.update(load_file(shard_path))
        elif os.path.exists(single_path):
            state_dict = load_file(single_path)
        else:
            logger.warning('No DiT weights found at %s', dit_dir)
            return
        missing, unexpected = self.model.load_state_dict(
            state_dict, strict=False)
        if missing:
            logger.info('DiT missing keys: %s', missing)
        if unexpected:
            logger.info('DiT unexpected keys: %s', unexpected)
        logger.info('Loaded DiT weights from %s', dit_dir)

    def _setup_trainable(self, architecture, lora_rank, lora_alpha,
                         lora_target_modules):
        if architecture == 'lora':
            from peft import LoraConfig, get_peft_model
            for p in self.model.parameters():
                p.requires_grad = False
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights=True,
                target_modules=lora_target_modules.split(','),
            )
            self.model = get_peft_model(self.model, lora_config)
            for param in self.model.parameters():
                param.data = param.to(torch.float32)
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
        # For "full" training, everything in self.model stays trainable.

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        prompt_embs: torch.Tensor,
        latents: torch.Tensor,
        clip_feas: torch.Tensor,
        ys: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        embodiment_ids: torch.Tensor,
        **kwargs,
    ) -> Dict:
        """Training forward pass with flow-matching loss.

        Args:
            prompt_embs: ``[B, seq_len, D]`` T5 text embeddings.
            latents: ``[B, C, T_lat, H_lat, W_lat]`` VAE-encoded video.
            clip_feas: ``[B, D_clip]`` CLIP image features.
            ys: ``[B, C_y, T_lat, H_lat, W_lat]`` conditioning input
                (mask + VAE-encoded first frame).
            states: ``[B, num_state_tokens, state_dim]``.
            actions: ``[B, action_horizon, action_dim]`` in **[-1,1]**.
            action_masks: ``[B, action_horizon, action_dim]`` boolean.
            embodiment_ids: ``[B]`` integer embodiment category.

        Returns:
            dict with ``loss``, ``dynamics_loss``, ``action_loss``.
        """
        device = actions.device

        # --- Flow-matching noise ---
        noise = torch.randn_like(latents)
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)

        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps,
                                    (noise.shape[0], noise.shape[1]))

        # Align block timesteps
        timestep_id_block = timestep_id[:,
                                        1:].reshape(timestep_id.shape[0], -1,
                                                    self.num_frame_per_block)
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]
        timestep_id_block = timestep_id_block.reshape(
            timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block],
                                   dim=1)

        _, num_lat_frames, num_channels, lat_h, lat_w = noise.shape
        frame_seqlen = int(lat_h * lat_w / 4)
        seq_len = num_lat_frames * frame_seqlen

        timestep = self.scheduler.timesteps[timestep_id].to(device)
        noisy_latents = self.scheduler.add_noise(
            latents.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(
            latents, noise, timestep).transpose(1, 2)

        # --- Action noise ---
        noise_action = torch.randn_like(actions)
        timestep_action_id = timestep_id_block.repeat(
            1,
            1,
            actions.shape[1] // (noise.shape[1] - 1) if
            (noise.shape[1] - 1) > 0 else 1,
        )
        timestep_action_id = timestep_action_id.reshape(
            timestep_action_id.shape[0], -1)
        if timestep_action_id.shape[1] != actions.shape[1]:
            timestep_action_id = torch.randint(
                0, self.scheduler.num_train_timesteps,
                (actions.shape[0], actions.shape[1]))

        timestep_action = self.scheduler.timesteps[timestep_action_id].to(
            device)
        noisy_actions = self.scheduler.add_noise(
            actions.flatten(0, 1),
            noise_action.flatten(0, 1),
            timestep_action.flatten(0, 1),
        ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
        training_target_action = self.scheduler.training_target(
            actions, noise_action, timestep_action)

        # --- DiT forward ---
        with torch.amp.autocast(
                dtype=torch.bfloat16, device_type=torch.device(device).type):
            video_noise_pred, action_noise_pred = self.model(
                noisy_latents.transpose(1, 2),
                timestep=timestep,
                clip_feature=clip_feas,
                y=ys,
                context=prompt_embs,
                seq_len=seq_len,
                state=states.to(torch.bfloat16),
                embodiment_id=embodiment_ids,
                action=noisy_actions,
                timestep_action=timestep_action,
                clean_x=latents.transpose(1, 2),
            )

            # --- Compute losses ---
            dynamics_loss = F.mse_loss(
                video_noise_pred.float(),
                training_target.float(),
                reduction='none',
            ).mean(dim=(1, 3, 4))
            weight_dyn = (
                dynamics_loss * self.scheduler.training_weight(
                    timestep.flatten(0, 1)).unflatten(
                        0, (noise.shape[0], noise.shape[1])).to(device))
            weighted_dynamics_loss = weight_dyn.mean()

            action_loss_raw = F.mse_loss(
                action_noise_pred.float(),
                training_target_action.float(),
                reduction='none',
            ) * action_masks.float()
            weight_act = (
                action_loss_raw.mean(dim=2) *
                self.scheduler.training_weight(timestep_action.flatten(
                    0, 1)).unflatten(0, (noise_action.shape[0],
                                         noise_action.shape[1])).to(device))
            weighted_action_loss = weight_act.mean()

            loss = weighted_dynamics_loss + weighted_action_loss

        return dict(
            loss=loss,
            dynamics_loss=weighted_dynamics_loss,
            action_loss=weighted_action_loss,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_action(
        self,
        prompt_embs: torch.Tensor,
        latents: torch.Tensor,
        clip_feas: torch.Tensor,
        ys: torch.Tensor,
        states: torch.Tensor,
        embodiment_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Single-step action prediction (simplified, non-autoregressive)."""
        raise NotImplementedError(
            'DreamZero inference is not yet implemented in fluxvla. '
            'Use the original dreamzero codebase for inference.')

    # ------------------------------------------------------------------
    # FSDP / DDP helpers
    # ------------------------------------------------------------------
    def get_fsdp_wrapping_policy(self) -> Callable:
        CausalWanModel, _ = _import_dreamzero_modules()
        # Only wrap the trainable DiT.
        return partial(
            _module_wrap_policy,
            module_classes={CausalWanModel},
        )
