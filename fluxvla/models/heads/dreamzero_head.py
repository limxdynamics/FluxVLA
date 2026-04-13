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
        num_inference_steps: int = 4,
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
        self.num_inference_steps = num_inference_steps
        self.train_architecture = train_architecture
        self.skip_pretrained_loading = skip_pretrained_loading
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.use_cache = False

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

        self.reset_inference_state()
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
            dynamics_loss *
            self.scheduler.training_weight(timestep.flatten(0, 1)).unflatten(
                0, (noise.shape[0], noise.shape[1])).to(device))
        weighted_dynamics_loss = weight_dyn.mean()

        action_loss_raw = F.mse_loss(
            action_noise_pred.float(),
            training_target_action.float(),
            reduction='none',
        ) * action_masks.float()
        weight_act = (
            action_loss_raw.mean(dim=2) * self.scheduler.training_weight(
                timestep_action.flatten(0, 1)).unflatten(
                    0,
                    (noise_action.shape[0], noise_action.shape[1])).to(device))
        weighted_action_loss = weight_act.mean()

        loss = weighted_dynamics_loss + weighted_action_loss

        return dict(
            loss=loss,
            dynamics_loss=weighted_dynamics_loss,
            action_loss=weighted_action_loss,
        )

    def _create_kv_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        return [
            torch.zeros(
                2,
                batch_size,
                0,
                num_heads,
                head_dim,
                dtype=dtype,
                device=device,
            ) for _ in range(self.model.num_layers)
        ]

    def reset_inference_state(self) -> None:
        self.inference_kv_cache = None
        self.inference_clip_feas = None
        self.inference_ys = None
        self.inference_prompt_embs = None
        self.current_start_frame = 0

    def _should_reset_inference_state(
        self,
        prompt_embs: torch.Tensor,
        clip_feas: torch.Tensor,
        ys: torch.Tensor,
    ) -> bool:
        if self.inference_kv_cache is None:
            return True
        if (self.inference_prompt_embs is None
                or self.inference_clip_feas is None
                or self.inference_ys is None):
            return True
        if self.inference_prompt_embs.shape != prompt_embs.shape:
            return True
        if self.inference_clip_feas.shape != clip_feas.shape:
            return True
        if self.inference_ys.shape != ys.shape:
            return True
        if not torch.equal(self.inference_prompt_embs, prompt_embs):
            return True
        if not torch.equal(self.inference_clip_feas, clip_feas):
            return True
        if not torch.equal(self.inference_ys, ys):
            return True
        if getattr(self.model, 'local_attn_size', -1) != -1 \
                and self.current_start_frame >= self.model.local_attn_size:
            return True
        return False

    def _append_reference_frames(
        self,
        prompt_embs: torch.Tensor,
        reference_latents: torch.Tensor,
        clip_feas: torch.Tensor,
        ys: torch.Tensor,
        start_frame: int,
    ) -> None:
        if reference_latents.shape[2] == 0:
            return

        device = reference_latents.device
        batch_size = reference_latents.shape[0]
        timestep = torch.zeros(
            batch_size,
            reference_latents.shape[2],
            dtype=torch.int64,
            device=device,
        )
        frame_seqlen = int(reference_latents.shape[-2] *
                           reference_latents.shape[-1] / 4)
        _, _, updated_kv_cache = self.model(
            reference_latents,
            timestep=timestep,
            clip_feature=clip_feas,
            y=ys,
            context=prompt_embs,
            seq_len=reference_latents.shape[2] * frame_seqlen,
            action=None,
            timestep_action=None,
            state=None,
            embodiment_id=None,
            kv_cache=self.inference_kv_cache,
            crossattn_cache=None,
            current_start_frame=start_frame,
        )
        self.inference_kv_cache = updated_kv_cache

    def _sample_action_block(
        self,
        prompt_embs: torch.Tensor,
        clip_feas: torch.Tensor,
        ys: torch.Tensor,
        states: torch.Tensor,
        embodiment_ids: torch.Tensor,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
        denoise_frames: int,
        latents_dtype: torch.dtype,
        latents_shape: tuple[int, int, int, int],
        num_inference_steps: int,
    ) -> torch.Tensor:
        from fluxvla.models.third_party_models.dreamzero.modules.flow_unipc_multistep_scheduler import \
            FlowUniPCMultistepScheduler  # noqa: E501

        device = states.device
        b = states.shape[0]
        num_channels, lat_h, lat_w, frame_seqlen = latents_shape

        noisy_latents = torch.randn(
            b,
            num_channels,
            denoise_frames,
            lat_h,
            lat_w,
            device=device,
            dtype=latents_dtype,
        )
        noisy_actions = torch.randn(
            b,
            self.action_horizon,
            self.max_action_dim,
            device=device,
            dtype=latents_dtype,
        )

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(
            num_inference_steps,
            device=device,
            shift=5.0,
        )
        sample_scheduler_action.set_timesteps(
            num_inference_steps,
            device=device,
            shift=5.0,
        )

        y_future_start = min(current_start_frame, ys.shape[2])
        y_future_end = min(current_start_frame + denoise_frames, ys.shape[2])
        y_future = ys[:, :, y_future_start:y_future_end]
        if y_future.shape[2] < denoise_frames:
            y_future = ys[:, :, -denoise_frames:]

        denoise_seq_len = denoise_frames * frame_seqlen

        for step_index in range(len(sample_scheduler.timesteps)):
            video_timestep = sample_scheduler.timesteps[step_index]
            action_timestep = sample_scheduler_action.timesteps[step_index]

            t_video = video_timestep.expand(b, denoise_frames)
            t_action = action_timestep.expand(b, self.action_horizon)

            video_noise_pred, action_noise_pred, _ = self.model(
                noisy_latents,
                timestep=t_video,
                clip_feature=clip_feas,
                y=y_future,
                context=prompt_embs,
                seq_len=denoise_seq_len,
                state=states.to(torch.bfloat16),
                embodiment_id=embodiment_ids,
                action=noisy_actions,
                timestep_action=t_action,
                kv_cache=kv_cache,
                crossattn_cache=None,
                current_start_frame=current_start_frame,
            )

            noisy_latents = sample_scheduler.step(
                model_output=video_noise_pred.float(),
                timestep=video_timestep,
                sample=noisy_latents.float(),
                step_index=step_index,
                return_dict=False,
            )[0]
            noisy_actions = sample_scheduler_action.step(
                model_output=action_noise_pred.float(),
                timestep=action_timestep,
                sample=noisy_actions.float(),
                step_index=step_index,
                return_dict=False,
            )[0]

            noisy_latents = noisy_latents.to(dtype=latents_dtype)
            noisy_actions = noisy_actions.to(dtype=latents_dtype)

        return noisy_actions

    def _predict_action_stateless(
        self,
        prompt_embs: torch.Tensor,
        latents: torch.Tensor,
        clip_feas: torch.Tensor,
        ys: torch.Tensor,
        states: torch.Tensor,
        embodiment_ids: torch.Tensor,
        num_inference_steps: int,
        observed_latent_frames: int,
    ) -> torch.Tensor:
        local_kv_cache = self._create_kv_cache(
            batch_size=states.shape[0],
            dtype=latents.dtype,
            device=states.device,
        )
        observed_latents = latents[:, :, :observed_latent_frames]
        self.inference_kv_cache = local_kv_cache
        self._append_reference_frames(
            prompt_embs=prompt_embs,
            reference_latents=observed_latents[:, :, :1],
            clip_feas=clip_feas,
            ys=ys[:, :, :1],
            start_frame=0,
        )
        local_kv_cache = self.inference_kv_cache
        self.inference_kv_cache = None

        denoise_frames = self.num_frame_per_block
        if observed_latent_frames <= 1:
            denoise_frames = 1

        return self._sample_action_block(
            prompt_embs=prompt_embs,
            clip_feas=clip_feas,
            ys=ys,
            states=states,
            embodiment_ids=embodiment_ids,
            kv_cache=local_kv_cache,
            current_start_frame=1,
            denoise_frames=denoise_frames,
            latents_dtype=latents.dtype,
            latents_shape=(
                latents.shape[1],
                latents.shape[3],
                latents.shape[4],
                int(latents.shape[3] * latents.shape[4] / 4),
            ),
            num_inference_steps=num_inference_steps,
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
        num_inference_steps: Optional[int] = None,
        observed_latent_frames: Optional[int] = None,
        reset_history: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Joint video+action denoising with persistent causal history."""
        # Incoming latents are [B, T_lat, C, H_lat, W_lat] from DreamZeroVLA;
        # convert to model-facing [B, C, T_lat, H_lat, W_lat].
        latents = latents.transpose(1, 2)
        _, _, num_lat_frames, lat_h, lat_w = latents.shape

        if observed_latent_frames is None:
            observed_latent_frames = num_lat_frames
        observed_latent_frames = max(
            1, min(observed_latent_frames, num_lat_frames))

        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps

        use_cache = kwargs.get('use_cache', self.use_cache)
        if not use_cache:
            self.reset_inference_state()
            return self._predict_action_stateless(
                prompt_embs=prompt_embs,
                latents=latents,
                clip_feas=clip_feas,
                ys=ys,
                states=states,
                embodiment_ids=embodiment_ids,
                num_inference_steps=num_inference_steps,
                observed_latent_frames=observed_latent_frames,
            )

        device = states.device
        observed_latents = latents[:, :, :observed_latent_frames]
        latents_shape = (
            latents.shape[1],
            lat_h,
            lat_w,
            int(lat_h * lat_w / 4),
        )

        if reset_history or self._should_reset_inference_state(
                prompt_embs=prompt_embs, clip_feas=clip_feas, ys=ys):
            self.reset_inference_state()
            self.inference_kv_cache = self._create_kv_cache(
                batch_size=states.shape[0],
                dtype=latents.dtype,
                device=device,
            )
            self.inference_clip_feas = clip_feas
            self.inference_ys = ys
            self.inference_prompt_embs = prompt_embs

        if self.current_start_frame == 0:
            self._append_reference_frames(
                prompt_embs=prompt_embs,
                reference_latents=observed_latents[:, :, :1],
                clip_feas=self.inference_clip_feas,
                ys=self.inference_ys[:, :, :1],
                start_frame=0,
            )
            self.current_start_frame = 1

        if (self.current_start_frame != 1 and observed_latent_frames > 1):
            reference_latents = observed_latents[:, :,
                                                 -self.num_frame_per_block:]
            reference_start_frame = max(
                1,
                self.current_start_frame - reference_latents.shape[2],
            )
            y_ref_end = min(
                self.current_start_frame,
                self.inference_ys.shape[2],
            )
            y_ref_start = max(0, y_ref_end - reference_latents.shape[2])
            y_reference = self.inference_ys[:, :, y_ref_start:y_ref_end]
            if y_reference.shape[2] == 0:
                y_reference = self.inference_ys[:, :, :reference_latents.
                                                shape[2]]
            self._append_reference_frames(
                prompt_embs=prompt_embs,
                reference_latents=reference_latents,
                clip_feas=self.inference_clip_feas,
                ys=y_reference,
                start_frame=reference_start_frame,
            )

        denoise_frames = self.num_frame_per_block
        if observed_latent_frames <= 1:
            denoise_frames = 1

        noisy_actions = self._sample_action_block(
            prompt_embs=prompt_embs,
            clip_feas=self.inference_clip_feas,
            ys=self.inference_ys,
            states=states,
            embodiment_ids=embodiment_ids,
            kv_cache=self.inference_kv_cache,
            current_start_frame=self.current_start_frame,
            denoise_frames=denoise_frames,
            latents_dtype=latents.dtype,
            latents_shape=latents_shape,
            num_inference_steps=num_inference_steps,
        )

        self.current_start_frame += denoise_frames
        return noisy_actions

    # ------------------------------------------------------------------
    # FSDP / DDP helpers
    # ------------------------------------------------------------------
    def get_fsdp_wrapping_policy(self) -> Callable:
        from importlib import import_module

        _chunk = import_module(
            'fluxvla.models.third_party_models.dreamzero.modules.'
            'wan_video_dit_action_casual_chunk')
        CausalWanAttentionBlock = _chunk.CausalWanAttentionBlock

        # Wrap at the block level so FSDP shards each DiT block
        # individually, significantly reducing peak memory.
        return partial(
            _module_wrap_policy,
            module_classes={CausalWanAttentionBlock},
        )
