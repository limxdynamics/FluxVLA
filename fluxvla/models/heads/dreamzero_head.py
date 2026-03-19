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
from einops import rearrange
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torch.distributions import Beta
from transformers import AutoTokenizer

from fluxvla.engines import HEADS

logger = logging.getLogger(__name__)


def _import_dreamzero_modules():
    """Lazily import DreamZero modules so the rest of fluxvla still works
    even when optional dependencies (flash-attn, etc.) are missing."""
    from fluxvla.models.third_party_models.dreamzero.modules.flow_match_scheduler import \
        FlowMatchScheduler  # noqa: E501
    from fluxvla.models.third_party_models.dreamzero.modules.wan_video_dit_action_casual_chunk import \
        CausalWanModel  # noqa: E501
    from fluxvla.models.third_party_models.dreamzero.modules.wan_video_image_encoder import \
        WanImageEncoder  # noqa: E501
    from fluxvla.models.third_party_models.dreamzero.modules.wan_video_text_encoder import \
        WanTextEncoder  # noqa: E501
    from fluxvla.models.third_party_models.dreamzero.modules.wan_video_vae import \
        WanVideoVAE  # noqa: E501
    return (CausalWanModel, WanTextEncoder, WanImageEncoder, WanVideoVAE,
            FlowMatchScheduler)


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

    This head manages its own text encoder (T5 / umt5-xxl), image encoder
    (CLIP), VAE, and DiT diffusion model.  It is designed to be used with
    ``DreamZeroVLA`` which passes raw images and text descriptions.

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
        tokenizer_path: HuggingFace name or local path for T5 tokenizer.
        max_text_len: Maximum token length for T5 inputs.
        tiled: Whether to use tiled VAE encoding.
        skip_pretrained_loading: If True, skip loading *all* pretrained
            weights (T5, CLIP, VAE, DiT) – useful for unit testing.
        wan_model_path: Path to Wan 2.1 checkpoint directory.
        text_encoder_path: Path to T5 encoder weights (``.pth``).
        image_encoder_path: Path to CLIP weights (``.pth``).
        vae_path: Path to VAE weights (``.pth``).
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
        tokenizer_path: str = 'google/umt5-xxl',
        max_text_len: int = 512,
        tiled: bool = False,
        tile_size_height: int = 34,
        tile_size_width: int = 34,
        tile_stride_height: int = 18,
        tile_stride_width: int = 16,
        skip_pretrained_loading: bool = False,
        wan_model_path: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        image_encoder_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        use_gradient_checkpointing: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        (CausalWanModel, WanTextEncoder, WanImageEncoder, WanVideoVAE,
         FlowMatchScheduler) = _import_dreamzero_modules()

        self.action_dim = action_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.max_state_dim = max_state_dim
        self.num_frames = num_frames
        self.num_frame_per_block = num_frame_per_block
        self.tiled = tiled
        self.tile_size_height = tile_size_height
        self.tile_size_width = tile_size_width
        self.tile_stride_height = tile_stride_height
        self.tile_stride_width = tile_stride_width
        self.noise_s = noise_s
        self.train_architecture = train_architecture
        self.skip_pretrained_loading = skip_pretrained_loading
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block

        # ----- build sub-models -----
        self.text_encoder = WanTextEncoder(
            text_encoder_pretrained_path=text_encoder_path)
        self.image_encoder = WanImageEncoder(
            image_encoder_pretrained_path=image_encoder_path)
        self.vae = WanVideoVAE(vae_pretrained_path=vae_path)
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

        self.normalize_video = torch.nn.Identity()
        self._build_normalize_video()

        # ----- noise distributions -----
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)

        # ----- T5 tokenizer (for encoding task descriptions) -----
        self._tokenizer_path = tokenizer_path
        self._max_text_len = max_text_len
        self._tokenizer = None  # lazy init

        # ----- load pretrained weights -----
        if not skip_pretrained_loading:
            self._load_pretrained_weights(text_encoder_path,
                                          image_encoder_path, vae_path,
                                          wan_model_path)

        # ----- freeze encoders, set trainable -----
        self._setup_trainable(train_architecture, lora_rank, lora_alpha,
                              lora_target_modules)

        if use_gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        self.scheduler.set_timesteps(1000, training=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_normalize_video(self):
        from torchvision.transforms import v2
        self.normalize_video = v2.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path)
        return self._tokenizer

    def _load_pretrained_weights(self, text_enc_path, img_enc_path, vae_path,
                                 dit_path):
        """Load pretrained weights for T5, CLIP, VAE, and DiT."""
        try:
            path = _ensure_file(text_enc_path,
                                'models_t5_umt5-xxl-enc-bf16.pth')
            self.text_encoder.load_state_dict(
                torch.load(path, map_location='cpu'))
            logger.info('Loaded T5 text encoder weights.')
        except Exception as e:
            logger.warning('Could not load T5 weights: %s', e)

        try:
            path = _ensure_file(
                img_enc_path,
                'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')
            self.image_encoder.model.load_state_dict(
                torch.load(path, map_location='cpu'), strict=False)
            logger.info('Loaded CLIP image encoder weights.')
        except Exception as e:
            logger.warning('Could not load CLIP weights: %s', e)

        try:
            path = _ensure_file(vae_path, 'Wan2.1_VAE.pth')
            self.vae.model.load_state_dict(
                torch.load(path, map_location='cpu'))
            logger.info('Loaded VAE weights.')
        except Exception as e:
            logger.warning('Could not load VAE weights: %s', e)

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
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

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

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def encode_prompt(self, input_ids, attention_mask):
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def encode_video(self, input_video):
        vae_dev = next(self.vae.parameters()).device
        if self.vae.model.training:
            self.vae.eval()
        input_video = input_video.to(device=vae_dev, dtype=torch.bfloat16)
        with torch.no_grad():
            latents = self.vae.encode(
                input_video,
                tiled=self.tiled,
                tile_size=(self.tile_size_height, self.tile_size_width),
                tile_stride=(self.tile_stride_height, self.tile_stride_width))
        return latents

    def encode_image(self, image, num_frames, height, width):
        device = image.device
        with torch.amp.autocast(
                dtype=torch.bfloat16, device_type=torch.device(device).type):
            batch_size = image.shape[0]
            clip_context = self.image_encoder.encode_image(image)
            msk = torch.ones(
                batch_size, num_frames, height // 8, width // 8, device=device)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                msk[:, 1:],
            ],
                               dim=1)
            msk = msk.view(batch_size, msk.shape[1] // 4, 4, height // 8,
                           width // 8)
            msk = msk.transpose(1, 2)
            image_input = image.transpose(1, 2)
            image_zeros = torch.zeros(
                batch_size,
                3,
                num_frames - 1,
                height,
                width,
                dtype=torch.bfloat16,
                device=device)
            input_video = torch.concat([image_input, image_zeros], dim=2)
            input_video = input_video.to(
                device=next(self.vae.parameters()).device,
                dtype=torch.bfloat16)
            with torch.no_grad():
                y = self.vae.encode(input_video)
            new_image = y[:, :, 0:1]
            y = torch.concat([msk, y], dim=1)
        return clip_context, y, new_image

    def tokenize_text(self, texts):
        """Tokenize a list of raw text strings with T5."""
        tok_out = self.tokenizer(
            texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self._max_text_len,
        )
        return tok_out.input_ids, tok_out.attention_mask

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        task_description: list,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        embodiment_ids: torch.Tensor,
        **kwargs,
    ) -> Dict:
        """Training forward pass with flow-matching loss.

        Args:
            images: Video frames ``[B, C, T, H, W]`` in float [0,1] range
                or uint8 [0,255].
            task_description: List of *B* raw text strings.
            states: ``[B, num_state_tokens, state_dim]``.
            actions: ``[B, action_horizon, action_dim]`` in **[-1,1]**.
            action_masks: ``[B, action_horizon, action_dim]`` boolean.
            embodiment_ids: ``[B]`` integer embodiment category.

        Returns:
            dict with ``loss``, ``dynamics_loss``, ``action_loss``.
        """
        self.set_frozen_modules_to_eval_mode()
        device = actions.device

        # --- 1. Normalise video to [-1, 1] if needed ---
        videos = images
        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
        if videos.max() > 1.0:
            videos = videos / 255.0
        b, c, t, h, w = videos.shape
        videos_flat = rearrange(videos, 'b c t h w -> (b t) c h w')
        videos_flat = self.normalize_video(videos_flat)
        videos = rearrange(videos_flat, '(b t) c h w -> b c t h w', b=b, t=t)
        videos = videos.to(dtype=torch.bfloat16)

        # --- 2. Encode text with T5 ---
        text_ids, text_mask = self.tokenize_text(task_description)
        text_ids = text_ids.to(device)
        text_mask = text_mask.to(device)
        prompt_embs = self.encode_prompt(text_ids, text_mask)

        # --- 3. Encode video with VAE ---
        latents = self.encode_video(videos)

        # --- 4. Encode conditioning image with CLIP ---
        first_frame = videos[:, :, :1].transpose(1, 2)  # [B, 1, C, H, W]
        clip_feas, ys, _ = self.encode_image(first_frame, t, h, w)

        latents = latents.to(device)
        clip_feas = clip_feas.to(device)
        ys = ys.to(device)
        prompt_embs = prompt_embs.to(device)

        # --- 5. Flow-matching noise ---
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

        # --- 6. DiT forward ---
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

            # --- 7. Compute losses ---
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
        images: torch.Tensor,
        task_description: list,
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
        (CausalWanModel, WanTextEncoder, WanImageEncoder, WanVideoVAE,
         _) = _import_dreamzero_modules()
        return partial(
            _module_wrap_policy,
            module_classes={
                CausalWanModel, WanTextEncoder, WanImageEncoder, WanVideoVAE
            },
        )
