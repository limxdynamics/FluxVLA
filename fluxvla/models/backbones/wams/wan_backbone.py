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
from typing import Optional

import torch
import torch.nn as nn

from fluxvla.engines import WAM_BACKBONES

logger = logging.getLogger(__name__)


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


def _import_wan_encoder_modules():
    """Lazily import Wan encoder modules."""
    from fluxvla.models.third_party_models.dreamzero.modules.wan_video_image_encoder import \
        WanImageEncoder  # noqa: E501
    from fluxvla.models.third_party_models.dreamzero.modules.wan_video_text_encoder import \
        WanTextEncoder  # noqa: E501
    from fluxvla.models.third_party_models.dreamzero.modules.wan_video_vae import \
        WanVideoVAE  # noqa: E501
    return WanTextEncoder, WanImageEncoder, WanVideoVAE


@WAM_BACKBONES.register_module()
class WanBackbone(nn.Module):
    """Wan 2.1 encoder backbone: T5 text encoder, CLIP image encoder, VAE.

    These modules are always frozen during training and serve as the
    encoding frontend for DreamZero.

    Args:
        text_encoder_path: Path to T5 encoder weights (``.pth``).
        image_encoder_path: Path to CLIP weights (``.pth``).
        vae_path: Path to VAE weights (``.pth``).
        tiled: Whether to use tiled VAE encoding.
        skip_pretrained_loading: If True, skip loading pretrained weights.
    """

    def __init__(
        self,
        text_encoder_path: Optional[str] = None,
        image_encoder_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        tiled: bool = False,
        tile_size_height: int = 34,
        tile_size_width: int = 34,
        tile_stride_height: int = 18,
        tile_stride_width: int = 16,
        skip_pretrained_loading: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        WanTextEncoder, WanImageEncoder, WanVideoVAE = (
            _import_wan_encoder_modules())

        self.tiled = tiled
        self.tile_size_height = tile_size_height
        self.tile_size_width = tile_size_width
        self.tile_stride_height = tile_stride_height
        self.tile_stride_width = tile_stride_width

        self.text_encoder = WanTextEncoder(
            text_encoder_pretrained_path=text_encoder_path)
        self.image_encoder = WanImageEncoder(
            image_encoder_pretrained_path=image_encoder_path)
        self.vae = WanVideoVAE(vae_pretrained_path=vae_path)

        if not skip_pretrained_loading:
            self._load_pretrained_weights(text_encoder_path,
                                          image_encoder_path, vae_path)

        # Freeze all parameters
        self.requires_grad_(False)

    def _load_pretrained_weights(self, text_enc_path, img_enc_path, vae_path):
        """Load pretrained weights for T5, CLIP, and VAE."""
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

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()

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

    def forward(self, video, input_ids, attention_mask):
        self.set_frozen_modules_to_eval_mode()

        _, _, num_frames, height, width = video.shape
        first_frame = video[:, :, :1].transpose(1, 2)

        prompt_embs = self.encode_prompt(input_ids, attention_mask)
        latents = self.encode_video(video)
        clip_feas, image_cond, new_image = self.encode_image(
            first_frame, num_frames, height, width)

        device = video.device
        return dict(
            prompt_embs=prompt_embs.to(device),
            latents=latents.to(device),
            clip_feas=clip_feas.to(device),
            image_cond=image_cond.to(device),
            new_image=new_image.to(device),
        )


__all__ = ['WanBackbone']
