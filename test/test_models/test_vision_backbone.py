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

import gc
import os
import unittest

import numpy as np
import pytest
import torch

from fluxvla.engines import build_vision_backbone_from_cfg

DINO_CKPT_PATH = './checkpoints/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors'  # noqa: E501
SIGLIP_CKPT_PATH = './checkpoints/ViT-SO400M-14-SigLIP/open_clip_model.safetensors'  # noqa: E501


@pytest.mark.skipif(
    not os.path.exists(DINO_CKPT_PATH),
    reason=f'Checkpoint not found: {DINO_CKPT_PATH}')
class TestHFCausalVisionBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='DinoSigLIPViTBackbone',
            vision_backbone_id='dinosiglip-vit-so-224px',
            dino_config=dict(
                model_id='dino',
                file=  # noqa: E251
                DINO_CKPT_PATH),
            siglip_config=dict(
                model_id='siglip_224',
                file=  # noqa: E251
                SIGLIP_CKPT_PATH))
        self.siglip_vit = build_vision_backbone_from_cfg(self.cfg).cuda().to(
            torch.bfloat16)

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_siglip_vit_forward(self):
        input_dino = torch.from_numpy(
            np.load(
                'test/data/models/vision_backbones/input_dino.npy')).cuda().to(
                    torch.bfloat16)

        input_siglip = torch.from_numpy(
            np.load('test/data/models/vision_backbones/input_siglip.npy')
        ).cuda().to(torch.bfloat16)
        pixel_values = torch.cat([input_dino, input_siglip], dim=1)
        output = self.siglip_vit(pixel_values).float()
        expected_output = torch.from_numpy(
            np.load(
                'test/data/models/vision_backbones/output_dinosiglipvit.npy')
        ).cuda()
        assert torch.allclose(
            output[:, ::10, ::10], expected_output, rtol=1e-3, atol=1e-1)
