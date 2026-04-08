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
import unittest

import numpy as np
import pytest
import torch

from fluxvla.engines import build_projector_from_cfg


class TestHFCausalLLMBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.mlp_projector_cfg = {
            'type': 'MLPProjector',
            'vision_dim': 16,
            'llm_dim': 32,
        }
        self.linear_projector_cfg = {
            'type': 'LinearProjector',
            'in_dim': 16,
            'out_dim': 32,
        }
        self.fused_projector_cfg = {
            'type': 'FusedMLPProjector',
            'fused_vision_dim': 16,
            'llm_dim': 32,
        }
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        import tensorflow as tf
        tf.random.set_seed(0)
        self.mlp_projector = build_projector_from_cfg(
            self.mlp_projector_cfg).cuda()
        self.linear_projector = build_projector_from_cfg(
            self.linear_projector_cfg).cuda()
        self.fused_projector = build_projector_from_cfg(
            self.fused_projector_cfg).cuda()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_mlp_projector(self):
        img_patches = np.random.rand(1, 8, 16).astype(np.float32)
        output = self.mlp_projector(torch.from_numpy(img_patches).cuda())
        self.assertEqual(output.shape, (1, 8, 32))

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_linear_projector(self):
        img_patches = np.random.rand(1, 8, 16).astype(np.float32)
        output = self.linear_projector(torch.from_numpy(img_patches).cuda())
        self.assertEqual(output.shape, (1, 8, 32))

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_fused_projector(self):
        img_patches = np.random.rand(1, 8, 16).astype(np.float32)
        output = self.fused_projector(torch.from_numpy(img_patches).cuda())
        self.assertEqual(output.shape, (1, 8, 32))
