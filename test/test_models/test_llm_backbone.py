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

from fluxvla.engines import build_llm_backbone_from_cfg

LLAMA2_CKPT_PATH = './checkpoints/Llama-2-7b-hf'


@pytest.mark.skipif(
    not os.path.exists(LLAMA2_CKPT_PATH),
    reason=f'Checkpoint not found: {LLAMA2_CKPT_PATH}')
class TestLLaMaLLMBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = {
            'type': 'LLaMa2LLMBackbone',
            'llm_backbone_id': 'llama2-7b-pure_causal',
            'llm_family': 'llama',
            'llm_path': LLAMA2_CKPT_PATH,
            'llm_max_length': 2048,
            'hf_token': None,
            'inference_mode': False
        }
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        import tensorflow as tf
        tf.random.set_seed(0)
        self.llm_backbone = build_llm_backbone_from_cfg(self.cfg).cuda()
        self.llm_backbone.llm

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_llama_backbone_forward(self):
        attention_mask = np.load(
            './test/data/models/llm_backbones/llama/attention_mask.npy',
            allow_pickle=True)
        inputs_embeds = np.load(
            './test/data/models/llm_backbones/llama/inputs_embeds.npy',
            allow_pickle=True)
        labels = np.load(
            './test/data/models/llm_backbones/llama/labels.npy',
            allow_pickle=True)
        logits = np.load(
            './test/data/models/llm_backbones/llama/logits.npy',
            allow_pickle=True)
        inputs_embeds = torch.from_numpy(inputs_embeds).cuda()
        outputs = self.llm_backbone.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.from_numpy(attention_mask).cuda(),
            labels=torch.from_numpy(labels).cuda())
        self.assertTrue(
            torch.allclose(
                torch.from_numpy(logits).mean(),
                outputs['logits'].cpu().mean(),
                rtol=1e-2))


class TestGemmaLLMBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = {
            'type':
            'GemmaLLMBackbone',
            'llm_backbone_id':
            'gemma-2b_causal',
            'llm_family':
            'gemma',
            'llm_path':
            None,  # noqa: E501
            'llm_config':
            dict(
                vocab_size=257152,
                hidden_size=2048,
                intermediate_size=16384,
                num_hidden_layers=18,
                num_attention_heads=8,
                num_key_value_heads=1,
                head_dim=256,
                max_position_embeddings=4096,
                hidden_act='gelu',
                rms_norm_eps=1e-6,
                tie_word_embeddings=False),
            'llm_max_length':
            2048,
            'hf_token':
            None,
            'inference_mode':
            False,
            'tokenizer_length':
            257152
        }
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        import tensorflow as tf
        tf.random.set_seed(0)
        self.llm_backbone = build_llm_backbone_from_cfg(self.cfg).cuda()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_llama_backbone_forward(self):
        input_ids = np.load(
            './test/data/models/llm_backbones/gemma/input_ids.npy',
            allow_pickle=True)
        input_ids = torch.from_numpy(input_ids).cuda().unsqueeze(0)
        outputs = self.llm_backbone(input_ids=input_ids)
        self.assertEqual(outputs['logits'].shape, (1, 180, 257152))


class TestQWen2LLMBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = {
            'type':
            'Qwen2LLMBackbone',
            'llm_backbone_id':
            'qwen2-0.5b',
            'llm_family':
            'qwen2',
            'llm_path':
            None,
            'llm_config':
            dict(
                vocab_size=151936,
                hidden_size=896,
                intermediate_size=4864,
                num_hidden_layers=24,
                num_attention_heads=14,
                num_key_value_heads=2,
                max_position_embeddings=32768,
                hidden_act='silu',
                output_hidden_states=True,
                rms_norm_eps=1e-6),
        }
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        import tensorflow as tf
        tf.random.set_seed(0)
        self.llm_backbone = build_llm_backbone_from_cfg(self.cfg).cuda()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_qwen_backbone_forward(self):
        inputs_embeds = np.load(
            'test/data/models/llm_backbones/qwen/inputs_embeds.npy',
            allow_pickle=True)
        inputs_embeds = torch.from_numpy(inputs_embeds).cuda()
        outputs = self.llm_backbone(inputs_embeds=inputs_embeds)
        self.assertEqual(outputs['hidden_states'][-1].shape, (4, 390, 896))
