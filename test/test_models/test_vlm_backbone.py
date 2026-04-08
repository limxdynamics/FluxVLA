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

from fluxvla.engines import build_vlm_backbone_from_cfg

QWEN2_5_VL_CKPT_PATH = './checkpoints/Qwen2.5-VL-3B-Instruct'


class TestPaligemmaBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='PaliGemma',
            vlm_backbone_id='paligemma_3b_pt_224',
            vlm_config=dict(
                vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type='paligemma',
                pad_token_id=0,
                projection_dim=2048,
                text_config=dict(
                    attention_bias=False,
                    attention_dropout=0.0,
                    head_dim=256,
                    hidden_act='gelu_pytorch_tanh',
                    hidden_activation='gelu_pytorch_tanh',
                    hidden_size=2048,
                    initializer_range=0.02,
                    intermediate_size=16384,
                    max_position_embeddings=8192,
                    model_type='gemma',
                    num_attention_heads=8,
                    num_hidden_layers=18,
                    num_image_tokens=256,
                    num_key_value_heads=1,
                    rms_norm_eps=1e-06,
                    rope_theta=10000.0,
                    torch_dtype='float32',
                    use_cache=True,
                    vocab_size=257152,
                ),
                transformers_version='4.52.4',
                vision_config=dict(
                    attention_dropout=0.0,
                    hidden_act='gelu_pytorch_tanh',
                    hidden_size=1152,
                    image_size=224,
                    intermediate_size=4304,
                    layer_norm_eps=1e-06,
                    model_type='siglip_vision_model',
                    num_attention_heads=16,
                    num_channels=3,
                    num_hidden_layers=27,
                    num_image_tokens=256,
                    patch_size=14,
                    projection_dim=2048,
                    projector_hidden_act='gelu_fast',
                    torch_dtype='float32',
                    vision_use_head=False,
                )))
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        import tensorflow as tf
        tf.random.set_seed(0)
        self.vlm_backbone = build_vlm_backbone_from_cfg(
            self.cfg).cuda().to(dtype=torch.bfloat16)

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_paligemma_backbone_forward(self):
        images = np.load(
            './test/data/models/vlm_backbones/paligemma/images.npy',
            allow_pickle=True)
        images = torch.from_numpy(images).cuda().reshape(2, 6, 224, 224)
        img_masks = np.load(
            './test/data/models/vlm_backbones/paligemma/img_masks.npy',
            allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            './test/data/models/vlm_backbones/paligemma/lang_tokens.npy',
            allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            './test/data/models/vlm_backbones/paligemma/lang_masks.npy',
            allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        with torch.no_grad():
            # Forward pass
            embs, pad_masks, attn_masks = self.vlm_backbone(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks)
            embs_target = np.load(
                './test/data/models/vlm_backbones/paligemma/embs.npy',
                allow_pickle=True)
            pad_masks_target = np.load(
                './test/data/models/vlm_backbones/paligemma/pad_masks.npy',
                allow_pickle=True)
            attn_masks_target = np.load(
                './test/data/models/vlm_backbones/paligemma/attn_masks.npy',
                allow_pickle=True)

        self.assertTrue(
            torch.allclose(
                embs.float()[:, ::10, ::10],
                torch.from_numpy(embs_target).cuda(),
                rtol=1e-4,
                atol=1e-1))

        self.assertTrue(
            torch.allclose(
                pad_masks,
                torch.from_numpy(pad_masks_target).cuda(),
                rtol=1e-3,
                atol=1e-3))
        self.assertTrue(
            torch.allclose(
                attn_masks,
                torch.from_numpy(attn_masks_target).cuda(),
                rtol=1e-3,
                atol=1e-3))


@pytest.mark.skipif(
    not os.path.exists(QWEN2_5_VL_CKPT_PATH),
    reason=f'Checkpoint not found: {QWEN2_5_VL_CKPT_PATH}')
class TestQWenVLBackbone(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='QWen2_5VL',
            vlm_backbone_id='qwen2_5_3b_vl_pt_224',
            vlm_path=  # noqa: E251
            QWEN2_5_VL_CKPT_PATH,  # noqa: E501
            vlm_config=dict(
                type='Qwen2_5_VLForConditionalGeneration',
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151645,
                vision_start_token_id=151652,
                vision_end_token_id=151653,
                vision_token_id=151654,
                image_token_id=151655,
                video_token_id=151656,
                hidden_act='silu',
                hidden_size=2048,
                initializer_range=0.02,
                intermediate_size=11008,
                max_position_embeddings=128000,
                max_window_layers=70,
                model_type='qwen2_5_vl',
                num_attention_heads=16,
                num_hidden_layers=36,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=32768,
                tie_word_embeddings=True,
                torch_dtype='bfloat16',
                transformers_version='4.41.2',
                use_cache=True,
                use_sliding_window=False,
                vision_config=dict(
                    depth=32,
                    hidden_act='silu',
                    hidden_size=1280,
                    intermediate_size=3420,
                    num_heads=16,
                    in_chans=3,
                    out_hidden_size=2048,
                    patch_size=14,
                    spatial_merge_size=2,
                    spatial_patch_size=14,
                    window_size=112,
                    fullatt_block_indexes=[7, 15, 23, 31],
                    tokens_per_second=2,
                    temporal_patch_size=2),
                rope_scaling=dict(type='mrope', mrope_section=[16, 24, 24]),
                vocab_size=151936))
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        import tensorflow as tf
        tf.random.set_seed(0)
        self.vlm_backbone = build_vlm_backbone_from_cfg(
            self.cfg).cuda().to(dtype=torch.bfloat16)

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_qwenvl_backbone_forward(self):
        images = np.load(
            './test/data/models/vlm_backbones/qwen_vl/images.npy',
            allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        lang_tokens = np.load(
            './test/data/models/vlm_backbones/qwen_vl/lang_tokens.npy',
            allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            './test/data/models/vlm_backbones/qwen_vl/lang_masks.npy',
            allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        img_masks = np.load(
            './test/data/models/vlm_backbones/qwen_vl/img_masks.npy',
            allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        image_grid_thw = np.load(
            './test/data/models/vlm_backbones/qwen_vl/image_grid_thw.npy',
            allow_pickle=True)
        image_grid_thw = torch.from_numpy(image_grid_thw).cuda()
        with torch.no_grad():
            # Forward pass
            embs, pad_masks, _ = self.vlm_backbone(
                images=images,
                img_masks=img_masks,
                image_grid_thw=image_grid_thw,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks)
            embs_target = np.load(
                'test/data/models/vlm_backbones/qwen_vl/last_hidden_state.npy',
                allow_pickle=True)
            pad_masks_target = np.load(
                './test/data/models/vlm_backbones/qwen_vl/pad_masks.npy',
                allow_pickle=True)

        self.assertTrue(
            torch.allclose(
                embs.float()[:, ::10, ::10],
                torch.from_numpy(embs_target).cuda(),
                rtol=1e-4,
                atol=1e-1))

        self.assertTrue(
            torch.allclose(
                pad_masks,
                torch.from_numpy(pad_masks_target).cuda(),
                rtol=1e-3,
                atol=1e-3))
