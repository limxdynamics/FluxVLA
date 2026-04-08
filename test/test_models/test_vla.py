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
import torch.nn.functional as F

from fluxvla.engines import build_vla_from_cfg, set_seed_everywhere

OPENVLA_CKPT_PATH = './checkpoints/openvla-7b-finetuned-libero-10'
LLAMA2_CKPT_PATH = './checkpoints/Llama-2-7b-hf'
DINO_CKPT_PATH = './checkpoints/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors'  # noqa: E501
SIGLIP_CKPT_PATH = './checkpoints/ViT-SO400M-14-SigLIP/open_clip_model.safetensors'  # noqa: E501
PI0_CKPT_PATH = './checkpoints/pi0_base/model.safetensors'
PI05_CKPT_PATH = './checkpoints/pi05_base/model.safetensors'
GR00T_CKPT_PATH = './checkpoints/GR00T-N1.5-3B'
DREAMZERO_CKPT_PATH = './checkpoints/DreamZero-AgiBot'
SMOLVLA_CKPT_PATH = './checkpoints/smolvla_base/model.safetensors'
OPENVLA_DATA_DIR = 'test/data/models/vlas/openvla'
LLAVAVLA_DATA_DIR = 'test/data/models/vlas/llavavla'
GR00T_DATA_DIR = 'test/data/models/vlas/gr00t'
PI0_DATA_DIR = 'test/data/models/vlas/pi0'
PI05_DATA_DIR = 'test/data/models/vlas/pi05'
DREAMZERO_DATA_DIR = 'test/data/models/vlas/dreamzero'
DREAMZERO_NUM_INFERENCE_STEPS = 2


@pytest.mark.skipif(
    not os.path.exists(OPENVLA_CKPT_PATH)
    or not os.path.exists(LLAMA2_CKPT_PATH)
    or not os.path.exists(DINO_CKPT_PATH)
    or not os.path.exists(SIGLIP_CKPT_PATH),
    reason=f'Checkpoint not found: {OPENVLA_CKPT_PATH}')
class TestOpenVLA(unittest.TestCase):

    def setUp(self):
        #  TODO: Find a way to test use_flash_attention
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='OpenVLA',
            vision_backbone=dict(
                type='DinoSigLIPViTBackbone',
                vision_backbone_id='dinosiglip-vit-so-224px',
                dino_config=dict(
                    model_id='dino',
                    file=  # noqa: E251
                    DINO_CKPT_PATH),
                image_resize_strategy='resize-naive',
                siglip_config=dict(
                    model_id='siglip_224',
                    file=  # noqa: E251
                    SIGLIP_CKPT_PATH)),
            llm_backbone=dict(
                type='LLaMa2LLMBackbone',
                llm_backbone_id='llama2-7b-pure_causal',
                llm_family='llama',
                llm_path=  # noqa: E251
                LLAMA2_CKPT_PATH,  # noqa: E501
                llm_max_length=2048,
                hf_token=None,
                inference_mode=False),
            projector=dict(
                type='FusedMLPProjector', fused_vision_dim=2176, llm_dim=4096),
            tokenizer=dict(
                type='ActionTokenizer',
                model_path=  # noqa: E251
                OPENVLA_CKPT_PATH,  # noqa: E501
                bins=256,
                min_action=-1,
                max_action=1,
            ),
            pretrained_name_or_path=None,  # noqa: E501
            vla_head=dict(
                type='OpenVLAHead', norm_stats=None, vocab_size=32000),
            freeze_vision_backbone=False,
            freeze_llm_backbone=False,
            freeze_projector=False)
        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).cuda()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_forward(self):
        input_ids = torch.from_numpy(
            np.load(os.path.join(OPENVLA_DATA_DIR, 'input_ids.npy'))).cuda()
        attention_mask = torch.from_numpy(
            np.load(os.path.join(OPENVLA_DATA_DIR,
                                 'attention_mask.npy'))).cuda()
        pixel_values_dino = torch.from_numpy(
            np.load(os.path.join(OPENVLA_DATA_DIR,
                                 'pixel_values_dino.npy'))).cuda()
        pixel_values_siglip = torch.from_numpy(
            np.load(os.path.join(OPENVLA_DATA_DIR,
                                 'pixel_values_siglip.npy'))).cuda()
        pixel_values = torch.cat([pixel_values_dino, pixel_values_siglip],
                                 dim=1)
        labels = torch.from_numpy(
            np.load(os.path.join(OPENVLA_DATA_DIR, 'labels.npy'))).cuda()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                output, _ = self.vla.forward_model(input_ids, attention_mask,
                                                   pixel_values.bfloat16(),
                                                   labels)
        expected_loss = np.load(os.path.join(OPENVLA_DATA_DIR, 'loss.npy'))
        expected_logits = np.load(os.path.join(OPENVLA_DATA_DIR, 'logits.npy'))
        self.assertAlmostEqual(
            output['loss'].cpu().detach().numpy(), expected_loss, delta=1e-2)
        self.assertTrue(
            np.allclose(
                output['logits'].cpu().float().detach().numpy()[:, ::10, ::10],
                expected_logits,
                rtol=1e-3,
                atol=1e-1))


@pytest.mark.skipif(
    not os.path.exists(GR00T_CKPT_PATH),
    reason=f'Checkpoint not found: {GR00T_CKPT_PATH}')
class TestGr00t(unittest.TestCase):

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='LlavaVLA',
            pretrained_name_or_path=  # noqa: E251
            './checkpoints/GR00T-N1.5-3B',
            vlm_backbone=dict(
                type='EagleBackbone',
                vlm_path=  # noqa: E251
                'fluxvla/models/third_party_models/eagle2_hg_model'),
            vla_head=dict(
                type='FlowMatchingHead',
                state_dim=64,
                hidden_size=1024,
                input_embedding_dim=1536,
                num_layers=1,
                num_heads=4,
                num_inference_timesteps=4,
                traj_length=10,
                action_dim=32,
                ori_action_dim=7),
            freeze_vlm_backbone=False,
            name_mapping={
                'vlm_backbone.vlm': 'backbone.eagle_model',
                'vla_head': 'action_head'
            },
            freeze_projector=False)
        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).cuda()
        self.vla.from_pretrained()
        self.vla.eval()

    def test_forward(self):
        images = np.load(
            os.path.join(LLAVAVLA_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda().repeat_interleave(
            8, dim=2).repeat_interleave(
                8, dim=3)
        img_masks = np.load(
            os.path.join(LLAVAVLA_DATA_DIR, 'img_masks.npy'),
            allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(GR00T_DATA_DIR, 'lang_tokens.npy'),
            allow_pickle=True).repeat(2, 0)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(GR00T_DATA_DIR, 'lang_tokens.npy'),
            allow_pickle=True).repeat(2, 0)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = torch.from_numpy(
            np.load(os.path.join(LLAVAVLA_DATA_DIR, 'states.npy'))).cuda()
        states = F.pad(states, (0, 64 - states.shape[-1]))
        actions = torch.from_numpy(
            np.load(os.path.join(LLAVAVLA_DATA_DIR, 'actions.npy'))).cuda()
        actions = F.pad(actions, (0, 32 - actions.shape[-1]))
        embodiment_ids = torch.ones((2)).cuda().long()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                output = self.vla.forward(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states.bfloat16(),
                    actions=actions.bfloat16(),
                    action_masks=torch.ones((1, 10)).cuda(),
                    embodiment_ids=embodiment_ids)
        self.assertTrue(
            np.allclose(
                output['loss'].cpu().detach().numpy(), 0.5135, atol=1e-2))

    def test_predict_action(self):
        images = np.load(
            os.path.join(LLAVAVLA_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda().repeat_interleave(
            8, dim=2).repeat_interleave(
                8, dim=3)
        img_masks = np.load(
            os.path.join(LLAVAVLA_DATA_DIR, 'img_masks.npy'),
            allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(GR00T_DATA_DIR, 'lang_tokens.npy'),
            allow_pickle=True).repeat(2, 0)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(GR00T_DATA_DIR, 'lang_tokens.npy'),
            allow_pickle=True).repeat(2, 0)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = torch.from_numpy(
            np.load(os.path.join(LLAVAVLA_DATA_DIR, 'states.npy'))).cuda()
        states = F.pad(states, (0, 64 - states.shape[-1]))
        pred_actions_target = np.load(
            os.path.join(GR00T_DATA_DIR, 'pred_actions.npy'),
            allow_pickle=True)
        embodiment_ids = torch.ones((2)).cuda().long()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                pred_actions = self.vla.predict_action(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states,
                    embodiment_ids=embodiment_ids)
        self.assertTrue(
            np.allclose(
                pred_actions.cpu().detach().numpy(),
                pred_actions_target,
                atol=1e-2))


@pytest.mark.skipif(
    not os.path.exists(PI0_CKPT_PATH),
    reason=f'Checkpoint not found: {PI0_CKPT_PATH}')
class TestPI0FlowMatching(unittest.TestCase):

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='PI0FlowMatching',
            llm_backbone=dict(
                type='ConditionGemmaModel',
                adarms_cond_dim=None,
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
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
                num_key_value_heads=1,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype='float32',
                use_cache=True,
                vocab_size=257152,
            ),
            vision_backbone=dict(
                type='SigLIPViTBackbone',
                vision_backbone_id='siglip_224',
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
                    patch_size=14,
                    projection_dim=2048,
                    projector_hidden_act='gelu_fast',
                    torch_dtype='float32',
                    vision_use_head=False,
                ),
            ),
            projector=dict(
                type='LinearProjector',
                in_dim=1152,
                out_dim=2048,
            ),
            proj_width=1024,
            n_action_steps=50,
            state_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
            action_in_proj=dict(
                type='LinearProjector', in_dim=32, out_dim=1024),
            action_out_proj=dict(
                type='LinearProjector', in_dim=1024, out_dim=32),
            action_time_mlp_in=dict(
                type='LinearProjector', in_dim=2048, out_dim=1024),
            action_time_mlp_out=dict(
                type='LinearProjector', in_dim=1024, out_dim=1024),
            max_action_dim=32,
            llm_expert=dict(
                type='ConditionGemmaModel',
                attention_bias=False,
                adarms_cond_dim=None,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act='gelu_pytorch_tanh',
                hidden_activation='gelu_pytorch_tanh',
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type='gemma',
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype='float32',
                transformers_version='4.48.1',
                use_adarms=False,
                use_cache=True,
                vocab_size=257152),
            freeze_llm_backbone=False,
            freeze_vision_backbone=False,
            pretrained_name_or_path=  # noqa: E251
            './checkpoints/pi0_base/model.safetensors',  # noqa: E501
            name_mapping={
                'llm_backbone':
                'paligemma_with_expert.paligemma.model.language_model',
                'vision_backbone.vision':
                'paligemma_with_expert.paligemma.model.vision_tower',
                'projector.projector':
                'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',  # noqa: E501
                'llm_expert':
                'paligemma_with_expert.gemma_expert.model',
                'action_time_mlp_in.projector':
                'action_time_mlp_in',
                'action_time_mlp_out.projector':
                'action_time_mlp_out',
                'state_proj.projector':
                'state_proj',
                'action_in_proj.projector':
                'action_in_proj',
                'action_out_proj.projector':
                'action_out_proj',
                'llm_backbone.embed_tokens':
                'paligemma_with_expert.paligemma.lm_head',
            },
            params_to_change_dtype=[
                'llm_expert.llm.model.layers',
                'vlm_backbone.vlm.model.language_model.layers',
                'vlm_backbone.vlm.model.vision_tower',
                'vlm_backbone.vlm.model.multi_modal_projector',
            ],
            ori_action_dim=7,
        )
        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).cuda()
        self.vla.from_pretrained()
        self.vla.eval()

    def test_prefix_forward(self):
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        embs_target = np.load(
            os.path.join(PI0_DATA_DIR, 'prefix_embs.npy'), allow_pickle=True)
        pad_masks_target = np.load(
            os.path.join(PI0_DATA_DIR, 'prefix_pad_masks.npy'),
            allow_pickle=True)
        att_masks_target = np.load(
            os.path.join(PI0_DATA_DIR, 'prefix_att_masks.npy'),
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                embs, pad_masks, att_masks = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks)
        self.assertTrue(
            np.allclose(
                embs.float().cpu().detach().numpy()[:, ::10, ::10],
                embs_target,
                atol=1e-1))
        self.assertTrue(
            np.allclose(pad_masks.float().cpu().detach().numpy(),
                        pad_masks_target))
        self.assertTrue(
            np.allclose(att_masks.float().cpu().detach().numpy(),
                        att_masks_target))

    def test_suffix_forward(self):
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_time.npy'), allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_x_t.npy'), allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        suffix_embs_target = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_embs.npy'), allow_pickle=True)
        suffix_pad_masks_target = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_pad_masks.npy'),
            allow_pickle=True)
        suffix_att_masks_target = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_att_masks.npy'),
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                    adarms_cond,
                ) = self.vla.embed_suffix(states, x_t, time)

        self.assertTrue(
            np.allclose(
                suffix_embs.float().cpu().detach().numpy()[:, :, ::10],
                suffix_embs_target,
                atol=1e-2))
        self.assertTrue(
            np.allclose(suffix_pad_masks.float().cpu().detach().numpy(),
                        suffix_pad_masks_target))
        self.assertTrue(
            np.allclose(suffix_att_masks.float().cpu().detach().numpy(),
                        suffix_att_masks_target))

    def test_forward(self):
        from fluxvla.engines.utils.model_utils import make_att_2d_masks
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_time.npy'), allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            os.path.join(PI0_DATA_DIR, 'suffix_x_t.npy'), allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        actions_target = np.load(
            os.path.join(PI0_DATA_DIR, 'actions.npy'), allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                ) = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                )

                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                    adarms_cond,
                ) = self.vla.embed_suffix(states, x_t, time)

                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks],
                                      dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks],
                                      dim=1)

                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1

                att_2d_masks_4d = self.vla._prepare_attention_masks_4d(
                    att_2d_masks)

                suffix_out, _ = self.vla.forward_model(
                    inputs_embeds=[prefix_embs, suffix_embs],
                    attention_masks=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    fill_kv_cache=None,
                    adarms_cond=[None, adarms_cond],
                    time=time)

                actions = self.vla.action_out_proj(
                    suffix_out[:, -self.vla.n_action_steps:])

                self.assertTrue(
                    np.allclose(
                        actions.float().cpu().detach().numpy(),
                        actions_target,
                        atol=1e-1))

    def test_predict_actions(self):
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        noise = np.load(
            os.path.join(PI0_DATA_DIR, 'noise.npy'), allow_pickle=True)
        noise = torch.from_numpy(noise).cuda()
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        pred_actions_target = np.load(
            os.path.join(PI0_DATA_DIR, 'pred_actions.npy'), allow_pickle=True)

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float32, enabled=True):
                actions = self.vla.predict_action(
                    images=images,
                    states=states,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    noise=noise)

        self.assertTrue(
            np.allclose(
                actions.float().cpu().detach().numpy(),
                pred_actions_target,
                atol=5e-1))


@pytest.mark.skipif(
    not os.path.exists(PI05_CKPT_PATH),
    reason=f'Checkpoint not found: {PI05_CKPT_PATH}')
class TestPI05FlowMatching(unittest.TestCase):

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='PI05FlowMatching',
            llm_backbone=dict(
                type='ConditionGemmaModel',
                adarms_cond_dim=None,
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
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
                num_key_value_heads=1,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype='float32',
                use_cache=True,
                vocab_size=257152,
            ),
            vision_backbone=dict(
                type='SigLIPViTBackbone',
                vision_backbone_id='siglip_224',
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
                    patch_size=14,
                    projection_dim=2048,
                    projector_hidden_act='gelu_fast',
                    torch_dtype='float32',
                    vision_use_head=False,
                ),
            ),
            projector=dict(
                type='LinearProjector',
                in_dim=1152,
                out_dim=2048,
            ),
            proj_width=1024,
            n_action_steps=10,
            action_in_proj=dict(
                type='LinearProjector', in_dim=32, out_dim=1024),
            action_out_proj=dict(
                type='LinearProjector', in_dim=1024, out_dim=32),
            time_mlp_in=dict(
                type='LinearProjector', in_dim=1024, out_dim=1024),
            time_mlp_out=dict(
                type='LinearProjector', in_dim=1024, out_dim=1024),
            max_action_dim=32,
            llm_expert=dict(
                type='ConditionGemmaModel',
                attention_bias=False,
                adarms_cond_dim=1024,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act='gelu_pytorch_tanh',
                hidden_activation='gelu_pytorch_tanh',
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type='gemma',
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype='float32',
                transformers_version='4.48.1',
                use_adarms=True,
                use_cache=True,
                vocab_size=257152),
            freeze_llm_backbone=False,
            freeze_vision_backbone=False,
            pretrained_name_or_path=  # noqa: E251
            PI05_CKPT_PATH,  # noqa: E501
            name_mapping={
                'llm_backbone':
                'paligemma_with_expert.paligemma.model.language_model',
                'llm_backbone.embed_tokens':
                'paligemma_with_expert.paligemma.lm_head',
                'vision_backbone.vision':
                'paligemma_with_expert.paligemma.model.vision_tower',
                'projector.projector': 'paligemma_with_expert.paligemma.model.'
                'multi_modal_projector.linear',
                'llm_expert': ('paligemma_with_expert.gemma_expert.'
                               'model'),
                'time_mlp_in.projector': 'time_mlp_in',
                'time_mlp_out.projector': 'time_mlp_out',
                'action_in_proj.projector': 'action_in_proj',
                'action_out_proj.projector': 'action_out_proj',
            },
            params_to_change_dtype=[
                'llm_expert.llm.model.layers',
                'vlm_backbone.vlm.model.language_model.layers',
                'vlm_backbone.vlm.model.vision_tower',
                'vlm_backbone.vlm.model.multi_modal_projector',
            ])
        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).cuda()
        self.vla.from_pretrained()
        self.vla.eval()

    def test_prefix_forward(self):
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        embs_target = np.load(
            os.path.join(PI05_DATA_DIR, 'prefix_embs.npy'), allow_pickle=True)
        pad_masks_target = np.load(
            os.path.join(PI05_DATA_DIR, 'prefix_pad_masks.npy'),
            allow_pickle=True)
        att_masks_target = np.load(
            os.path.join(PI05_DATA_DIR, 'prefix_att_masks.npy'),
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                embs, pad_masks, att_masks = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks)
        self.assertTrue(
            np.allclose(
                embs.float().cpu().detach().numpy()[:, ::10, ::10],
                embs_target,
                atol=5e-1))
        self.assertTrue(
            np.allclose(pad_masks.float().cpu().detach().numpy(),
                        pad_masks_target))
        self.assertTrue(
            np.allclose(att_masks.float().cpu().detach().numpy(),
                        att_masks_target))

    def test_suffix_forward(self):
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_time.npy'), allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_x_t.npy'), allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        suffix_embs_target = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_embs.npy'), allow_pickle=True)
        suffix_pad_masks_target = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_pad_masks.npy'),
            allow_pickle=True)
        suffix_att_masks_target = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_att_masks.npy'),
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                    adarms_cond,
                ) = self.vla.embed_suffix(states, x_t, time)

        self.assertTrue(
            np.allclose(
                suffix_embs.float().cpu().detach().numpy()[:, :, ::10],
                suffix_embs_target,
                atol=1e-2))
        self.assertTrue(
            np.allclose(suffix_pad_masks.float().cpu().detach().numpy(),
                        suffix_pad_masks_target))
        self.assertTrue(
            np.allclose(suffix_att_masks.float().cpu().detach().numpy(),
                        suffix_att_masks_target))

    def test_forward(self):
        from fluxvla.engines.utils.model_utils import make_att_2d_masks
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_time.npy'), allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_x_t.npy'), allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        actions_target = np.load(
            os.path.join(PI05_DATA_DIR, 'actions.npy'), allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                ) = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                )

                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                    adarms_cond,
                ) = self.vla.embed_suffix(states, x_t, time)

                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks],
                                      dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks],
                                      dim=1)

                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1

                att_2d_masks_4d = self.vla._prepare_attention_masks_4d(
                    att_2d_masks)

                suffix_out, _ = self.vla.forward_model(
                    inputs_embeds=[prefix_embs, suffix_embs],
                    attention_masks=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    fill_kv_cache=None,
                    adarms_cond=[None, adarms_cond],
                    time=time)

                actions = self.vla.action_out_proj(suffix_out)

                self.assertTrue(
                    np.allclose(
                        actions.float().cpu().detach().numpy(),
                        actions_target,
                        atol=1e-1))

    def test_predict_actions(self):
        images = np.load(
            os.path.join(PI05_DATA_DIR, 'images.npy'), allow_pickle=True)
        images = torch.from_numpy(images).cuda()
        img_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'img_masks.npy'), allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_tokens.npy'), allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            os.path.join(PI05_DATA_DIR, 'lang_masks.npy'), allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        noise = np.load(
            os.path.join(PI05_DATA_DIR, 'noise.npy'), allow_pickle=True)
        noise = torch.from_numpy(noise).cuda()
        states = np.load(
            os.path.join(PI05_DATA_DIR, 'suffix_state.npy'), allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        pred_actions_target = np.load(
            os.path.join(PI05_DATA_DIR, 'pred_actions.npy'), allow_pickle=True)

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                actions = self.vla.predict_action(
                    images=images,
                    states=states,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    noise=noise)

        self.assertTrue(
            np.allclose(
                actions.float().cpu().detach().numpy(),
                pred_actions_target,
                atol=1e-1))


@unittest.skipUnless(
    torch.cuda.is_available() and os.path.exists(DREAMZERO_CKPT_PATH),
    'DreamZero checkpoint not available or CUDA is not available')
class TestDreamZero(unittest.TestCase):
    """Compare DreamZero forward outputs with reference implementation IO."""

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

        self.cfg = dict(
            type='DreamZeroVLA',
            num_views=2,
            frame_window_size=9,
            pretrained_name_or_path=DREAMZERO_CKPT_PATH,
            vlm_backbone=dict(
                type='WanBackbone',
                text_encoder_path=None,
                image_encoder_path=None,
                vae_path=None,
                tiled=False,
                skip_pretrained_loading=True,
            ),
            vla_head=dict(
                type='DreamZeroHead',
                action_dim=7,
                max_action_dim=32,
                action_horizon=10,
                max_state_dim=64,
                num_frames=9,
                num_frame_per_block=2,
                num_action_per_block=10,
                num_state_per_block=1,
                frame_seqlen=8,
                hidden_size=1024,
                input_embedding_dim=1536,
                dit_dim=5120,
                dit_ffn_dim=13824,
                dit_num_heads=40,
                dit_num_layers=40,
                dit_freq_dim=256,
                dit_in_dim=36,
                dit_out_dim=16,
                max_num_embodiments=32,
                noise_beta_alpha=1.5,
                noise_beta_beta=1.0,
                noise_s=0.999,
                num_inference_steps=DREAMZERO_NUM_INFERENCE_STEPS,
                train_architecture='full',
                skip_pretrained_loading=True,
                wan_model_path=None,
                use_gradient_checkpointing=True,
            ),
            name_mapping={
                'vla_head.model': 'action_head.model',
                'vlm_backbone.text_encoder': 'action_head.text_encoder',
                'vlm_backbone.image_encoder': 'action_head.image_encoder',
                'vlm_backbone.vae': 'action_head.vae',
            },
            strict_mapping=False,
            freeze_llm_backbone=True,
            freeze_vlm_backbone=True,
            freeze_projector=True,
        )

        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).bfloat16().cuda()
        self.vla.from_pretrained()
        self.vla.eval()

    def _load_tensor(self, root, name):
        return np.load(os.path.join(root, f'{name}.npy'), allow_pickle=True)

    def test_forward(self):
        if not os.path.isdir(DREAMZERO_DATA_DIR):
            self.skipTest(
                f'DreamZero data dir not found: {DREAMZERO_DATA_DIR}')

        images = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'images')).cuda().to(torch.bfloat16)
        lang_tokens = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'lang_tokens')).cuda().long()
        lang_masks = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR, 'lang_masks')).cuda().long()
        states = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'states')).cuda().to(torch.bfloat16)
        actions = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'actions')).cuda().to(torch.bfloat16)
        action_masks = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'action_masks')).cuda().bool()
        embodiment_ids = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'embodiment_ids')).cuda().long()

        loss_ref = self._load_tensor(DREAMZERO_DATA_DIR, 'loss')
        dynamics_loss_ref = self._load_tensor(DREAMZERO_DATA_DIR,
                                              'dynamics_loss')
        action_loss_ref = self._load_tensor(DREAMZERO_DATA_DIR, 'action_loss')

        set_seed_everywhere(0)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                output = self.vla.forward(
                    images=images,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states,
                    actions=actions,
                    action_masks=action_masks,
                    embodiment_ids=embodiment_ids,
                )

        self.assertTrue(
            np.allclose(
                output['loss'].float().cpu().numpy(), loss_ref, atol=1e-3))
        self.assertTrue(
            np.allclose(
                output['dynamics_loss'].float().cpu().numpy(),
                dynamics_loss_ref,
                atol=1e-3))
        self.assertTrue(
            np.allclose(
                output['action_loss'].float().cpu().numpy(),
                action_loss_ref,
                atol=1e-3))

    def test_predict_action(self):
        if not os.path.isdir(DREAMZERO_DATA_DIR):
            self.skipTest(
                f'DreamZero data dir not found: {DREAMZERO_DATA_DIR}')

        pred_actions_path = os.path.join(DREAMZERO_DATA_DIR,
                                         'pred_actions.npy')
        if not os.path.exists(pred_actions_path):
            self.skipTest(
                f'DreamZero pred_actions not found: {pred_actions_path}')

        images = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'images')).cuda().to(torch.bfloat16)
        lang_tokens = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'lang_tokens')).cuda().long()
        lang_masks = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR, 'lang_masks')).cuda().long()
        states = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'states')).cuda().to(torch.bfloat16)
        embodiment_ids = torch.from_numpy(
            self._load_tensor(DREAMZERO_DATA_DIR,
                              'embodiment_ids')).cuda().long()

        pred_actions_ref = self._load_tensor(DREAMZERO_DATA_DIR,
                                             'pred_actions')

        set_seed_everywhere(0)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                pred_actions = self.vla.predict_action(
                    images=images,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states,
                    embodiment_ids=embodiment_ids,
                )

        self.assertTrue(
            np.allclose(
                pred_actions.float().cpu().numpy(),
                pred_actions_ref,
                atol=1e-2))


@pytest.mark.skipif(
    not os.path.exists(SMOLVLA_CKPT_PATH),
    reason=f'Checkpoint not found: {SMOLVLA_CKPT_PATH}')
class TestSmolVLAFlowMatching(unittest.TestCase):

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.cfg = dict(
            type='SmolVLAFlowMatching',
            vlm_backbone=dict(
                type='SmolVLMBackbone',
                vision_config=dict(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    image_size=512,
                    patch_size=16,
                    intermediate_size=3072,
                    hidden_act='gelu_pytorch_tanh',
                    layer_norm_eps=1e-6,
                ),
                text_config=dict(
                    hidden_size=960,
                    num_hidden_layers=32,
                    num_attention_heads=15,
                    num_key_value_heads=5,
                    head_dim=64,
                    intermediate_size=2560,
                    vocab_size=49280,
                    rms_norm_eps=1e-5,
                    hidden_act='silu',
                    max_position_embeddings=8192,
                ),
                scale_factor=4,
                num_vlm_layers=16,
            ),
            llm_expert=dict(
                type='SmolVLMExpert',
                hidden_size=720,
                num_hidden_layers=16,
                num_attention_heads=15,
                num_key_value_heads=5,
                head_dim=64,
                intermediate_size=-1,
                vocab_size=49280,
                attention_bias=False,
                rms_norm_eps=1e-5,
                hidden_act='silu',
                max_position_embeddings=8192,
                attention_mode='cross_attn',
                vlm_kv_dim=320,
                self_attn_every_n_layers=2,
            ),
            state_proj=dict(type='LinearProjector', in_dim=32, out_dim=960),
            action_in_proj=dict(
                type='LinearProjector', in_dim=32, out_dim=720),
            action_out_proj=dict(
                type='LinearProjector', in_dim=720, out_dim=32),
            action_time_mlp_in=dict(
                type='LinearProjector', in_dim=1440, out_dim=720),
            action_time_mlp_out=dict(
                type='LinearProjector', in_dim=720, out_dim=720),
            freeze_vlm_backbone=True,
            max_action_dim=32,
            ori_action_dim=7,
            chunk_size=50,
            num_steps=10,
            add_image_special_tokens=False,
            use_cache=True,
            pretrained_name_or_path=SMOLVLA_CKPT_PATH,
            name_mapping={
                'vlm_backbone.vlm': 'model.vlm_with_expert.vlm.model',
                'llm_expert.expert': 'model.vlm_with_expert.lm_expert',
                'state_proj.projector': 'model.state_proj',
                'action_in_proj.projector': 'model.action_in_proj',
                'action_out_proj.projector': 'model.action_out_proj',
                'action_time_mlp_in.projector': 'model.action_time_mlp_in',
                'action_time_mlp_out.projector': 'model.action_time_mlp_out',
            })
        set_seed_everywhere(0)
        self.vla = build_vla_from_cfg(self.cfg).cuda()
        self.vla.from_pretrained()
        self.vla.eval()

    def _load_images(self):
        """Load single uint8 image and reconstruct (B, N_cam*3, H, W)."""
        img = np.load(
            'test/data/models/vlas/smolvla/image.npy')  # (3,H,W) uint8
        img_t = torch.from_numpy(img).float() / 255.0  # (3, H, W)
        B, N_cam = 2, 2
        images = img_t[None].repeat(B * N_cam, 1, 1, 1)  # (B*N_cam, 3, H, W)
        return images.reshape(B, N_cam * 3, img.shape[1], img.shape[2]).cuda()

    def test_prefix_forward(self):
        images = self._load_images()
        img_masks = np.load(
            'test/data/models/vlas/smolvla/img_masks.npy', allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            'test/data/models/vlas/smolvla/lang_tokens.npy', allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            'test/data/models/vlas/smolvla/lang_masks.npy', allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = np.load(
            'test/data/models/vlas/smolvla/states.npy', allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        embs_target = np.load(
            'test/data/models/vlas/smolvla/prefix_embs.npy', allow_pickle=True)
        pad_masks_target = np.load(
            'test/data/models/vlas/smolvla/prefix_pad_masks.npy',
            allow_pickle=True)
        att_masks_target = np.load(
            'test/data/models/vlas/smolvla/prefix_att_masks.npy',
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                embs, pad_masks, att_masks = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states)
        embs_np = embs.float().cpu().detach().numpy()[:, ::10, ::10]
        embs_diff = np.abs(embs_np - embs_target)
        pad_np = pad_masks.float().cpu().detach().numpy()
        att_np = att_masks.float().cpu().detach().numpy()
        print(f'\n[prefix] embs max_diff={embs_diff.max():.6f}'
              f', mean_diff={embs_diff.mean():.6f}')
        print(f'[prefix] pad_masks max_diff='
              f'{np.max(np.abs(pad_np - pad_masks_target)):.6f}')
        print(f'[prefix] att_masks max_diff='
              f'{np.max(np.abs(att_np - att_masks_target)):.6f}')
        self.assertTrue(np.allclose(embs_np, embs_target, atol=5e-1))
        self.assertTrue(
            np.allclose(pad_masks.float().cpu().detach().numpy(),
                        pad_masks_target))
        self.assertTrue(
            np.allclose(att_masks.float().cpu().detach().numpy(),
                        att_masks_target))

    def test_suffix_forward(self):
        time = np.load(
            'test/data/models/vlas/smolvla/suffix_time.npy', allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            'test/data/models/vlas/smolvla/suffix_x_t.npy', allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        suffix_embs_target = np.load(
            'test/data/models/vlas/smolvla/suffix_embs.npy', allow_pickle=True)
        suffix_pad_masks_target = np.load(
            'test/data/models/vlas/smolvla/suffix_pad_masks.npy',
            allow_pickle=True)
        suffix_att_masks_target = np.load(
            'test/data/models/vlas/smolvla/suffix_att_masks.npy',
            allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                ) = self.vla.embed_suffix(x_t, time)

        suffix_embs_np = suffix_embs.float().cpu().detach().numpy()[:, :, ::10]
        s_diff = np.abs(suffix_embs_np - suffix_embs_target)
        s_pad_np = suffix_pad_masks.float().cpu().detach().numpy()
        s_att_np = suffix_att_masks.float().cpu().detach().numpy()
        print(f'\n[suffix] embs max_diff={s_diff.max():.6f}'
              f', mean_diff={s_diff.mean():.6f}')
        print(f'[suffix] pad_masks max_diff='
              f'{np.max(np.abs(s_pad_np - suffix_pad_masks_target)):.6f}')
        print(f'[suffix] att_masks max_diff='
              f'{np.max(np.abs(s_att_np - suffix_att_masks_target)):.6f}')
        self.assertTrue(
            np.allclose(suffix_embs_np, suffix_embs_target, atol=1e-2))
        self.assertTrue(
            np.allclose(suffix_pad_masks.float().cpu().detach().numpy(),
                        suffix_pad_masks_target))
        self.assertTrue(
            np.allclose(suffix_att_masks.float().cpu().detach().numpy(),
                        suffix_att_masks_target))

    def test_forward(self):
        from fluxvla.engines.utils.model_utils import make_att_2d_masks
        images = self._load_images()
        img_masks = np.load(
            'test/data/models/vlas/smolvla/img_masks.npy', allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            'test/data/models/vlas/smolvla/lang_tokens.npy', allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            'test/data/models/vlas/smolvla/lang_masks.npy', allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = np.load(
            'test/data/models/vlas/smolvla/states.npy', allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            'test/data/models/vlas/smolvla/suffix_time.npy', allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            'test/data/models/vlas/smolvla/suffix_x_t.npy', allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()
        actions_target = np.load(
            'test/data/models/vlas/smolvla/actions.npy', allow_pickle=True)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                (
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                ) = self.vla.embed_prefix(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    states=states,
                )

                (
                    suffix_embs,
                    suffix_pad_masks,
                    suffix_att_masks,
                ) = self.vla.embed_suffix(x_t, time)

                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks],
                                      dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks],
                                      dim=1)

                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1

                suffix_out, _ = self.vla.forward_model(
                    attention_mask=att_2d_masks,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False)

                suffix_out = suffix_out[:, -self.vla.chunk_size:]
                suffix_out = suffix_out.to(dtype=torch.float32)
                actions = self.vla.action_out_proj(suffix_out)

                actions_np = actions.float().cpu().detach().numpy()
                a_diff = np.abs(actions_np - actions_target)
                print(f'\n[forward] actions max_diff='
                      f'{a_diff.max():.6f}'
                      f', mean_diff={a_diff.mean():.6f}')
                self.assertTrue(
                    np.allclose(actions_np, actions_target, atol=1e-1))

    def test_predict_actions(self):
        images = self._load_images()
        img_masks = np.load(
            'test/data/models/vlas/smolvla/img_masks.npy', allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            'test/data/models/vlas/smolvla/lang_tokens.npy', allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            'test/data/models/vlas/smolvla/lang_masks.npy', allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        noise = np.load(
            'test/data/models/vlas/smolvla/noise.npy', allow_pickle=True)
        noise = torch.from_numpy(noise).cuda()
        states = np.load(
            'test/data/models/vlas/smolvla/states.npy', allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        pred_actions_target = np.load(
            'test/data/models/vlas/smolvla/pred_actions.npy',
            allow_pickle=True)

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                actions = self.vla.predict_action(
                    images=images,
                    states=states,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    noise=noise)

        pred_np = actions.float().cpu().detach().numpy()
        p_diff = np.abs(pred_np - pred_actions_target)
        print(f'\n[predict] pred_actions max_diff='
              f'{p_diff.max():.6f}'
              f', mean_diff={p_diff.mean():.6f}')
        self.assertTrue(np.allclose(pred_np, pred_actions_target, atol=5e-1))

    def test_vlm_output_consistency(self):
        """Verify that joint forward and prefill+decode produce consistent
        suffix outputs."""
        from fluxvla.engines.utils.model_utils import make_att_2d_masks
        images = self._load_images()
        img_masks = np.load(
            'test/data/models/vlas/smolvla/img_masks.npy', allow_pickle=True)
        img_masks = torch.from_numpy(img_masks).cuda()
        lang_tokens = np.load(
            'test/data/models/vlas/smolvla/lang_tokens.npy', allow_pickle=True)
        lang_tokens = torch.from_numpy(lang_tokens).cuda()
        lang_masks = np.load(
            'test/data/models/vlas/smolvla/lang_masks.npy', allow_pickle=True)
        lang_masks = torch.from_numpy(lang_masks).cuda()
        states = np.load(
            'test/data/models/vlas/smolvla/states.npy', allow_pickle=True)
        states = torch.from_numpy(states).cuda()
        time = np.load(
            'test/data/models/vlas/smolvla/suffix_time.npy', allow_pickle=True)
        time = torch.from_numpy(time).cuda()
        x_t = np.load(
            'test/data/models/vlas/smolvla/suffix_x_t.npy', allow_pickle=True)
        x_t = torch.from_numpy(x_t).cuda()

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                prefix_embs, prefix_pad_masks, prefix_att_masks = (
                    self.vla.embed_prefix(images, img_masks, lang_tokens,
                                          lang_masks, states))
                suffix_embs, suffix_pad_masks, suffix_att_masks = (
                    self.vla.embed_suffix(x_t, time))

                # Path 1: joint forward (training path)
                pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks],
                                      dim=1)
                att_masks = torch.cat([prefix_att_masks, suffix_att_masks],
                                      dim=1)
                joint_att_2d = make_att_2d_masks(pad_masks, att_masks)
                joint_pos_ids = torch.cumsum(pad_masks, dim=1) - 1
                suffix_out_joint, _ = self.vla.forward_model(
                    joint_att_2d, joint_pos_ids, [prefix_embs, suffix_embs])

                # Path 2: prefill + decode (inference path)
                prefix_att_2d = make_att_2d_masks(prefix_pad_masks,
                                                  prefix_att_masks)
                prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                _, past_kv = self.vla.forward_model(
                    prefix_att_2d,
                    prefix_pos_ids, [prefix_embs, None],
                    use_cache=True)

                suffix_len = suffix_pad_masks.shape[1]
                batch_size = prefix_pad_masks.shape[0]
                prefix_len = prefix_pad_masks.shape[1]
                prefix_pad_2d = prefix_pad_masks[:, None, :].expand(
                    batch_size, suffix_len, prefix_len)
                suffix_att_2d = make_att_2d_masks(suffix_pad_masks,
                                                  suffix_att_masks)
                decode_att = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                decode_pos = (
                    prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1)
                suffix_out_decode, _ = self.vla.forward_model(
                    decode_att,
                    decode_pos, [None, suffix_embs],
                    past_key_values=past_kv,
                    use_cache=True)

        out_joint = suffix_out_joint.float().cpu().detach().numpy()
        out_decode = suffix_out_decode.float().cpu().detach().numpy()
        max_diff = np.max(np.abs(out_joint - out_decode))
        mean_diff = np.mean(np.abs(out_joint - out_decode))
        print(f'\nSuffix output consistency: max_diff={max_diff:.6f}, '
              f'mean_diff={mean_diff:.6f}')
        self.assertTrue(
            np.allclose(out_joint, out_decode, atol=1e-1),
            f'Suffix outputs diverge: max_diff={max_diff}')
