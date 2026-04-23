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

import os
import unittest

import numpy as np
import torch

XVLA_CKPT_PATH = './checkpoints/X-VLA-Pt'


# ===================================================================
# DreamZeroHead – tiny DiT forward & predict_action
# ===================================================================
@unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
class TestDreamZeroHeadTiny(unittest.TestCase):
    """Test DreamZeroHead forward and predict_action with a tiny DiT."""

    @classmethod
    def setUpClass(cls):
        from fluxvla.models.heads import DreamZeroHead
        cls.head = DreamZeroHead(
            action_dim=7,
            max_action_dim=32,
            action_horizon=10,
            max_state_dim=64,
            num_frames=9,
            num_frame_per_block=2,
            num_action_per_block=10,
            num_state_per_block=1,
            frame_seqlen=128,
            hidden_size=32,
            input_embedding_dim=1536,
            dit_dim=64,
            dit_ffn_dim=128,
            dit_num_heads=4,
            dit_num_layers=2,
            dit_freq_dim=64,
            dit_in_dim=36,
            dit_out_dim=16,
            max_num_embodiments=1,
            skip_pretrained_loading=True,
            wan_model_path=None,
            use_gradient_checkpointing=False,
        ).cuda().bfloat16()

    def _make_inputs(self, batch_size=2):
        """Create fake encoded inputs.

        Head predict_action expects latents as [B, T_lat, C, H_lat, W_lat].
        Head forward expects latents as [B, C, T_lat, H_lat, W_lat].
        ys must match latents spatial dims [B, C_y, T_lat, H_lat, W_lat].
        """
        device = 'cuda'
        dtype = torch.bfloat16
        T_lat, C_lat, H_lat, W_lat = 3, 16, 32, 16
        latents = torch.randn(
            batch_size, T_lat, C_lat, H_lat, W_lat, device=device, dtype=dtype)
        prompt_embs = torch.randn(
            batch_size, 512, 4096, device=device, dtype=dtype)
        clip_feas = torch.randn(
            batch_size, 1, 1280, device=device, dtype=dtype)
        ys = torch.randn(
            batch_size, 20, T_lat, H_lat, W_lat, device=device, dtype=dtype)
        states = torch.randn(batch_size, 1, 64, device=device, dtype=dtype)
        embodiment_ids = torch.zeros(
            batch_size, dtype=torch.long, device=device)
        return dict(
            prompt_embs=prompt_embs,
            latents=latents,
            clip_feas=clip_feas,
            ys=ys,
            states=states,
            embodiment_ids=embodiment_ids,
        )

    def test_forward_produces_loss(self):
        """Training forward should return a dict with 'loss' key."""
        inputs = self._make_inputs()
        actions = torch.randn(2, 10, 32, device='cuda', dtype=torch.bfloat16)
        action_masks = torch.ones(2, 10, 32, dtype=torch.bool, device='cuda')
        action_masks[:, :, 7:] = False

        latents_bctHW = inputs['latents'].transpose(1, 2)
        with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
            out = self.head(
                prompt_embs=inputs['prompt_embs'],
                latents=latents_bctHW,
                clip_feas=inputs['clip_feas'],
                ys=inputs['ys'],
                states=inputs['states'],
                actions=actions,
                action_masks=action_masks,
                embodiment_ids=inputs['embodiment_ids'],
            )
        self.assertIn('loss', out)
        self.assertFalse(torch.isnan(out['loss']), 'Loss should not be NaN')

    def test_predict_action_shape(self):
        """predict_action should return [B, action_horizon, max_action_dim]."""
        inputs = self._make_inputs()
        with torch.no_grad():
            actions = self.head.predict_action(num_inference_steps=2, **inputs)
        self.assertEqual(actions.shape, (2, 10, 32))
        self.assertFalse(
            torch.isnan(actions).any(),
            'Predicted actions should not contain NaN')

    def test_predict_action_deterministic_with_seed(self):
        """Same seed should give same output."""
        inputs = self._make_inputs(batch_size=1)
        results = []
        for _ in range(2):
            self.head.reset_inference_state()
            torch.manual_seed(42)
            with torch.no_grad():
                a = self.head.predict_action(num_inference_steps=2, **inputs)
            results.append(a.clone())
        torch.testing.assert_close(results[0], results[1])

    def test_predict_action_stateless_by_default(self):
        inputs = self._make_inputs(batch_size=1)

        torch.manual_seed(123)
        with torch.no_grad():
            first = self.head.predict_action(num_inference_steps=2, **inputs)

        torch.manual_seed(123)
        with torch.no_grad():
            second = self.head.predict_action(num_inference_steps=2, **inputs)

        torch.testing.assert_close(first, second)
        self.assertIsNone(self.head.inference_kv_cache)
        self.assertEqual(self.head.current_start_frame, 0)

    def test_predict_action_cache_mode_reuses_history(self):
        inputs = self._make_inputs(batch_size=1)

        self.head.reset_inference_state()
        torch.manual_seed(123)
        with torch.no_grad():
            first = self.head.predict_action(
                num_inference_steps=2,
                use_cache=True,
                **inputs,
            )

        torch.manual_seed(123)
        with torch.no_grad():
            second = self.head.predict_action(
                num_inference_steps=2,
                use_cache=True,
                **inputs,
            )

        self.head.reset_inference_state()
        torch.manual_seed(123)
        with torch.no_grad():
            reset = self.head.predict_action(
                num_inference_steps=2,
                use_cache=True,
                **inputs,
            )

        self.assertFalse(torch.allclose(first, second))
        torch.testing.assert_close(first, reset)

    def test_predict_action_cache_mode_single_frame_forces_reset(self):
        inputs = self._make_inputs(batch_size=1)
        single_frame_inputs = dict(inputs)
        single_frame_inputs['latents'] = inputs['latents'][:, :1]
        single_frame_inputs['ys'] = inputs['ys'][:, :, :1]

        self.head.current_start_frame = 5
        self.head.inference_kv_cache = self.head._create_kv_cache(
            batch_size=1,
            dtype=inputs['latents'].dtype,
            device=inputs['latents'].device,
        )
        self.head.inference_prompt_embs = inputs['prompt_embs']
        self.head.inference_clip_feas = inputs['clip_feas']
        self.head.inference_ys = inputs['ys']

        torch.manual_seed(7)
        with torch.no_grad():
            self.head.predict_action(
                num_inference_steps=2,
                observed_latent_frames=1,
                reset_history=True,
                use_cache=True,
                **single_frame_inputs,
            )

        self.assertEqual(self.head.current_start_frame, 2)
        self.assertIsNotNone(self.head.inference_kv_cache)

        self.head.reset_inference_state()


@unittest.skipUnless(
    torch.cuda.is_available() and os.path.exists(XVLA_CKPT_PATH),
    f'CUDA not available or checkpoint not found: {XVLA_CKPT_PATH}')
class TestXVLAHeadGolden(unittest.TestCase):

    DATA_DIR = 'test/data/models/heads/xvla_head'

    @classmethod
    def setUpClass(cls):
        from fluxvla.engines import build_vla_from_cfg, set_seed_everywhere
        cfg = dict(
            type='XVLAVla',
            pretrained_name_or_path=XVLA_CKPT_PATH,
            vlm_backbone=dict(
                type='Florence2Backbone',
                vlm_path=XVLA_CKPT_PATH,
                dtype='bf16',
            ),
            vla_head=dict(
                type='XVLAFlowMatchingHead',
                hidden_size=1024,
                multi_modal_input_size=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4.0,
                num_domains=30,
                dim_action=20,
                dim_propio=20,
                len_soft_prompts=32,
                dim_time=32,
                max_len_seq=512,
                use_hetero_proj=False,
                num_actions=30,
                num_inference_steps=10,
                action_mode='ee6d',
            ),
            freeze_vlm_backbone=False,
            name_mapping={
                'vlm_backbone.vlm': 'vlm',
                'vla_head.transformer': 'transformer',
            },
        )
        set_seed_everywhere(0)
        cls.vla = build_vla_from_cfg(cfg).cuda()
        cls.vla.from_pretrained()
        cls.vla.vla_head = cls.vla.vla_head.to(dtype=torch.bfloat16)
        cls.vla.eval()
        cls.head = cls.vla.vla_head

    def _load(self, name, dtype=None):
        arr = np.load(f'{self.DATA_DIR}/{name}.npy', allow_pickle=True)
        tensor = torch.from_numpy(arr).cuda()
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor

    def test_forward(self):
        input_features = self._load('input_features', torch.bfloat16)
        attention_mask = self._load('attention_mask')
        aux_visual_inputs = self._load('aux_visual_inputs', torch.bfloat16)
        states = self._load('states', torch.bfloat16)
        actions = self._load('actions', torch.bfloat16)
        action_masks = self._load('action_masks')
        embodiment_ids = self._load('embodiment_ids').long()

        position_loss_ref = np.load(f'{self.DATA_DIR}/position_loss.npy')
        rotate6d_loss_ref = np.load(f'{self.DATA_DIR}/rotate6D_loss.npy')
        gripper_loss_ref = np.load(f'{self.DATA_DIR}/gripper_loss.npy')
        loss_ref = np.load(f'{self.DATA_DIR}/loss.npy')

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                out = self.head(
                    input_features=input_features,
                    states=states,
                    attention_mask=attention_mask,
                    actions=actions,
                    action_masks=action_masks,
                    embodiment_ids=embodiment_ids,
                    aux_visual_inputs=aux_visual_inputs,
                )

        self.assertTrue(
            np.allclose(
                out['position_loss'].float().cpu().numpy(),
                position_loss_ref,
                atol=1e-2))
        self.assertTrue(
            np.allclose(
                out['rotate6D_loss'].float().cpu().numpy(),
                rotate6d_loss_ref,
                atol=1e-2))
        self.assertTrue(
            np.allclose(
                out['gripper_loss'].float().cpu().numpy(),
                gripper_loss_ref,
                atol=1e-2))
        self.assertTrue(
            np.allclose(
                out['loss'].float().cpu().numpy(),
                loss_ref,
                atol=1e-2))

    def test_predict_action(self):
        input_features = self._load('input_features', torch.bfloat16)
        attention_mask = self._load('attention_mask')
        aux_visual_inputs = self._load('aux_visual_inputs', torch.bfloat16)
        states = self._load('states', torch.bfloat16)
        embodiment_ids = self._load('embodiment_ids').long()
        pred_actions_ref = np.load(f'{self.DATA_DIR}/pred_actions.npy')

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
                pred_actions = self.head.predict_action(
                    input_features=input_features,
                    states=states,
                    attention_mask=attention_mask,
                    embodiment_ids=embodiment_ids,
                    aux_visual_inputs=aux_visual_inputs,
                )

        self.assertTrue(
            np.allclose(
                pred_actions.float().cpu().numpy(),
                pred_actions_ref,
                atol=1e-2))


if __name__ == '__main__':
    unittest.main()
