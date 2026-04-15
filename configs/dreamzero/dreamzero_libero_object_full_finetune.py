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

# ===================================================================
# DreamZero – LIBERO object full fine-tune config
#
# Video setup:
#   frame_window_size = 9 (current frame + 8 future frames for
#   dynamics supervision).  The first frame is the conditioning
#   observation; the remaining frames are prediction targets for
#   the video dynamics loss.
#   VAE temporal compression: latent_frames = 1 + (T-1)//4
#   T=9 → 3 latent frames → 1 conditioning + 2 = 1 block of
#   num_frame_per_block=2.
#
# Image layout : 2 views (agentview + wrist) @ 128x128 each
#                tiled vertically → 256×128
# VAE latent   : 32×16 (spatial /8)
# After patch  : 16×8  (patch 2×2)
# frame_seqlen : 16 * 8 = 128
#
# Pretrained weights are loaded from pretrained_name_or_path
# (DreamZero-AgiBot safetensors) with name_mapping to remap
# checkpoint keys to the fluxvla module structure.
# Sub-model pretrained_name_or_path=None because weights come
# from the unified checkpoint, not per-component .pth files.
# ===================================================================

_ckpt_root = './checkpoints'
_tokenizer = _ckpt_root + '/Wan2.1-I2V-14B-480P/google/umt5-xxl'

_frame_window_size = 9

model = dict(
    type='DreamZeroVLA',
    num_views=2,
    frame_window_size=_frame_window_size,
    pretrained_name_or_path=  # noqa: E251
    _ckpt_root + '/DreamZero-AgiBot',
    wam_backbone=dict(
        type='WanBackbone',
        text_encoder_path=None,
        image_encoder_path=None,
        vae_path=None,
        tiled=False,
    ),
    vla_head=dict(
        type='DreamZeroHead',
        # ----- action / state dims -----
        action_dim=7,
        max_action_dim=32,
        action_horizon=10,
        max_state_dim=64,
        # ----- video / latent -----
        num_frames=_frame_window_size,
        num_frame_per_block=2,
        num_action_per_block=10,
        num_state_per_block=1,
        frame_seqlen=128,
        # ----- DiT architecture (Wan 14B) -----
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
        # ----- noise schedule -----
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_inference_steps=16,
        # ----- pretrained paths -----
        use_gradient_checkpointing=True,
    ),
    name_mapping={
        'vla_head.model': 'action_head.model',
        'wam_backbone.text_encoder': 'action_head.text_encoder',
        'wam_backbone.image_encoder': 'action_head.image_encoder',
        'wam_backbone.vae': 'action_head.vae',
    },
)

train_dataloader = dict(
    per_device_batch_size=2,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name='libero_object_no_noops',
        datasets=dict(
            type='ParquetDataset',
            data_root_path='./datasets/libero_object_no_noops_lerobotv2.1',
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state',
                        'timestamp',
                        'actions',
                        'info',
                        'stats',
                        'action_masks',
                    ],
                    video_keys=[
                        'observation.images.image',
                        'observation.images.wrist_image',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    },
                    embodiment_id=0,
                ),
                dict(type='ParquetPrompter', use_conversation=False),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=_tokenizer,
                    ),
                    max_len=512,
                ),
                dict(type='ResizeImages', height=128, width=128),
                dict(type='SimpleNormalizeImages'),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,
                    state_dim=32,
                    state_key='proprio',
                    action_key='action',
                    norm_type='mean_std',
                ),
                dict(
                    type='PrepareVideoForDreamZero',
                    num_views=2,
                    frame_window_size=_frame_window_size,
                ),
            ],
            action_window_size=10,
            action_key='action',
            use_delta=False,
            statistic_name='libero_object_no_noops',
            window_start_idx=0,
            frame_window_size=_frame_window_size,
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=12,
    learning_rate=1e-5,
    weight_decay=1e-5,
    max_grad_norm=1.0,
    collator=dict(
        type='DictCollator',
        keys=[
            'states',
            'images',
            'img_masks',
            'actions',
            'action_masks',
            'embodiment_ids',
            'frame_masks',
            'lang_tokens',
            'lang_masks',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats', 'timestamp'],
    ),
    sampler=None,
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.05,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='full-shard',
    change_key_name=False,
)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_object',
    model_family='dreamzero',
    eval_chunk_size=10,
    resize_size=128,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 128, 128], [3, 128, 128]],
                means=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
                stds=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
            ),
            dict(
                type='LiberoProprioFromInputs',
                norm_type='mean_std',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                state_dim=32,
                out_key='states',
            ),
            dict(
                type='LiberoPromptFromInputs',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=_tokenizer,
                ),
                max_len=512,
                use_conversation=False,
            ),
            dict(
                type='PrepareVideoForDreamZero',
                num_views=2,
                frame_window_size=1,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='mean_std',
        action_dim=7,
    ),
)
