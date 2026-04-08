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

model = dict(
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

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleInferenceBackbone',
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model'),
    vla_head=dict(
        type='FlowMatchingInferenceHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=10,
        action_dim=32,
        ori_action_dim=7,
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_attention_heads=32,
            num_layers=16,
            output_dim=1024,
            positional_embeddings=None)))

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name='libero_goal_no_noops',
        datasets=dict(
            type='ParquetDataset',
            data_root_path=  # noqa: E251
            'datasets/libero_goal_no_noops_lerobotv2.1',  # noqa: E501
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    embodiment_id=2,
                    parquet_keys=[
                        'observation.state', 'timestamp', 'actions', 'info',
                        'stats', 'action_masks'
                    ],
                    video_keys=[
                        'observation.images.image',
                        'observation.images.wrist_image',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions']
                    }),
                dict(type='ParquetPrompter'),
                dict(
                    type='ProcessPromptsWithImage',
                    max_len=600,
                    num_images=2,
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=  # noqa: E251
                        'fluxvla/models/third_party_models/eagle2_hg_model',
                        # special_tokens={'pad_token': '<PAD>'}
                    )),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='NormalizeImages',
                    means=[[123.515625, 116.04492188, 103.59375],
                           [123.515625, 116.04492188, 103.59375]],
                    stds=[[58.27148438, 57.02636719, 57.27539062],
                          [58.27148438, 57.02636719, 57.27539062]],
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,
                    state_dim=64,
                    state_key='proprio',
                    action_key='action',
                    norm_type='mean_std')
            ],
            action_window_size=10,
            action_key='action',
            use_delta=False,
            statistic_name='libero_goal_no_noops',
            window_start_idx=0)))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=18,
    learning_rate=1.5e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',
        # special_tokens={'pad_token': '<PAD>'}
    ),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks',
            'embodiment_ids'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.03,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_goal',
    model_family='pi0',
    eval_chunk_size=10,
    resize_size=224,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                embodiment_id=2,
                img_keys=['agentview_image', 'robot0_eye_in_hand_image']),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224]],
                means=[[123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375]],
                stds=[[58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062]],
            ),
            dict(
                type='ProcessPromptsWithImage',
                max_len=600,
                num_images=2,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    'fluxvla/models/third_party_models/eagle2_hg_model',
                    # special_tokens={'pad_token': '<PAD>'}
                )),
            dict(
                type='LiberoProprioFromInputs',
                state_dim=64,
                norm_type='mean_std',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                out_key='states'),
        ]),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='mean_std',
    ),
)
