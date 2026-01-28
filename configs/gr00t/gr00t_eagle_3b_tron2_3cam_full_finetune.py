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
        num_steps=32,
        traj_length=10,
        action_dim=32,
        ori_action_dim=18),
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
        type='EagleBackbone',
        dtype=None,
        vlm_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model'),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_steps=32,
        num_inference_timesteps=4,
        traj_length=10,
        action_dim=32,
        ori_action_dim=18),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={'observation.state': ['proprio']},
        statistic_keys=['observation.state', 'action'],
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=[  # noqa: E501
                    './datasets/RealRobot_Tron2_lerobot/tron2_example',  # noqa: E501
                ],
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=30,
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=[
                            'observation.images.cam_high',
                            'observation.images.cam_left_wrist',
                            'observation.images.cam_right_wrist'
                        ],
                        name_mappings={'observation.state': ['states']}),
                    dict(type='ParquetPrompter'),
                    dict(
                        type='ProcessPromptsWithImage',
                        max_len=900,
                        num_images=3,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            'fluxvla/models/third_party_models/eagle2_hg_model',  # noqa: E501
                            # special_tokens={'pad_token': '<PAD>'}
                        )),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(
                        type='NormalizeImages',
                        means=[[123.515625, 116.04492188, 103.59375],
                               [123.515625, 116.04492188, 103.59375],
                               [123.515625, 116.04492188, 103.59375]],
                        stds=[[58.27148438, 57.02636719, 57.27539062],
                              [58.27148438, 57.02636719, 57.27539062],
                              [58.27148438, 57.02636719, 57.27539062]],
                    ),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_dim=64,
                        action_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='mean_std')
                ],
                action_key='action',
                window_start_idx=0,
                action_window_size=32)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=7,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        'fluxvla/models/third_party_models/eagle2_hg_model',  # noqa: E501
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
    lr_scheduler_type='constant',
    warmup_ratio=0.0,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

inference = dict(
    type='Tron2InferenceRunner',
    seed=7,
    task_descriptions={
        '1': 'complete the task',
    },
    publish_rate=30,
    action_chunk=32,
    mixed_precision_dtype='bf16',
    dataset=dict(
        type='PrivateInferenceDataset',
        embodiment_id=30,
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        transforms=[
            dict(
                type='ProcessPromptsWithImage',
                max_len=900,
                num_images=3,
                tokenizer=dict(type='PretrainedTokenizer'
                               # special_tokens={'pad_token': '<PAD>'}
                               )),
            dict(type='ResizeImages', height=224, width=224),
            dict(
                type='NormalizeImages',
                means=[[123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375]],
                stds=[[58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062]],
            ),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='mean_std')
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=18,
    ),
    operator=dict(
        type='Tron2Operator',
        img_left_topic='/camera/left/color/image_rect_raw',
        img_right_topic='/camera/right/color/image_rect_raw',
        img_top_topic='/camera/top/color/image_raw',
        joint_state_topic='/joint_states',
        gripper_state_topic='/gripper_state',
        ee_pose_left_topic='/left_arm/ee_pose',
        ee_pose_right_topic='/right_arm/ee_pose',
        trajectory_exec_mode='servoj',
    ))
