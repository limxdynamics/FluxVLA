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
        torch_dtype='bfloat16',
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
        torch_dtype='bfloat16',
    ),
    state_proj=dict(type='LinearProjector', in_dim=32, out_dim=960),
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=720),
    action_out_proj=dict(type='LinearProjector', in_dim=720, out_dim=32),
    action_time_mlp_in=dict(type='LinearProjector', in_dim=1440, out_dim=720),
    action_time_mlp_out=dict(type='LinearProjector', in_dim=720, out_dim=720),
    freeze_vlm_backbone=True,
    max_action_dim=32,
    ori_action_dim=7,
    chunk_size=50,
    num_steps=10,
    add_image_special_tokens=False,
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/smolvla_base/model.safetensors',  # noqa: E501
    name_mapping={
        'vlm_backbone.vlm': 'model.vlm_with_expert.vlm.model',
        'llm_expert.expert': 'model.vlm_with_expert.lm_expert',
        'state_proj.projector': 'model.state_proj',
        'action_in_proj.projector': 'model.action_in_proj',
        'action_out_proj.projector': 'model.action_out_proj',
        'action_time_mlp_in.projector': 'model.action_time_mlp_in',
        'action_time_mlp_out.projector': 'model.action_time_mlp_out',
    })

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
        statistic_name='libero_spatial_no_noops',
        datasets=dict(
            type='ParquetDataset',
            data_root_path=  # noqa: E251
            './datasets/libero_spatial_no_noops_lerobotv2.1',  # noqa: E501
            transforms=[
                dict(
                    type='ProcessParquetInputs',
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
                dict(
                    type='ParquetPrompter',
                    use_conversation=False,
                    add_new_line=True),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=  # noqa: E251
                        './checkpoints/SmolVLM2-500M-Video-Instruct',  # noqa: E501
                    )),
                dict(type='ResizeImagesTopLeftPad', height=512, width=512),
                dict(type='SimpleNormalizeImages'),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,
                    state_dim=32,
                    state_key='proprio',
                    action_key='action',
                    norm_type='mean_std')
            ],
            action_window_size=50,
            action_key='action',
            use_delta=False,
            statistic_name='libero_spatial_no_noops',
            window_start_idx=0)))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=36,
    learning_rate=1e-4,
    weight_decay=0.0,
    max_grad_norm=1.0,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'timestamp', 'images', 'img_masks', 'lang_tokens',
            'lang_masks', 'actions', 'action_masks'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        './checkpoints/SmolVLM2-500M-Video-Instruct',  # noqa: E501
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.03,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    sharding_strategy='no-shard',
    change_key_name=False)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_spatial',
    model_family='smolvla',
    eval_chunk_size=10,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
                resize_size=512,
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='letterbox-top-left',
                input_sizes=[[3, 512, 512], [3, 512, 512]],
                means=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
                stds=[[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
                letterbox_fill=[0, 0, 0],
            ),
            dict(
                type='LiberoPromptFromInputs',
                use_conversation=False,
                add_new_line=True,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    './checkpoints/SmolVLM2-500M-Video-Instruct',  # noqa: E501
                )),
            dict(
                type='LiberoProprioFromInputs',
                norm_type='mean_std',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                state_dim=32,
                out_key='states'),
        ]),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='mean_std',
        action_dim=7,
    ),
)
