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

# GR00T VLA: Qwen3-0.6B LLM + Qwen3-VL 2B Vision
#               + linear projection 1024->2048.
_qwen3vl_vla_ckpt = './checkpoints/gr00t_qwen3vl_0.6b_libero'
_qwen3vl_vlm_config = dict(
    architectures=['Qwen3VLAForConditionalGeneration'],
    dtype='bfloat16',
    eos_token_id=151645,
    image_token_id=151655,
    model_type='qwen3_vl',
    pad_token_id=151643,
    pos_skipping_range=4096,
    text_config=dict(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=151643,
        dtype='bfloat16',
        eos_token_id=151645,
        head_dim=128,
        hidden_act='silu',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=3072,
        max_position_embeddings=262144,
        model_type='qwen3_vl_text',
        num_attention_heads=16,
        num_hidden_layers=28,
        num_key_value_heads=8,
        pad_token_id=None,
        rms_norm_eps=1e-06,
        rope_parameters=dict(
            mrope_interleaved=True,
            mrope_section=[24, 20, 20],
            rope_theta=5000000,
            rope_type='default'),
        tie_word_embeddings=True,
        use_cache=True,
        vocab_size=151936),
    tie_word_embeddings=True,
    use_another_LLM_path='',
    use_pos_skipping=False,
    vision_config=dict(
        deepstack_visual_indexes=[5, 11, 17],
        depth=24,
        dtype='bfloat16',
        hidden_act='gelu_pytorch_tanh',
        hidden_size=1024,
        in_channels=3,
        initializer_range=0.02,
        intermediate_size=4096,
        model_type='qwen3_vl',
        num_heads=16,
        num_position_embeddings=2304,
        out_hidden_size=1024,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2))

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=_qwen3vl_vla_ckpt,
    name_mapping=None,
    strict_mapping=False,
    # Qwen3-VL-0.6B (native 1024) + linear projection 1024->2048
    # to match GR00T-N1.5 action head
    vlm_backbone=dict(
        type='Qwen3VL',
        vlm_backbone_id='qwen3_0.6b_vl_pt',
        vlm_path=None,
        vlm_config=_qwen3vl_vlm_config,
        use_projection=True,
        projection_output_dim=2048,
        projection_type='linear',
        attn_implementation='sdpa'),
    # vla_head dims aligned with GR00T-N1.5-3B so pretrained head can load
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        backbone_embedding_dim=2048,
        vl_self_attention_cfg=dict(
            attention_head_dim=64,
            num_attention_heads=32,  # 32*64=2048
            num_layers=4,
            dropout=0.2,
            final_dropout=True,
            positional_embeddings=None),
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            num_attention_heads=32,  # 32*48=1536 (DiT inner_dim)
            cross_attention_dim=2048,
            num_layers=16,
            output_dim=1024,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            positional_embeddings=None),
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=10,
        action_dim=32,
        ori_action_dim=7),
    freeze_vlm_backbone=False,
    freeze_projector=False)

# Eval: same architecture as model; weights from checkpoint only (no GR00T)
inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path=_qwen3vl_vla_ckpt,
    name_mapping=None,
    vlm_backbone=dict(
        type='Qwen3VL',
        vlm_backbone_id='qwen3_0.6b_vl_pt',
        vlm_path=None,
        vlm_config=_qwen3vl_vlm_config,
        use_projection=True,
        projection_output_dim=2048,
        projection_type='linear',
        attn_implementation='sdpa'),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        backbone_embedding_dim=2048,
        vl_self_attention_cfg=dict(
            attention_head_dim=64,
            num_attention_heads=32,
            num_layers=4,
            dropout=0.2,
            final_dropout=True,
            positional_embeddings=None),
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            num_attention_heads=32,
            cross_attention_dim=2048,
            num_layers=16,
            output_dim=1024,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            positional_embeddings=None),
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        traj_length=10,
        action_dim=32,
        ori_action_dim=7),
    freeze_vlm_backbone=False,
    freeze_projector=False)

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
            data_root_path=[  # noqa: E251
                'datasets/libero_spatial_no_noops_lerobotv2.1',
            ],
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
                    type='ProcessPrompts',
                    tokenizer=dict(
                        type='PretrainedTokenizer',
                        model_path=  # noqa: E251
                        './checkpoints/gr00t_qwen3vl_0.6b_libero/tokenizer/',  # noqa: E501
                    )),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='QWen2VLImageTransform',
                    min_pixels=56 * 56,
                    max_pixels=28 * 28 * 1280,
                    patch_size=16,  # Qwen3-VL uses 16 (Qwen2-VL uses 14)
                    temporal_patch_size=2,
                    merge_size=2,
                    # Qwen3-VL same image pipeline as Qwen2-VL:
                    # do_normalize=True by default
                    image_mean=[0.48145466, 0.4578275,
                                0.40821073],  # OPENAI_CLIP_MEAN
                    image_std=[0.26862954, 0.26130258,
                               0.27577711]),  # OPENAI_CLIP_STD
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
            statistic_name='libero_spatial_no_noops',
            window_start_idx=0)))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=24,
    learning_rate=1.5e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        './checkpoints/gr00t_qwen3vl_0.6b_libero/tokenizer/',  # noqa: E501
    ),
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks',
            'embodiment_ids', 'image_grid_thw'
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
    sharding_strategy='full-shard',
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_spatial',
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
            dict(type='ConvertPILImageToNumpyArray'),
            dict(
                type='QWen2VLImageTransform',
                min_pixels=56 * 56,
                max_pixels=28 * 28 * 1280,
                patch_size=16,  # Qwen3-VL uses 16
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.48145466, 0.4578275,
                            0.40821073],  # OPENAI_CLIP_MEAN
                image_std=[0.26862954, 0.26130258,
                           0.27577711],  # OPENAI_CLIP_STD
                img_key='pixel_values',
                to_tensor=True),
            dict(
                type='LiberoPromptFromInputs',
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    './checkpoints/gr00t_qwen3vl_0.6b_libero/tokenizer/',  # noqa: E501
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
