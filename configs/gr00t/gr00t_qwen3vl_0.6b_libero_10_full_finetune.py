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
        statistic_name='libero_10_no_noops',
        dataset_statistics=dict(
            libero_10_no_noops=dict(
                proprio=dict(
                    mean=[
                        -0.0419132679050224, 0.034591788297521735,
                        0.8265881844959498, 2.90259518190321,
                        -0.5570652600832564, -0.16592166873533284,
                        0.02845031351083622, -0.02880236273799356
                    ],
                    std=[
                        0.03756502182067285, 0.05091765880150317,
                        0.09107525593038836, 0.12327524826514363,
                        0.4418352294043351, 0.12490994022681218,
                        0.004662133639412193, 0.00460807817987938
                    ],
                    min=[
                        -0.48278069496154785, -0.3309336006641388,
                        0.44550687074661255, 1.1323540210723877,
                        -3.6312508583068848, -1.842738389968872,
                        -0.005453015677630901, -0.04112039878964424
                    ],
                    max=[
                        0.2103137969970703, 0.38887521624565125,
                        1.333192229270935, 3.7248642444610596, 3.5618896484375,
                        1.3863215446472168, 0.041575800627470016,
                        0.0013126095291227102
                    ],
                    q01=[
                        -0.1855636807291125, -0.16145669766439186,
                        0.7064185725262808, 2.5678211534702324,
                        -1.2430377303522737, -0.5195810482339626,
                        0.01022917473133343, -0.03999379658232052
                    ],
                    q99=[
                        0.05938728483051665, 0.2361478409238694,
                        0.9397258571145816, 3.2118708728143526,
                        0.49082919816100534, 0.2100883989120329,
                        0.040047131839991014, -0.011104049991952391
                    ]),
                timestamp=dict(
                    mean=[7.007510548523206],
                    std=[4.457129586378845],
                    min=[0.0],
                    max=[25.2],
                    q01=None,
                    q99=None),
                action=dict(
                    mean=[
                        0.01905656634877842, 0.05672475971568838,
                        -0.056239289430234256, 0.004756678478841528,
                        0.002797492338491304, -0.00714607048416358,
                        0.54599156235075
                    ],
                    std=[
                        0.10588348353857541, 0.13552477199270377,
                        0.13886650724555177, 0.01433739270759898,
                        0.02038583948325967, 0.033299202425577934,
                        0.1881810653484855
                    ],
                    min=[
                        -0.9375, -0.9375, -0.9375, -0.23642857372760773,
                        -0.3053571283817291, -0.3642857074737549, 0.0
                    ],
                    max=[
                        0.9375, 0.9375, 0.9375, 0.32892856001853943,
                        0.36964285373687744, 0.375, 1.0
                    ],
                    q01=[
                        -0.4997477764535965, -0.6992653512084763,
                        -0.6543309163615124, -0.07417070079989778,
                        -0.11898748445770971, -0.15976085962510805, 0.0
                    ],
                    q99=[
                        0.658747846713789, 0.7333480638990948,
                        0.768601965587579, 0.09784501244893279,
                        0.12943469061349036, 0.15137893471596325, 1.0
                    ]))),
        datasets=dict(
            type='ParquetDataset',
            data_root_path=[  # noqa: E251
                'datasets/libero_10_no_noops_lerobotv2.1',
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
            statistic_name='libero_10_no_noops',
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
    task_suite_name='libero_10',
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
                    './checkpoints/gr00t_qwen3vl_0.6b_libero/',  # noqa: E501
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
