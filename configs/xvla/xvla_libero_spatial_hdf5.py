import os

_XVLA_PATH = './pretrained/X-VLA'
if not os.path.exists(_XVLA_PATH):
    _XVLA_PATH = '/mnt/data/cpfs/users/yanis/X-VLA/pretrained/X-VLA-Pt'
_META_PATH = '/mnt/data/cpfs/users/yanis/X-VLA/data/libero_spatial_meta.json'

model = dict(
    type='XVLAVla',
    pretrained_name_or_path=_XVLA_PATH,
    vlm_backbone=dict(
        type='Florence2Backbone',
        vlm_path=_XVLA_PATH,
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

train_dataloader = dict(
    per_device_batch_size=16,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        seed=7,
        # No normalization stats needed for X-VLA (no mean/std norm on actions).
        statistic_keys=[],
        name_mappings={},
        statistic_name='private',
        datasets=dict(
            type='LiberoHDF5Dataset',
            meta_path=_META_PATH,
            num_actions=30,
            num_views=3,
            embodiment_id=3,
            training=True,
            statistic_name='private',
            image_size=224,
            transforms=[
                dict(
                    type='ProcessXVLAPrompts',
                    tokenizer_path=_XVLA_PATH,
                    max_len=50,
                ),
            ],
        ),
    ),
)

runner = dict(
    type='FSDPTrainRunner',
    max_steps=100000,
    learning_rate=1e-4,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'images', 'img_masks',
            'lang_tokens', 'lang_masks',
            'actions', 'action_masks', 'embodiment_ids',
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats', 'timestamp'],
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1,
    ),
    lr_scheduler_type='xvla-freeze-warmup',
    freeze_steps=1000,
    warmup_steps=2000,
    save_iter_interval=50000,
    lr_coef=0.1,
    betas=(0.9, 0.95),
    warmup_ratio=0.03,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False,
)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_spatial',
    model_family='pi0',
    eval_chunk_size=30,
    resize_size=224,
    num_trials_per_task=50,
    num_steps_wait=10,
    task_horizons=dict(
        libero_spatial=800,
        libero_object=800,
        libero_goal=800,
        libero_10=900,
        libero_90=800,
    ),
    use_xvla_client_semantics=True,
    seed=7,
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                embodiment_id=3,
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
                flip_all_views=False,
                num_padding_imgs=1,
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224], [3, 224, 224]],
                means=[[123.675, 116.28, 103.53],
                       [123.675, 116.28, 103.53],
                       [123.675, 116.28, 103.53]],
                stds=[[58.395, 57.12, 57.375],
                      [58.395, 57.12, 57.375],
                      [58.395, 57.12, 57.375]],
            ),
            dict(
                type='ProcessXVLAPrompts',
                tokenizer_path=_XVLA_PATH,
                max_len=50,
            ),
            dict(
                type='LiberoEE6DProprioFromInputs',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                out_key='states',
                target_dim=20,
            ),
        ],
    ),
    denormalize_action=dict(
        type='DenormalizeXVLALiberoAction',
        gripper_binarize=True,
    ),
)
