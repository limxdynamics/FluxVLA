clip_pretrained_name_or_path = './checkpoints/clip-vit-base-patch32'
data_root_path = './datasets/SARM_manual_test_10Episodes_lerobotv3.0'

current_transforms = [
    dict(type='ResizeImageSequence', height=224, width=224),
    dict(
        type='NormalizeImageSequence',
        means=[0.48145466, 0.4578275, 0.40821073],
        stds=[0.26862954, 0.26130258, 0.27577711],
        scale_to_unit_interval=True,
    ),
    dict(type='PadStates', max_state_dim=32),
    dict(
        type='TokenizeText',
        model_name_or_path=clip_pretrained_name_or_path,
        max_length=77),
]

current_collator = dict(
    type='DictCollator',
    keys=[
        'images', 'states', 'lengths', 'episode_index', 'current_index',
        'sparse_targets', 'dense_targets', 'text_input_ids',
        'text_attention_mask'
    ],
    meta_keys=['task_description'],
)

model = dict(
    type='SARMRewardModel',
    annotation_mode='dense_only',
    data_root_path=data_root_path,
    n_obs_steps=8,
    frame_gap=30,
    max_rewind_steps=4,
    llm_backbone=dict(
        type='SARMBackbone',
        pretrained_name_or_path=clip_pretrained_name_or_path,
        hidden_dim=768,
        num_heads=12,
        num_layers=8,
        max_state_dim=32,
        dropout=0.1,
        num_cameras=1,
        freeze_clip_backbone=True,
    ),
    freeze_llm_backbone=False,
)

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=2,
    dataset=dict(
        type='SARMDataset',
        data_root_path=data_root_path,
        video_keys=['observation.images.cam_high'],
        annotation_mode='dense_only',
        n_obs_steps=8,
        frame_gap=30,
        max_rewind_steps=4,
        rewind_probability=0.8,
        state_key='observation.state',
        training=True,
        transforms=current_transforms,
    ),
)

runner = dict(
    type='DDPTrainRunner',
    max_steps=5000,
    learning_rate=5e-5,
    weight_decay=1e-3,
    max_grad_norm=1.0,
    collator=current_collator,
    sampler='distributed',
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', ),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=20,
    ),
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.03,
    enable_gradient_checkpointing=False,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
)

inference_dataset = dict(
    type='SARMDataset',
    data_root_path=data_root_path,
    video_keys=['observation.images.cam_high'],
    annotation_mode='dense_only',
    n_obs_steps=8,
    frame_gap=30,
    max_rewind_steps=0,
    rewind_probability=0.0,
    state_key='observation.state',
    training=False,
    transforms=current_transforms,
)
