# ============================================================
# GR00T-N1.5 Eagle 3B：Robocasa GR1 24 任务 × 每任务 30 episode 子集后训练
# ============================================================
#
# 数据：先运行（全量源目录多为 workspace 上 robocasa_lerobot_V2.1）
#   python scripts/sample_robocasa_fluxvla_subset.py \\
#       --src /mnt/workspace/mnt/data/yiming/fluxvla/datasets/robocasa_lerobot_V2.1 \\
#       --dst .../robocasa_gr1_24tasks_first30ep --per-task 30 --strategy first
# DLC 上通过 ROBOCASA_DATASET_ROOT 指向含 24 个子目录的数据根，
# launch 脚本会把其链到 ./datasets/robocasa_fluxvla，故下列相对路径不变。
#
# statistic_name 须与 eval.unnorm_key 一致；
#
# ============================================================

model = dict(
    type='LlavaVLA',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        dtype='bf16',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900),
        # GR00T N1.5 RoboCasa uses --tune-visual with tune_llm=False:
        # train Eagle vision tower, keep Eagle language_model frozen.
        tune_llm=False,
        tune_visual=True),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_inference_timesteps=4,
        num_steps=16,
        traj_length=16,
        action_dim=32,
        ori_action_dim=29),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

_STAT = 'robocasa_gr1_24tasks_30ep'
_OFFICIAL_GR1_STATS_PATH = 'work_dirs/official_groot_gr1_dataset_statistics.json'

train_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_STAT,
        dataset_statistics_path=_OFFICIAL_GR1_STATS_PATH,
        datasets=[
            dict(
                type='ParquetDataset',
                data_root_path=[
                    './datasets/robocasa_fluxvla/PnPBottleToCabinetClose',
                    './datasets/robocasa_fluxvla/PnPCanToDrawerClose',
                    './datasets/robocasa_fluxvla/PnPCupToDrawerClose',
                    './datasets/robocasa_fluxvla/PnPMilkToMicrowaveClose',
                    './datasets/robocasa_fluxvla/PnPPotatoToMicrowaveClose',
                    './datasets/robocasa_fluxvla/PnPWineToCabinetClose',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromCuttingboardToBasketSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromCuttingboardToPanSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromCuttingboardToPotSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromPlacematToBasketSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromPlacematToBowlSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromPlacematToPlateSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromPlacematToTieredshelfSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromPlateToBowlSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromPlateToCardboardboxSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromPlateToPanSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromPlateToPlateSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromTrayToCardboardboxSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromTrayToPlateSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromTrayToPotSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromTrayToTieredbasketSplitA',
                    './datasets/robocasa_fluxvla/PosttrainPnPNovelFromTrayToTieredshelfSplitA',
                ],
                statistic_name=_STAT,
                action_key='action',
                use_delta=False,
                window_start_idx=0,
                transforms=[
                    dict(
                        type='ProcessParquetInputs',
                        embodiment_id=24,
                        parquet_keys=[
                            'observation.state', 'timestamp', 'actions',
                            'info', 'stats', 'action_masks'
                        ],
                        video_keys=['observation.images.ego_view'],
                        name_mappings={
                            'observation.state': ['states'],
                            'actions': ['actions']
                        }),
                    dict(type='RobocasaGR1N15Bridge'),
                    dict(type='ParquetPrompter'),
                    dict(
                        type='ProcessPromptsWithImage',
                        max_len=900,
                        num_images=1,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path='fluxvla/models/third_party_models/eagle2_hg_model')),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(
                        type='NormalizeImages',
                        means=[[127.5, 127.5, 127.5]],
                        stds=[[127.5, 127.5, 127.5]]),
                    dict(
                        type='NormalizeStatesAndActions',
                        state_dim=64,
                        action_dim=32,
                        state_key='proprio',
                        action_key='action',
                        norm_type='min_max',
                        normalize_states=False)
                ],
                action_window_size=16)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=12,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    save_iter_interval=500,
    save_epoch_interval=1,
    max_keep_ckpts=5,
    save_full_model=True,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path='fluxvla/models/third_party_models/eagle2_hg_model'),
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
    sharding_strategy='no-shard',
    change_key_name=False)

inference_model = dict(
    type='LlavaVLA',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleInferenceBackbone',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model',
        vlm_config=dict(max_input_seq_len=900)),
    vla_head=dict(
        type='FlowMatchingInferenceHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_layers=1,
        num_heads=4,
        num_steps=16,
        num_inference_timesteps=4,
        traj_length=16,
        ori_action_dim=29,
        action_dim=32,
        max_input_seq_len=900,
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

eval = dict(
    type='RobocasaEvalRunner',
    model_family='groot',
    task_list=[
        'gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env',
        'gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env',
    ],
    eval_chunk_size=16,
    max_episode_steps=720,
    num_trials_per_task=20,
    seed=7,
    unnorm_key=_STAT,
    dataset=dict(
        type='RobocasaEvalDataset',
        unnorm_key=_STAT,
        transforms=[
            dict(type='ProcessRobocasaEvalInputs',
                 img_key='video.ego_view_bg_crop_pad_res256_freq20',
                 resize_size=224,
                 center_crop_scale=0.95,
                 normalize=False,
                 embodiment_id=24),
            dict(type='RobocasaGR1N15Bridge'),
            dict(
                type='NormalizeStatesAndActions',
                state_dim=64,
                state_key='proprio',
                action_key='action',
                norm_type='min_max',
                normalize_states=False),
            dict(
                type='ProcessPromptsWithImage',
                max_len=900,
                num_images=1,
                return_text=True,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path='fluxvla/models/third_party_models/eagle2_hg_model')),
            dict(
                type='NormalizeImages',
                means=[[127.5, 127.5, 127.5]],
                stds=[[127.5, 127.5, 127.5]]),
        ]),
    denormalize_action=dict(
        type='DenormalizeRobocasaAction',
        norm_type='min_max',
        action_dim=29,
        clip_actions=False,
        stats_order='fluxvla'),
)
