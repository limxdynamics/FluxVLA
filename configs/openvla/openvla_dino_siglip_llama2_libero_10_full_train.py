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
    type='OpenVLA',
    arch_specifier='no-align+fused-gelu-mlp',
    vision_backbone=dict(
        type='DinoSigLIPViTBackbone',
        vision_backbone_id='dinosiglip-vit-so-224px',
        dino_config=dict(
            model_id='dino',
            file=  # noqa: E251
            './checkpoints/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors'  # noqa: E501
        ),
        image_resize_strategy='resize-naive',
        siglip_config=dict(
            model_id='siglip_224',
            file=  # noqa: E251
            './checkpoints/ViT-SO400M-14-SigLIP/open_clip_model.safetensors'  # noqa: E501
        )),
    llm_backbone=dict(
        type='LLaMa2LLMBackbone',
        llm_backbone_id='llama2-7b-pure_causal',
        llm_family='llama',
        llm_path=  # noqa: E251
        './checkpoints/Llama-2-7b-hf',  # noqa: E501
        llm_max_length=2048,
        hf_token=None,
        inference_mode=False),
    projector=dict(
        type='FusedMLPProjector', fused_vision_dim=2176, llm_dim=4096),
    tokenizer=dict(
        type='ActionTokenizer',
        model_path=  # noqa: E251
        './checkpoints/openvla-7b-finetuned-libero-10',  # noqa: E501
        bins=256,
        min_action=-1,
        max_action=1,
    ),
    pretrained_name_or_path=  # noqa: E251
    None,  # noqa: E501
    vla_head=dict(type='OpenVLAHead', norm_stats=None, vocab_size=32000),
    freeze_vision_backbone=False,
    freeze_llm_backbone=False,
    freeze_projector=False)

train_dataloader = dict(
    per_device_batch_size=8,
    dataset=dict(
        type='RLDSDataset',
        data_root_dir=  # noqa: E251
        './datasets/modified_libero_rlds',
        data_mix=[('libero_10_no_noops', 1.0)],
        batch_transform=dict(
            type='RLDSBatchTransform',
            load_camera_views=['image_primary', 'image_primary'],
            action_tokenizer=dict(
                type='ActionTokenizer',
                model_path=  # noqa: E251
                './checkpoints/openvla-7b-finetuned-libero-10',  # noqa: E501
                bins=256,
                min_action=-1,
                max_action=1,
            ),
            base_tokenizer=dict(
                type='PretrainedTokenizer',
                model_path=  # noqa: E251
                './checkpoints/openvla-7b-finetuned-libero-10',  # noqa: E501
                # special_tokens={'pad_token': '<PAD>'}
            ),
            prompter=dict(
                type='PurePrompter',
                model_family='openvla',
            ),
            img_transform=dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224]],
                means=[[123.515625, 116.04492188, 103.59375], [128, 128, 128]],
                stds=[[58.27148438, 57.02636719, 57.27539062], [128, 128,
                                                                128]],
            ),
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            future_action_window_size=0,
            skip_unlabeled=True,
            goal_relabeling_strategy='uniform',
        ),
        frame_transform_kwargs=dict(
            resize_size=(224, 224),
            num_parallel_calls=16,
        ),
        shuffle_buffer_size=256000,
        train=True,
        image_aug=False,
        action_proprio_normalization_type='bounds_q99'))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=24,
    learning_rate=2e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sampler=None,
    collator=dict(
        type='PaddedCollatorForActionPrediction',
        model_max_length=2048,
        pad_token_id=0,
        padding_side='right',
        pixel_values_dtype='fp16',
        ignore_idx=-100),
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
    mixed_precision_dtype='bf16')

eval = dict(
    type='LiberoEvalRunner',
    model_family='openvla',
    task_suite_name='libero_10',
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'agentview_image'],
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224]],
                means=[[123.515625, 116.04492188, 103.59375], [128, 128, 128]],
                stds=[[58.27148438, 57.02636719, 57.27539062], [128, 128,
                                                                128]],
            ),
            dict(
                type='LiberoPromptFromInputs',
                prompt_suffix=' ',
                max_len=None,
                tokenizer=dict(
                    type='PretrainedTokenizer',
                    model_path=  # noqa: E251
                    './checkpoints/openvla-7b-finetuned-libero-10',  # noqa: E501
                    # special_tokens={'pad_token': '<PAD>'}
                )),
        ]),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='quantile',
        action_norm_mask=[True, True, True, True, True, True, False],
    ),
    resize_size=224,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7)
