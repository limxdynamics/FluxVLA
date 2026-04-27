# Origin: Modified from
# Upstream-Repo: openvla/openvla
# Upstream-Path: prismatic/vla/datasets/rlds/dataset.py
# Upstream-Ref: c8f03f48af692657d3060c19588038c7220e9af9
# SPDX-License-Identifier: MIT
# Notes: Attribution normalized; no functional change.

import copy
import json
from typing import Any, Dict, List, Union

import numpy as np
import tensorflow as tf
import torch
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import IterableDataset

from fluxvla.engines import DATASETS, build_transform_from_cfg
from fluxvla.engines.utils.eval_utils import (crop_and_resize,
                                              get_libero_image, quat2axisangle)
from .utils.data_utils import (get_oxe_dataset_kwargs_and_weights,
                               make_interleaved_dataset,
                               make_oxe_dataset_kwargs)


@DATASETS.register_module()
class RLDSDataset(IterableDataset):

    def __init__(
        self,
        data_root_dir: str,
        data_mix: List,
        batch_transform: Union[dict, ConfigDict, Config],
        traj_transform_kwargs: Dict = None,
        frame_transform_kwargs: Dict = None,
        load_camera_views: List = ('primary', ),
        load_depth: bool = False,
        load_proprio: bool = False,
        load_language: bool = True,
        action_proprio_normalization_type: str = 'normal',
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Initialize the RLDS dataset.

        Args:
            data_root_dir (str): Path to the root directory of the dataset.
            data_mix (List): List of tuples specifying the dataset mix.
            batch_transform (Union[dict, ConfigDict, Config]):
                Configuration for the batch transformation.
            traj_transform_kwargs (Dict): Additional arguments for
                trajectory transformation.
            frame_transform_kwargs (Dict): Additional arguments for
                frame transformation.
            load_camera_views (List[str]): List of camera views to load.
            load_depth (bool): Whether to load depth information.
            load_proprio (bool): Whether to load proprioceptive information.
            load_language (bool): Whether to load language information.
            action_proprio_normalization_type (str): Type of normalization
                for proprioceptive actions.
            resize_resolution (Tuple[int, int]): Resolution to resize images
                to.
            shuffle_buffer_size (int, optional): Buffer size for shuffling.
                Defaults to 256000.
            train (bool, optional): Whether to use the training split.
                Defaults to True.
            image_aug (bool, optional): Whether to apply image augmentations.
                Defaults to False.
        """

        self.batch_transform = build_transform_from_cfg(batch_transform)
        """Lightweight wrapper around RLDS TFDS Pipeline for use with
        PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix = data_root_dir, data_mix

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            data_mix,
            load_camera_views=load_camera_views,
            load_depth=load_depth,
            load_proprio=load_proprio,
            load_language=load_language,
            action_proprio_normalization_type=  # noqa: E251
            action_proprio_normalization_type,  # noqa: E251
        )
        rlds_config = dict(
            traj_transform_kwargs=traj_transform_kwargs,
            frame_transform_kwargs=frame_transform_kwargs,
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(data_mix),
            traj_read_threads=len(data_mix),
            train=train,
        )
        included_datasets, filtered_mixture_spec = set(), []
        for d_name, d_weight in data_mix:
            if d_name in included_datasets:
                # TODO: Change to logging
                print(f'Skipping Duplicate Dataset: {(d_name, d_weight)!r}')
                continue

            included_datasets.add(d_name)
            filtered_mixture_spec.append((d_name, d_weight))

        # Assemble Dataset Config (kwargs) and Weights
        per_dataset_kwargs, sampling_weights = [], []
        for d_name, d_weight in filtered_mixture_spec:
            try:
                per_dataset_kwargs.append(
                    make_oxe_dataset_kwargs(
                        d_name,
                        data_root_dir,
                        load_camera_views,
                        load_depth,
                        load_proprio,
                        load_language,
                        action_proprio_normalization_type,
                    ))
                sampling_weights.append(d_weight)

            except ValueError as e:
                # TODO: Change to logging
                print(f'Skipping {d_name} due to Error: {e}')

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config['frame_transform_kwargs'].update({
                'image_augment_kwargs':
                dict(
                    random_resized_crop=dict(
                        scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        'random_resized_crop',
                        'random_brightness',
                        'random_contrast',
                        'random_saturation',
                        'random_hue',
                    ],
                )
            }),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = \
            make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():

            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError(
            'IterableDataset does not implement map-style __getitem__; \
                see __iter__ instead!')


@DATASETS.register_module()
class LiberoEvalDataset:
    """Dataset for evaluating Libero with a VLA processor.
    This dataset processes images and prompts for evaluation purposes.
    It resizes images, applies center cropping if specified, and builds
    prompts for the VLA model.

    Args:
        norm_stats (str or Dict): Normalization statistics, which can be a
            JSON string or a dictionary containing 'mean', 'std', 'q01',
            and 'q99' for each feature.
            If a string, it should be a JSON representation of the
            normalization statistics.
        task_suite_name (str): Name of the task suite for evaluation.
        img_keys (List[str]): List of keys to extract images from the input.
            Defaults to ['agentview_image']. Note that the first key
            is used as replay image.
        processor (ConfigDict): Configuration for the VLA processor.
        center_crop (bool): Whether to apply center cropping to images.
            Defaults to False.
        resize_size (int): Size to resize images to. Defaults to 224.
        max_length (int): Maximum length of the input tokens.
            Defaults to 180.
        use_quantiles (bool): Whether to use quantiles for normalization.
            Defaults to True.
    """

    def __init__(self,
                 norm_stats: str,
                 task_suite_name: str,
                 img_transform: ConfigDict,
                 tokenizer: ConfigDict,
                 img_keys: List[str] = ['agentview_image'],
                 center_crop: bool = False,
                 resize_size: int = 224,
                 max_len: int = 180,
                 use_quantiles=True,
                 pad_token_id: int = 0,
                 prompt_suffix='',
                 load_proprio=True,
                 use_pil=True) -> None:
        from fluxvla.engines import (build_tokenizer_from_cfg,
                                     build_transform_from_cfg)
        self.img_transform = build_transform_from_cfg(img_transform)
        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.task_suite_name = task_suite_name
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats
        self.img_keys = img_keys
        self.center_crop = center_crop
        self.resize_size = resize_size
        self.max_len = max_len
        self.use_quantiles = use_quantiles
        self.pad_token_id = pad_token_id
        self.prompt_suffix = prompt_suffix
        self.load_proprio = load_proprio
        self.use_pil = use_pil

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the observation for evaluation."""
        imgs = list()
        for img_key in self.img_keys:
            if img_key not in inputs:
                raise KeyError(f'Inputs are missing image key {img_key!r}.')
            imgs.append(get_libero_image(inputs, self.resize_size, img_key))
        replay_img = copy.deepcopy(imgs[0])
        robot0_eef_pos = inputs['robot0_eef_pos'],
        robot0_eef_quat = inputs['robot0_eef_quat'],
        robot0_gripper_qpos = inputs['robot0_gripper_qpos']
        task_description = inputs['task_description']
        images = list()
        if self.use_pil:
            for img in imgs:
                image = Image.fromarray(img)
                image = image.convert('RGB')

                # (If trained with image augmentations)
                # Center crop image and then
                # resize back up to original size.
                # IMPORTANT: Let's say crop scale == 0.9. To get the new height
                # and width (post-crop), multiply
                # the original height and width by sqrt(0.9) -- not 0.9!
                if self.center_crop:
                    batch_size = 1
                    crop_scale = 0.9

                    # Convert to TF Tensor and record original data type
                    # (should be tf.uint8)
                    image = tf.convert_to_tensor(np.array(image))
                    orig_dtype = image.dtype

                    # Convert to data type tf.float32 and values between [0,1]
                    image = tf.image.convert_image_dtype(image, tf.float32)

                    # Crop and then resize back to original size
                    image = crop_and_resize(image, crop_scale, batch_size)

                    # Convert back to original data type
                    image = tf.clip_by_value(image, 0, 1)
                    image = tf.image.convert_image_dtype(
                        image, orig_dtype, saturate=True)

                    # Convert back to PIL Image
                    image = Image.fromarray(image.numpy())
                    image = image.convert('RGB')
                images.append(image)
        else:
            images = imgs
        image_out = self.img_transform(images)

        if isinstance(image_out, tuple):
            pixel_values, image_grid_thw = image_out
        elif isinstance(image_out, dict):
            pixel_values = image_out.get('pixel_values', None)
            image_grid_thw = image_out.get('image_grid_thw', None)
        else:
            pixel_values = image_out
            image_grid_thw = None
        # Build VLA prompt
        prompt = 'In: What action should the robot take to {}?\nOut:{}'.format(
            task_description.lower(), self.prompt_suffix)

        # Process inputs.
        tokens = torch.tensor(self.tokenizer(prompt)['input_ids']).unsqueeze(0)
        token_mask = [True] * len(tokens[0])
        tokens_len = len(tokens[0])
        if self.max_len is not None:
            if tokens_len < self.max_len:
                token_padding = [self.pad_token_id] * (
                    self.max_len - tokens_len)
                mask_padding = [False] * (self.max_len - tokens_len)
                tokens = tokens[0].cpu().numpy().tolist() + token_padding
                token_mask = token_mask + mask_padding
            else:
                tokens = tokens[:self.max_len]
                token_mask = token_mask[:self.max_len]
        # Image to tensor
        tokens = torch.tensor(tokens)
        if self.load_proprio:
            states = np.concatenate(
                (robot0_eef_pos[0], quat2axisangle(robot0_eef_quat[0]),
                 robot0_gripper_qpos))
            norm_stats = self.norm_stats[self.task_suite_name + '_no_noops']

            if self.use_quantiles:
                states = self._normalize_quantile(states,
                                                  norm_stats['proprio'])
            else:
                states = self._normalize(states, norm_stats['proprio'])

        batch = dict(
            images=pixel_values.cuda().unsqueeze(0),  # noqa: E501
            img_masks=torch.tensor([[True, True]]).cuda(),  # noqa: E501
            lang_tokens=tokens.unsqueeze(0).cuda(),
            lang_masks=torch.tensor(token_mask).unsqueeze(0).cuda())
        if self.load_proprio:
            batch['states'] = torch.from_numpy(
                states).bfloat16().cuda().unsqueeze(0)
        if image_grid_thw is not None:
            batch['image_grid_thw'] = image_grid_thw.unsqueeze(0)
        return batch, replay_img

    def _normalize(self, normalized_states: np.ndarray, stats: Dict):
        assert 'min' in stats and stats['min'] is not None
        assert 'max' in stats and stats['max'] is not None
        state_high = np.array(stats['max'])
        state_low = np.array(stats['min'])
        mask = np.array(stats['mask'])
        states = np.where(
            mask,
            np.clip(
                2 * (normalized_states - state_low) /
                (state_high - state_low + 1e-8) - 1, -1, 1), normalized_states)
        return states

    def _normalize_quantile(self, normalized_states: np.ndarray, stats: Dict):
        assert 'q01' in stats and stats['q01'] is not None
        assert 'q99' in stats and stats['q99'] is not None  # noqa: E501
        state_high = np.array(stats['q99'])
        state_low = np.array(stats['q01'])
        if 'mask' in stats:
            mask = np.array(stats['mask'])
        else:
            mask = np.ones_like(state_high, dtype=bool)
        states = np.where(
            mask,
            np.clip(
                2 * (normalized_states - state_low) /
                (state_high - state_low + 1e-8) - 1, -1, 1), normalized_states)
        return states
