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
import json
import os
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import concatenate_datasets, load_dataset
from fluxvla.engines import DATASETS, build_transform_from_cfg


@DATASETS.register_module()
class ParquetDataset(Dataset):

    def __init__(self,
                 data_root_path: Union[str, List[str]],
                 transforms: List[Dict],
                 action_window_size: int = 9,
                 action_key: str = 'observation.state',
                 use_delta: bool = False,
                 statistic_name: str = 'private',
                 window_start_idx: int = 1) -> None:
        """Initialize the Parquet dataset.

        Args:
            data_root_path (Union[str, List[str]]): Path(s) to the root
                directory(ies). The metadata will be loaded from
                `data_root_path/meta` and data from `data_root_path/data`.
                If a list is provided, multiple datasets will be loaded and
                concatenated.
            transforms (List[Dict]): List of transformation configurations.
            batch_transform (Union[dict, ConfigDict, Config]):
                Configuration for the batch transformation.
            episodes (list[int]): List of episode indices to include
                in the dataset.
            local_files_only (bool): Whether to use local files only.
            action_horizon (int): The number of time steps for the
                action sequence.
            video_backend (str, optional): Backend for
                video processing.
                Defaults to None.
            action_key (str): Key for the action data.
            use_delta (bool): Whether to use delta actions.
                Defaults to False.
            statistic_name (str): Name for the statistics collection.
                Defaults to 'private'.
            window_start_idx (int): Start index for the action window.
                Defaults to 1.
        """
        super().__init__()
        self.action_window_size = action_window_size
        if isinstance(data_root_path, str):
            data_root_path = [data_root_path]
        self.data_root_path = data_root_path

        meta_root = [os.path.join(path, 'meta') for path in data_root_path]
        data_root = [os.path.join(path, 'data') for path in data_root_path]

        # Merge multiple meta_root
        all_stats = []
        all_tasks = []
        all_episodes = []
        info_list = []

        for root in meta_root:
            assert os.path.exists(os.path.join(root, 'info.json')), \
                f'Metadata file not found at {os.path.join(root, "info.json")}'  # noqa: E501
            with open(os.path.join(root, 'info.json'), 'rb') as f:
                info_list.append(json.load(f))

            assert os.path.exists(
                os.path.join(root, 'episodes_stats.jsonl')), \
                f'Statistics file not found at {os.path.join(root, "episodes_stats.jsonl")}'  # noqa: E501
            with open(
                    os.path.join(root, 'episodes_stats.jsonl'),
                    'r',
                    encoding='utf-8') as f:
                all_stats.extend([json.loads(line) for line in f])

            assert os.path.exists(os.path.join(root, 'tasks.jsonl')), \
                f'Tasks file not found at {os.path.join(root, "tasks.jsonl")}'
            with open(
                    os.path.join(root, 'tasks.jsonl'), 'r',
                    encoding='utf-8') as f:
                all_tasks.append([json.loads(line) for line in f])

            assert os.path.exists(os.path.join(root, 'episodes.jsonl')), \
                f'Episodes file not found at {os.path.join(root, "episodes.jsonl")}'  # noqa: E501
            with open(
                    os.path.join(root, 'episodes.jsonl'), 'r',
                    encoding='utf-8') as f:
                all_episodes.extend([json.loads(line) for line in f])

        self.info = info_list
        self.stats = all_stats
        self.tasks = all_tasks
        self.episodes = all_episodes
        # Summarize all data_root
        datasets = []
        dataset_sizes = []  # Record the size of each dataset
        for root in data_root:
            hf_dataset = load_dataset('parquet', data_dir=root, split='train')
            dataset_sizes.append(len(hf_dataset))
            datasets.append(hf_dataset)
        hf_dataset = concatenate_datasets(datasets)
        # Compute cumulative sizes for fast index lookup
        self.dataset_cumulative_sizes = np.cumsum([0] + dataset_sizes)
        self.dataset = hf_dataset
        self.transforms = list()
        self.action_key = action_key
        self.use_delta = use_delta
        self.statistic_name = statistic_name
        self.window_start_idx = window_start_idx
        for transform in transforms:
            self.transforms.append(build_transform_from_cfg(transform))

    def _rand_another(self):
        """Randomly select another index from the dataset."""
        return np.random.randint(0, len(self.dataset))

    def _get_dataset_index(self, index: int) -> int:
        """Get which dataset in data_root list the index belongs to.

        Args:
            index (int): The index in the concatenated dataset.

        Returns:
            int: The index of the dataset in data_root list (0-based).
        """
        if self.dataset_cumulative_sizes is None:
            return 0
        # Use binary search to find the index of the dataset in data_root list
        dataset_idx = np.searchsorted(
            self.dataset_cumulative_sizes, index, side='right') - 1
        return dataset_idx

    def __getitem__(self, index, dataset_statistics):
        data = self.dataset[index]
        # Determine which dataset the data belongs to
        dataset_idx = self._get_dataset_index(index)
        while (index == len(self.dataset) - 1
               or self.dataset[index]['episode_index'] !=
               self.dataset[index + 1]['episode_index']
               or self._get_dataset_index(index + 1) != dataset_idx or
               self.tasks[dataset_idx][self.dataset[index +
                                                    1]['task_index']]['task']
               == 'empty' or
               self.tasks[dataset_idx][self.dataset[index +
                                                    1]['task_index']]['task']
               == 'static'):

            index = self._rand_another()
            data = self.dataset[index]
            # Recalculate dataset_idx
            dataset_idx = self._get_dataset_index(index)
        actions = list()
        action_masks = list()
        window_idx = self.window_start_idx
        while len(actions) < self.action_window_size:
            if (index + window_idx < len(self.dataset)
                    and data['episode_index']
                    == self.dataset[index + window_idx]['episode_index']
                    and  # noqa: E501
                    self._get_dataset_index(index + window_idx) == dataset_idx
                    and  # noqa: E501
                    self.tasks[dataset_idx][self.dataset[index + window_idx]
                                            ['task_index']]['task'] != 'empty'
                    and  # noqa: E501
                    self.tasks[dataset_idx][self.dataset[index + window_idx]
                                            ['task_index']]['task'] !=
                    'static'):  # noqa: E501
                if self.use_delta:
                    actions.append(
                        np.array(self.dataset[index +
                                              window_idx][self.action_key]) -
                        np.array(self.dataset[index + window_idx -
                                              1][self.action_key]).tolist())
                else:
                    actions.append(self.dataset[index +
                                                window_idx][self.action_key])
                action_masks.append(1)
            elif index + window_idx >= len(
                    self.dataset) or self.tasks[dataset_idx][self.dataset[
                        index + window_idx]['task_index']]['task'] == 'empty':
                for _ in range(self.action_window_size - len(actions)):
                    actions.append(actions[-1])
                    action_masks.append(0)
                break
            elif self.tasks[dataset_idx][self.dataset[index + window_idx]
                                         ['task_index']]['task'] == 'static':
                window_idx += 1
                continue
            else:
                if len(actions) > 0:
                    actions.append(actions[-1])
                else:
                    actions.append(data[self.action_key])
                action_masks.append(0)
            window_idx += 1
        data['info'] = self.info[dataset_idx]
        data['stats'] = dataset_statistics[self.statistic_name]
        data['actions'] = np.array(actions, dtype=np.float32)
        data['action_masks'] = np.array(action_masks, dtype=np.float32)
        data['task_description'] = self.tasks[dataset_idx][data['task_index']][
            'task']  # noqa: E501
        data['data_root'] = self.data_root_path[dataset_idx]
        for transform in self.transforms:
            data = transform(data)

        return data

    def __len__(self):
        return len(self.dataset)

        # Additional initialization can be added here if needed.


@DATASETS.register_module()
class LiberoParquetEvalDataset:
    """Evaluation dataset pipeline for Libero using Parquet-style transforms.

    This mirrors the behavior of `LiberoEvalDataset` in `rlds_dataset.py`,
    but composes processing via a list of transforms similar to
    `ParquetDataset`.

    Args:
        norm_stats (str | Dict): Normalization stats dict or path to JSON.
        task_suite_name (str): Name of Libero task suite (for stats keying).
        tokenizer (Dict): Tokenizer config for `build_tokenizer_from_cfg`.
        transforms (List[Dict]): List of transform configs applied in order.
    """

    def __init__(self,
                 norm_stats: Any,
                 task_suite_name: str,
                 transforms: List[Dict],
                 num_padding_imgs: int = 0) -> None:

        # Build image/token transforms (parquet-style sequential list)
        self.transforms = [build_transform_from_cfg(t) for t in transforms]
        self.task_suite_name = task_suite_name
        self.num_padding_imgs = num_padding_imgs
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Compose transforms chain (parquet-style) starting from raw inputs
        data: Dict[str, Any] = dict(inputs)
        if self.norm_stats is not None:
            norm_stats = self.norm_stats[self.task_suite_name + '_no_noops']
        else:
            norm_stats = None
        data['norm_stats'] = norm_stats
        for t in self.transforms:
            data = t(data)
        replay_img = data.get('replay_img', None)

        assert 'lang_tokens' in data and 'lang_masks' in data, \
            'Prompt transform must provide lang_tokens and lang_masks'
        tokens = torch.tensor(data['lang_tokens'])
        token_mask = data['lang_masks'].tolist() if hasattr(
            data['lang_masks'], 'tolist') else list(data['lang_masks'])

        # Proprio
        img_masks = data.get('img_masks', None)
        pixel_values = data['pixel_values']
        if img_masks is None:
            # Fallback: all True masks based on pixel_values shape
            num_imgs = pixel_values.shape[0] // 3
            img_masks = [True] * num_imgs
        else:
            img_masks = list(img_masks)
        # Add padding images with zero values and False masks
        if self.num_padding_imgs > 0:
            padding_img = pixel_values.new_zeros(3, pixel_values.shape[-2],
                                                 pixel_values.shape[-1])
            padding_imgs = padding_img.repeat(self.num_padding_imgs, 1, 1)
            pixel_values = torch.cat([pixel_values, padding_imgs], dim=0)
            img_masks.extend([False] * self.num_padding_imgs)
        batch: Dict[str, Any] = dict(
            images=pixel_values.cuda().unsqueeze(0),
            img_masks=torch.tensor([img_masks]).cuda(),
            lang_tokens=tokens.unsqueeze(0).cuda(),
            lang_masks=torch.tensor(token_mask).unsqueeze(0).cuda(),
        )

        if 'states' in data:
            batch['states'] = torch.from_numpy(
                data['states']).bfloat16().cuda().unsqueeze(0)
        if 'embodiment_ids' in data:
            batch['embodiment_ids'] = torch.from_numpy(
                data['embodiment_ids']).int().cuda().unsqueeze(0)

        if data.get('image_grid_thw', None) is not None:
            batch['image_grid_thw'] = data['image_grid_thw'].unsqueeze(0)

        return batch, replay_img


@DATASETS.register_module()
class PrivateInferenceDataset:
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
                 transforms: List[Dict],
                 model_path: str,
                 img_keys: List[str] = ['agentview_image'],
                 center_crop: bool = False,
                 resize_size: int = 224,
                 max_len: int = 180,
                 use_quantiles=True,
                 embodiment_id: int = None) -> None:
        from fluxvla.engines import build_transform_from_cfg
        self.transforms = list()
        for transform in transforms:
            transform['model_path'] = model_path
            self.transforms.append(build_transform_from_cfg(transform))
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
        self.embodiment_id = embodiment_id

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the observation for evaluation."""
        imgs = list()
        for img_key in self.img_keys:
            if img_key not in data:
                raise KeyError(f'Image key `{img_key}` not found in inputs!')
            imgs.append(data[img_key].transpose(2, 0, 1))  # HWC to CHW
        inputs = dict(
            images=imgs,
            task_description=data.get('task_description',
                                      'No task description provided'),
            stats=self.norm_stats['private'],
            states=data['qpos'])
        for transform in self.transforms:
            inputs = transform(inputs)

        batch = dict(
            images=torch.from_numpy(
                inputs['images']).unsqueeze(0).cuda(),  # noqa: E501
            img_masks=torch.tensor([[True for _ in range(len(self.img_keys))]
                                    ]).cuda(),  # noqa: E501
            lang_tokens=torch.from_numpy(
                inputs['lang_tokens']).unsqueeze(0).cuda(),
            lang_masks=torch.from_numpy(
                inputs['lang_masks']).unsqueeze(0).cuda(),
            states=torch.from_numpy(
                inputs['states']).float().cuda().unsqueeze(0))
        if self.embodiment_id is not None:
            batch['embodiment_ids'] = torch.from_numpy(
                np.array(self.embodiment_id)).int().cuda().unsqueeze(0)
        return batch

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
