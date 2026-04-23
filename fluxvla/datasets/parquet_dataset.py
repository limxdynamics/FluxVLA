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
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import concatenate_datasets, load_dataset
from fluxvla.engines import DATASETS, build_transform_from_cfg


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_parquet_records(paths: List[Path]) -> List[Dict[str, Any]]:
    if not paths:
        return []
    dataset = load_dataset(
        'parquet', data_files=[str(path) for path in paths], split='train')
    return [dict(record) for record in dataset]


def _load_episode_records(meta_root: Path) -> List[Dict[str, Any]]:
    legacy_path = meta_root / 'episodes.jsonl'
    if legacy_path.exists():
        return _load_jsonl_records(legacy_path)

    episodes_dir = meta_root / 'episodes'
    parquet_files = sorted(episodes_dir.rglob('*.parquet'))
    if parquet_files:
        return _load_parquet_records(parquet_files)

    raise FileNotFoundError(f'Episodes metadata not found under {meta_root}')


def _load_stats_records(meta_root: Path) -> List[Dict[str, Any]]:
    v3_stats_path = meta_root / 'stats.json'
    if v3_stats_path.exists():
        return [{'stats': _load_json(v3_stats_path)}]

    legacy_path = meta_root / 'episodes_stats.jsonl'
    if legacy_path.exists():
        return _load_jsonl_records(legacy_path)

    raise FileNotFoundError(f'Statistics metadata not found under {meta_root}')


def _extract_task_text(record: Dict[str, Any]) -> str:
    for key in ('task', 'tasks', '__index_level_0__'):
        value = record.get(key)
        if value is not None:
            return str(value)

    for key, value in record.items():
        if key == 'task_index' or value is None:
            continue
        if isinstance(value, str):
            return value

    for key, value in record.items():
        if key != 'task_index' and value is not None:
            return str(value)

    return ''


def _load_task_mapping(meta_root: Path) -> Dict[int, str]:
    v3_path = meta_root / 'tasks.parquet'
    if v3_path.exists():
        records = _load_parquet_records([v3_path])
    else:
        legacy_path = meta_root / 'tasks.jsonl'
        if not legacy_path.exists():
            raise FileNotFoundError(
                f'Tasks metadata not found under {meta_root}')
        records = _load_jsonl_records(legacy_path)

    tasks = {}
    for fallback_index, record in enumerate(records):
        task_index = record.get('task_index', fallback_index)
        if isinstance(task_index, np.ndarray) and task_index.ndim == 0:
            task_index = task_index.item()
        if isinstance(task_index, torch.Tensor) and task_index.numel() == 1:
            task_index = task_index.item()
        tasks[int(task_index)] = _extract_task_text(record)
    return tasks


@DATASETS.register_module()
class ParquetDataset(Dataset):

    def __init__(self,
                 data_root_path: Union[str, List[str]],
                 transforms: List[Dict],
                 action_window_size: int = 9,
                 action_key: str = 'observation.state',
                 use_delta: bool = False,
                 statistic_name: str = 'private',
                 window_start_idx: int = 1,
                 frame_window_size: int = 1) -> None:
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
        episodes_by_dataset = []
        info_list = []

        for root in meta_root:
            meta_path = Path(root)
            info_path = meta_path / 'info.json'
            assert os.path.exists(info_path), \
                f'Metadata file not found at {info_path}'
            info = _load_json(info_path)
            if 'features' in info:
                for feature in info['features'].values():
                    if isinstance(feature.get('shape'), list):
                        feature['shape'] = tuple(feature['shape'])
            info_list.append(info)

            all_stats.extend(_load_stats_records(meta_path))
            all_tasks.append(_load_task_mapping(meta_path))

            episode_records = _load_episode_records(meta_path)
            episodes_by_dataset.append(episode_records)
            all_episodes.extend(episode_records)

        self.info = info_list
        self.stats = all_stats
        self.tasks = all_tasks
        self.episodes = all_episodes
        self.episodes_by_dataset = episodes_by_dataset
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
        self.frame_window_size = frame_window_size
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

    def _resolve_task_description(self, dataset_idx: int, data: Dict) -> str:
        raw_task = data.get('task')
        if isinstance(raw_task, str):
            return raw_task
        if isinstance(raw_task, (list, tuple)) and len(raw_task) == 1:
            only_value = raw_task[0]
            if isinstance(only_value, str):
                return only_value
            if isinstance(only_value, (int, np.integer)):
                return self.tasks[dataset_idx].get(int(only_value), '')
        if isinstance(raw_task, np.ndarray) and raw_task.ndim == 0:
            raw_task = raw_task.item()
        if isinstance(raw_task, torch.Tensor) and raw_task.numel() == 1:
            raw_task = raw_task.item()
        if isinstance(raw_task, (int, np.integer)):
            return self.tasks[dataset_idx].get(int(raw_task), '')

        raw_task_index = data.get('task_index')
        if isinstance(raw_task_index, np.ndarray) and raw_task_index.ndim == 0:
            raw_task_index = raw_task_index.item()
        if isinstance(raw_task_index,
                      torch.Tensor) and raw_task_index.numel() == 1:
            raw_task_index = raw_task_index.item()
        if isinstance(raw_task_index, (int, np.integer)):
            return self.tasks[dataset_idx].get(int(raw_task_index), '')

        return ''

    def __getitem__(self, index, dataset_statistics):
        data = self.dataset[index]
        # Determine which dataset the data belongs to
        dataset_idx = self._get_dataset_index(index)
        while (index == len(self.dataset) - 1
               or self.dataset[index]['episode_index'] !=
               self.dataset[index + 1]['episode_index']
               or self._get_dataset_index(index + 1) != dataset_idx
               or self._resolve_task_description(
                   dataset_idx, self.dataset[index + 1]) == 'empty'
               or self._resolve_task_description(
                   dataset_idx, self.dataset[index + 1]) == 'static'):

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
                    self._resolve_task_description(
                        dataset_idx, self.dataset[index + window_idx]) !=
                    'empty' and  # noqa: E501
                    self._resolve_task_description(
                        dataset_idx,
                        self.dataset[index + window_idx]) != 'static'):
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
                    self.dataset) or self._resolve_task_description(
                        dataset_idx,
                        self.dataset[index + window_idx]) == 'empty':
                for _ in range(self.action_window_size - len(actions)):
                    actions.append(actions[-1])
                    action_masks.append(0)
                break
            elif self._resolve_task_description(
                    dataset_idx, self.dataset[index + window_idx]) == 'static':
                window_idx += 1
                continue
            else:
                if len(actions) > 0:
                    actions.append(actions[-1])
                else:
                    actions.append(data[self.action_key])
                action_masks.append(0)
            window_idx += 1
        # Collect forward-looking frame timestamps for video models
        if self.frame_window_size > 1:
            frame_timestamps = [data['timestamp']]
            frame_masks = [1]
            for fi in range(1, self.frame_window_size):
                future_idx = index + fi
                if (future_idx < len(self.dataset)
                        and self.dataset[future_idx]['episode_index']
                        == data['episode_index'] and
                        self._get_dataset_index(future_idx) == dataset_idx):
                    frame_timestamps.append(
                        self.dataset[future_idx]['timestamp'])
                    frame_masks.append(1)
                else:
                    frame_timestamps.append(frame_timestamps[-1])
                    frame_masks.append(0)
            data['frame_timestamps'] = frame_timestamps
            data['frame_masks'] = np.array(frame_masks, dtype=np.float32)

        data['info'] = self.info[dataset_idx]
        data['stats'] = dataset_statistics[self.statistic_name]
        data['actions'] = np.array(actions, dtype=np.float32)
        data['action_masks'] = np.array(action_masks, dtype=np.float32)
        data['task_description'] = self._resolve_task_description(
            dataset_idx, data)
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

        batch['reset_history'] = bool(data.get('is_new_episode', False))

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
                raise KeyError(
                    'Image key {!r} not found in inputs!'.format(img_key))
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
