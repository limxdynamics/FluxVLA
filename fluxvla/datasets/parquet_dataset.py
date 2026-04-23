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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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
        info_list = []

        for root in meta_root:
            info_path = os.path.join(root, 'info.json')
            assert os.path.exists(info_path), \
                f'Metadata file not found at {info_path}'
            with open(os.path.join(root, 'info.json'), 'rb') as f:
                info_list.append(json.load(f))

            stats_path = os.path.join(root, 'episodes_stats.jsonl')
            assert os.path.exists(stats_path), \
                f'Statistics file not found at {stats_path}'
            with open(
                    os.path.join(root, 'episodes_stats.jsonl'),
                    'r',
                    encoding='utf-8') as f:
                all_stats.extend([json.loads(line) for line in f])

            tasks_path = os.path.join(root, 'tasks.jsonl')
            assert os.path.exists(tasks_path), \
                f'Tasks file not found at {tasks_path}'
            with open(tasks_path, 'r', encoding='utf-8') as f:
                all_tasks.append([json.loads(line) for line in f])

            episodes_path = os.path.join(root, 'episodes.jsonl')
            assert os.path.exists(episodes_path), \
                f'Episodes file not found at {episodes_path}'
            with open(episodes_path, 'r', encoding='utf-8') as f:
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
        data['task_description'] = self.tasks[dataset_idx][data['task_index']][
            'task']  # noqa: E501
        data['data_root'] = self.data_root_path[dataset_idx]
        for transform in self.transforms:
            data = transform(data)

        return data

    def __len__(self):
        return len(self.dataset)

        # Additional initialization can be added here if needed.


@dataclass
class _EpisodeRecord:
    episode_index: int
    episode_chunk: int
    task_description: str
    timestamps: np.ndarray
    traj_20d: np.ndarray


def _load_jsonl(path: Path) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def _load_parquet_columns(path: Path,
                          columns: Sequence[str]) -> Dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(columns))
    data = table.to_pydict()
    return {key: np.asarray(value) for key, value in data.items()}


@DATASETS.register_module()
class LiberoLeRobotEE6DDataset(Dataset):
    """Map-style X-VLA dataset backed by LeRobot parquet and videos.

    The dataset stays within LeRobot v2 storage conventions, but restores
    X-VLA's training contract from `abs_action_6d` supervision:

    1. single-arm 10D EE6D trajectory expanded to 20D by zero-padding arm-2
    2. 1 second future window interpolated to `num_actions + 1` steps
    3. current step becomes `states`, remaining steps become `actions`
    4. camera frames are decoded from LeRobot videos instead of HDF5 blobs
    """

    def __init__(
        self,
        data_root_path: str,
        transforms: List[Dict],
        num_actions: int = 30,
        num_views: int = 3,
        image_keys: Optional[List[str]] = None,
        embodiment_id: int = 3,
        training: bool = True,
        statistic_name: str = 'private',
        image_size: int = 224,
        future_window_seconds: float = 1.0,
        frame_tolerance_s: float = 0.1,
        drop_incomplete_future: bool = True,
        image_frame_offset: int = 1,
        episode_indices: Optional[List[int]] = None,
        max_episodes: Optional[int] = None,
        static_threshold: float = 1e-5,
    ) -> None:
        super().__init__()
        from torchvision import transforms as tv_transforms
        from torchvision.transforms import InterpolationMode

        self.data_root = Path(data_root_path)
        self.num_actions = num_actions
        self.num_views = num_views
        self.embodiment_id = embodiment_id
        self.statistic_name = statistic_name
        self.image_size = image_size
        self.future_window_seconds = future_window_seconds
        self.frame_tolerance_s = frame_tolerance_s
        self.drop_incomplete_future = drop_incomplete_future
        self.image_frame_offset = image_frame_offset
        self.static_threshold = static_threshold
        self.image_keys = list(image_keys) if image_keys is not None else [
            'observation.images.image',
            'observation.images.wrist_image',
        ]
        if not self.image_keys:
            raise ValueError('`image_keys` must contain at least one video key.')
        if self.image_frame_offset < 0:
            raise ValueError('`image_frame_offset` must be non-negative.')

        meta_root = self.data_root / 'meta'
        info_path = meta_root / 'info.json'
        if not info_path.exists():
            raise FileNotFoundError(f'Metadata file not found: {info_path}')
        with open(info_path, 'r', encoding='utf-8') as f:
            self.info = json.load(f)

        feature_info = self.info.get('features', {})
        if 'abs_action_6d' not in feature_info:
            raise KeyError(
                '`abs_action_6d` is required in LeRobot info.json features '
                'for XVLA training.',
            )
        if feature_info['abs_action_6d'].get('shape') != [10]:
            raise ValueError(
                'LeRobot feature `abs_action_6d` must have shape [10].',
            )
        if 'video_path' not in self.info:
            raise KeyError('LeRobot info.json must define `video_path`.')

        self.episodes_meta = _load_jsonl(meta_root / 'episodes.jsonl')
        self.tasks_meta = _load_jsonl(meta_root / 'tasks.jsonl')
        stats_path = meta_root / 'episodes_stats.jsonl'
        self.stats = _load_jsonl(stats_path) if stats_path.exists() else []

        selected_episodes = self.episodes_meta
        if episode_indices is not None:
            episode_set = set(episode_indices)
            selected_episodes = [
                episode for episode in selected_episodes
                if int(episode['episode_index']) in episode_set
            ]
        if max_episodes is not None:
            selected_episodes = selected_episodes[:max_episodes]
        if not selected_episodes:
            raise ValueError('No episodes selected for LiberoLeRobotEE6DDataset.')

        self.records = [
            self._load_episode_record(episode_meta)
            for episode_meta in selected_episodes
        ]
        self._index = []
        for episode_slot, record in enumerate(self.records):
            self._index.extend(self._build_episode_index(record, episode_slot))
        if not self._index:
            raise ValueError(
                'No valid training samples found in the selected episodes.',
            )

        aug = [
            tv_transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
        ]
        if training:
            aug.append(
                tv_transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.0,
                ), )
        aug.extend([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])
        self.image_aug = tv_transforms.Compose(aug)
        self.transforms = [build_transform_from_cfg(t) for t in transforms]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int, dataset_statistics=None) -> Dict:
        from PIL import Image

        del dataset_statistics

        episode_slot, step_idx = self._index[index]
        record = self.records[episode_slot]
        cur = float(record.timestamps[step_idx])
        query = np.linspace(
            cur,
            min(cur + self.future_window_seconds, float(record.timestamps[-1])),
            self.num_actions + 1,
            dtype=np.float32,
        )
        traj_seq = self._interp_traj(record.timestamps, record.traj_20d,
                                     query)

        image_idx = min(step_idx + self.image_frame_offset,
                        len(record.timestamps) - 1)
        image_timestamp = float(record.timestamps[image_idx])

        images = []
        img_masks = []
        for video_key in self.image_keys[:self.num_views]:
            frame = self._decode_video_frame(
                self._video_path(record.episode_chunk, record.episode_index,
                                 video_key),
                image_timestamp,
                self.frame_tolerance_s,
            )
            pil_img = Image.fromarray(frame).convert('RGB')
            images.append(self.image_aug(pil_img).numpy())
            img_masks.append(True)

        if images:
            padding_image = np.zeros_like(images[0], dtype=np.float32)
        else:
            padding_image = np.zeros(
                (3, self.image_size, self.image_size),
                dtype=np.float32,
            )
        while len(images) < self.num_views:
            images.append(padding_image.copy())
            img_masks.append(False)

        data = {
            'states': traj_seq[0].copy(),
            'images': np.stack(images, axis=0),
            'img_masks': np.asarray(img_masks, dtype=bool),
            'actions': traj_seq[1:].copy(),
            'action_masks': np.ones((self.num_actions,), dtype=np.float32),
            'task_description': record.task_description,
            'embodiment_ids': np.array(self.embodiment_id),
            'stats': {},
            'info': self.info,
            'timestamp': cur,
            'prompt': '',
        }

        for transform in self.transforms:
            data = transform(data)

        return data

    def _load_episode_record(self, episode_meta: Dict) -> _EpisodeRecord:
        episode_index = int(episode_meta['episode_index'])
        episode_chunk = episode_index // int(self.info['chunks_size'])
        parquet_path = self.data_root / self.info['data_path'].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        if not parquet_path.exists():
            raise FileNotFoundError(
                f'Parquet file for episode {episode_index} not found: '
                f'{parquet_path}',
            )

        columns = _load_parquet_columns(parquet_path,
                                        ['timestamp', 'abs_action_6d'])
        timestamps = np.asarray(columns['timestamp'], dtype=np.float32)
        if timestamps.ndim != 1:
            timestamps = timestamps.reshape(-1)
        if timestamps.shape[0] < 2:
            raise ValueError(
                f'Episode {episode_index} must have at least 2 timestamps.',
            )
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                f'Episode {episode_index} timestamps must be strictly '
                'increasing for interpolation.',
            )

        abs_action_6d = np.asarray(columns['abs_action_6d'], dtype=np.float32)
        if abs_action_6d.ndim != 2 or abs_action_6d.shape[1] != 10:
            raise ValueError(
                f'Episode {episode_index} has invalid `abs_action_6d` shape '
                f'{abs_action_6d.shape}; expected [T, 10].',
            )
        if abs_action_6d.shape[0] != timestamps.shape[0]:
            raise ValueError(
                f'Length mismatch in episode {episode_index}: '
                f'{abs_action_6d.shape[0]} actions vs {timestamps.shape[0]} '
                'timestamps.',
            )

        traj_20d = np.zeros((abs_action_6d.shape[0], 20), dtype=np.float32)
        traj_20d[:, :9] = abs_action_6d[:, :9]
        traj_20d[:, 9:10] = (abs_action_6d[:, 9:10] > 0.0).astype(np.float32)

        tasks = episode_meta.get('tasks', [])
        if not tasks:
            raise ValueError(
                f'Episode {episode_index} must contain at least one task '
                'description in episodes.jsonl.',
            )

        return _EpisodeRecord(
            episode_index=episode_index,
            episode_chunk=episode_chunk,
            task_description=str(tasks[0]),
            timestamps=timestamps,
            traj_20d=traj_20d,
        )

    def _build_episode_index(self, record: _EpisodeRecord,
                             episode_slot: int) -> List[tuple[int, int]]:
        timestamps = record.timestamps
        traj = record.traj_20d
        max_start = len(timestamps) - 1
        if self.drop_incomplete_future:
            valid_steps = np.where(
                timestamps + self.future_window_seconds <=
                timestamps[-1] + 1e-6)[0]
        else:
            valid_steps = np.arange(max_start, dtype=np.int64)

        sample_index = []
        for idx in valid_steps.tolist():
            if idx + 1 >= len(traj):
                continue
            if idx + self.image_frame_offset >= len(timestamps):
                continue
            if np.abs(traj[idx + 1] - traj[idx]).max() < self.static_threshold:
                continue
            sample_index.append((episode_slot, idx))
        return sample_index

    def _video_path(self, episode_chunk: int, episode_index: int,
                    video_key: str) -> Path:
        return self.data_root / self.info['video_path'].format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_index,
        )

    @staticmethod
    def _interp_traj(timestamps: np.ndarray, traj: np.ndarray,
                     query: np.ndarray) -> np.ndarray:
        out = np.empty((len(query), traj.shape[1]), dtype=np.float32)
        for dim in range(traj.shape[1]):
            out[:, dim] = np.interp(
                query,
                timestamps,
                traj[:, dim],
                left=float(traj[0, dim]),
                right=float(traj[-1, dim]),
            )
        return out

    @staticmethod
    def _decode_video_frame(video_path: Path,
                            timestamp: float,
                            tolerance_s: float,
                            backend: str = 'pyav') -> np.ndarray:
        import torchvision

        if not video_path.exists():
            raise FileNotFoundError(f'Video file not found: {video_path}')

        keyframes_only = False
        torchvision.set_video_backend(backend)
        if backend == 'pyav':
            keyframes_only = True

        reader = torchvision.io.VideoReader(str(video_path), 'video')
        reader.seek(float(timestamp), keyframes_only=keyframes_only)

        loaded_frames = []
        loaded_ts = []
        for frame in reader:
            loaded_frames.append(frame['data'])
            loaded_ts.append(float(frame['pts']))
            if frame['pts'] >= timestamp:
                break

        if backend == 'pyav':
            reader.container.close()

        if not loaded_frames:
            raise RuntimeError(
                f'Failed to decode any frame from {video_path} near '
                f'{timestamp:.4f}s.',
            )

        loaded_ts_arr = np.asarray(loaded_ts, dtype=np.float32)
        nearest_idx = int(np.argmin(np.abs(loaded_ts_arr - timestamp)))
        nearest_ts = float(loaded_ts_arr[nearest_idx])
        if abs(nearest_ts - timestamp) > tolerance_s:
            raise RuntimeError(
                f'Closest frame in {video_path} is too far from query '
                f'({nearest_ts:.4f}s vs {timestamp:.4f}s, '
                f'tol={tolerance_s}).',
            )

        frame = loaded_frames[nearest_idx].permute(1, 2, 0).cpu().numpy()
        return np.ascontiguousarray(frame)


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
                 num_padding_imgs: int = 0,
                 allow_private_stats_fallback: bool = False) -> None:

        # Build image/token transforms (parquet-style sequential list)
        self.transforms = [build_transform_from_cfg(t) for t in transforms]
        self.task_suite_name = task_suite_name
        self.num_padding_imgs = num_padding_imgs
        self.allow_private_stats_fallback = allow_private_stats_fallback
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)
        else:
            self.norm_stats = norm_stats

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Compose transforms chain (parquet-style) starting from raw inputs
        data: Dict[str, Any] = dict(inputs)
        if self.norm_stats is not None:
            stats_key = self.task_suite_name + '_no_noops'
            if stats_key in self.norm_stats:
                norm_stats = self.norm_stats[stats_key]
            elif (self.allow_private_stats_fallback
                  and 'private' in self.norm_stats):
                norm_stats = self.norm_stats['private']
            else:
                raise KeyError(
                    f"Normalization stats key '{stats_key}' not found.")
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
