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

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchvision

from fluxvla.datasets.parquet_dataset import ParquetDataset
from fluxvla.datasets.utils.sarm_utils import (apply_rewind_augmentation,
                                               compute_absolute_indices,
                                               find_stage_and_tau,
                                               load_episode_annotations,
                                               load_temporal_proportions)
from fluxvla.engines import DATASETS


@DATASETS.register_module()
class SARMDataset(ParquetDataset):
    """SARM dataset built on top of :class:`ParquetDataset`.

    It reuses the base class for loading dataset metadata and the concatenated
    Hugging Face parquet dataset, and layers SARM-specific logic on top:

    * multi-frame observation sequence construction from ``frame_gap``
    * video decoding via ``torchvision`` from ``videos/<key>/...mp4``
    * sparse/dense stage-aware target computation
    * optional rewind augmentation at training time
    """

    def __init__(self,
                 data_root_path: Union[str, List[str]],
                 video_keys: List[str],
                 transforms: List[Dict],
                 annotation_mode: str = 'single_stage',
                 n_obs_steps: int = 8,
                 frame_gap: int = 30,
                 max_rewind_steps: int = 4,
                 rewind_probability: float = 0.8,
                 state_key: str = 'observation.state',
                 training: bool = True) -> None:
        super().__init__(
            data_root_path=data_root_path,
            transforms=transforms,
            action_window_size=1,
            action_key=state_key,
            use_delta=False,
            statistic_name='sarm',
            window_start_idx=0,
        )
        self.video_keys = video_keys
        self.annotation_mode = annotation_mode
        self.n_obs_steps = n_obs_steps
        self.frame_gap = frame_gap
        self.max_rewind_steps = max_rewind_steps
        self.rewind_probability = rewind_probability
        self.state_key = state_key
        self.training = training
        # At inference we emit exactly one sample per frame (n_obs_steps + 1
        # obs frames, no rewind tail) to stay aligned with how gr00t-style
        # ParquetDataset iterates over a dataset at inference time.
        rewind_reserve = max_rewind_steps if training else 0
        self.total_frames = 1 + n_obs_steps + rewind_reserve

        # ParquetDataset base class already loaded ``self.info``,
        # ``self.tasks`` (as list[task_index -> task text]),
        # ``self.episodes`` (flat), ``self.dataset`` (HF dataset concatenated
        # across data roots), ``self.dataset_cumulative_sizes`` and
        # ``self.transforms``. Convert / re-index them into the forms SARM
        # expects and prepare SARM-specific annotation caches.
        self.meta_roots = [
            os.path.join(path, 'meta') for path in self.data_root_path
        ]
        self.episodes_meta: List[Dict[int, Dict]] = []
        for dataset_episodes in self.episodes_by_dataset:
            records: Dict[int, Dict] = {}
            for fallback_episode_index, record in enumerate(dataset_episodes):
                episode_index = int(
                    record.get('episode_index', fallback_episode_index))
                records[episode_index] = record
            self.episodes_meta.append(records)

        self.episode_ranges = {}
        episode_ids = np.asarray(self.dataset['episode_index'])
        previous_key = None
        episode_start = 0
        for index, episode_id in enumerate(episode_ids):
            dataset_idx = self._get_dataset_index(index)
            current_key = (dataset_idx, int(episode_id))
            if previous_key is None:
                previous_key = current_key
                episode_start = index
                continue
            if current_key != previous_key:
                self.episode_ranges[previous_key] = (episode_start, index)
                previous_key = current_key
                episode_start = index
        if previous_key is not None:
            self.episode_ranges[previous_key] = (episode_start,
                                                 len(self.dataset))

        self.sparse_annotations = {}
        self.dense_annotations = {}
        self.sparse_temporal_meta = {}
        self.dense_temporal_meta = {}
        for dataset_idx, meta_root in enumerate(self.meta_roots):
            if annotation_mode == 'single_stage':
                self.sparse_temporal_meta[dataset_idx] = (['task'], {
                    'task': 1.0
                })
                continue
            if annotation_mode == 'dual':
                sparse_names, sparse_props = load_temporal_proportions(
                    meta_root, 'sparse')
                self.sparse_temporal_meta[dataset_idx] = (sparse_names, {
                    name: prop
                    for name, prop in zip(sparse_names, sparse_props)
                })
                self.sparse_annotations[
                    dataset_idx] = load_episode_annotations(
                        meta_root, 'sparse')
            else:
                self.sparse_temporal_meta[dataset_idx] = (['task'], {
                    'task': 1.0
                })
            dense_names, dense_props = load_temporal_proportions(
                meta_root, 'dense')
            self.dense_temporal_meta[dataset_idx] = (dense_names, {
                name: prop
                for name, prop in zip(dense_names, dense_props)
            })
            self.dense_annotations[dataset_idx] = load_episode_annotations(
                meta_root, 'dense')

    def _decode_video_frames_torchvision(
            self,
            video_path: Union[Path, str],
            timestamps: List[float],
            tolerance_s: float = 0.1,
            backend: str = 'pyav') -> torch.Tensor:
        video_path = str(video_path)
        first_ts = timestamps[0]
        last_ts = timestamps[-1]

        def _load_candidates(seek_ts: float) -> tuple[List[torch.Tensor],
                                                      List[float]]:
            keyframes_only = backend == 'pyav'
            torchvision.set_video_backend(backend)
            reader = torchvision.io.VideoReader(video_path, 'video')
            reader.seek(seek_ts, keyframes_only=keyframes_only)

            loaded_frames: List[torch.Tensor] = []
            loaded_ts: List[float] = []
            try:
                for frame in reader:
                    current_ts = float(frame['pts'])
                    loaded_frames.append(frame['data'])
                    loaded_ts.append(current_ts)
                    if current_ts >= last_ts:
                        break
            finally:
                if backend == 'pyav':
                    reader.container.close()
            return loaded_frames, loaded_ts

        def _match_candidates(loaded_frames: List[torch.Tensor],
                              loaded_ts: List[float]) -> Optional[torch.Tensor]:
            if not loaded_ts:
                return None
            query_ts = torch.tensor(timestamps, dtype=torch.float32)
            loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
            dist = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
            min_dist, argmin = dist.min(1)
            if not (min_dist <= tolerance_s).all():
                return None
            return torch.stack([loaded_frames[idx] for idx in argmin])

        loaded_frames, loaded_ts = _load_candidates(first_ts)
        matched_frames = _match_candidates(loaded_frames, loaded_ts)
        if matched_frames is None and backend == 'pyav' and first_ts > 0:
            # torchvision warns that accurate seek is not implemented for pyav.
            loaded_frames, loaded_ts = _load_candidates(0.0)
            matched_frames = _match_candidates(loaded_frames, loaded_ts)
        if matched_frames is None:
            raise ValueError(
                f'Failed to find frames within tolerance for {video_path}')
        return matched_frames

    def _build_video_path(self, dataset_idx: int, episode_index: int,
                          video_key: str) -> str:
        info = self.info[dataset_idx]
        episode_meta = self.episodes_meta[dataset_idx][episode_index]
        video_root_path = info['video_path']
        format_kwargs = {
            'video_key': video_key,
            'episode_index': episode_index,
        }

        def _first_available_int(keys: List[str]) -> Optional[int]:
            for key in keys:
                if key in episode_meta:
                    return int(episode_meta[key])
            return None

        chunk_index = _first_available_int([
            f'videos/{video_key}/chunk_index',
            'meta/episodes/chunk_index',
            'data/chunk_index',
            'chunk_index',
        ])
        file_index = _first_available_int([
            f'videos/{video_key}/file_index',
            'meta/episodes/file_index',
            'data/file_index',
            'file_index',
        ])
        if chunk_index is not None:
            format_kwargs['chunk_index'] = chunk_index
        if file_index is not None:
            format_kwargs['file_index'] = file_index
        if 'chunks_size' in info:
            format_kwargs['episode_chunk'] = episode_index // int(
                info['chunks_size'])
        return os.path.join(self.data_root_path[dataset_idx],
                            video_root_path.format(**format_kwargs))

    def _get_annotation(self, dataset_idx: int, episode_index: int,
                        annotation_type: str) -> Optional[Dict]:
        if annotation_type == 'sparse':
            return self.sparse_annotations.get(dataset_idx,
                                               {}).get(episode_index)
        return self.dense_annotations.get(dataset_idx, {}).get(episode_index)

    def _compute_targets(self, dataset_idx: int, episode_index: int,
                         episode_length: int, valid_indices: List[int],
                         annotation_type: str) -> np.ndarray:
        targets = np.zeros(self.total_frames, dtype=np.float32)
        if annotation_type == 'sparse':
            global_names, temporal_props = self.sparse_temporal_meta[
                dataset_idx]
            if self.annotation_mode in ['single_stage', 'dense_only']:
                annotation = None
            else:
                annotation = self._get_annotation(dataset_idx, episode_index,
                                                  'sparse')
        else:
            global_names, temporal_props = self.dense_temporal_meta[
                dataset_idx]
            annotation = self._get_annotation(dataset_idx, episode_index,
                                              'dense')

        subtask_names = None if annotation is None else annotation.get(
            'subtask_names')
        subtask_start_frames = None if annotation is None else annotation.get(
            'subtask_start_frames')
        subtask_end_frames = None if annotation is None else annotation.get(
            'subtask_end_frames')

        for target_idx, rel_frame in enumerate(valid_indices):
            targets[target_idx] = find_stage_and_tau(
                rel_frame,
                episode_length,
                subtask_names,
                subtask_start_frames,
                subtask_end_frames,
                global_names,
                temporal_props,
                return_combined=True,
            )
        return targets

    def __getitem__(self, index: int, dataset_statistics=None) -> Dict:
        del dataset_statistics
        data = self.dataset[index]
        dataset_idx = self._get_dataset_index(index)
        episode_index = int(data['episode_index'])
        episode_start, episode_end = self.episode_ranges[(dataset_idx,
                                                          episode_index)]
        episode_length = episode_end - episode_start
        obs_indices, _ = compute_absolute_indices(
            index,
            episode_start,
            episode_end,
            self.n_obs_steps,
            frame_gap=self.frame_gap)
        obs_indices = obs_indices.tolist()

        rewind_step = 0
        rewind_indices = []
        if self.training and self.max_rewind_steps > 0 and np.random.random(
        ) < self.rewind_probability:
            rewind_step, rewind_indices = apply_rewind_augmentation(
                index,
                episode_start,
                self.n_obs_steps,
                self.max_rewind_steps,
                frame_gap=self.frame_gap,
            )

        sequence_indices = obs_indices + rewind_indices
        valid_length = len(sequence_indices)
        while len(sequence_indices) < self.total_frames:
            sequence_indices.append(sequence_indices[-1])

        sequence_rows = [self.dataset[seq_idx] for seq_idx in sequence_indices]
        timestamps = [float(row['timestamp']) for row in sequence_rows]
        states = np.stack([
            np.asarray(row[self.state_key], dtype=np.float32)
            for row in sequence_rows
        ],
                          axis=0)

        images_per_camera = []
        for video_key in self.video_keys:
            video_path = self._build_video_path(dataset_idx, episode_index,
                                                video_key)
            frames = self._decode_video_frames_torchvision(
                video_path, timestamps)
            images_per_camera.append(frames.numpy())
        images = np.stack(images_per_camera, axis=1)

        relative_indices = [
            seq_idx - episode_start
            for seq_idx in sequence_indices[:valid_length]
        ]
        output = {
            'images': images,
            'states': states,
            'task_description':
            self._resolve_task_description(dataset_idx, data),
            'lengths': np.asarray(valid_length, dtype=np.int64),
            'episode_index': np.asarray(episode_index, dtype=np.int64),
            'current_index': np.asarray(index, dtype=np.int64),
        }
        output['sparse_targets'] = self._compute_targets(
            dataset_idx, episode_index, episode_length, relative_indices,
            'sparse')
        if self.annotation_mode in ['dense_only', 'dual']:
            output['dense_targets'] = self._compute_targets(
                dataset_idx, episode_index, episode_length, relative_indices,
                'dense')

        for transform in self.transforms:
            output = transform(output)
        return output
