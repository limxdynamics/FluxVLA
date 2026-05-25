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
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset


def compute_tau(current_frame: int | float, subtask_start: int | float,
                subtask_end: int | float) -> float:
    """Compute normalized progress within one subtask span.

    Args:
        current_frame (int | float): Frame whose progress is requested.
        subtask_start (int | float): Inclusive start frame for the subtask.
        subtask_end (int | float): Inclusive end frame for the subtask.

    Returns:
        float: Progress clipped to ``[0.0, 1.0]``.
    """
    duration = subtask_end - subtask_start
    if duration <= 0:
        return 1.0
    return float(np.clip((current_frame - subtask_start) / duration, 0.0, 1.0))


def find_stage_and_tau(
    current_frame: int,
    episode_length: int,
    subtask_names: Optional[List[str]],
    subtask_start_frames: Optional[List[int]],
    subtask_end_frames: Optional[List[int]],
    global_subtask_names: List[str],
    temporal_proportions: Optional[Dict[str, float]],
    return_combined: bool = False,
) -> Tuple[int, float] | float:
    """Map a frame to a SARM stage index and in-stage progress.

    Args:
        current_frame (int): Frame index relative to the current episode.
        episode_length (int): Number of frames in the episode.
        subtask_names (Optional[List[str]]): Per-episode subtask names.
        subtask_start_frames (Optional[List[int]]): Per-episode subtask start
            frames.
        subtask_end_frames (Optional[List[int]]): Per-episode subtask end
            frames.
        global_subtask_names (List[str]): Dataset-level ordered subtask names.
        temporal_proportions (Optional[Dict[str, float]]): Dataset priors kept
            for API compatibility.
        return_combined (bool): If ``True``, return ``stage + tau`` instead of
            ``(stage, tau)``.

    Returns:
        Tuple[int, float] | float: Stage/tau pair or combined scalar target.
    """
    del temporal_proportions
    stage_idx, tau = 0, 0.0
    num_stages = len(global_subtask_names)

    if num_stages == 1:
        tau = min(1.0, max(0.0, current_frame / max(episode_length - 1, 1)))
    elif (subtask_names is None or subtask_start_frames is None
          or subtask_end_frames is None):
        pass
    elif current_frame < subtask_start_frames[0]:
        pass
    elif current_frame > subtask_end_frames[-1]:
        stage_idx, tau = num_stages - 1, 0.999
    else:
        found = False
        for name, start, end in zip(
                subtask_names,
                subtask_start_frames,
                subtask_end_frames,
                strict=True):
            if start <= current_frame <= end:
                stage_idx = global_subtask_names.index(
                    name) if name in global_subtask_names else 0
                tau = compute_tau(current_frame, start, end)
                found = True
                break
        if not found:
            for idx in range(len(subtask_names) - 1):
                if subtask_end_frames[
                        idx] < current_frame < subtask_start_frames[idx + 1]:
                    name = subtask_names[idx]
                    stage_idx = global_subtask_names.index(
                        name) if name in global_subtask_names else idx
                    tau = 1.0
                    break

    if return_combined:
        if stage_idx >= num_stages - 1 and tau >= 1.0:
            return num_stages - 1 + 0.999
        return stage_idx + tau
    return stage_idx, tau


def compute_absolute_indices(
        frame_idx: int,
        ep_start: int,
        ep_end: int,
        n_obs_steps: int,
        frame_gap: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute observation indices around a center frame.

    Args:
        frame_idx (int): Center frame index in the global dataset.
        ep_start (int): Inclusive global start index for the episode.
        ep_end (int): Exclusive global end index for the episode.
        n_obs_steps (int): Number of neighboring observations, excluding the
            center frame.
        frame_gap (int): Frame stride between neighboring observations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Clamped indices and a mask where
        ``1`` marks indices that were clamped out of bounds.
    """
    half_steps = n_obs_steps // 2
    past_deltas = [-frame_gap * idx for idx in range(half_steps, 0, -1)]
    future_deltas = [frame_gap * idx for idx in range(1, half_steps + 1)]
    delta_indices = past_deltas + [0] + future_deltas

    frames = []
    out_of_bounds = []
    for delta in delta_indices:
        target_idx = frame_idx + delta
        clamped_idx = max(ep_start, min(ep_end - 1, target_idx))
        frames.append(clamped_idx)
        out_of_bounds.append(1 if target_idx != clamped_idx else 0)

    return torch.tensor(frames), torch.tensor(out_of_bounds)


def apply_rewind_augmentation(
    frame_idx: int,
    ep_start: int,
    n_obs_steps: int,
    max_rewind_steps: int,
    frame_gap: int = 30,
    rewind_step: Optional[int] = None,
) -> Tuple[int, List[int]]:
    """Sample extra historical frames for SARM rewind augmentation.

    Args:
        frame_idx (int): Center frame index in the global dataset.
        ep_start (int): Inclusive global start index for the episode.
        n_obs_steps (int): Number of neighboring observations before rewind.
        max_rewind_steps (int): Maximum number of rewind frames to append.
        frame_gap (int): Frame stride between neighboring observations.
        rewind_step (Optional[int]): Optional deterministic rewind length.

    Returns:
        Tuple[int, List[int]]: Selected rewind length and global rewind frame
        indices.
    """
    half_steps = n_obs_steps // 2
    earliest_obs_frame = frame_idx - half_steps * frame_gap
    if earliest_obs_frame <= ep_start:
        return 0, []

    available_history = earliest_obs_frame - ep_start
    max_valid_step = available_history // frame_gap
    max_rewind = min(max_rewind_steps, max(0, max_valid_step))
    if max_rewind <= 0:
        return 0, []

    rewind_step = random.randint(
        1, max_rewind) if rewind_step is None else min(rewind_step, max_rewind)
    if rewind_step == 0:
        return 0, []

    rewind_indices = []
    for idx in range(1, rewind_step + 1):
        rewind_idx = earliest_obs_frame - idx * frame_gap
        rewind_indices.append(max(ep_start, rewind_idx))
    return rewind_step, rewind_indices


def pad_state_to_max_dim(state: torch.Tensor,
                         max_state_dim: int) -> torch.Tensor:
    """Pad or truncate the state feature dimension.

    Args:
        state (torch.Tensor): State tensor with feature dimension last.
        max_state_dim (int): Desired feature dimension.

    Returns:
        torch.Tensor: State tensor whose last dimension is ``max_state_dim``.
    """
    current_dim = state.shape[-1]
    if current_dim >= max_state_dim:
        return state[..., :max_state_dim]
    padding = (0, max_state_dim - current_dim)
    return F.pad(state, padding, mode='constant', value=0)


def temporal_proportions_to_breakpoints(
    temporal_proportions: Dict[str, float] | List[float],
    subtask_names: Optional[List[str]] = None,
) -> List[float]:
    """Convert temporal stage proportions into cumulative breakpoints.

    Args:
        temporal_proportions (Dict[str, float] | List[float]): Stage
            proportions by name or ordered list.
        subtask_names (Optional[List[str]]): Optional explicit order for dict
            inputs.

    Returns:
        List[float]: Breakpoints from ``0.0`` to ``1.0``.
    """
    if isinstance(temporal_proportions, dict):
        if subtask_names is not None:
            proportions = [
                temporal_proportions.get(name, 0.0) for name in subtask_names
            ]
        else:
            proportions = list(temporal_proportions.values())
    else:
        proportions = list(temporal_proportions)

    total = sum(proportions)
    if total > 0 and abs(total - 1.0) > 1e-6:
        proportions = [value / total for value in proportions]

    breakpoints = [0.0]
    cumsum = 0.0
    for value in proportions:
        cumsum += value
        breakpoints.append(cumsum)
    breakpoints[-1] = 1.0
    return breakpoints


def normalize_stage_tau(
    x: float | torch.Tensor,
    num_stages: Optional[int] = None,
    breakpoints: Optional[List[float]] = None,
    temporal_proportions: Dict[str, float] | List[float] | None = None,
    subtask_names: Optional[List[str]] = None,
) -> float | torch.Tensor:
    """Normalize ``stage + tau`` targets with temporal priors.

    Args:
        x (float | torch.Tensor): Encoded SARM value where the integer part is
            the stage index and the fractional part is in-stage progress.
        num_stages (Optional[int]): Number of equally weighted stages.
        breakpoints (Optional[List[float]]): Explicit cumulative
            breakpoints.
        temporal_proportions (Dict[str, float] | List[float] | None): Stage
            duration priors used to derive breakpoints.
        subtask_names (Optional[List[str]]): Optional dict ordering for
            temporal proportions.

    Returns:
        float | torch.Tensor: Progress normalized to ``[0.0, 1.0]``.
    """
    resolved_breakpoints = breakpoints
    if resolved_breakpoints is not None:
        current_breakpoints = cast(List[float], resolved_breakpoints)
        num_stages = len(current_breakpoints) - 1
        resolved_breakpoints = current_breakpoints
    elif temporal_proportions is not None:
        resolved_breakpoints = temporal_proportions_to_breakpoints(
            temporal_proportions, subtask_names)
        num_stages = len(resolved_breakpoints) - 1
    elif num_stages is not None:
        resolved_breakpoints = [
            idx / num_stages for idx in range(num_stages + 1)
        ]
    else:
        raise ValueError(
            'Either num_stages, breakpoints, or temporal_proportions '
            'must be provided')
    assert resolved_breakpoints is not None
    assert num_stages is not None

    if isinstance(x, torch.Tensor):
        result = torch.zeros_like(x)
        for stage_idx in range(num_stages):
            mask = (x >= stage_idx) & (x < stage_idx + 1)
            tau_in_stage = x - stage_idx
            stage_start = resolved_breakpoints[stage_idx]
            stage_width = (
                resolved_breakpoints[stage_idx + 1] -
                resolved_breakpoints[stage_idx])
            result[mask] = stage_start + tau_in_stage[mask] * stage_width
        result[x >= num_stages] = 1.0
        return result.clamp(0.0, 1.0)

    if x < 0:
        return 0.0
    if x >= num_stages:
        return 1.0
    stage = int(x)
    tau = x - stage
    return resolved_breakpoints[stage] + tau * (
        resolved_breakpoints[stage + 1] - resolved_breakpoints[stage])


def _annotation_prefixes(annotation_type: str) -> List[str]:
    """Column-name prefixes used to discover SARM annotations on the
    standard LeRobot episodes metadata (``episodes.jsonl`` /
    ``meta/episodes/*.parquet``).

    Sparse annotations also fall back to the unprefixed column names to stay
    backwards compatible with older sparse-only datasets.
    """
    prefixes = [f'{annotation_type}_']
    if annotation_type == 'sparse':
        prefixes.append('')
    return prefixes


def _extract_annotation_row(record: Dict,
                            annotation_type: str) -> Optional[Dict]:
    for prefix in _annotation_prefixes(annotation_type):
        names = record.get(f'{prefix}subtask_names')
        start_frames = record.get(f'{prefix}subtask_start_frames')
        end_frames = record.get(f'{prefix}subtask_end_frames')
        if names is not None:
            return {
                'subtask_names': names,
                'subtask_start_frames': start_frames,
                'subtask_end_frames': end_frames,
            }
    return None


def load_temporal_proportions(
        meta_root: str | Path,
        annotation_type: str) -> Tuple[List[str], List[float]]:
    """Load dataset-level SARM temporal proportions.

    Args:
        meta_root (str | Path): Dataset ``meta`` directory.
        annotation_type (str): Annotation prefix such as ``sparse`` or
            ``dense``.

    Returns:
        Tuple[List[str], List[float]]: Ordered subtask names and proportions.

    Raises:
        FileNotFoundError: If the expected temporal proportions file is
            missing.
    """
    meta_root = Path(meta_root)
    file_path = meta_root / f'temporal_proportions_{annotation_type}.json'
    if not file_path.exists():
        raise FileNotFoundError(
            f'Temporal proportions file not found: {file_path}')
    with open(file_path, 'r', encoding='utf-8') as handle:
        proportions_dict = json.load(handle)
    names = list(proportions_dict.keys())
    props = [float(proportions_dict[name]) for name in names]
    return names, props


def load_episode_annotations(meta_root: str | Path,
                             annotation_type: str) -> Dict[int, Dict]:
    """Load per-episode SARM annotations from the standard LeRobot episodes
    metadata.

    Supported sources (in priority order):

    * ``meta/episodes.jsonl`` (LeRobotDataset v2.1)
    * ``meta/episodes/**/*.parquet`` (LeRobotDataset v3.x)

    Both forms store annotations as extra columns on the episodes table, for
    example ``sparse_subtask_names`` / ``dense_subtask_start_frames`` /
    ``dense_subtask_end_frames``. This matches how the official LeRobot SARM
    code (``lerobot.policies.sarm``) reads annotations via
    ``dataset_meta.episodes.to_pandas()``.

    Args:
        meta_root (str | Path): Dataset ``meta`` directory.
        annotation_type (str): Annotation prefix such as ``sparse`` or
            ``dense``.

    Returns:
        Dict[int, Dict]: Mapping from episode index to subtask names and frame
        spans.
    """
    meta_root = Path(meta_root)
    episodes_path = meta_root / 'episodes.jsonl'
    if episodes_path.exists():
        records: Dict[int, Dict] = {}
        with open(episodes_path, 'r', encoding='utf-8') as handle:
            for episode_index, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                annotation = _extract_annotation_row(record, annotation_type)
                if annotation is None:
                    continue
                records[episode_index] = annotation
        return records

    episodes_dir = meta_root / 'episodes'
    if not episodes_dir.exists():
        return {}

    parquet_files = sorted(episodes_dir.rglob('*.parquet'))
    if not parquet_files:
        return {}

    datasets = [
        load_dataset('parquet', data_files=str(parquet_file), split='train')
        for parquet_file in parquet_files
    ]
    episodes_dataset = concatenate_datasets(
        [cast(HFDataset, dataset) for dataset in datasets])

    records = {}
    for raw_record in episodes_dataset:
        record = cast(Dict[str, object], raw_record)
        annotation = _extract_annotation_row(record, annotation_type)
        if annotation is None:
            continue
        episode_index = int(cast(int | float | str, record['episode_index']))
        records[episode_index] = annotation
    return records
