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
"""Compute SARM progress parquet files for RA-BC training."""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from infer_sarm_progress import (_as_float, _as_int,
                                 _register_sarm_runtime_modules,
                                 _resolve_head_modes)
from mmengine import Config, DictAction
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

ProgressRows = DefaultDict[Tuple[int, int], List[Dict[str, Any]]]


def parse_args():
    """Parse command-line arguments for SARM RA-BC progress computation."""
    parser = argparse.ArgumentParser(
        description='Compute SARM progress parquet files for RA-BC.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument(
        '--output-path',
        type=str,
        default='./sarm_progress.parquet',
        help='Output parquet path. Defaults to ./sarm_progress.parquet.')
    parser.add_argument(
        '--head-mode',
        type=str,
        default='sparse',
        choices=['sparse', 'dense', 'both'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Compute every Nth frame and interpolate missing frames.')
    parser.add_argument(
        '--frame-index',
        type=int,
        default=None,
        help='Sequence frame to score. Defaults to the SARM center frame.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config key-value pairs in xxx=yyy format')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be >= 1')
    if args.batch_size < 1:
        parser.error('--batch-size must be >= 1')
    return args


def _episode_ranges(dataset: Dataset) -> List[Tuple[int, int, int, int]]:
    ranges = getattr(dataset, 'episode_ranges', None)
    if not ranges:
        return [(0, 0, 0, len(dataset))]
    return [
        (int(dataset_idx), int(episode_idx), int(start), int(end))
        for (dataset_idx, episode_idx), (
            start, end) in sorted(ranges.items(), key=lambda item: item[1][0])
    ]


def _indices_for_stride(dataset: Dataset, stride: int) -> List[int]:
    if stride == 1:
        return list(range(len(dataset)))

    indices = []
    for _, _, start, end in _episode_ranges(dataset):
        episode_indices = list(range(start, end, stride))
        if end > start and (end - 1) not in episode_indices:
            episode_indices.append(end - 1)
        indices.extend(episode_indices)
    return sorted(set(indices))


def _dataset_index(dataset: Dataset, global_index: int) -> int:
    getter = getattr(dataset, '_get_dataset_index', None)
    if callable(getter):
        return int(getter(global_index))
    return 0


def _frame_index_by_global_index(dataset: Dataset) -> Dict[int, int]:
    frame_indices = {}
    for _, _, start, end in _episode_ranges(dataset):
        for global_index in range(start, end):
            frame_indices[global_index] = global_index - start
    return frame_indices


def _interpolate_episode(records: List[Dict[str, Any]],
                         episode_indices: Iterable[int],
                         head_modes: List[str]) -> List[Dict[str, Any]]:
    if not records:
        return []

    records_by_index = {int(record['index']): record for record in records}
    computed_indices = np.asarray(
        sorted(records_by_index.keys()), dtype=np.int64)
    episode_indices = list(episode_indices)
    episode_start = min(episode_indices)
    output_rows = []

    for global_index in episode_indices:
        if global_index in records_by_index:
            output_rows.append(dict(records_by_index[global_index]))
            continue

        row = dict(records[0])
        row['index'] = int(global_index)
        row['frame_index'] = int(global_index - episode_start)
        for head_mode in head_modes:
            progress_key = f'progress_{head_mode}'
            values = np.asarray([
                records_by_index[idx].get(progress_key, np.nan)
                for idx in computed_indices
            ],
                                dtype=np.float32)
            mask = np.isfinite(values)
            if mask.sum() == 0:
                row[progress_key] = np.nan
            elif mask.sum() == 1:
                row[progress_key] = float(values[mask][0])
            else:
                row[progress_key] = float(
                    np.interp(global_index, computed_indices[mask],
                              values[mask]))
        output_rows.append(row)
    return output_rows


def _build_output_rows(dataset: Dataset, computed_rows: ProgressRows,
                       head_modes: List[str],
                       stride: int) -> List[Dict[str, Any]]:
    rows = []
    for dataset_index, episode_index, start, end in _episode_ranges(dataset):
        episode_rows = computed_rows.get((dataset_index, episode_index), [])
        if stride == 1:
            rows.extend(episode_rows)
            continue
        rows.extend(
            _interpolate_episode(episode_rows, range(start, end), head_modes))
    return sorted(rows, key=lambda row: row['index'])


def _write_progress_parquet(output_path: str, rows: List[Dict[str, Any]],
                            reward_model_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No SARM progress rows were produced.')

    columns: Dict[str, List[Any]] = {
        key: [row.get(key) for row in rows]
        for key in rows[0].keys()
    }
    table = pa.Table.from_pydict(columns)
    metadata = dict(table.schema.metadata or {})
    metadata[b'reward_model_path'] = str(reward_model_path).encode()
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


def main():
    """Run SARM inference and write an RA-BC progress parquet."""
    args = parse_args()
    builders = _register_sarm_runtime_modules()
    build_collator_from_cfg = builders['build_collator_from_cfg']
    build_dataset_from_cfg = builders['build_dataset_from_cfg']
    build_vla_from_cfg = builders['build_vla_from_cfg']

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset_cfg = cfg.get('inference_dataset', cfg.train_dataloader.dataset)
    dataset_cfg = dataset_cfg.copy()
    dataset_cfg['training'] = False
    dataset = cast(Dataset, build_dataset_from_cfg(dataset_cfg))
    compute_indices = _indices_for_stride(dataset, args.stride)
    compute_dataset = Subset(dataset, compute_indices)

    collator = build_collator_from_cfg(cfg.runner.collator.copy())
    model = cast(Any, build_vla_from_cfg(cfg.model))
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(
        checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device(args.device)).eval()

    head_modes = _resolve_head_modes(model, args.head_mode)
    frame_index = args.frame_index
    if frame_index is None:
        frame_index = int(getattr(model, 'n_obs_steps', 0)) // 2
    frame_indices = _frame_index_by_global_index(dataset)

    dataloader = DataLoader(
        compute_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator)

    computed_rows: ProgressRows = defaultdict(list)
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='SARM RA-BC progress'):
            progress_by_mode = {}
            for head_mode in head_modes:
                progress_by_mode[head_mode] = model.predict_progress(
                    images=batch['images'],
                    text_input_ids=batch['text_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    states=batch['states'],
                    lengths=batch['lengths'],
                    head_mode=head_mode,
                    frame_index=frame_index,
                ).detach().cpu()

            batch_size = len(batch['episode_index'])
            for idx in range(batch_size):
                global_index = _as_int(batch['current_index'][idx])
                episode_index = _as_int(batch['episode_index'][idx])
                dataset_index = _dataset_index(dataset, global_index)
                row = {
                    'index': global_index,
                    'dataset_index': dataset_index,
                    'episode_index': episode_index,
                    'frame_index': frame_indices.get(global_index,
                                                     global_index),
                }
                for head_mode in head_modes:
                    row[f'progress_{head_mode}'] = _as_float(
                        progress_by_mode[head_mode][idx])
                computed_rows[(dataset_index, episode_index)].append(row)

    rows = _build_output_rows(dataset, computed_rows, head_modes, args.stride)
    _write_progress_parquet(args.output_path, rows, args.ckpt_path)


if __name__ == '__main__':
    main()
