#!/usr/bin/env python
# flake8: noqa
# isort: skip_file
# yapf: disable
"""Parse per-episode sparse annotation info from a LeRobot dataset."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _load_episodes_df(episodes_dir: Path) -> pd.DataFrame:
    parquet_files = sorted(episodes_dir.glob('*/*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(
            f'No episodes parquet files found in {episodes_dir}')

    dfs = []
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        dfs.append(table.to_pandas())
    return pd.concat(dfs, ignore_index=True)


def _normalize_list(value: Any) -> list | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, (np.generic, )):
        return [value.item()]
    return [value]


def _compute_temporal_proportions(
        names: list | None, starts: list | None,
        ends: list | None) -> dict[str, float] | None:
    if not names or not starts or not ends:
        return None

    if len(names) != len(starts) or len(names) != len(ends):
        return None

    durations_by_name: dict[str, float] = {}
    ordered_names: list[str] = []
    total_duration = 0.0

    for name, start, end in zip(names, starts, ends, strict=True):
        start_val = float(start)
        end_val = float(end)
        duration = max(0.0, end_val - start_val)
        name_str = str(name)
        if name_str not in durations_by_name:
            durations_by_name[name_str] = 0.0
            ordered_names.append(name_str)
        durations_by_name[name_str] += duration
        total_duration += duration

    if total_duration <= 0:
        return None

    return {
        name: durations_by_name[name] / total_duration
        for name in ordered_names
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Parse sparse subtask info per episode.')
    parser.add_argument(
        '--dataset-root',
        type=Path,
        required=True,
        help=
        'Dataset root directory containing meta/episodes and meta/temporal_proportions_sparse.json',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Optional output JSON path. If not set, writes to dataset root.',
    )
    args = parser.parse_args()

    episodes_dir = args.dataset_root / 'meta' / 'episodes'
    episodes_df = _load_episodes_df(episodes_dir)

    records = []
    for _, row in episodes_df.iterrows():
        sparse_names = _normalize_list(row.get('sparse_subtask_names'))
        sparse_starts = _normalize_list(row.get('sparse_subtask_start_times'))
        sparse_ends = _normalize_list(row.get('sparse_subtask_end_times'))
        num_sparse_stages = len(sparse_names) if sparse_names else None
        temporal_props = _compute_temporal_proportions(sparse_names,
                                                       sparse_starts,
                                                       sparse_ends)

        ep_index = row.get('episode_index')
        if ep_index is None or (isinstance(ep_index, float)
                                and pd.isna(ep_index)):
            ep_index_value = None
        else:
            ep_index_value = int(ep_index)

        record = {
            'episode_index': ep_index_value,
            'num_sparse_stages': num_sparse_stages,
            'sparse_subtask_names': sparse_names,
            'sparse_temporal_proportions': temporal_props,
        }
        records.append(record)

    output_path = args.output or (args.dataset_root /
                                  'sparse_episode_info.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
