#!/usr/bin/env python
# flake8: noqa
# isort: skip_file
# yapf: disable

# Copyright 2025 FluxVLA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Write manual SARM subtask annotations into a standard LeRobot dataset.

Supports both LeRobot v2.1 (``meta/episodes.jsonl``) and v3.x
(``meta/episodes/*/*.parquet``). Reads a per-episode annotation *spec*
(JSON / JSONL) and writes the following standard columns back onto the
episodes metadata:

- ``sparse_subtask_names`` / ``sparse_subtask_start_frames`` / ``sparse_subtask_end_frames``
- ``dense_subtask_names``  / ``dense_subtask_start_frames``  / ``dense_subtask_end_frames``
- (optional) ``*_start_times`` / ``*_end_times`` if the spec provides them

It also (re)computes
``meta/temporal_proportions_{sparse,dense}.json`` from the written spans so
downstream SARM training picks up consistent class priors — exactly the file
FluxVLA's ``sarm_utils.load_temporal_proportions`` expects.

This is the manual companion to
:mod:`tools.sarm_annotate.subtask_annotation` (the VLM-based pipeline): the
resulting columns are byte-for-byte compatible, so the same dataset can be
consumed by :mod:`fluxvla.datasets.sarm_dataset` and by the upstream
``lerobot.policies.sarm`` implementation.

----------------------------------------------------------------------------
Spec format
----------------------------------------------------------------------------

The spec is a JSON file (``.json`` list *or* ``.jsonl``) with one entry per
episode. Each entry must have ``episode_index`` and at least one of the
``sparse`` / ``dense`` blocks. All frame indices are **inclusive** and
0-based; times (if provided) are in seconds.

Minimal single-stage example (covers the whole episode automatically — no
frame bookkeeping required):

    [
      {"episode_index": 0, "sparse": "auto"},
      {"episode_index": 1, "sparse": "auto"}
    ]

Full multi-stage example:

    [
      {
        "episode_index": 0,
        "sparse": {
          "names":        ["reach", "grasp", "place"],
          "start_frames": [0,        60,      150],
          "end_frames":   [59,      149,      199]
        },
        "dense": {
          "names":        ["move_to_cup", "close_gripper", "lift", "move_to_plate", "lower", "open_gripper"],
          "start_frames": [0,   40,  60, 100, 150, 185],
          "end_frames":   [39,  59,  99, 149, 184, 199],
          "start_times":  [0.0, 0.8, 1.2, 2.0, 3.0, 3.7],
          "end_times":    [0.78, 1.18, 1.98, 2.98, 3.68, 3.98]
        }
      }
    ]

Rules:

- ``"sparse": "auto"`` / ``"dense": "auto"`` creates a single ``"task"`` stage
  spanning ``[0, length - 1]`` for that episode (and uses the dataset's
  ``length`` / ``dataset_from_index``-``dataset_to_index`` columns to infer
  length). Useful for bootstrapping ``single_stage`` / ``dense_only`` SARM
  training without writing any frame bookkeeping by hand.
- ``--default-sparse auto`` / ``--default-dense auto`` applies the same
  behaviour to every episode *not* explicitly listed in the spec (pass an
  empty spec file ``[]`` to annotate a whole dataset with a single ``"task"``
  stage in one line).

----------------------------------------------------------------------------
Usage
----------------------------------------------------------------------------

    # Single-stage "task" on every episode (no spec needed)
    python tools/sarm_annotate/write_manual_stages.py \\
        --dataset-root /path/to/dataset \\
        --default-sparse auto

    # Manual multi-stage from spec
    python tools/sarm_annotate/write_manual_stages.py \\
        --dataset-root /path/to/dataset \\
        --spec my_stages.json

    # Dense-only, auto sparse fallback
    python tools/sarm_annotate/write_manual_stages.py \\
        --dataset-root /path/to/dataset \\
        --spec my_dense_stages.json \\
        --default-sparse auto
"""

from __future__ import annotations
import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq

EPISODES_JSONL = 'episodes.jsonl'
EPISODES_DIR = 'episodes'
SPARSE_PROPS = 'temporal_proportions_sparse.json'
DENSE_PROPS = 'temporal_proportions_dense.json'

StagesBlock = Dict[str, Optional[List[Any]]]

# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------


def _load_spec(spec_path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    """Load ``[{episode_index, sparse?, dense?}, ...]`` into a dict keyed by
    episode_index. Accepts both ``.json`` (list) and ``.jsonl`` (one object
    per line)."""
    if spec_path is None:
        return {}
    if not spec_path.exists():
        raise FileNotFoundError(f'Spec file not found: {spec_path}')

    text = spec_path.read_text(encoding='utf-8').strip()
    if not text:
        return {}

    entries: List[Dict[str, Any]]
    if spec_path.suffix == '.jsonl':
        entries = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
    else:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = [parsed]
        entries = parsed

    out: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        if 'episode_index' not in entry:
            raise ValueError(f"Spec entry missing 'episode_index': {entry}")
        out[int(entry['episode_index'])] = entry
    return out


def _resolve_block(
    value: Any,
    episode_length: int,
) -> Optional[StagesBlock]:
    """Normalise a user-provided block into a canonical stage block.

    ``value`` can be:
      * ``None`` / missing  -> returns ``None`` (column left untouched)
      * ``"auto"``          -> single ``"task"`` stage spanning the episode
      * dict with ``names`` / ``start_frames`` / ``end_frames`` (+ optional
        ``start_times`` / ``end_times``) -> validated and returned as-is.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value != 'auto':
            raise ValueError(f'Unsupported stage shorthand: {value!r}')
        if episode_length <= 0:
            return None
        return {
            'names': ['task'],
            'start_frames': [0],
            'end_frames': [int(episode_length) - 1],
            'start_times': None,
            'end_times': None,
        }

    if not isinstance(value, dict):
        raise ValueError(
            f'Unsupported stage block type: {type(value).__name__}')

    names = value.get('names')
    starts = value.get('start_frames')
    ends = value.get('end_frames')
    if names is None or starts is None or ends is None:
        raise ValueError(
            "Stage block must contain 'names', 'start_frames', 'end_frames'")
    if not (len(names) == len(starts) == len(ends)):
        raise ValueError(f'Stage block lengths disagree: names={len(names)}, '
                         f'starts={len(starts)}, ends={len(ends)}')

    return {
        'names': [str(n) for n in names],
        'start_frames': [int(s) for s in starts],
        'end_frames': [int(e) for e in ends],
        'start_times': ([float(t) for t in value['start_times']]
                        if value.get('start_times') is not None else None),
        'end_times': ([float(t) for t in value['end_times']]
                      if value.get('end_times') is not None else None),
    }


# ---------------------------------------------------------------------------
# Temporal proportions
# ---------------------------------------------------------------------------


def _accumulate_proportions(
    accumulator: 'OrderedDict[str, float]',
    block: Optional[StagesBlock],
) -> float:
    """Fold one episode's block into the running name -> total-duration map.
    Returns the duration contributed by this block."""
    if block is None:
        return 0.0
    added = 0.0
    for name, start, end in zip(block['names'], block['start_frames'],
                                block['end_frames']):
        duration = max(0, int(end) - int(start) + 1)
        if duration == 0:
            continue
        accumulator[name] = accumulator.get(name, 0.0) + duration
        added += duration
    return added


def _write_proportions(meta_dir: Path, kind: str,
                       accumulator: 'OrderedDict[str, float]') -> None:
    total = sum(accumulator.values())
    if total <= 0:
        return
    out = {name: dur / total for name, dur in accumulator.items()}
    target = meta_dir / (SPARSE_PROPS if kind == 'sparse' else DENSE_PROPS)
    target.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {target}')


# ---------------------------------------------------------------------------
# v2.1 writer (meta/episodes.jsonl)
# ---------------------------------------------------------------------------


def _episode_length_v21(record: Dict[str, Any]) -> int:
    if 'length' in record and record['length'] is not None:
        return int(record['length'])
    if 'dataset_from_index' in record and 'dataset_to_index' in record:
        return int(record['dataset_to_index']) - int(
            record['dataset_from_index'])
    raise ValueError(
        f"Episode record has no 'length' nor 'dataset_{{from,to}}_index': "
        f"{record.get('episode_index')}")


def _apply_block_v21(record: Dict[str, Any], kind: str,
                     block: Optional[StagesBlock]) -> None:
    if block is None:
        return
    record[f'{kind}_subtask_names'] = block['names']
    record[f'{kind}_subtask_start_frames'] = block['start_frames']
    record[f'{kind}_subtask_end_frames'] = block['end_frames']
    if block.get('start_times') is not None:
        record[f'{kind}_subtask_start_times'] = block['start_times']
    if block.get('end_times') is not None:
        record[f'{kind}_subtask_end_times'] = block['end_times']


def _rewrite_jsonl(
    jsonl_path: Path,
    spec: Dict[int, Dict[str, Any]],
    default_sparse: Any,
    default_dense: Any,
) -> Tuple['OrderedDict[str, float]', 'OrderedDict[str, float]', int]:
    sparse_acc: 'OrderedDict[str, float]' = OrderedDict()
    dense_acc: 'OrderedDict[str, float]' = OrderedDict()
    updated = 0

    records: List[Dict[str, Any]] = []
    with jsonl_path.open('r', encoding='utf-8') as handle:
        for fallback_idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            ep_idx = int(record.get('episode_index', fallback_idx))
            record.setdefault('episode_index', ep_idx)
            records.append(record)

    for record in records:
        ep_idx = int(record['episode_index'])
        length = _episode_length_v21(record)
        entry = spec.get(ep_idx, {})

        sparse_block = _resolve_block(
            entry.get('sparse', default_sparse), length)
        dense_block = _resolve_block(entry.get('dense', default_dense), length)

        if sparse_block is not None or dense_block is not None:
            updated += 1
        _apply_block_v21(record, 'sparse', sparse_block)
        _apply_block_v21(record, 'dense', dense_block)
        _accumulate_proportions(sparse_acc, sparse_block)
        _accumulate_proportions(dense_acc, dense_block)

    tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    tmp_path.replace(jsonl_path)
    print(f'Rewrote {jsonl_path} ({updated} episodes updated)')
    return sparse_acc, dense_acc, updated


# ---------------------------------------------------------------------------
# v3.x writer (meta/episodes/*/*.parquet)
# ---------------------------------------------------------------------------


def _episode_length_v3(row: pd.Series) -> int:
    if 'length' in row and pd.notna(row['length']):
        return int(row['length'])
    if {'dataset_from_index', 'dataset_to_index'}.issubset(row.index):
        return int(row['dataset_to_index']) - int(row['dataset_from_index'])
    raise ValueError(
        f"Episode row has no 'length' nor 'dataset_{{from,to}}_index': "
        f"{row.get('episode_index')}")


def _apply_block_v3(df: pd.DataFrame, col_prefix: str,
                    blocks: List[Optional[StagesBlock]]) -> None:
    name_col = f'{col_prefix}_subtask_names'
    start_col = f'{col_prefix}_subtask_start_frames'
    end_col = f'{col_prefix}_subtask_end_frames'
    start_t_col = f'{col_prefix}_subtask_start_times'
    end_t_col = f'{col_prefix}_subtask_end_times'

    df[name_col] = [b['names'] if b else None for b in blocks]
    df[start_col] = [b['start_frames'] if b else None for b in blocks]
    df[end_col] = [b['end_frames'] if b else None for b in blocks]
    if any(b and b.get('start_times') is not None for b in blocks):
        df[start_t_col] = [
            b['start_times'] if
            (b and b.get('start_times') is not None) else None for b in blocks
        ]
    if any(b and b.get('end_times') is not None for b in blocks):
        df[end_t_col] = [
            b['end_times'] if (b and b.get('end_times') is not None) else None
            for b in blocks
        ]


def _rewrite_parquet(
    parquet_files: List[Path],
    spec: Dict[int, Dict[str, Any]],
    default_sparse: Any,
    default_dense: Any,
) -> Tuple['OrderedDict[str, float]', 'OrderedDict[str, float]', int]:
    sparse_acc: 'OrderedDict[str, float]' = OrderedDict()
    dense_acc: 'OrderedDict[str, float]' = OrderedDict()
    total_updated = 0

    for path in parquet_files:
        df = pq.read_table(path).to_pandas()
        sparse_blocks: List[Optional[StagesBlock]] = []
        dense_blocks: List[Optional[StagesBlock]] = []
        updated_here = 0

        for _, row in df.iterrows():
            ep_idx = int(row['episode_index'])
            length = _episode_length_v3(row)
            entry = spec.get(ep_idx, {})
            sparse_block = _resolve_block(
                entry.get('sparse', default_sparse), length)
            dense_block = _resolve_block(
                entry.get('dense', default_dense), length)
            sparse_blocks.append(sparse_block)
            dense_blocks.append(dense_block)
            if sparse_block is not None or dense_block is not None:
                updated_here += 1
            _accumulate_proportions(sparse_acc, sparse_block)
            _accumulate_proportions(dense_acc, dense_block)

        _apply_block_v3(df, 'sparse', sparse_blocks)
        _apply_block_v3(df, 'dense', dense_blocks)
        tmp_path = path.with_suffix('.tmp.parquet')
        df.to_parquet(
            tmp_path, index=False, engine='pyarrow', compression='snappy')
        tmp_path.replace(path)
        print(f'Rewrote {path} ({updated_here} episodes updated)')
        total_updated += updated_here

    return sparse_acc, dense_acc, total_updated


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Manually write SARM sparse/dense subtask annotations into a '
            'standard LeRobot dataset (v2.1 or v3.x).'))
    parser.add_argument(
        '--dataset-root',
        type=Path,
        required=True,
        help="Dataset root containing a 'meta/' folder.")
    parser.add_argument(
        '--spec',
        type=Path,
        default=None,
        help='Per-episode stage spec (.json list or .jsonl).')
    parser.add_argument(
        '--default-sparse',
        default=None,
        help='Fallback for episodes missing from spec. '
        "Use 'auto' for a single 'task' stage.")
    parser.add_argument(
        '--default-dense',
        default=None,
        help='Fallback for episodes missing from spec. '
        "Use 'auto' for a single 'task' stage.")
    parser.add_argument(
        '--skip-proportions',
        action='store_true',
        help='Do not (re)write meta/temporal_proportions_*.json.')
    args = parser.parse_args()

    meta_dir = args.dataset_root / 'meta'
    if not meta_dir.exists():
        raise FileNotFoundError(f'meta/ not found under {args.dataset_root}')

    spec = _load_spec(args.spec)

    if spec:
        print(f'Loaded {len(spec)} spec entries from {args.spec}')
    if args.default_sparse:
        print(f'Default sparse fallback: {args.default_sparse!r}')
    if args.default_dense:
        print(f'Default dense fallback: {args.default_dense!r}')

    jsonl_path = meta_dir / EPISODES_JSONL
    episodes_dir = meta_dir / EPISODES_DIR

    if jsonl_path.exists():
        sparse_acc, dense_acc, updated = _rewrite_jsonl(
            jsonl_path, spec, args.default_sparse, args.default_dense)
    elif episodes_dir.exists():
        parquet_files = sorted(episodes_dir.glob('*/*.parquet'))
        if not parquet_files:
            raise FileNotFoundError(
                f'No episodes parquet under {episodes_dir}')
        sparse_acc, dense_acc, updated = _rewrite_parquet(
            parquet_files, spec, args.default_sparse, args.default_dense)
    else:
        raise FileNotFoundError(
            f'Neither {jsonl_path} nor {episodes_dir} exists — not a '
            'standard LeRobot dataset?')

    if not args.skip_proportions:
        _write_proportions(meta_dir, 'sparse', sparse_acc)
        _write_proportions(meta_dir, 'dense', dense_acc)

    print(f'Done. {updated} episode annotation(s) written.')


if __name__ == '__main__':
    main()
