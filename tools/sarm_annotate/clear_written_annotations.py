#!/usr/bin/env python
"""Clear annotation fields written into a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ANNOTATION_COLUMNS = [
    "dense_subtask_names",
    "dense_subtask_start_times",
    "dense_subtask_end_times",
    "dense_subtask_start_frames",
    "dense_subtask_end_frames",
    "sparse_subtask_names",
    "sparse_subtask_start_times",
    "sparse_subtask_end_times",
    "sparse_subtask_start_frames",
    "sparse_subtask_end_frames",
    "subtask_names",
    "subtask_start_times",
    "subtask_end_times",
    "subtask_start_frames",
    "subtask_end_frames",
]

DERIVED_FILES = [
    "meta/temporal_proportions_dense.json",
    "meta/temporal_proportions_sparse.json",
    "dense_episode_info.json",
    "sparse_episode_info.json",
    "subtask_episode_info.json",
]


def has_value(v) -> bool:
    if v is None:
        return False
    return not (isinstance(v, float) and pd.isna(v))


def clear_parquet_annotations(parquet_path: Path, apply: bool) -> tuple[int, int]:
    df = pd.read_parquet(parquet_path)
    existing = [c for c in ANNOTATION_COLUMNS if c in df.columns]
    if not existing:
        return 0, 0

    non_null_cells = 0
    for col in existing:
        non_null_cells += int(df[col].map(has_value).sum())

    if apply and non_null_cells > 0:
        for col in existing:
            df[col] = pd.Series([None] * len(df), dtype=object)
        df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

    return len(existing), non_null_cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear written dense/sparse/subtask annotations.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this flag, only prints what would be cleared.",
    )
    args = parser.parse_args()

    episodes_dir = args.dataset_root / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episodes parquet found under: {episodes_dir}")

    total_cols = 0
    total_cells = 0
    for parquet_path in parquet_files:
        cols, cells = clear_parquet_annotations(parquet_path, apply=args.apply)
        total_cols += cols
        total_cells += cells
        print(f"{parquet_path}: columns={cols}, non_null_cells={cells}")

    print(f"Total: columns_seen={total_cols}, non_null_cells={total_cells}")

    for rel in DERIVED_FILES:
        path = args.dataset_root / rel
        if path.exists():
            if args.apply:
                path.unlink()
                print(f"Removed: {path}")
            else:
                print(f"Would remove: {path}")

    if not args.apply:
        print("Dry-run mode. Add --apply to execute.")


if __name__ == "__main__":
    main()
