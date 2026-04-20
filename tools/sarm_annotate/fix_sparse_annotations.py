#!/usr/bin/env python
"""Fix sparse annotations in a LeRobot dataset to single-stage 'task'.

- Sets meta/temporal_proportions_sparse.json to {"task": 1.0}
- For each meta/episodes/*/*.parquet:
  * sparse_subtask_names -> ["task"]
  * sparse_subtask_start_frames -> [0]
  * sparse_subtask_end_frames -> [length - 1] (falls back to dataset_to_index - dataset_from_index - 1)
  * sparse_subtask_start_times / end_times (if present) -> None

Usage:
  python fix_sparse_annotations.py --dataset-root /path/to/dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def fix_temporal_proportions(meta_dir: Path) -> None:
    out_path = meta_dir / "temporal_proportions_sparse.json"
    data = {"task": 1.0}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


def _compute_lengths(df: pd.DataFrame) -> pd.Series:
    if "length" in df.columns:
        return df["length"].astype(int)
    if {"dataset_from_index", "dataset_to_index"}.issubset(df.columns):
        return (df["dataset_to_index"] - df["dataset_from_index"]).astype(int)
    raise ValueError("Could not compute episode length; missing columns")


def fix_episodes(episodes_dir: Path) -> None:
    parquet_files = sorted(episodes_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {episodes_dir}")

    for p in parquet_files:
        print(f"Fixing {p}")
        table = pq.read_table(p)
        df = table.to_pandas()

        lengths = _compute_lengths(df)
        df["sparse_subtask_names"] = [["task"] for _ in range(len(df))]
        df["sparse_subtask_start_frames"] = [[0] for _ in range(len(df))]
        df["sparse_subtask_end_frames"] = [[int(L) - 1 if int(L) > 0 else 0] for L in lengths]

        # Optional time columns
        for col in ["sparse_subtask_start_times", "sparse_subtask_end_times"]:
            if col in df.columns:
                df[col] = [None for _ in range(len(df))]

        # Write back
        tmp_path = p.with_suffix(".fixed.tmp.parquet")
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(p)
        print(f"Rewrote {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix sparse annotations to single-stage 'task'.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root path")
    args = parser.parse_args()

    meta_dir = args.dataset_root / "meta"
    episodes_dir = meta_dir / "episodes"

    fix_temporal_proportions(meta_dir)
    fix_episodes(episodes_dir)

    print("Done")


if __name__ == "__main__":
    main()
