#!/usr/bin/env python
# flake8: noqa
# isort: skip_file
# yapf: disable

from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

# Make sure the sibling ``subtask_annotation.py`` in this directory is
# importable whether we run as a package or a standalone script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:  # package-style import (``python -m tools.sarm_annotate.run_vlm_dense_subset``)
    from .subtask_annotation import VideoAnnotator, process_single_episode
except ImportError:  # direct ``python tools/sarm_annotate/run_vlm_dense_subset.py``
    from subtask_annotation import (  # type: ignore[no-redef]
        VideoAnnotator, process_single_episode,
    )

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def resolve_dataset_spec(dataset_source: str) -> tuple[str, Path | None]:
    source_path = Path(dataset_source).expanduser()
    if source_path.exists():
        source_path = source_path.resolve()
        info_path = source_path / 'meta' / 'info.json'
        if not info_path.exists():
            raise ValueError(
                f"Local dataset path does not look like a LeRobot dataset: {source_path}"
            )
        return source_path.name, source_path
    return dataset_source, None


def timestamp_to_seconds(timestamp: str) -> float:
    parts = timestamp.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0])


def serialize_annotation(annotation: Any) -> list[dict[str, float | str]]:
    return [{
        'name': subtask.name,
        'start': float(timestamp_to_seconds(subtask.timestamps.start)),
        'end': float(timestamp_to_seconds(subtask.timestamps.end)),
    } for subtask in annotation.subtasks]


def round_timing(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def compute_episode_duration_s(dataset_meta: Any, episode_index: int,
                               video_key: str) -> float:
    start = float(dataset_meta.episodes[f"videos/{video_key}/from_timestamp"]
                  [episode_index])
    end = float(dataset_meta.episodes[f"videos/{video_key}/to_timestamp"]
                [episode_index])
    return end - start


def build_timing_record(episode_index: int) -> dict[str, Any]:
    return {
        'episode_index': int(episode_index),
        'episode_duration_s': None,
        'timing_weight': None,
        'dense_vlm_time_s': None,
        'total_vlm_time_s': 0.0,
        'dense_success': False,
        'dense_error': None,
    }


def finalize_timing_record(record: dict[str, Any]) -> dict[str, Any]:
    dense_time = record['dense_vlm_time_s']
    total_time = 0.0 if dense_time is None else dense_time
    record['episode_duration_s'] = round_timing(record['episode_duration_s'])
    record['timing_weight'] = round_timing(record['timing_weight'])
    record['dense_vlm_time_s'] = round_timing(dense_time)
    record['total_vlm_time_s'] = round(total_time, 6)
    return record


def format_timing_line(record: dict[str, Any]) -> str:
    pieces = [
        f"Episode {record['episode_index']}",
        f"duration={record['episode_duration_s']:.3f}s"
        if record['episode_duration_s'] is not None else None,
        f"total={record['total_vlm_time_s']:.3f}s",
    ]
    if record['dense_vlm_time_s'] is not None:
        pieces.append(f"dense={record['dense_vlm_time_s']:.3f}s")
    return ' | '.join(piece for piece in pieces if piece is not None)


def compute_weighted_average(records: list[dict[str, Any]],
                             field_name: str) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for record in records:
        value = record.get(field_name)
        weight = record.get('timing_weight')
        if value is None or weight is None or weight <= 0:
            continue
        weighted_sum += value * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return round(weighted_sum / total_weight, 6)


def compute_unweighted_average(records: list[dict[str, Any]],
                               field_name: str) -> float | None:
    values = [
        record[field_name] for record in records
        if record.get(field_name) is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def save_timing_summary(
    output_path: Path,
    dataset_source: str,
    dataset_root: Path,
    repo_id: str,
    video_key: str,
    num_workers: int,
    episode_indices: list[int],
    timing_records: list[dict[str, Any]],
) -> Path:
    sorted_records = sorted(
        timing_records, key=lambda item: item['episode_index'])
    successful_records = [
        record for record in sorted_records if record['total_vlm_time_s'] > 0
    ]

    summary = {
        'dataset_source':
        dataset_source,
        'repo_id':
        repo_id,
        'dataset_root':
        str(dataset_root),
        'video_key':
        video_key,
        'num_workers':
        num_workers,
        'episode_indices':
        episode_indices,
        'num_timed_episodes':
        len(sorted_records),
        'num_successful_timed_episodes':
        len(successful_records),
        'average_total_vlm_time_s':
        compute_weighted_average(successful_records, 'total_vlm_time_s'),
        'average_dense_vlm_time_s':
        compute_weighted_average(sorted_records, 'dense_vlm_time_s'),
        'unweighted_average_total_vlm_time_s':
        compute_unweighted_average(successful_records, 'total_vlm_time_s'),
        'unweighted_average_dense_vlm_time_s':
        compute_unweighted_average(sorted_records, 'dense_vlm_time_s'),
        'episodes':
        sorted_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"Saved timing summary to: {output_path}")
    if summary['average_total_vlm_time_s'] is not None:
        print(
            f"Weighted average total VLM time per episode: {summary['average_total_vlm_time_s']:.3f}s"
        )
    if summary['average_dense_vlm_time_s'] is not None:
        print(
            f"Weighted average dense VLM time per episode: {summary['average_dense_vlm_time_s']:.3f}s"
        )
    return output_path


def worker_process_dense_episodes(
    worker_id: int,
    gpu_id: int,
    episode_indices: list[int],
    repo_id: str,
    dataset_root: str | None,
    video_key: str,
    dense_subtask_list: list[str],
    model_name: str,
    torch_dtype: torch.dtype,
) -> tuple[dict[int, list[dict[str, float | str]]], list[dict[str, Any]]]:
    del worker_id

    device = f"cuda:{gpu_id}"
    dataset = LeRobotDataset(repo_id, root=dataset_root, download_videos=False)
    dense_annotator = VideoAnnotator(dense_subtask_list, model_name, device,
                                     torch_dtype)

    dense_annotations: dict[int, list[dict[str, float | str]]] = {}
    timing_records: list[dict[str, Any]] = []

    for episode_index in episode_indices:
        record = build_timing_record(episode_index)
        record['episode_duration_s'] = compute_episode_duration_s(
            dataset.meta, episode_index, video_key)
        record['timing_weight'] = record['episode_duration_s']

        start_time = time.perf_counter()
        _, dense_annotation, dense_error = process_single_episode(
            episode_index,
            dataset.root,
            dataset.meta,
            video_key,
            dataset.fps,
            dense_annotator,
        )
        record['dense_vlm_time_s'] = time.perf_counter() - start_time

        if dense_annotation is not None:
            dense_annotations[episode_index] = serialize_annotation(
                dense_annotation)
            record['dense_success'] = True
        elif dense_error:
            record['dense_error'] = dense_error

        record = finalize_timing_record(record)
        timing_records.append(record)
        print(format_timing_line(record))

    return dense_annotations, timing_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run dense VLM annotation on a selected episode subset')
    parser.add_argument(
        '--dataset-source',
        type=str,
        required=True,
        help='Dataset repo id or local dataset path')
    parser.add_argument(
        '--episodes',
        type=int,
        nargs='+',
        required=True,
        help='Global episode indices to annotate')
    parser.add_argument(
        '--dense-subtasks',
        type=str,
        required=True,
        help='Comma-separated dense subtasks')
    parser.add_argument(
        '--video-key', type=str, required=True, help='Dataset video key')
    parser.add_argument(
        '--model', type=str, required=True, help='VLM model path or name')
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save dense annotation JSON')
    parser.add_argument(
        '--timing-output',
        type=str,
        required=True,
        help='Path to save timing summary JSON')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel VLM workers')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        default=None,
        help='Explicit GPU IDs to use')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Fallback device for single-worker mode')
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['bfloat16', 'float16', 'float32'],
        help='Torch dtype used for VLM inference',
    )
    args = parser.parse_args()

    repo_id, dataset_root = resolve_dataset_spec(args.dataset_source)
    dense_subtask_list = [
        item.strip() for item in args.dense_subtasks.split(',')
        if item.strip()
    ]
    if not dense_subtask_list:
        raise ValueError('--dense-subtasks must contain at least one subtask')

    output_path = Path(args.output_path).expanduser().resolve()
    timing_output = Path(args.timing_output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timing_output.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        download_videos=True,
        episodes=args.episodes)
    if args.video_key not in (dataset.meta.video_keys or []):
        raise ValueError(
            f"Video key '{args.video_key}' not found. Available: {dataset.meta.video_keys}"
        )

    torch_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }[args.dtype]
    selected_episode_indices = sorted(set(args.episodes))

    dense_annotations: dict[int, list[dict[str, float | str]]] = {}
    timing_records: list[dict[str, Any]] = []

    if args.num_workers > 1 and torch.cuda.is_available():
        gpu_ids = args.gpu_ids or list(
            range(min(args.num_workers, torch.cuda.device_count())))
        worker_count = len(gpu_ids)
        print(f"Parallel processing with {worker_count} workers")

        episodes_per_worker = [[] for _ in range(worker_count)]
        for index, episode_index in enumerate(selected_episode_indices):
            episodes_per_worker[index % worker_count].append(episode_index)

        with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context('spawn')) as executor:
            futures = [
                executor.submit(
                    worker_process_dense_episodes,
                    worker_id,
                    gpu_ids[worker_id],
                    episodes_per_worker[worker_id],
                    repo_id,
                    str(dataset_root) if dataset_root is not None else None,
                    args.video_key,
                    dense_subtask_list,
                    args.model,
                    torch_dtype,
                ) for worker_id in range(worker_count)
                if episodes_per_worker[worker_id]
            ]

            for future in as_completed(futures):
                worker_annotations, worker_timing_records = future.result()
                dense_annotations.update(worker_annotations)
                timing_records.extend(worker_timing_records)
    else:
        device = args.device
        if device == 'cuda' and torch.cuda.is_available():
            device = 'cuda:0'

        dense_annotator = VideoAnnotator(dense_subtask_list, args.model,
                                         device, torch_dtype)
        for episode_index in selected_episode_indices:
            record = build_timing_record(episode_index)
            record['episode_duration_s'] = compute_episode_duration_s(
                dataset.meta, episode_index, args.video_key)
            record['timing_weight'] = record['episode_duration_s']

            start_time = time.perf_counter()
            _, dense_annotation, dense_error = process_single_episode(
                episode_index,
                dataset.root,
                dataset.meta,
                args.video_key,
                dataset.fps,
                dense_annotator,
            )
            record['dense_vlm_time_s'] = time.perf_counter() - start_time

            if dense_annotation is not None:
                dense_annotations[episode_index] = serialize_annotation(
                    dense_annotation)
                record['dense_success'] = True
            elif dense_error:
                record['dense_error'] = dense_error

            record = finalize_timing_record(record)
            timing_records.append(record)
            print(format_timing_line(record))

    missing_episode_indices = [
        episode_index for episode_index in selected_episode_indices
        if episode_index not in dense_annotations
    ]
    if missing_episode_indices:
        raise RuntimeError(
            f"Dense VLM annotation failed for episodes: {missing_episode_indices}"
        )

    annotations_payload = {
        'dataset_source': args.dataset_source,
        'repo_id': repo_id,
        'dataset_root': str(dataset.root),
        'video_key': args.video_key,
        'episode_indices': selected_episode_indices,
        'dense_subtasks': dense_subtask_list,
        'annotations': {
            str(index): dense_annotations[index]
            for index in sorted(dense_annotations)
        },
    }
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(annotations_payload, file, indent=2, ensure_ascii=False)
    print(f"Saved dense annotations to: {output_path}")

    save_timing_summary(
        output_path=timing_output,
        dataset_source=args.dataset_source,
        dataset_root=dataset.root,
        repo_id=repo_id,
        video_key=args.video_key,
        num_workers=args.num_workers,
        episode_indices=selected_episode_indices,
        timing_records=timing_records,
    )


if __name__ == '__main__':
    main()
