#!/usr/bin/env python
# flake8: noqa
# isort: skip_file
# yapf: disable

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""SARM subtask annotation with per-episode VLM timing statistics.

This is a timing-instrumented copy of the original subtask annotation entrypoint.
It preserves the original annotation flow while collecting:

1. Sparse VLM time per episode, when sparse VLM annotation is enabled.
2. Dense VLM time per episode, when dense VLM annotation is enabled.
3. Total VLM time per episode.
4. Aggregate averages across successfully processed episodes.

The script also accepts either a Hugging Face dataset repo id or a full local
dataset path in --repo-id.
"""

import argparse
import json
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

# Sibling-module import: subtask_annotation.py lives next to this file inside
# ``tools/sarm_annotate/``. Make the import work both when this file is run as
# a script (``python tools/sarm_annotate/subtask_annotation_timing.py``) and
# when it is imported as part of the ``tools.sarm_annotate`` package.
try:  # when executed as part of a package
    from .subtask_annotation import (
        VideoAnnotator, compute_temporal_proportions,
        generate_auto_sparse_annotations, load_annotations_from_dataset,
        process_single_episode, save_annotations_to_dataset,
        visualize_annotations)
except ImportError:  # when executed as a standalone script
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from subtask_annotation import (  # type: ignore[no-redef]
        VideoAnnotator, compute_temporal_proportions,
        generate_auto_sparse_annotations, load_annotations_from_dataset,
        process_single_episode, save_annotations_to_dataset,
        visualize_annotations,
    )

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def resolve_dataset_spec(dataset_source: str) -> tuple[str, Path | None]:
    """Resolve either a repo id or a full local dataset path."""
    source_path = Path(dataset_source).expanduser()
    if source_path.exists():
        source_path = source_path.resolve()
        info_path = source_path / 'meta' / 'info.json'
        if not info_path.exists():
            raise ValueError(
                f'Local dataset path does not look like a LeRobot dataset: {source_path}'
            )
        return source_path.name, source_path
    return dataset_source, None


def round_timing(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def compute_episode_duration_s(dataset_meta: Any, episode_index: int,
                               video_key: str) -> float:
    start = float(dataset_meta.episodes[f'videos/{video_key}/from_timestamp']
                  [episode_index])
    end = float(dataset_meta.episodes[f'videos/{video_key}/to_timestamp']
                [episode_index])
    return end - start


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


def save_proportions(
    dataset_root: Path,
    annotations: dict[int, Any],
    fps: int,
    prefix: str,
    subtask_list: list[str] | None = None,
    is_auto: bool = False,
) -> None:
    props: dict[str, float] = {
        'task': 1.0
    } if is_auto else compute_temporal_proportions(annotations, fps,
                                                   subtask_list)
    path = dataset_root / 'meta' / f'temporal_proportions_{prefix}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(props, file, indent=2, ensure_ascii=False)
    print(f'Saved {prefix} temporal proportions')


def build_timing_record(episode_index: int) -> dict[str, Any]:
    return {
        'episode_index': episode_index,
        'episode_duration_s': None,
        'timing_weight': None,
        'sparse_vlm_time_s': None,
        'dense_vlm_time_s': None,
        'total_vlm_time_s': 0.0,
        'sparse_success': False,
        'dense_success': False,
        'sparse_error': None,
        'dense_error': None,
    }


def finalize_timing_record(record: dict[str, Any]) -> dict[str, Any]:
    sparse_time = record['sparse_vlm_time_s']
    dense_time = record['dense_vlm_time_s']
    total_time = 0.0
    if sparse_time is not None:
        total_time += sparse_time
    if dense_time is not None:
        total_time += dense_time

    record['episode_duration_s'] = round_timing(record['episode_duration_s'])
    record['timing_weight'] = round_timing(record['timing_weight'])
    record['sparse_vlm_time_s'] = round_timing(sparse_time)
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
    if record['sparse_vlm_time_s'] is not None:
        pieces.append(f"sparse={record['sparse_vlm_time_s']:.3f}s")
    if record['dense_vlm_time_s'] is not None:
        pieces.append(f"dense={record['dense_vlm_time_s']:.3f}s")
    return ' | '.join(piece for piece in pieces if piece is not None)


def save_timing_summary(
    output_path: Path,
    dataset_source: str,
    dataset_root: Path,
    repo_id: str,
    video_key: str,
    num_workers: int,
    random_sample_count: int | None,
    random_seed: int,
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
        'random_sample_count':
        random_sample_count,
        'random_seed':
        random_seed,
        'num_timed_episodes':
        len(sorted_records),
        'num_successful_timed_episodes':
        len(successful_records),
        'average_total_vlm_time_s':
        compute_weighted_average(successful_records, 'total_vlm_time_s'),
        'average_sparse_vlm_time_s':
        compute_weighted_average(sorted_records, 'sparse_vlm_time_s'),
        'average_dense_vlm_time_s':
        compute_weighted_average(sorted_records, 'dense_vlm_time_s'),
        'unweighted_average_total_vlm_time_s':
        compute_unweighted_average(successful_records, 'total_vlm_time_s'),
        'unweighted_average_sparse_vlm_time_s':
        compute_unweighted_average(sorted_records, 'sparse_vlm_time_s'),
        'unweighted_average_dense_vlm_time_s':
        compute_unweighted_average(sorted_records, 'dense_vlm_time_s'),
        'episodes':
        sorted_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f'Saved timing summary to: {output_path}')
    if summary['average_total_vlm_time_s'] is not None:
        print(
            f"Weighted average total VLM time per episode: {summary['average_total_vlm_time_s']:.3f}s"
        )
    if summary['average_sparse_vlm_time_s'] is not None:
        print(
            f"Weighted average sparse VLM time per episode: {summary['average_sparse_vlm_time_s']:.3f}s"
        )
    if summary['average_dense_vlm_time_s'] is not None:
        print(
            f"Weighted average dense VLM time per episode: {summary['average_dense_vlm_time_s']:.3f}s"
        )

    return output_path


def worker_process_episodes_timed(
    worker_id: int,
    gpu_id: int,
    episode_indices: list[int],
    repo_id: str,
    dataset_root: str | None,
    video_key: str,
    sparse_subtask_list: list[str] | None,
    dense_subtask_list: list[str] | None,
    model_name: str,
    torch_dtype: torch.dtype,
) -> tuple[dict[int, Any], dict[int, Any] | None, list[dict[str, Any]]]:
    """Worker for parallel processing across GPUs with episode-level timing."""
    del worker_id

    device = f'cuda:{gpu_id}'
    dataset = LeRobotDataset(repo_id, root=dataset_root, download_videos=False)

    sparse_annotator = (
        VideoAnnotator(sparse_subtask_list, model_name, device, torch_dtype)
        if sparse_subtask_list else None)
    dense_annotator = (
        VideoAnnotator(
            dense_subtask_list,
            model_name,
            device,
            torch_dtype,
            sparse_annotator.model if sparse_annotator else None,
            sparse_annotator.processor if sparse_annotator else None,
        ) if dense_subtask_list else None)

    sparse_annotations: dict[int, Any] = {}
    dense_annotations: dict[int,
                            Any] | None = {} if dense_subtask_list else None
    timing_records: list[dict[str, Any]] = []

    for ep_idx in episode_indices:
        record = build_timing_record(ep_idx)
        record['episode_duration_s'] = compute_episode_duration_s(
            dataset.meta, ep_idx, video_key)
        record['timing_weight'] = record['episode_duration_s']

        if sparse_annotator:
            start_time = time.perf_counter()
            _, sparse_ann, sparse_err = process_single_episode(
                ep_idx, dataset.root, dataset.meta, video_key, dataset.fps,
                sparse_annotator)
            record['sparse_vlm_time_s'] = time.perf_counter() - start_time
            if sparse_ann:
                sparse_annotations[ep_idx] = sparse_ann
                record['sparse_success'] = True
            elif sparse_err:
                record['sparse_error'] = sparse_err

        if dense_annotator:
            start_time = time.perf_counter()
            _, dense_ann, dense_err = process_single_episode(
                ep_idx, dataset.root, dataset.meta, video_key, dataset.fps,
                dense_annotator)
            record['dense_vlm_time_s'] = time.perf_counter() - start_time
            if dense_ann and dense_annotations is not None:
                dense_annotations[ep_idx] = dense_ann
                record['dense_success'] = True
            elif dense_err:
                record['dense_error'] = dense_err

        record = finalize_timing_record(record)
        timing_records.append(record)
        print(format_timing_line(record))

    return sparse_annotations, dense_annotations, timing_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description='SARM subtask annotation with per-episode VLM timing')
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='Dataset repo id or full local dataset path')
    parser.add_argument(
        '--sparse-subtasks',
        type=str,
        default=None,
        help='Comma-separated sparse subtask names')
    parser.add_argument(
        '--dense-subtasks',
        type=str,
        default=None,
        help='Comma-separated dense subtask names')
    parser.add_argument(
        '--dense-only',
        action='store_true',
        help="Dense-only mode with auto-generated sparse 'task' stage")
    parser.add_argument(
        '--episodes',
        type=int,
        nargs='+',
        default=None,
        help='Episode indices to annotate')
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-VL-30B-A3B-Instruct',
        help='VLM model')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip already annotated episodes')
    parser.add_argument(
        '--video-key',
        type=str,
        default=None,
        help='Video key (default: first available)')
    parser.add_argument(
        '--push-to-hub', action='store_true', help='Push to HuggingFace Hub')
    parser.add_argument(
        '--output-repo-id',
        type=str,
        default=None,
        help='Output repo ID for push')
    parser.add_argument(
        '--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Parallel workers for multi-GPU')
    parser.add_argument(
        '--gpu-ids', type=int, nargs='+', default=None, help='GPU IDs to use')
    parser.add_argument(
        '--random-sample-count',
        type=int,
        default=None,
        help=
        'Randomly sample this many episodes from the candidate set before annotation',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed used when --random-sample-count is set',
    )
    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='Only visualize existing annotations (no generation)',
    )
    parser.add_argument(
        '--num-visualizations',
        type=int,
        default=5,
        help='Number of episodes to visualize (default: 5)',
    )
    parser.add_argument(
        '--visualize-type',
        type=str,
        default='sparse',
        choices=['sparse', 'dense', 'both'],
        help='Type of annotations to visualize (default: sparse)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./subtask_viz',
        help='Output directory for visualizations (default: ./subtask_viz)',
    )
    parser.add_argument(
        '--timing-output',
        type=str,
        default=None,
        help=
        'Optional JSON output path for timing summary (default: <output-dir>/vlm_timing_summary.json)',
    )

    args = parser.parse_args()

    repo_id, dataset_root = resolve_dataset_spec(args.repo_id)

    print(f'Loading dataset: {args.repo_id}')
    dataset = LeRobotDataset(repo_id, root=dataset_root, download_videos=True)
    fps = dataset.fps

    if not dataset.meta.video_keys:
        raise ValueError('No video keys found')

    video_key = (
        args.video_key if args.video_key in (dataset.meta.video_keys or [])
        else dataset.meta.video_keys[0])
    print(f'Using camera: {video_key}, FPS: {fps}')

    if args.visualize_only:
        print('Visualization-only mode')
        sparse_annotations = load_annotations_from_dataset(
            dataset.root, prefix='sparse')
        dense_annotations = load_annotations_from_dataset(
            dataset.root, prefix='dense')

        if not sparse_annotations and not dense_annotations:
            print('Error: No annotations found. Run annotation first.')
            return

        print(
            f'Found {len(sparse_annotations)} sparse, {len(dense_annotations)} dense annotations'
        )
        visualize_annotations(
            dataset=dataset,
            sparse_annotations=sparse_annotations,
            dense_annotations=dense_annotations if dense_annotations else None,
            video_key=video_key,
            output_dir=Path(args.output_dir),
            num_episodes=args.num_visualizations,
            annotation_type=args.visualize_type,
            episode_indices=args.episodes,
        )
        return

    if args.dense_only and not args.dense_subtasks:
        print('Error: --dense-only requires --dense-subtasks')
        return
    if args.dense_subtasks and not args.sparse_subtasks and not args.dense_only:
        print(
            'Error: --dense-subtasks requires --sparse-subtasks or --dense-only'
        )
        return

    sparse_subtask_list = ([
        item.strip() for item in args.sparse_subtasks.split(',')
    ] if args.sparse_subtasks else None)
    dense_subtask_list = [
        item.strip() for item in args.dense_subtasks.split(',')
    ] if args.dense_subtasks else None
    auto_sparse = sparse_subtask_list is None
    dense_mode = dense_subtask_list is not None
    torch_dtype = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }[args.dtype]

    episode_indices = args.episodes or list(range(dataset.meta.total_episodes))
    existing_annotations = load_annotations_from_dataset(
        dataset.root, prefix='sparse')
    if args.skip_existing:
        episode_indices = [
            episode for episode in episode_indices
            if episode not in existing_annotations
        ]

    if args.random_sample_count is not None:
        if args.random_sample_count <= 0:
            raise ValueError('--random-sample-count must be positive')
        sample_count = min(args.random_sample_count, len(episode_indices))
        sampler = random.Random(args.random_seed)
        episode_indices = sorted(sampler.sample(episode_indices, sample_count))
        print(
            f'Randomly sampled {len(episode_indices)} episodes with seed {args.random_seed}: {episode_indices}'
        )

    if not episode_indices:
        print('All episodes already annotated!')
        return
    print(f'Annotating {len(episode_indices)} episodes')

    gpu_ids = args.gpu_ids or list(
        range(
            min(args.num_workers,
                torch.cuda.device_count()
                if torch.cuda.is_available() else 1)))
    args.num_workers = len(gpu_ids)

    sparse_annotations = existing_annotations.copy()
    dense_annotations: dict[int, Any] | None = {} if dense_mode else None
    timing_records: list[dict[str, Any]] = []

    if auto_sparse:
        sparse_annotations.update(
            generate_auto_sparse_annotations(dataset, episode_indices,
                                             video_key))
        save_annotations_to_dataset(
            dataset.root, sparse_annotations, fps, prefix='sparse')
        print(
            f"Auto-generated {len(episode_indices)} sparse 'task' annotations")

    need_vlm = (not auto_sparse) or dense_mode

    if need_vlm:
        if args.num_workers > 1:
            print(f'Parallel processing with {args.num_workers} workers')
            episodes_per_worker = [[] for _ in range(args.num_workers)]
            for index, episode_idx in enumerate(episode_indices):
                episodes_per_worker[index %
                                    args.num_workers].append(episode_idx)

            with ProcessPoolExecutor(
                    max_workers=args.num_workers,
                    mp_context=mp.get_context('spawn')) as executor:
                futures = [
                    executor.submit(
                        worker_process_episodes_timed,
                        worker_id,
                        gpu_ids[worker_id],
                        episodes_per_worker[worker_id],
                        repo_id,
                        str(dataset_root)
                        if dataset_root is not None else None,
                        video_key,
                        sparse_subtask_list,
                        dense_subtask_list,
                        args.model,
                        torch_dtype,
                    ) for worker_id in range(args.num_workers)
                    if episodes_per_worker[worker_id]
                ]

                for future in as_completed(futures):
                    try:
                        worker_sparse, worker_dense, worker_timing = future.result(
                        )
                        sparse_annotations.update(worker_sparse)
                        if dense_mode and worker_dense and dense_annotations is not None:
                            dense_annotations.update(worker_dense)
                        timing_records.extend(worker_timing)
                        save_annotations_to_dataset(
                            dataset.root,
                            sparse_annotations,
                            fps,
                            prefix='sparse')
                        if dense_mode and dense_annotations is not None:
                            save_annotations_to_dataset(
                                dataset.root,
                                dense_annotations,
                                fps,
                                prefix='dense')
                    except Exception as err:
                        raise RuntimeError(f'Worker failed: {err}') from err
        else:
            sparse_annotator = (
                VideoAnnotator(sparse_subtask_list, args.model, args.device,
                               torch_dtype)
                if not auto_sparse and sparse_subtask_list else None)
            dense_annotator = (
                VideoAnnotator(
                    dense_subtask_list,
                    args.model,
                    args.device,
                    torch_dtype,
                    sparse_annotator.model if sparse_annotator else None,
                    sparse_annotator.processor if sparse_annotator else None,
                ) if dense_mode else None)

            for index, episode_idx in enumerate(episode_indices):
                print(
                    f'Episode {episode_idx} ({index + 1}/{len(episode_indices)})'
                )
                record = build_timing_record(episode_idx)
                record['episode_duration_s'] = compute_episode_duration_s(
                    dataset.meta, episode_idx, video_key)
                record['timing_weight'] = record['episode_duration_s']

                if sparse_annotator:
                    start_time = time.perf_counter()
                    _, sparse_ann, sparse_err = process_single_episode(
                        episode_idx, dataset.root, dataset.meta, video_key,
                        fps, sparse_annotator)
                    record['sparse_vlm_time_s'] = time.perf_counter(
                    ) - start_time
                    if sparse_ann:
                        sparse_annotations[episode_idx] = sparse_ann
                        record['sparse_success'] = True
                        save_annotations_to_dataset(
                            dataset.root,
                            sparse_annotations,
                            fps,
                            prefix='sparse')
                    elif sparse_err:
                        record['sparse_error'] = sparse_err
                        print(f'Sparse failed: {sparse_err}')

                if dense_annotator:
                    start_time = time.perf_counter()
                    _, dense_ann, dense_err = process_single_episode(
                        episode_idx, dataset.root, dataset.meta, video_key,
                        fps, dense_annotator)
                    record['dense_vlm_time_s'] = time.perf_counter(
                    ) - start_time
                    if dense_ann and dense_annotations is not None:
                        dense_annotations[episode_idx] = dense_ann
                        record['dense_success'] = True
                        save_annotations_to_dataset(
                            dataset.root,
                            dense_annotations,
                            fps,
                            prefix='dense')
                    elif dense_err:
                        record['dense_error'] = dense_err
                        print(f'Dense failed: {dense_err}')

                record = finalize_timing_record(record)
                timing_records.append(record)
                print(format_timing_line(record))

    save_proportions(dataset.root, sparse_annotations, fps, 'sparse',
                     sparse_subtask_list, auto_sparse)
    if dense_mode and dense_annotations:
        save_proportions(dataset.root, dense_annotations, fps, 'dense',
                         dense_subtask_list)

    print(
        f'\nComplete! {len(sparse_annotations)} sparse, {len(dense_annotations or {})} dense annotations'
    )

    if need_vlm:
        timing_output = Path(
            args.timing_output) if args.timing_output else Path(
                args.output_dir) / 'vlm_timing_summary.json'
        save_timing_summary(
            output_path=timing_output,
            dataset_source=args.repo_id,
            dataset_root=dataset.root,
            repo_id=repo_id,
            video_key=video_key,
            num_workers=args.num_workers,
            random_sample_count=args.random_sample_count,
            random_seed=args.random_seed,
            timing_records=timing_records,
        )

    if args.num_visualizations > 0:
        print(f'\nGenerating {args.num_visualizations} visualizations...')
        visualize_type = 'both' if dense_mode else 'sparse'
        visualize_annotations(
            dataset=dataset,
            sparse_annotations=sparse_annotations,
            dense_annotations=dense_annotations,
            video_key=video_key,
            output_dir=Path(args.output_dir),
            num_episodes=args.num_visualizations,
            annotation_type=visualize_type,
        )

    if args.push_to_hub:
        try:
            dataset.push_to_hub(push_videos=True)
            print(f'Pushed to {args.output_repo_id or repo_id}')
        except Exception as err:
            print(f'Push failed: {err}')


if __name__ == '__main__':
    main()
