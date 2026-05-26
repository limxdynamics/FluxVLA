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

import argparse
import importlib
import json
import logging
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple, cast

import numpy as np
import torch
from mmengine import Config, DictAction
from torch.utils.data import DataLoader, Dataset

# SARM inference is PyTorch-only; set this before any transitive Hugging Face
# imports can probe or initialize TensorFlow.
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('DATASETS_VERBOSITY', 'error')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
for logger_name in ('datasets', 'datasets.config', 'transformers',
                    'transformers.utils.import_utils'):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VizRecordsByMode = Dict[str, DefaultDict[int, List[Dict]]]


def _ensure_namespace_package(name: str, path: Path) -> types.ModuleType:
    """Install a namespace package without executing its ``__init__`` file."""
    module = sys.modules.get(name)
    if module is not None and hasattr(module, '__path__'):
        return module
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _install_lightweight_fluxvla_imports() -> Dict[str, Any]:
    """Expose only the registry utilities needed by SARM inference."""
    package_paths = {
        'fluxvla':
        REPO_ROOT / 'fluxvla',
        'fluxvla.collators':
        REPO_ROOT / 'fluxvla' / 'collators',
        'fluxvla.datasets':
        REPO_ROOT / 'fluxvla' / 'datasets',
        'fluxvla.datasets.utils':
        REPO_ROOT / 'fluxvla' / 'datasets' / 'utils',
        'fluxvla.engines':
        REPO_ROOT / 'fluxvla' / 'engines',
        'fluxvla.engines.utils':
        REPO_ROOT / 'fluxvla' / 'engines' / 'utils',
        'fluxvla.models':
        REPO_ROOT / 'fluxvla' / 'models',
        'fluxvla.models.backbones':
        REPO_ROOT / 'fluxvla' / 'models' / 'backbones',
        'fluxvla.models.backbones.llms':
        REPO_ROOT / 'fluxvla' / 'models' / 'backbones' / 'llms',
        'fluxvla.models.vlas':
        REPO_ROOT / 'fluxvla' / 'models' / 'vlas',
        'fluxvla.tokenizers':
        REPO_ROOT / 'fluxvla' / 'tokenizers',
    }
    for name, path in package_paths.items():
        _ensure_namespace_package(name, path)

    root = importlib.import_module('fluxvla.engines.utils.root')
    builder = importlib.import_module('fluxvla.engines.utils.builder')
    overwatch = importlib.import_module('fluxvla.engines.utils.overwatch')

    registry_names = [
        'TOKENIZERS',
        'TRANSFORMS',
        'DATASETS',
        'LLM_BACKBONES',
        'VISION_BACKBONES',
        'PROJECTORS',
        'HEADS',
        'VLAS',
        'RUNNERS',
        'COLLATORS',
        'METRICS',
        'PROCESSORS',
        'VLM_BACKBONES',
        'OPERATORS',
    ]
    builder_names = [
        'build_collator_from_cfg',
        'build_dataset_from_cfg',
        'build_from_cfg',
        'build_head_from_cfg',
        'build_llm_backbone_from_cfg',
        'build_projector_from_cfg',
        'build_tokenizer_from_cfg',
        'build_transform_from_cfg',
        'build_vision_backbone_from_cfg',
        'build_vla_from_cfg',
        'build_vlm_backbone_from_cfg',
    ]
    engines_pkg = sys.modules['fluxvla.engines']
    utils_pkg = sys.modules['fluxvla.engines.utils']
    for attr in registry_names:
        value = getattr(root, attr)
        setattr(engines_pkg, attr, value)
        setattr(utils_pkg, attr, value)
    for attr in builder_names:
        value = getattr(builder, attr)
        setattr(engines_pkg, attr, value)
        setattr(utils_pkg, attr, value)
    setattr(engines_pkg, 'initialize_overwatch',
            overwatch.initialize_overwatch)
    setattr(utils_pkg, 'initialize_overwatch', overwatch.initialize_overwatch)
    return {name: getattr(builder, name) for name in builder_names}


def _register_sarm_transforms() -> None:
    """Register SARM-only transforms without importing unrelated modules."""
    from fluxvla.engines import TRANSFORMS

    class ResizeImageSequence:
        """Resize image sequences while preserving leading dimensions."""

        def __init__(self, height: int, width: int, *args, **kwargs):
            del args, kwargs
            self.height = height
            self.width = width

        def __call__(self, data: Dict) -> Dict:
            import cv2
            assert 'images' in data, "Input data must contain 'images' key"
            images = np.asarray(data['images'])
            original_shape = images.shape
            if images.ndim < 4:
                raise ValueError(
                    'Input image sequence must have at least 4 dimensions')
            flat_images = images.reshape(-1, original_shape[-3],
                                         original_shape[-2],
                                         original_shape[-1])

            resized_images = []
            for image in flat_images:
                resized_images.append(
                    cv2.resize(
                        image.transpose(1, 2, 0), (self.width, self.height),
                        interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1))

            data['images'] = np.stack(
                resized_images, axis=0).reshape(*original_shape[:-2],
                                                self.height, self.width)
            return data

    class NormalizeImageSequence:
        """Normalize image sequences while preserving leading dimensions."""

        def __init__(self,
                     means: List,
                     stds: List,
                     scale_to_unit_interval: bool = False,
                     *args,
                     **kwargs):
            del args, kwargs
            self.means = np.asarray(means, dtype=np.float32)
            self.stds = np.asarray(stds, dtype=np.float32)
            self.scale_to_unit_interval = scale_to_unit_interval

        def __call__(self, data: Dict) -> Dict:
            assert 'images' in data, "Input data must contain 'images' key"
            images = np.asarray(data['images'])
            original_shape = images.shape
            flat_images = images.reshape(-1, original_shape[-3],
                                         original_shape[-2],
                                         original_shape[-1]).astype(np.float32)
            if self.scale_to_unit_interval:
                flat_images = flat_images / 255.0

            means = self.means
            stds = self.stds
            if means.ndim == 1:
                means = np.broadcast_to(means[None, :],
                                        (flat_images.shape[0], 3))
            if stds.ndim == 1:
                stds = np.broadcast_to(stds[None, :],
                                       (flat_images.shape[0], 3))
            if means.shape[0] == 1:
                means = np.broadcast_to(means, (flat_images.shape[0], 3))
            if stds.shape[0] == 1:
                stds = np.broadcast_to(stds, (flat_images.shape[0], 3))
            if (means.shape[0] != flat_images.shape[0]
                    or stds.shape[0] != flat_images.shape[0]):
                raise ValueError(
                    'Means/stds must have length 1 or match the number '
                    'of images after flattening.')

            normalized_images = []
            for idx, image in enumerate(flat_images):
                normalized_images.append((image - means[idx][:, None, None]) /
                                         (stds[idx][:, None, None] + 1e-8))

            data['images'] = np.stack(
                normalized_images, axis=0).reshape(original_shape)
            return data

    class PadStates:
        """Pad or truncate SARM state vectors."""

        def __init__(self, max_state_dim: int = 32):
            self.max_state_dim = max_state_dim

        def __call__(self, data: Dict) -> Dict:
            states = np.asarray(data['states'], dtype=np.float32)
            current_dim = states.shape[-1]
            if current_dim >= self.max_state_dim:
                data['states'] = states[..., :self.max_state_dim]
                return data
            padded_shape = (*states.shape[:-1], self.max_state_dim)
            padded = np.zeros(padded_shape, dtype=np.float32)
            padded[..., :current_dim] = states
            data['states'] = padded
            return data

    class TokenizeText:
        """Tokenize task text for CLIP-based SARM inference."""

        def __init__(self,
                     tokenizer: Dict,
                     max_length: int = 77,
                     text_key: str = 'task_description',
                     output_ids_key: str = 'text_input_ids',
                     output_attention_mask_key: str = 'text_attention_mask'):
            from fluxvla.engines import build_tokenizer_from_cfg
            from fluxvla.engines.utils.hf_hub import resolve_hf_local_path
            tokenizer = dict(tokenizer)
            model_path = tokenizer.get('model_path')
            if isinstance(model_path, str):
                tokenizer['model_path'] = resolve_hf_local_path(model_path)
            self.tokenizer = build_tokenizer_from_cfg(tokenizer)
            self.max_length = max_length
            self.text_key = text_key
            self.output_ids_key = output_ids_key
            self.output_attention_mask_key = output_attention_mask_key

        def __call__(self, data: Dict) -> Dict:
            encoded = self.tokenizer(
                data[self.text_key],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='np',
            )
            data[self.output_ids_key] = encoded['input_ids'][0].astype(
                np.int64)
            data[self.output_attention_mask_key] = encoded['attention_mask'][
                0].astype(np.int64)
            return data

    TRANSFORMS.register_module(module=ResizeImageSequence, force=True)
    TRANSFORMS.register_module(module=NormalizeImageSequence, force=True)
    TRANSFORMS.register_module(module=PadStates, force=True)
    TRANSFORMS.register_module(module=TokenizeText, force=True)


def _register_sarm_runtime_modules() -> Dict[str, Any]:
    """Register only modules required by the SARM inference config."""
    builders = _install_lightweight_fluxvla_imports()
    _register_sarm_transforms()
    for module_name in [
            'fluxvla.collators.dict_collator',
            'fluxvla.datasets.sarm_dataset',
            'fluxvla.models.backbones.llms.sarm',
            'fluxvla.models.vlas.sarm_reward_model',
            'fluxvla.tokenizers.pretrained_tokenizer',
    ]:
        importlib.import_module(module_name)
    return builders


def parse_args():
    """Parse command-line arguments for SARM progress inference.

    Returns:
        argparse.Namespace: Parsed inference arguments.
    """
    parser = argparse.ArgumentParser(
        description='Infer SARM progress over a LeRobot v2.1 or v3.x dataset.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument(
        '--head-mode',
        type=str,
        default='sparse',
        choices=['sparse', 'dense', 'both'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='Generate SARM prediction visualizations without writing JSONL.')
    parser.add_argument(
        '--num-visualizations',
        type=int,
        default=5,
        help='Number of episodes to visualize after inference. Set to 0 to '
        'skip visualization.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./sarm_viz',
        help='Directory for post-training SARM prediction visualizations.')
    parser.add_argument(
        '--num-display-frames',
        type=int,
        default=8,
        help='Number of thumbnail frames to show in each visualization.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, '
        'the key-value pair in xxx=yyy format')
    args = parser.parse_args()
    if not args.visualize_only and args.output_path is None:
        parser.error(
            '--output-path is required unless --visualize-only is set')
    if args.num_visualizations < 0:
        parser.error('--num-visualizations must be non-negative')
    if args.num_display_frames <= 0:
        parser.error('--num-display-frames must be positive')
    return args


def _resolve_head_modes(model: Any, head_mode: str) -> List[str]:
    """Resolve requested SARM heads against model capabilities.

    Args:
        model (Any): SARM reward model.
        head_mode (str): Requested head mode.

    Returns:
        List[str]: Head modes to run.
    """
    uses_dual_heads = bool(getattr(model, 'uses_dual_heads', False))
    if head_mode == 'both':
        modes = ['sparse']
        if uses_dual_heads:
            modes.append('dense')
        return modes
    if head_mode == 'dense' and not uses_dual_heads:
        raise ValueError('Dense SARM head is unavailable for this model.')
    return [head_mode]


def _num_stages(model: Any, head_mode: str) -> int:
    if head_mode == 'sparse':
        return int(model.num_sparse_stages)
    return int(model.num_dense_stages)


def _stage_labels(model: Any, head_mode: str) -> List[str]:
    names = getattr(model, f'{head_mode}_subtask_names', None)
    if names:
        return [str(name) for name in names]
    return [f'Stage {idx + 1}' for idx in range(_num_stages(model, head_mode))]


def _temporal_proportions(model: Any, head_mode: str):
    return getattr(model, f'{head_mode}_temporal_proportions', None)


def _target_key(head_mode: str) -> str:
    return f'{head_mode}_targets'


def _as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _normalize_target_value(model: Any, head_mode: str,
                            target_value: float) -> float:
    from fluxvla.datasets.utils.sarm_utils import normalize_stage_tau
    if not np.isfinite(target_value):
        return float('nan')
    return float(
        normalize_stage_tau(
            target_value,
            num_stages=_num_stages(model, head_mode),
            temporal_proportions=_temporal_proportions(model, head_mode),
            subtask_names=_stage_labels(model, head_mode),
        ))


def _to_numpy_image(image_sequence, frame_index: int) -> np.ndarray:
    """Convert one transformed SARM image sequence to a displayable frame."""
    image = image_sequence.detach().cpu().numpy() if isinstance(
        image_sequence, torch.Tensor) else np.asarray(image_sequence)
    if image.ndim == 5:
        frame = image[min(frame_index, image.shape[0] - 1), 0]
    elif image.ndim == 4:
        frame = image[min(frame_index, image.shape[0] - 1)]
    elif image.ndim == 3:
        frame = image
    else:
        raise ValueError(f'Unsupported image shape for visualization: '
                         f'{image.shape}')

    if frame.shape[0] in [1, 3]:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        frame = frame - np.nanmin(frame)
        scale = np.nanmax(frame)
        if scale > 0:
            frame = frame / scale
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame


def _visual_episode_indices(dataset: Dataset,
                            num_visualizations: int) -> set[int]:
    episode_ranges = getattr(dataset, 'episode_ranges', None)
    if not episode_ranges:
        return set()
    ordered_ranges = sorted(
        episode_ranges.items(), key=lambda item: item[1][0])
    return {
        int(episode_index)
        for (_, episode_index), _ in ordered_ranges[:num_visualizations]
    }


def _render_episode_visualization(records: List[Dict], stage_labels: List[str],
                                  output_path: Path, num_display_frames: int,
                                  title: str) -> None:
    """Render a LeRobot-style SARM prediction visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    records = sorted(records, key=lambda item: item['current_index'])
    frame_positions = np.arange(len(records))
    progress_preds = np.asarray(
        [record['pred_progress'] for record in records], dtype=np.float32)
    stage_probs = np.stack([record['stage_probs'] for record in records])
    gt_progress = np.asarray([record['gt_progress'] for record in records],
                             dtype=np.float32)
    gt_stages = np.asarray([record['gt_stage'] for record in records],
                           dtype=np.float32)

    num_stages = stage_probs.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)
    ax_progress = fig.add_subplot(gs[0])
    ax_stages = fig.add_subplot(gs[1])
    ax_frames = fig.add_subplot(gs[2])

    ax_progress.plot(
        frame_positions,
        progress_preds,
        linewidth=2,
        color='#2E86AB',
        label='Predicted')
    ax_progress.fill_between(
        frame_positions, 0, progress_preds, alpha=0.25, color='#2E86AB')
    if np.isfinite(gt_progress).any():
        ax_progress.plot(
            frame_positions,
            gt_progress,
            linewidth=2,
            linestyle='--',
            color='#28A745',
            label='Ground Truth')
    ax_progress.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax_progress.set_ylabel('Progress')
    ax_progress.set_title(title, fontweight='bold')
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc='upper left')
    ax_progress.grid(True, alpha=0.3)

    ax_stages.stackplot(
        frame_positions,
        *[stage_probs[:, idx] for idx in range(num_stages)],
        colors=colors,
        alpha=0.8,
        labels=stage_labels)
    for idx in range(1, len(gt_stages)):
        if (np.isfinite(gt_stages[idx - 1]) and np.isfinite(gt_stages[idx])
                and gt_stages[idx] != gt_stages[idx - 1]):
            ax_stages.axvline(
                x=idx, color='black', linestyle='-', alpha=0.7, linewidth=1.5)
    ax_stages.set_xlabel('Frame')
    ax_stages.set_ylabel('Stage Probability')
    ax_stages.set_ylim(0, 1)
    ax_stages.legend(loc='upper left', ncol=min(num_stages, 5), fontsize=8)
    ax_stages.grid(True, alpha=0.3)

    ax_frames.axis('off')
    sample_count = min(num_display_frames, len(records))
    sample_indices = np.linspace(0, len(records) - 1, sample_count, dtype=int)
    frames = [records[idx]['image'] for idx in sample_indices]
    combined = np.concatenate(frames, axis=1)
    ax_frames.imshow(combined)
    width = frames[0].shape[1]
    for col, record_idx in enumerate(sample_indices):
        record = records[record_idx]
        stage_name = stage_labels[int(np.argmax(record['stage_probs']))][:16]
        pred_progress = format(record['pred_progress'], '.2f')
        ax_frames.text(
            col * width + width / 2,
            -8, f"Frame {record['current_index']}\n"
            f'{pred_progress}\n{stage_name}',
            ha='center',
            va='top',
            fontsize=7)
    ax_frames.set_title('Sample Frames', pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _visualize_sarm_predictions(viz_records_by_mode: VizRecordsByMode,
                                model: Any, output_dir: str,
                                num_display_frames: int) -> None:
    output_root = Path(output_dir)
    for head_mode, records_by_episode in viz_records_by_mode.items():
        labels = _stage_labels(model, head_mode)
        for episode_index, records in sorted(records_by_episode.items()):
            if not records:
                continue
            task_description = records[0]['task_description']
            output_path = (
                output_root /
                f'sarm_prediction_ep{episode_index}_{head_mode}.png')
            _render_episode_visualization(
                records=records,
                stage_labels=labels,
                output_path=output_path,
                num_display_frames=num_display_frames,
                title=f'{task_description} (Episode {episode_index}, '
                f'{head_mode})')


def _write_jsonl(output_path: str, records: List[Dict]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    """Run SARM progress inference and optional prediction visualizations."""
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

    collator_cfg = cfg.runner.collator.copy()
    collator = build_collator_from_cfg(collator_cfg)

    model = cast(Any, build_vla_from_cfg(cfg.model))
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(
        checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device(args.device)).eval()
    head_modes = _resolve_head_modes(model, args.head_mode)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator)

    results = []
    collect_visuals = args.num_visualizations > 0
    visual_episode_indices = _visual_episode_indices(dataset,
                                                     args.num_visualizations)
    viz_records_by_mode = {
        head_mode: defaultdict(list)
        for head_mode in head_modes
    }
    frame_index = int(getattr(model, 'n_obs_steps', 0))
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            progress_by_mode = {}
            stage_probs_by_mode = {}
            for head_mode in head_modes:
                output = model.predict_progress(
                    images=batch['images'],
                    text_input_ids=batch['text_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    states=batch['states'],
                    lengths=batch['lengths'],
                    head_mode=head_mode,
                    return_stage_probs=collect_visuals,
                )
                if collect_visuals:
                    progress_tensor, stage_probs_tensor = cast(
                        Tuple[torch.Tensor, torch.Tensor], output)
                    stage_probs_by_mode[head_mode] = (
                        stage_probs_tensor.detach().cpu())
                else:
                    progress_tensor = cast(torch.Tensor, output)
                progress_by_mode[head_mode] = progress_tensor.detach().cpu()

            batch_size = len(batch['episode_index'])
            for idx in range(batch_size):
                episode_index = _as_int(batch['episode_index'][idx])
                current_index = _as_int(batch['current_index'][idx])
                task_description = batch['task_description'][idx]
                record = {
                    'episode_index': episode_index,
                    'current_index': current_index,
                    'task_description': task_description,
                }
                if len(head_modes) == 1:
                    head_mode = head_modes[0]
                    record['pred_progress'] = float(
                        progress_by_mode[head_mode][idx])
                else:
                    for head_mode in head_modes:
                        record[f'pred_{head_mode}_progress'] = float(
                            progress_by_mode[head_mode][idx])
                results.append(record)

                if (not collect_visuals
                        or episode_index not in visual_episode_indices):
                    continue
                image = _to_numpy_image(batch['images'][idx], frame_index)
                for head_mode in head_modes:
                    target_value = float('nan')
                    key = _target_key(head_mode)
                    if key in batch:
                        target_frame = min(frame_index,
                                           batch[key].shape[1] - 1)
                        target_value = _as_float(batch[key][idx, target_frame])
                    viz_records_by_mode[head_mode][episode_index].append({
                        'current_index':
                        current_index,
                        'task_description':
                        task_description,
                        'pred_progress':
                        float(progress_by_mode[head_mode][idx]),
                        'stage_probs':
                        stage_probs_by_mode[head_mode][idx].numpy(),
                        'gt_progress':
                        _normalize_target_value(model, head_mode,
                                                target_value),
                        'gt_stage':
                        float(np.floor(target_value))
                        if np.isfinite(target_value) else float('nan'),
                        'image':
                        image,
                    })

    if not args.visualize_only:
        assert args.output_path is not None
        _write_jsonl(args.output_path, results)
    if collect_visuals:
        _visualize_sarm_predictions(
            viz_records_by_mode=viz_records_by_mode,
            model=model,
            output_dir=args.output_dir,
            num_display_frames=args.num_display_frames)


if __name__ == '__main__':
    main()
