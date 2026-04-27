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
"""End-to-end RTC test with real model, checkpoint, and training data.

Loads a model from config + checkpoint, takes training data batches,
and runs predict_action with/without RTC. Produces per-dimension
denoising process visualizations with prefix regions highlighted.

Usage:
    # GR00T / PI0 — run all modes
    python scripts/test_rtc.py \
        --config configs/gr00t/xxx.py \
        --checkpoint /path/to/checkpoint.pt \
        --prefix_len 5 \
        --output_dir work_dirs/rtc_test
"""

import argparse
import os
import time as time_mod

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config


def load_model(cfg, checkpoint_path, device):
    """Build model from config and load checkpoint.

    Args:
        cfg (Config): Model configuration.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: The loaded model in eval mode.
    """
    from fluxvla.engines.utils.builder import build_vla_from_cfg

    model = build_vla_from_cfg(cfg.model)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f'Loaded checkpoint: {checkpoint_path}')
    model = model.to(device).eval()
    return model


def load_batch(cfg, device):
    """Build dataset from config and get one batch.

    Args:
        cfg (Config): Dataset and runner configuration.
        device (torch.device): Device to move tensors to.

    Returns:
        dict: A single batch dict with tensors on the target device.
    """
    from torch.utils.data import DataLoader

    from fluxvla.engines.utils.builder import (build_collator_from_cfg,
                                               build_dataset_from_cfg)

    dataset = build_dataset_from_cfg(cfg.train_dataloader.dataset)
    collator = build_collator_from_cfg(cfg.runner.collator)
    loader = DataLoader(
        dataset, batch_size=1, collate_fn=collator, num_workers=0)
    batch = next(iter(loader))

    # Move tensors to device, cast float64 → float32
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float64:
                v = v.float()
            batch[k] = v.to(device)
    return batch


def predict_with_intermediates(model,
                               batch,
                               prefix_len=0,
                               prev_actions=None,
                               rtc_config=None,
                               seed=42):
    """Run predict_action and capture intermediate denoising steps.

    Monkey-patches the denoise loop to record x_t at each step.
    Supports both GR00T (denoise_step on model.vla_head) and
    PI0/PI0.5 (denoise_step directly on model).
    """
    # Determine where denoise_step lives
    if (hasattr(model, 'vla_head') and model.vla_head is not None
            and hasattr(model.vla_head, 'denoise_step')):
        target = model.vla_head  # GR00T / LlavaVLA
        is_pi0 = False
    else:
        target = model  # PI0 / PI0.5
        is_pi0 = True

    intermediates = []

    # Hook into denoise_step to capture actions at each step
    orig_denoise = target.denoise_step

    def hooked_denoise(*args, **kwargs):
        # For GR00T: args[0] is actions tensor
        # For PI0:   args[3] is x_t (states, prefix_pad_masks, pkv, x_t, ...)
        capture_idx = 3 if is_pi0 else 0
        intermediates.append(args[capture_idx].detach().cpu().float().clone())
        return orig_denoise(*args, **kwargs)

    # Ensure images is a list (predict_action expects List[Tensor])
    images = batch['images']
    if isinstance(images, torch.Tensor) and images.ndim == 5:
        images = [images[:, i] for i in range(images.shape[1])]

    # Build kwargs compatible with both model types
    predict_kwargs = dict(
        images=images,
        lang_tokens=batch['lang_tokens'],
        states=batch['states'],
        img_masks=batch.get('img_masks'),
        lang_masks=batch.get('lang_masks'),
        prev_actions=prev_actions,
        prefix_len=prefix_len,
        rtc_config=rtc_config,
    )
    if not is_pi0:
        # GR00T accepts embodiment_ids; PI0 does not
        predict_kwargs['embodiment_ids'] = batch.get('embodiment_ids')

    target.denoise_step = hooked_denoise
    try:
        torch.manual_seed(seed)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model.predict_action(**predict_kwargs)
    finally:
        target.denoise_step = orig_denoise

    # Append final prediction
    intermediates.append(pred.detach().cpu().clone())
    return pred, intermediates


def plot_denoising_per_dim(intermediates,
                           gt_actions,
                           prefix_len,
                           prev_actions=None,
                           title_prefix='',
                           save_path=None,
                           dim_names=None):
    """Plot denoising process for each action dimension.

    Args:
        intermediates: List of (1, T, D) tensors at each denoising step.
        gt_actions: (1, T, D) ground truth actions.
        prefix_len: Number of prefix steps (highlighted region).
        prev_actions: (1, remaining, D) previous chunk actions used as prefix.
        title_prefix: String prepended to figure title.
        save_path: Where to save the plot.
        dim_names: Optional list of dimension names.
    """
    n_steps = len(intermediates)
    action_dim = intermediates[0].shape[-1]
    T = intermediates[0].shape[1]

    if dim_names is None:
        dim_names = [f'dim {d}' for d in range(action_dim)]

    # Layout: one row per dimension
    n_cols = 2
    n_rows = (action_dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 3 * n_rows), squeeze=False)
    fig.suptitle(
        '{}Denoising Process (prefix_len={})'.format(title_prefix, prefix_len),
        fontsize=14)

    colors = plt.cm.viridis(np.linspace(0, 1, n_steps))

    for d in range(action_dim):
        ax = axes[d // n_cols, d % n_cols]

        # Prefix region
        if prefix_len > 0:
            ax.axvspan(
                0, prefix_len - 1, alpha=0.15, color='orange', label='prefix')

        # Ground truth
        gt = gt_actions[0, :, d].numpy()
        ax.plot(range(T), gt, 'k-', linewidth=2, alpha=0.4, label='GT')

        # Previous actions in prefix region
        if prev_actions is not None and prefix_len > 0:
            pa = prev_actions[0, :prefix_len, d].numpy()
            ax.plot(
                range(prefix_len),
                pa,
                'r-',
                linewidth=2.5,
                label='prev_actions',
                zorder=5)

        # Denoising steps (subsample if too many)
        max_traces = 8
        if n_steps > max_traces:
            indices = np.linspace(0, n_steps - 1, max_traces).astype(int)
        else:
            indices = range(n_steps)

        for idx in indices:
            step_frac = idx / max(n_steps - 1, 1)
            x = intermediates[idx][0, :, d].numpy()
            alpha = 0.3 + 0.7 * step_frac
            ax.plot(range(T), x, color=colors[idx], alpha=alpha, linewidth=0.8)

        # Final prediction (bold)
        final = intermediates[-1][0, :, d].numpy()
        ax.plot(range(T), final, 'b-', linewidth=1.5, label='final pred')

        ax.set_title(dim_names[d], fontsize=10)
        ax.set_xlabel('step')
        ax.legend(fontsize=7, loc='upper right')

    # Hide unused axes
    for d in range(action_dim, n_rows * n_cols):
        axes[d // n_cols, d % n_cols].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved: {save_path}')
    plt.close()


def plot_comparison(results,
                    mode_configs,
                    gt_actions,
                    prefix_len,
                    save_path=None):
    """Compare predictions across all run RTC modes.

    Args:
        results (dict): Mapping mode_name -> (pred, intermediates, elapsed).
        mode_configs (dict): Mapping mode_name -> (display_name, kwargs).
        gt_actions (torch.Tensor): Ground truth actions of shape (1, T, D).
        prefix_len (int): Number of prefix steps (highlighted region).
        save_path (str, optional): Where to save the plot.
    """
    # Style per mode: (color, linestyle)
    mode_styles = {
        'no_rtc': ('r', '--'),
        'prefix': ('b', '-'),
        'guidance': ('g', '-'),
        'guidance_vjp': ('m', '--'),
    }

    action_dim = gt_actions.shape[-1]
    T = gt_actions.shape[1]

    n_cols = 2
    n_rows = (action_dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 3 * n_rows), squeeze=False)
    fig.suptitle(
        'RTC Comparison (prefix_len={})'.format(prefix_len), fontsize=14)

    for d in range(action_dim):
        ax = axes[d // n_cols, d % n_cols]

        if prefix_len > 0:
            ax.axvspan(
                0, prefix_len - 1, alpha=0.15, color='orange', label='prefix')

        gt = gt_actions[0, :, d].numpy()
        ax.plot(range(T), gt, 'k-', linewidth=2, alpha=0.4, label='GT')

        for mode, (pred, _, _) in results.items():
            display = mode_configs[mode][0]
            color, ls = mode_styles.get(mode, ('tab:gray', '-'))
            ax.plot(
                range(T),
                pred[0, :, d].cpu().numpy(),
                color=color,
                linestyle=ls,
                linewidth=1,
                label=display,
                alpha=0.7)

        ax.set_title(f'dim {d}', fontsize=10)
        ax.legend(fontsize=7)

    for d in range(action_dim, n_rows * n_cols):
        axes[d // n_cols, d % n_cols].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved: {save_path}')
    plt.close()


ALL_RTC_MODES = ['no_rtc', 'prefix', 'guidance', 'guidance_vjp']


def main():
    """Entry point for RTC end-to-end testing.

    Loads model and data from config, runs predict_action under
    selected RTC modes, and generates per-dimension denoising
    visualizations.

    Use ``--modes`` to choose which modes to run.
    """
    parser = argparse.ArgumentParser(description='End-to-end RTC test')
    parser.add_argument(
        '--config', type=str, required=True, help='Config file path')
    parser.add_argument(
        '--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument(
        '--prefix_len', type=int, default=5, help='RTC prefix length')
    parser.add_argument('--output_dir', type=str, default='work_dirs/rtc_test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--modes',
        nargs='+',
        default=ALL_RTC_MODES,
        choices=ALL_RTC_MODES,
        help='RTC modes to test (default: all). '
        'E.g. --modes no_rtc guidance guidance_vjp')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('Loading model...')
    model = load_model(cfg, args.checkpoint, device)

    print('Loading data...')
    batch = load_batch(cfg, device)

    gt_actions = batch['actions'].cpu()
    prefix_len = args.prefix_len

    # Simulate prev_actions: use GT actions
    # (as if the previous chunk predicted them correctly)
    prev_actions = gt_actions.to(device)

    modes = args.modes
    print(f'Model: {type(model).__name__}')
    print(f'Action shape: {gt_actions.shape}, prefix_len={prefix_len}')
    print(f'Modes: {modes}')

    # -- Mode definitions -----------------------------------------------------
    # Each mode: (display_name, predict_with_intermediates kwargs)
    mode_configs = {
        'no_rtc': ('No RTC', {}),
        'prefix': ('RTC prefix',
                   dict(
                       prefix_len=prefix_len,
                       prev_actions=prev_actions,
                       rtc_config={
                           'method': 'prefix',
                           'enabled': True
                       },
                   )),
        'guidance': ('RTC guidance',
                     dict(
                         prefix_len=prefix_len,
                         prev_actions=prev_actions,
                         rtc_config={
                             'method': 'guidance',
                             'enabled': True,
                             'max_guidance_weight': 5.0,
                             'schedule': 'exp',
                             'decay_end': prefix_len * 2,
                         },
                     )),
        'guidance_vjp': ('RTC guidance+VJP',
                         dict(
                             prefix_len=prefix_len,
                             prev_actions=prev_actions,
                             rtc_config={
                                 'method': 'guidance',
                                 'enabled': True,
                                 'max_guidance_weight': 5.0,
                                 'schedule': 'exp',
                                 'decay_end': prefix_len * 2,
                                 'use_vjp': True,
                             },
                         )),
    }

    # -- Run selected modes ---------------------------------------------------
    results = {}  # name -> (pred, intermediates, elapsed)
    for mode in modes:
        display_name, kwargs = mode_configs[mode]
        print(f'\n--- {display_name} ---')
        t0 = time_mod.time()
        pred, inter = predict_with_intermediates(model, batch, **kwargs)
        elapsed = time_mod.time() - t0
        print(f'  Time: {elapsed:.3f}s')  # noqa: E231
        results[mode] = (pred, inter, elapsed)

    # -- Timing summary -------------------------------------------------------
    t_base = results[modes[0]][2]  # first mode as baseline
    print('\n=== Timing ===')
    for mode in modes:
        t = results[mode][2]
        display = mode_configs[mode][0]
        ratio = ' ({:.1f}x)'.format(t / t_base) if mode != modes[0] else ''
        print('  {:20s}: {:.3f}s{}'.format(display, t, ratio))

    # -- Overlap L2 (prefix region) -------------------------------------------
    print('\n=== Overlap L2 (prefix region) ===')
    for mode in modes:
        p = results[mode][0]
        display = mode_configs[mode][0]
        overlap = p[:, :prefix_len].cpu() - gt_actions[:, :prefix_len]
        l2 = overlap.pow(2).mean().sqrt().item()
        print(f'  {display:20s}: L2={l2:.4f}')  # noqa: E231

    # -- Per-dimension denoising visualization --------------------------------
    for mode in modes:
        display = mode_configs[mode][0]
        use_prefix = (mode != 'no_rtc')
        plot_denoising_per_dim(
            results[mode][1],
            gt_actions,
            prefix_len=prefix_len if use_prefix else 0,
            prev_actions=prev_actions.cpu() if use_prefix else None,
            title_prefix=f'{display}: ',
            save_path=os.path.join(args.output_dir, f'denoise_{mode}.png'))

    # -- Comparison plot ------------------------------------------------------
    plot_comparison(
        results,
        mode_configs,
        gt_actions,
        prefix_len,
        save_path=os.path.join(args.output_dir,
                               f'rtc_comparison_prefix_len_{prefix_len}.png'))

    print(f'\nAll plots saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
