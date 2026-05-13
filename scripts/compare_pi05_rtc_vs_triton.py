#!/usr/bin/env python3
"""Compare regular PI0.5 RTC inference against Triton RTC inference."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from mmengine import Config
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare regular RTC vs Triton RTC for PI0.5')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument(
        '--ckpt-path',
        default=None,
        help='Optional explicit state_dict checkpoint override.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--prompt-len', type=int, default=32)
    parser.add_argument('--prefix-len', type=int, default=4)
    parser.add_argument('--state-dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefix-atol', type=float, default=1e-4)
    parser.add_argument('--suffix-max-atol', type=float, default=1e-2)
    parser.add_argument('--suffix-mean-atol', type=float, default=1e-3)
    return parser.parse_args()


def resolve_normal_cfg(cfg: Config):
    if not hasattr(cfg, 'model'):
        raise ValueError('Config must define cfg.model.')
    model_cfg = cfg.model.copy()
    if model_cfg.get('type') != 'PI05FlowMatching':
        raise ValueError(
            f'Expected cfg.model.type == PI05FlowMatching, got '
            f'{model_cfg.get("type")}')
    return model_cfg


def resolve_triton_cfg(cfg: Config):
    if not hasattr(cfg, 'inference_model'):
        raise ValueError('Config must define cfg.inference_model.')
    model_cfg = cfg.inference_model.copy()
    model_type = model_cfg.get('type')
    if model_type == 'PI05FlowMatchingInference':
        model_cfg['type'] = 'PI05FlowMatchingRTCInference'
        print('[compare] Switched inference_model.type to '
              'PI05FlowMatchingRTCInference')
    elif model_type != 'PI05FlowMatchingRTCInference':
        raise ValueError(
            f'Unsupported inference model type: {model_type}')
    return model_cfg


def resolve_checkpoint_path(ckpt_path: str | None) -> Path | None:
    if not ckpt_path:
        return None
    resolved = Path(ckpt_path)
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {resolved}')
    return resolved


def build_model(model_cfg,
                config_pretrained_path: Path | None,
                explicit_ckpt_path: Path | None,
                device: torch.device,
                label: str):
    from fluxvla.engines import build_vla_from_cfg

    build_cfg = model_cfg.copy()
    if explicit_ckpt_path is not None:
        build_cfg['pretrained_name_or_path'] = None

    print(f'[compare] Building {label} model ...')
    model = build_vla_from_cfg(build_cfg)

    if explicit_ckpt_path is not None:
        print(f'[compare] Loading explicit checkpoint for {label}: '
              f'{explicit_ckpt_path}')
        if explicit_ckpt_path.suffix == '.safetensors':
            checkpoint = load_file(str(explicit_ckpt_path), device='cpu')
        else:
            checkpoint = torch.load(str(explicit_ckpt_path), map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
    elif config_pretrained_path is not None:
        print(f'[compare] Loading config pretrained weights for {label} via '
              'model.from_pretrained()')
        model.from_pretrained()
    else:
        print(f'[compare] No checkpoint loading requested for {label}.')

    return model.to(device).eval()


def make_synthetic_inputs(model, num_views: int, args, device: torch.device):
    n_action_steps = int(model.n_action_steps)
    action_dim = int(model.max_action_dim)
    image_size = model.vision_backbone.vision.vision_model.config.image_size
    vocab_size = model.llm_backbone.config.vocab_size
    bos_token_id = getattr(model.llm_backbone.config, 'bos_token_id', 2)

    if args.prefix_len < 0 or args.prefix_len > n_action_steps:
        raise ValueError(
            f'prefix_len must be in [0, {n_action_steps}], '
            f'got {args.prefix_len}')

    torch.manual_seed(args.seed)
    images_views = torch.randn(
        1, num_views, 3, image_size, image_size, device=device)
    images_normal = images_views.flatten(1, 2).contiguous()
    images_triton = images_normal.clone()

    lang_tokens = torch.full(
        (1, args.prompt_len),
        fill_value=bos_token_id,
        dtype=torch.long,
        device=device)
    if args.prompt_len > 1:
        tail = torch.arange(
            1, args.prompt_len, dtype=torch.long, device=device) % vocab_size
        lang_tokens[0, 1:] = tail
    lang_masks = torch.ones(
        1, args.prompt_len, dtype=torch.bool, device=device)
    img_masks_normal = torch.ones(
        1, num_views, dtype=torch.bool, device=device)
    states = torch.zeros(1, args.state_dim, dtype=torch.float32, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    noise = torch.randn(
        1,
        n_action_steps,
        action_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator)

    return {
        'images_normal': images_normal,
        'images_triton': images_triton,
        'lang_tokens': lang_tokens,
        'lang_masks': lang_masks,
        'img_masks_normal': img_masks_normal,
        'states': states,
        'noise': noise,
        'n_action_steps': n_action_steps,
        'action_dim': action_dim,
    }


def run_predict(fn, dtype: torch.dtype, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        pred = fn(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return pred.detach().float().cpu(), elapsed


def move_batch_to_device(batch: dict, device: torch.device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def release_cuda_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_stats(a: torch.Tensor, b: torch.Tensor, prefix_len: int):
    diff = (a - b).abs()
    stats = {
        'all_max': diff.max().item(),
        'all_mean': diff.mean().item(),
    }
    if prefix_len > 0:
        prefix_diff = diff[:, :prefix_len, :]
        stats['prefix_max'] = prefix_diff.max().item()
        stats['prefix_mean'] = prefix_diff.mean().item()
    else:
        stats['prefix_max'] = 0.0
        stats['prefix_mean'] = 0.0

    if prefix_len < a.shape[1]:
        suffix_diff = diff[:, prefix_len:, :]
        stats['suffix_max'] = suffix_diff.max().item()
        stats['suffix_mean'] = suffix_diff.mean().item()
    else:
        stats['suffix_max'] = 0.0
        stats['suffix_mean'] = 0.0
    return stats


def print_stats(label: str, stats: dict):
    print(f'[compare] {label}:')
    print(f'  all_max={stats["all_max"]:.6f}')
    print(f'  all_mean={stats["all_mean"]:.6f}')
    print(f'  prefix_max={stats["prefix_max"]:.6f}')
    print(f'  prefix_mean={stats["prefix_mean"]:.6f}')
    print(f'  suffix_max={stats["suffix_max"]:.6f}')
    print(f'  suffix_mean={stats["suffix_mean"]:.6f}')


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    normal_cfg = resolve_normal_cfg(cfg)
    triton_cfg = resolve_triton_cfg(cfg)

    explicit_ckpt_path = resolve_checkpoint_path(args.ckpt_path)
    # normal_pretrained_path = resolve_checkpoint_path(
    #     normal_cfg.get('pretrained_name_or_path'))
    # triton_pretrained_path = resolve_checkpoint_path(
    #     triton_cfg.get('pretrained_name_or_path'))
    
    normal_pretrained_path = resolve_checkpoint_path(args.ckpt_path)
    triton_pretrained_path = resolve_checkpoint_path(args.ckpt_path)
    num_views = int(triton_cfg.get('num_view', 0))
    if num_views <= 0:
        raise ValueError(f'Invalid num_views from inference config: {num_views}')

    device = torch.device(args.device)
    dtype_map = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
    }
    dtype = dtype_map[args.dtype]

    cpu_device = torch.device('cpu')
    normal_model = build_model(normal_cfg, normal_pretrained_path,
                               explicit_ckpt_path, device, 'normal')
    batch_cpu = make_synthetic_inputs(normal_model, num_views, args, cpu_device)
    batch = move_batch_to_device(batch_cpu, device)

    normal_plain, normal_plain_t = run_predict(
        normal_model.predict_action,
        dtype,
        images=batch['images_normal'],
        lang_tokens=batch['lang_tokens'],
        states=batch['states'],
        img_masks=batch['img_masks_normal'],
        lang_masks=batch['lang_masks'],
        noise=batch['noise'].clone(),
        prev_actions=None,
        prefix_len=0,
        rtc_config=None,
    )

    prev_actions_cpu = normal_plain[:, :args.prefix_len, :].clone()
    prev_actions = prev_actions_cpu.to(
        device=device, dtype=torch.float32)
    rtc_config = {'enabled': True, 'method': 'prefix'}

    normal_rtc, normal_rtc_t = run_predict(
        normal_model.predict_action,
        dtype,
        images=batch['images_normal'],
        lang_tokens=batch['lang_tokens'],
        states=batch['states'],
        img_masks=batch['img_masks_normal'],
        lang_masks=batch['lang_masks'],
        noise=batch['noise'].clone(),
        prev_actions=prev_actions,
        prefix_len=args.prefix_len,
        rtc_config=rtc_config,
    )
    release_cuda_model(normal_model)
    del batch

    triton_model = build_model(triton_cfg, triton_pretrained_path,
                               explicit_ckpt_path, device, 'triton')
    batch = move_batch_to_device(batch_cpu, device)

    triton_plain, triton_plain_t = run_predict(
        triton_model.predict_action,
        dtype,
        images=batch['images_triton'],
        lang_tokens=batch['lang_tokens'],
        states=batch['states'],
        img_masks=None,
        lang_masks=batch['lang_masks'],
        noise=batch['noise'].clone(),
        prev_actions=None,
        prefix_len=0,
        rtc_config=None,
    )
    prev_actions = prev_actions_cpu.to(device=device, dtype=torch.float32)
    triton_rtc, triton_rtc_t = run_predict(
        triton_model.predict_action,
        dtype,
        images=batch['images_triton'],
        lang_tokens=batch['lang_tokens'],
        states=batch['states'],
        img_masks=None,
        lang_masks=batch['lang_masks'],
        noise=batch['noise'].clone(),
        prev_actions=prev_actions,
        prefix_len=args.prefix_len,
        rtc_config=rtc_config,
    )
    release_cuda_model(triton_model)

    prefix_ref = prev_actions.cpu()
    normal_prefix_lock = (
        normal_rtc[:, :args.prefix_len, :] - prefix_ref).abs().max().item()
    triton_prefix_lock = (
        triton_rtc[:, :args.prefix_len, :] - prefix_ref).abs().max().item()

    plain_stats = compute_stats(normal_plain, triton_plain, prefix_len=0)
    rtc_stats = compute_stats(normal_rtc, triton_rtc, args.prefix_len)

    print('[compare] normal plain shape:', tuple(normal_plain.shape))
    print('[compare] triton plain shape:', tuple(triton_plain.shape))
    print('[compare] normal rtc shape:', tuple(normal_rtc.shape))
    print('[compare] triton rtc shape:', tuple(triton_rtc.shape))
    print(f'[compare] normal plain time: {normal_plain_t * 1000:.1f} ms')
    print(f'[compare] triton plain time: {triton_plain_t * 1000:.1f} ms')
    print(f'[compare] normal rtc time: {normal_rtc_t * 1000:.1f} ms')
    print(f'[compare] triton rtc time: {triton_rtc_t * 1000:.1f} ms')
    print(f'[compare] normal prefix lock max_abs_diff: '
          f'{normal_prefix_lock:.6f}')
    print(f'[compare] triton prefix lock max_abs_diff: '
          f'{triton_prefix_lock:.6f}')
    print_stats('plain normal_vs_triton', plain_stats)
    print_stats('rtc normal_vs_triton', rtc_stats)

    if normal_prefix_lock > args.prefix_atol:
        raise SystemExit(
            f'FAIL: normal RTC prefix lock diff {normal_prefix_lock:.6f} '
            f'exceeds prefix_atol={args.prefix_atol}')
    if triton_prefix_lock > args.prefix_atol:
        raise SystemExit(
            f'FAIL: triton RTC prefix lock diff {triton_prefix_lock:.6f} '
            f'exceeds prefix_atol={args.prefix_atol}')
    if rtc_stats['suffix_max'] > args.suffix_max_atol:
        raise SystemExit(
            f'FAIL: suffix_max {rtc_stats["suffix_max"]:.6f} exceeds '
            f'suffix_max_atol={args.suffix_max_atol}')
    if rtc_stats['suffix_mean'] > args.suffix_mean_atol:
        raise SystemExit(
            f'FAIL: suffix_mean {rtc_stats["suffix_mean"]:.6f} exceeds '
            f'suffix_mean_atol={args.suffix_mean_atol}')

    print('PASS')


if __name__ == '__main__':
    main()
