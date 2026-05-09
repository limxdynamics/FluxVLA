#!/usr/bin/env python3
"""Minimal local RTC verification for PI0.5 Triton inference.

Builds the inference model from config, loads a checkpoint, runs one plain
prediction and two RTC prefix-conditioned predictions with synthetic inputs,
then checks whether the prefix region is preserved.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from mmengine import Config
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Minimal local RTC verification for PI0.5 inference')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument(
        '--ckpt-path',
        default=None,
        help='Optional checkpoint override. Defaults to config value.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--prompt-len', type=int, default=32)
    parser.add_argument('--prefix-len', type=int, default=4)
    parser.add_argument('--state-dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--atol', type=float, default=1e-2)
    parser.add_argument(
        '--no-rtc-override',
        action='store_true',
        help='Do not auto-switch PI05FlowMatchingInference -> RTC class.')
    return parser.parse_args()


def resolve_model_cfg(cfg: Config, allow_rtc_override: bool):
    if not hasattr(cfg, 'inference_model'):
        raise ValueError('Config must define inference_model for this script.')

    model_cfg = cfg.inference_model.copy()
    model_type = model_cfg.get('type')
    if model_type == 'PI05FlowMatchingInference' and allow_rtc_override:
        model_cfg['type'] = 'PI05FlowMatchingRTCInference'
        print('[verify] Switched inference_model.type to '
              'PI05FlowMatchingRTCInference')
    elif model_type != 'PI05FlowMatchingRTCInference':
        raise ValueError(
            f'Unsupported inference model type for RTC verification: '
            f'{model_type}')
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
                device: torch.device):
    from fluxvla.engines import build_vla_from_cfg

    build_cfg = model_cfg.copy()
    if explicit_ckpt_path is not None:
        build_cfg['pretrained_name_or_path'] = None

    print('[verify] Building model ...')
    model = build_vla_from_cfg(build_cfg)

    if explicit_ckpt_path is not None:
        print(f'[verify] Loading explicit checkpoint: {explicit_ckpt_path}')
        if explicit_ckpt_path.suffix == '.safetensors':
            checkpoint = load_file(str(explicit_ckpt_path), device='cpu')
        else:
            checkpoint = torch.load(str(explicit_ckpt_path), map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        load_result = model.load_state_dict(state_dict, strict=True)
        if load_result is not None:
            print(f'[verify] load_state_dict result: {load_result}')
    elif config_pretrained_path is not None:
        print('[verify] Loading config pretrained weights via '
              'model.from_pretrained()')
        model.from_pretrained()
    else:
        print('[verify] No checkpoint loading requested.')

    model = model.to(device).eval()
    return model


def make_synthetic_inputs(model, args, device: torch.device):
    num_views = getattr(model, 'num_views', None)
    if num_views is None:
        raise ValueError('Model is missing num_views.')
    n_action_steps = int(model.n_action_steps)
    action_dim = int(model.max_action_dim)

    if args.prefix_len < 0 or args.prefix_len > n_action_steps:
        raise ValueError(
            f'prefix_len must be in [0, {n_action_steps}], '
            f'got {args.prefix_len}')

    image_size = model.vision_backbone.vision.vision_model.config.image_size
    vocab_size = model.llm_backbone.config.vocab_size
    bos_token_id = getattr(model.llm_backbone.config, 'bos_token_id', 2)

    torch.manual_seed(args.seed)
    images = torch.randn(
        1, num_views * 3, image_size, image_size, device=device)
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
        1, args.prompt_len, dtype=torch.long, device=device)
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
        'images': images,
        'lang_tokens': lang_tokens,
        'lang_masks': lang_masks,
        'states': states,
        'noise': noise,
        'n_action_steps': n_action_steps,
        'action_dim': action_dim,
    }


def run_predict(model, dtype: torch.dtype, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        pred = model.predict_action(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return pred.detach().float().cpu(), elapsed


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model_cfg = resolve_model_cfg(cfg, allow_rtc_override=not args.no_rtc_override)
    config_pretrained_path = resolve_checkpoint_path(
        model_cfg.get('pretrained_name_or_path'))
    explicit_ckpt_path = resolve_checkpoint_path(args.ckpt_path)
    device = torch.device(args.device)
    dtype_map = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
    }
    dtype = dtype_map[args.dtype]

    model = build_model(model_cfg, config_pretrained_path, explicit_ckpt_path,
                        device)
    batch = make_synthetic_inputs(model, args, device)

    base_kwargs = dict(
        images=batch['images'],
        lang_tokens=batch['lang_tokens'],
        states=batch['states'],
        img_masks=None,
        lang_masks=batch['lang_masks'],
    )

    plain_pred, plain_time = run_predict(
        model,
        dtype,
        **base_kwargs,
        noise=batch['noise'].clone(),
        prev_actions=None,
        prefix_len=0,
        rtc_config=None,
    )

    if args.prefix_len > 0:
        prev_actions = plain_pred[:, :args.prefix_len, :].to(
            device=device, dtype=torch.float32)
        rtc_config = {'enabled': True, 'method': 'prefix'}
    else:
        prev_actions = None
        rtc_config = None

    rtc_pred_1, rtc_time_1 = run_predict(
        model,
        dtype,
        **base_kwargs,
        noise=batch['noise'].clone(),
        prev_actions=prev_actions,
        prefix_len=args.prefix_len,
        rtc_config=rtc_config,
    )
    rtc_pred_2, rtc_time_2 = run_predict(
        model,
        dtype,
        **base_kwargs,
        noise=batch['noise'].clone(),
        prev_actions=prev_actions,
        prefix_len=args.prefix_len,
        rtc_config=rtc_config,
    )

    print('[verify] plain output shape:', tuple(plain_pred.shape))
    print('[verify] rtc output shape:', tuple(rtc_pred_1.shape))
    print(f'[verify] plain time: {plain_time * 1000:.1f} ms')
    print(f'[verify] rtc time (first replay): {rtc_time_1 * 1000:.1f} ms')
    print(f'[verify] rtc time (second replay): {rtc_time_2 * 1000:.1f} ms')

    if args.prefix_len > 0:
        prefix_ref = prev_actions.cpu()
        prefix_diff_1 = (
            rtc_pred_1[:, :args.prefix_len, :] - prefix_ref).abs().max().item()
        prefix_diff_2 = (
            rtc_pred_2[:, :args.prefix_len, :] - prefix_ref).abs().max().item()
        print(f'[verify] prefix max_abs_diff #1: {prefix_diff_1:.6f}')
        print(f'[verify] prefix max_abs_diff #2: {prefix_diff_2:.6f}')
        if prefix_diff_1 > args.atol or prefix_diff_2 > args.atol:
            raise SystemExit(
                f'FAIL: prefix diff exceeds atol={args.atol}. '
                f'Got {prefix_diff_1:.6f}, {prefix_diff_2:.6f}')

    print('PASS')


if __name__ == '__main__':
    main()
