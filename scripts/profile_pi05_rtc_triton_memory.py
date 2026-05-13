#!/usr/bin/env python3
"""Profile single PI0.5 RTC Triton inference memory without normal baseline."""

from __future__ import annotations

import argparse
import subprocess
import threading
import time
from pathlib import Path

import torch
from mmengine import Config
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-path', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--prompt-len', type=int, default=32)
    parser.add_argument('--prefix-len', type=int, default=5)
    parser.add_argument('--state-dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-interval', type=float, default=0.2)
    return parser.parse_args()


def read_used_mib(gpu_index: int = 0) -> int:
    out = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits', '-i', str(gpu_index)
    ], text=True)
    return int(out.strip().splitlines()[0].strip())


class MemorySampler:

    def __init__(self, interval: float = 0.2, gpu_index: int = 0):
        self.interval = interval
        self.gpu_index = gpu_index
        self.values = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.values.append((time.time(), read_used_mib(self.gpu_index)))
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def baseline(self):
        return self.values[0][1] if self.values else 0

    @property
    def peak(self):
        return max(v for _, v in self.values) if self.values else 0


def resolve_checkpoint_path(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def build_model(cfg, ckpt_path):
    from fluxvla.engines import build_vla_from_cfg

    model_cfg = cfg.inference_model.copy()
    if model_cfg.get('type') != 'PI05FlowMatchingRTCInference':
        raise ValueError(
            f'Expected PI05FlowMatchingRTCInference, got {model_cfg.get("type")}')

    explicit_ckpt = resolve_checkpoint_path(ckpt_path)
    build_cfg = model_cfg.copy()
    if explicit_ckpt is not None:
        build_cfg['pretrained_name_or_path'] = None

    print('[profile] Building Triton model on CPU ...')
    model = build_vla_from_cfg(build_cfg).eval()

    if explicit_ckpt is not None:
        print(f'[profile] Loading explicit checkpoint: {explicit_ckpt}')
        if explicit_ckpt.suffix == '.safetensors':
            checkpoint = load_file(str(explicit_ckpt), device='cpu')
        else:
            checkpoint = torch.load(str(explicit_ckpt), map_location='cpu')
        state_dict = (checkpoint['model'] if isinstance(checkpoint, dict)
                      and 'model' in checkpoint else checkpoint)
        model.load_state_dict(state_dict, strict=True)
    elif model_cfg.get('pretrained_name_or_path'):
        print('[profile] Loading config pretrained weights on CPU ...')
        model.from_pretrained()
    return model


def make_inputs(model, args, device):
    torch.manual_seed(args.seed)
    image_size = model.vision_backbone.vision.vision_model.config.image_size
    vocab_size = model.llm_backbone.config.vocab_size
    bos_token_id = getattr(model.llm_backbone.config, 'bos_token_id', 2)
    action_dim = int(model.max_action_dim)
    n_action_steps = int(model.n_action_steps)
    num_views = int(model.num_views)

    lang_tokens = torch.full(
        (1, args.prompt_len), bos_token_id, dtype=torch.long, device=device)
    if args.prompt_len > 1:
        lang_tokens[0, 1:] = (
            torch.arange(1, args.prompt_len, dtype=torch.long, device=device)
            % vocab_size)
    lang_masks = torch.ones(1, args.prompt_len, dtype=torch.long, device=device)
    images = torch.randn(1, num_views * 3, image_size, image_size, device=device)
    states = torch.zeros(1, args.state_dim, dtype=torch.float32, device=device)
    noise = torch.randn(
        1, n_action_steps, action_dim, dtype=torch.bfloat16, device=device)
    prev_actions = torch.randn(
        1, n_action_steps, action_dim, dtype=torch.float32, device=device)
    return dict(
        images=images,
        lang_tokens=lang_tokens,
        states=states,
        lang_masks=lang_masks,
        noise=noise,
        prev_actions=prev_actions,
        prefix_len=args.prefix_len,
        rtc_config={'enabled': True, 'method': 'prefix'},
    )


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)
    dtype = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
    }[args.dtype]

    baseline_before = read_used_mib(0)
    with MemorySampler(args.sample_interval, 0) as sampler:
        model = build_model(cfg, args.ckpt_path)
        inputs = make_inputs(model, args, device)
        print('[profile] Running first RTC Triton predict ...')
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
            pred = model.predict_action(**inputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print('[profile] Running replay RTC Triton predict ...')
        t1 = time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
            pred2 = model.predict_action(**inputs)
        torch.cuda.synchronize()
        replay_elapsed = time.perf_counter() - t1

    prefix_diff = (
        pred[:, :args.prefix_len] - inputs['prev_actions'][:, :args.prefix_len]
    ).abs().max().item()
    print(f'[profile] pred_shape={tuple(pred.shape)}')
    print(f'[profile] first_predict_ms={elapsed * 1000:.1f}')
    print(f'[profile] replay_predict_ms={replay_elapsed * 1000:.1f}')
    print(f'[profile] prefix_lock_max_abs_diff={prefix_diff:.6f}')
    print(f'[profile] second_pred_shape={tuple(pred2.shape)}')
    print(f'[mem] baseline_before_mib={baseline_before}')
    print(f'[mem] sampled_baseline_mib={sampler.baseline}')
    print(f'[mem] peak_mib={sampler.peak}')
    print(f'[mem] delta_from_process_start_mib={sampler.peak - sampler.baseline}')
    print(f'[mem] delta_from_preexisting_mib={sampler.peak - baseline_before}')
    print(f'[mem] samples={len(sampler.values)}')


if __name__ == '__main__':
    main()
