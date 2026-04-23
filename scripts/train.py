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
import gc
import json
import os
import sys

import draccus
import torch
import torch.distributed as dist
import yaml
from mmengine import Config, DictAction

from fluxvla.datasets.utils import (save_dataset_statistics,
                                    save_grouped_dataset_statistics)
from fluxvla.engines import (build_dataset_from_cfg, build_runner_from_cfg,
                             initialize_overwatch)

overwatch = initialize_overwatch(__name__)

_DISTRIBUTED_ENV_KEYS = (
    'GROUP_RANK',
    'GROUP_WORLD_SIZE',
    'LOCAL_RANK',
    'LOCAL_WORLD_SIZE',
    'MASTER_ADDR',
    'MASTER_PORT',
    'NODE_RANK',
    'RANK',
    'ROLE_NAME',
    'ROLE_RANK',
    'ROLE_WORLD_SIZE',
    'TORCHELASTIC_ERROR_FILE',
    'TORCHELASTIC_MAX_RESTARTS',
    'TORCHELASTIC_RESTART_COUNT',
    'TORCHELASTIC_RUN_ID',
    'WORLD_SIZE',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model with the given configuration.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the configuration file.',
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='The directory to save logs and checkpoints.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=  # noqa: E251
        'override some settings in the used config, the key-value pair in xxx=yyy format'  # noqa: E501
    )
    parser.add_argument(
        '--eval-after-train',
        action='store_true',
        help=  # noqa: E251
        'Whether to run evaluation after training. If set, the evaluation will be performed using the latest checkpoint.'  # noqa: E501
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to the checkpoint file to resume training from.',
    )
    args, unknown = parser.parse_known_args()
    return args, unknown


def _sync_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _clear_cuda_memory():
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, 'ipc_collect', None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except RuntimeError:
                pass


def _destroy_distributed_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _save_resolved_config(cfg, work_dir):
    config_yaml_path = os.path.join(work_dir, 'config.yaml')
    config_json_path = os.path.join(work_dir, 'config.json')

    with open(config_yaml_path, 'w') as f_yaml:
        draccus.dump(cfg.to_dict(), f_yaml)

    with open(config_yaml_path, 'r') as f_yaml, open(config_json_path,
                                                     'w') as f_json:
        yaml_cfg = yaml.safe_load(f_yaml)
        json.dump(yaml_cfg, f_json, indent=2)


def _resolve_eval_config_path(args):
    for candidate in (os.path.join(args.work_dir, 'config.json'),
                      os.path.join(args.work_dir, 'config.yaml'), args.config):
        if candidate is not None and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(args.config)


def _get_eval_relaunch_env():
    env = os.environ.copy()
    for key in _DISTRIBUTED_ENV_KEYS:
        env.pop(key, None)
    return env


def _relaunch_eval_in_fresh_process(args, eval_ckpt_path):
    """Replace the current process with a clean single-worker eval launch."""
    eval_script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'eval.py'))
    eval_config_path = _resolve_eval_config_path(args)
    eval_argv = [
        sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nnodes', '1', '--nproc-per-node', '1', eval_script_path, '--config',
        eval_config_path, '--ckpt-path',
        os.path.abspath(eval_ckpt_path)
    ]

    overwatch.info(
        'Re-launching evaluation in a fresh single-worker process to '
        'fully release training CUDA/FSDP state before eval.')
    os.execvpe(sys.executable, eval_argv, _get_eval_relaunch_env())


def _release_training_resources(runner, dataset):
    """Release training state before building evaluation objects."""
    overwatch.info('Cleaning up training resources before evaluation.')
    _sync_distributed()

    if runner is not None and hasattr(runner, 'cleanup'):
        runner.cleanup()

    if dataset is not None:
        for method_name in ('cleanup', 'close'):
            method = getattr(dataset, method_name, None)
            if callable(method):
                method()
                break

    runner = None
    dataset = None
    # Two rounds of gc to break any residual reference cycles between the
    # runner, FSDP handles, optimizer state and the config object.
    _clear_cuda_memory()

    _sync_distributed()
    return runner, dataset


def _resolve_eval_ckpt_path(ckpt_path):
    """Prefer the .safetensors sibling of a .pt checkpoint for evaluation.

    The .pt file produced during training contains the full optimizer state
    (AdamW fp32 momentum + variance) and scheduler state, which are not
    needed for inference. When eval-after-train spawns the eval runner on
    every rank, each rank would otherwise ``torch.load`` the entire .pt on
    CPU. On multi-GPU single-node setups this can easily exhaust system RAM
    and trigger the OOM killer (SIGKILL / exit code -9). The matching
    ``.safetensors`` file only stores model weights and is dramatically
    cheaper to load.
    """
    if ckpt_path is None:
        return ckpt_path
    if ckpt_path.endswith('.safetensors'):
        return ckpt_path
    if ckpt_path.endswith('.pt'):
        sf_candidate = ckpt_path[:-len('.pt')] + '.safetensors'
        if os.path.exists(sf_candidate):
            overwatch.info(
                f'Using safetensors checkpoint for evaluation: {sf_candidate} '
                f'(replaces {ckpt_path} to avoid loading optimizer state on '
                f'every rank).')
            return sf_candidate
    return ckpt_path


def train(args, cfg):
    """Train the model with the given configuration.

    Args:
        cfg (Config): The configuration object containing training settings.
    """
    os.makedirs(args.work_dir, exist_ok=True)
    dataset = build_dataset_from_cfg(cfg.train_dataloader.dataset)
    if overwatch.is_rank_zero() and hasattr(dataset, 'dataset_statistics'):
        save_dataset_statistics(dataset.dataset_statistics, args.work_dir)
    elif overwatch.is_rank_zero() and hasattr(dataset,
                                              'grouped_dataset_statistics'):
        # Handle grouped dataset statistics
        save_grouped_dataset_statistics(dataset.grouped_dataset_statistics,
                                        args.work_dir)
    if overwatch.is_rank_zero():
        _save_resolved_config(cfg, args.work_dir)
    if overwatch.is_rank_zero() and hasattr(
            dataset, 'batch_transform') and hasattr(
                dataset.batch_transform, 'base_tokenizer'):  # noqa: E501
        tokenizer = dataset.batch_transform.base_tokenizer
        tokenizer.save_pretrained(args.work_dir)
        overwatch.info(f'Saved tokenizer to {args.work_dir}')
    if hasattr(cfg.runner, 'metric'):
        cfg.runner.metric.run_dir = args.work_dir
    cfg.runner.cfg = cfg
    cfg.runner.args = args
    if args.resume_from is not None:
        cfg.runner.resume_from = args.resume_from
    runner = build_runner_from_cfg(cfg.runner)  # noqa: F841
    runner.run_setup(n_train_examples=len(dataset))
    ckpt_path = runner.run(dataset)
    if args.eval_after_train:
        if not hasattr(cfg, 'eval'):
            overwatch.warning(
                'No evaluation configuration found. Skipping evaluation.')
            return
        runner, dataset = _release_training_resources(runner, dataset)
        overwatch.info('Evaluation after training is enabled.')
        eval_ckpt_path = _resolve_eval_ckpt_path(ckpt_path)
        is_rank_zero = overwatch.is_rank_zero()
        _destroy_distributed_process_group()
        _clear_cuda_memory()
        if not is_rank_zero:
            return
        assert eval_ckpt_path is not None, (
            'Evaluation checkpoint path is missing after training.')
        _relaunch_eval_in_fresh_process(args, eval_ckpt_path)


if __name__ == '__main__':
    args, _ = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    train(args, cfg)
