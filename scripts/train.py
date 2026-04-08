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
import json
import os

import draccus
import torch
import yaml
from mmengine import Config, DictAction

from fluxvla.datasets.utils import (save_dataset_statistics,
                                    save_grouped_dataset_statistics)
from fluxvla.engines import (build_dataset_from_cfg, build_runner_from_cfg,
                             initialize_overwatch)

overwatch = initialize_overwatch(__name__)


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


def train(args, cfg):
    """Train the model with the given configuration.

    Args:
        cfg (Config): The configuration object containing training settings.
    """
    os.makedirs(args.work_dir, exist_ok=True)
    dataset = build_dataset_from_cfg(cfg.train_dataloader.dataset)
    if overwatch.is_rank_zero() and hasattr(dataset, 'dataset_statistics'):
        save_dataset_statistics(dataset.dataset_statistics, args.work_dir)
        draccus.dump(cfg.to_dict(),
                     open(os.path.join(args.work_dir, 'config.yaml'), 'w'))
        with open(os.path.join(args.work_dir, 'config.yaml'),
                  'r') as f_yaml, open(
                      os.path.join(args.work_dir, 'config.json'),
                      'w') as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
    elif overwatch.is_rank_zero() and hasattr(dataset,
                                              'grouped_dataset_statistics'):
        # Handle grouped dataset statistics
        save_grouped_dataset_statistics(dataset.grouped_dataset_statistics,
                                        args.work_dir)
        draccus.dump(cfg.to_dict(),
                     open(os.path.join(args.work_dir, 'config.yaml'), 'w'))
        with open(os.path.join(args.work_dir, 'config.yaml'),
                  'r') as f_yaml, open(
                      os.path.join(args.work_dir, 'config.json'),
                      'w') as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
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
        torch.cuda.empty_cache()
        overwatch.info('Evaluation after training is enabled.')
        assert 'eval' in cfg, 'Evaluation configuration is missing.'
        overwatch.info('Running Evaluation')
        cfg.eval.cfg = cfg
        cfg.eval.ckpt_path = ckpt_path
        if hasattr(cfg.eval, 'processor') and not hasattr(
                cfg.eval.processor, 'model_path'):
            cfg.eval.processor.model_path = ckpt_path
        overwatch.info('Running Evaluation')
        eval_runner = build_runner_from_cfg(cfg.eval)
        eval_runner.run_setup()
        eval_runner.run()
        overwatch.info('Evaluation completed.')


if __name__ == '__main__':
    args, _ = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    train(args, cfg)
