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
from typing import Any, cast

import torch
from mmengine import Config, DictAction
from torch.utils.data import DataLoader, Dataset

from fluxvla.engines import (build_collator_from_cfg, build_dataset_from_cfg,
                             build_vla_from_cfg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Infer SARM progress over a LeRobot v2.1 or v3.x dataset.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--head-mode', type=str, default='sparse')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, '
        'the key-value pair in xxx=yyy format')
    return parser.parse_args()


def main():
    args = parse_args()
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
    model.eval().cuda()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator)

    results = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            progress = model.predict_progress(
                images=batch['images'],
                text_input_ids=batch['text_input_ids'],
                text_attention_mask=batch['text_attention_mask'],
                states=batch['states'],
                lengths=batch['lengths'],
                head_mode=args.head_mode,
            ).detach().cpu().tolist()
            for idx, pred in enumerate(progress):
                results.append({
                    'episode_index':
                    int(batch['episode_index'][idx]),
                    'current_index':
                    int(batch['current_index'][idx]),
                    'task_description':
                    batch['task_description'][idx],
                    'pred_progress':
                    float(pred),
                })

    with open(args.output_path, 'w', encoding='utf-8') as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
