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

from typing import Dict, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from fluxvla.engines import COLLATORS
from fluxvla.engines.utils import str_to_dtype


@COLLATORS.register_module()
class PaddedCollatorForActionPrediction:
    """
    Collator for action prediction tasks. Pads input
        sequences and pixel values.

    Args:
        model_max_length (int): Maximum length of the input sequences.
        pad_token_id (int): ID of the padding token.
        padding_side (str): Side to pad on ("right" or "left").
        pixel_values_dtype (torch.dtype): Data type for pixel values.
        ignore_idx (int, optional): Index to ignore in the labels. Defaults
            to -100.
    """

    def __init__(self,
                 model_max_length: int,
                 pad_token_id: int,
                 padding_side: str = 'right',
                 pixel_values_dtype: str = 'fp32',
                 ignore_idx: int = -100) -> None:
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.pixel_values_dtype = str_to_dtype(pixel_values_dtype)
        self.ignore_idx = ignore_idx

    def __call__(
        self,
        instances: Sequence[Dict[str,
                                 torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function to pad input sequences and pixel values.

        Args:
            instances (Sequence[Dict[str, torch.Tensor]]): List of instances

        Raises:
            AssertionError: If the tokenizer's padding side is not "right".

        Returns:
            Dict[str, torch.Tensor]: Collated batch of data.
        """
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('lang_tokens', 'labels'))

        input_ids = [
            torch.tensor(id_arr) if isinstance(id_arr, np.ndarray) else id_arr
            for id_arr in input_ids
        ]
        labels = [
            torch.tensor(label_arr)
            if isinstance(label_arr, np.ndarray) else label_arr
            for label_arr in labels
        ]
        images = [instance['images'] for instance in instances]
        if 'dataset_name' in instances[0]:
            dataset_names = [
                instance['dataset_name'] for instance in instances
            ]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with
        # `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == 'right', \
            f'Invalid Tokenizer padding_side={self.padding_side!r}'
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_idx)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, :self.
                                      model_max_length], labels[:, :self.
                                                                model_max_length]  # noqa: E501

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in images
                    ]), 'Invalid VLA Example with `images = None`!'

        # Stack all `images` --> depending on type is
        # torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
        elif isinstance(images[0], np.ndarray):
            images = torch.stack(
                [torch.from_numpy(img_arr) for img_arr in images])
        elif isinstance(images[0], dict):
            images = {
                k:
                torch.stack([images[idx][k] for idx in range(len(input_ids))])
                for k in images[0]
            }
        else:
            raise ValueError(f'Unsupported pixel_values type = {type(images)}')

        output = dict(
            images=images,
            lang_tokens=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output['dataset_names'] = dataset_names
        return output
