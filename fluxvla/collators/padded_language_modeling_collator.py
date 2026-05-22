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

from typing import Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from fluxvla.engines import COLLATORS
from fluxvla.engines.utils import str_to_dtype


@COLLATORS.register_module()
class PaddedCollatorForLanguageModeling:
    """
    Collator for language modeling tasks. Pads input sequences
        and pixel values.

    Args:
        model_max_length (int): Maximum length of the input sequences.
        pad_token_id (int): ID of the padding token.
        default_image_resolution (Tuple[int, int, int]): Default image
            resolution (height, width, channels).
        padding_side (str): Side to pad on ("right" or "left").
        pixel_values_dtype (torch.dtype): Data type for pixel values.
        ignore_idx (int, optional): Index to ignore in the labels. Defaults
            to -100.
    """

    def __init__(self,
                 model_max_length: int,
                 pad_token_id: int,
                 default_image_resolution: Tuple[int, int, int],
                 padding_side: str = 'right',
                 pixel_values_dtype: str = 'fp32',
                 ignore_idx: int = -100) -> None:
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.default_image_resolution = default_image_resolution
        self.padding_side = padding_side
        self.pixel_values_dtype = str_to_dtype(pixel_values_dtype)
        self.ignore_idx = ignore_idx

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(
            self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(
        self,
        instances: Sequence[Dict[str,
                                 torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function to pad input sequences and pixel values.

        Args:
            instances (Sequence[Dict[str, torch.Tensor]]): List of instances
                to collate.

        Raises:
            ValueError: If the pixel values are not of type `torch.Tensor` or
                `Dict[str, torch.Tensor]`.

        Returns:
            Dict[str, torch.Tensor]: Collated batch of data.
        """
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('input_ids', 'labels'))
        pixel_values = [instance['pixel_values'] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"`
        # during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_idx)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, :self.
                                      model_max_length], labels[:, :self.
                                                                model_max_length]  # noqa: E501

        # Get `lang_masks` by checking for `pad_token_id`
        lang_masks = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of
        # `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor([
            idx for idx in range(len(pixel_values))
            if pixel_values[idx] is not None
        ],
                                          dtype=torch.long)

        # Stack all `pixel_values` --> depending on type (torch.Tensor,
        # or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack(
                [self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]],
                        torch.Tensor):
            pixel_values = torch.stack([
                pixel_values[idx]
                if idx in multimodal_indices else self.dummy_pixel_values
                for idx in range(len(input_ids))
            ])
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack([
                    pixel_values[idx][k]
                    if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ])
                for k in pv_example
            }
        else:
            raise ValueError(
                f'Unsupported `pixel_values` type = {type(pixel_values)}')

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            lang_masks=lang_masks,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )
