# Origin: Modified from
# Upstream-Repo: openvla/openvla
# Upstream-Path: prismatic/vla/datasets/rlds/obs_transforms.py
# Upstream-Ref: c8f03f48af692657d3060c19588038c7220e9af9
# SPDX-License-Identifier: MIT
# Notes: Attribution normalized; no functional change.

from typing import Any, Dict, List, Union

import torch
from mmengine.config import Config, ConfigDict
from PIL import Image

from fluxvla.engines import (TRANSFORMS, build_tokenizer_from_cfg,
                             build_transform_from_cfg)


@TRANSFORMS.register_module()
class RLDSBatchTransform:

    def __init__(self,
                 prompter: Union[dict, ConfigDict, Config],
                 img_transform: Dict,
                 load_camera_views: List[str] = ['image_primary'],
                 action_tokenizer: Union[dict, ConfigDict, Config] = None,
                 base_tokenizer: Union[dict, ConfigDict, Config] = None,
                 predict_stop_token: bool = True,
                 ignore_index: int = -100,
                 with_labels: bool = True,
                 max_len: int = None,
                 pad_token_id: int = 0) -> None:
        """
        RLDSBatchTransform is a callable that transforms a batch of RLDS data
        into the format expected by the OpenVLA collator/models.

        Args:
            action_tokenizer (ActionTokenizer): Tokenizer for actions.
            base_tokenizer (PreTrainedTokenizerBase): Base tokenizer for text.
            prompter (PromptBuilder): Prompt building utility.
            img_transform (Dict): Image transform or HF processor name.
            predict_stop_token (bool): Whether to predict stop token.
            ignore_index (int): Index to ignore in loss computation.
        """
        if action_tokenizer is None:
            self.action_tokenizer = None
        else:
            self.action_tokenizer = build_tokenizer_from_cfg(action_tokenizer)
        if base_tokenizer is None:
            self.base_tokenizer = None
        else:
            self.base_tokenizer = build_tokenizer_from_cfg(base_tokenizer)

        self.image_transform = build_transform_from_cfg(img_transform)
        self.load_camera_views = load_camera_views

        self.prompter = build_transform_from_cfg(prompter)
        self.predict_stop_token = predict_stop_token
        self.ignore_index = ignore_index
        self.max_len = max_len
        self.with_labels = with_labels
        self.pad_token_id = pad_token_id

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a RLDS batch to the format expected by OpenVLA models.

        Args:
            rlds_batch (Dict[str, Any]): A dictionary containing the RLDS
                batch data. Expected keys include:
                    - "dataset_name": Name of the dataset.
                    - "action": Action tokens.
                    - "observation": Contains "image_primary".
                    - "task": Contains "language_instruction".

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "images": Processed images as tensors.
                - "lang_tokens": Tokenized language instruction.
                - "lang_masks": Masks for the language tokens.
                - "dataset_name": Name of the dataset.
                - "actions": Action tokens as tensors.
                - "img_masks": Masks for the images.
                - "states" (optional): Proprioceptive states if available.
                - "labels" (optional): Labels for the actions.

        Notes:
            - Prompts are constructed from the language instruction and action.
            - Labels are masked to ignore non-action tokens in loss.
        """
        self.prompter.clear()
        dataset_name = rlds_batch['dataset_name']
        if 'proprio' in rlds_batch['observation']:
            states = rlds_batch['observation']['proprio']
        else:
            states = None
        action = rlds_batch['action']
        imgs = list()
        for view in self.load_camera_views:
            if view in rlds_batch['observation']:
                imgs.append(
                    Image.fromarray(rlds_batch['observation'][view][0]))
            else:
                raise ValueError(
                    f"View '{view}' not found in observation images.")
        lang = (rlds_batch['task']['language_instruction'].decode().lower())

        conversation = [
            {
                'from': 'human',
                'value': f'What action should the robot take to {lang}?'
            },
        ]
        if self.action_tokenizer is not None:
            conversation.append({
                'from': 'gpt',
                'value': self.action_tokenizer(action[0])
            })
        for turn in conversation:
            self.prompter.add_turn(turn['from'], turn['value'])

        # Tokenize prompt using base tokenizer
        tokens = self.base_tokenizer(
            self.prompter.get_prompt(), add_special_tokens=True).input_ids
        token_mask = [True] * len(tokens)
        tokens_len = len(tokens)
        labels = list(tokens)
        if self.max_len is not None:
            if tokens_len < self.max_len:
                token_padding = [self.pad_token_id] * (
                    self.max_len - tokens_len)
                mask_padding = [False] * (self.max_len - tokens_len)
                tokens = tokens + token_padding
                token_mask = token_mask + mask_padding
            else:
                tokens = tokens[:self.max_len]
                token_mask = token_mask[:self.max_len]
        # Image to tensor
        tokens = torch.tensor(tokens)
        labels = torch.tensor(labels)
        image_out = self.image_transform(dict(pixel_values=imgs))

        if isinstance(image_out, tuple):
            pixel_values, image_grid_thw = image_out
        elif isinstance(image_out, dict):
            pixel_values = image_out.get('pixel_values', None)
            image_grid_thw = image_out.get('image_grid_thw', None)
        else:
            pixel_values = image_out
            image_grid_thw = None

        # Mask non-action tokens in loss
        labels[:-(len(action[0]) + 1)] = self.ignore_index
        if not self.predict_stop_token:
            labels[-1] = self.ignore_index
        img_masks = torch.tensor([True, True])
        ret_dict = dict(
            images=pixel_values,
            lang_tokens=tokens,
            lang_masks=torch.tensor(token_mask),
            dataset_name=dataset_name,
            actions=torch.from_numpy(action),
            img_masks=img_masks)
        if image_grid_thw is not None:
            ret_dict['image_grid_thw'] = image_grid_thw
        if states is not None:
            ret_dict['states'] = torch.from_numpy(states[0])

        if self.with_labels:
            ret_dict['labels'] = labels
        return ret_dict
