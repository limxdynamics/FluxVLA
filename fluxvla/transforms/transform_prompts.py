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

import os
from typing import Dict, List, Optional

import numpy as np

from fluxvla.engines import TRANSFORMS


@TRANSFORMS.register_module()
class ProcessPrompts():
    """Process and tokenize prompts for language models.
    This class handles the tokenization of prompts using a specified
    tokenizer, ensuring that the tokenized output adheres to a maximum
    length. It can also optionally prepare labels for language modeling
    tasks by creating a copy of the tokenized input.

    Args:
        tokenizer (Dict): Configuration for building the tokenizer.
        max_len (int, optional): Maximum length for tokenized prompts.
            Defaults to 180.
        with_labels (bool, optional): Whether to include labels in
            the output. Defaults to False.
    """

    def __init__(self,
                 tokenizer: Dict,
                 model_path: str = None,
                 max_len: int = 180,
                 with_labels: bool = False,
                 with_state: bool = False,
                 ignore_index: int = -100):
        from fluxvla.engines import build_tokenizer_from_cfg
        if model_path is not None:
            tokenizer['model_path'] = os.path.join(model_path, 'tokenizer')
        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.max_len = max_len
        self.with_labels = with_labels
        self.with_state = with_state
        self.ignore_index = ignore_index

    def __call__(self, inputs):
        """Tokenize and process the prompt in the input data.
        The method tokenizes the 'prompt' field in the input dictionary,
        applies padding or truncation to meet the maximum length requirement,
        and optionally creates a 'labels' field for language modeling tasks.

        Args:
            inputs (Dict): Input data containing a 'prompt' key.
        """
        assert 'prompt' in inputs, "Data must contain 'prompt' key."
        if self.with_state:
            assert 'state' in inputs, "Data must contain 'state' key."
            state = inputs['state']
            tokens = self.tokenizer(
                inputs['prompt'], state=state,
                add_special_tokens=True)['input_ids']
        else:
            tokens = self.tokenizer(
                inputs['prompt'], add_special_tokens=True)['input_ids']
        token_mask = [True] * len(tokens)
        tokens_len = len(tokens)
        if self.max_len is not None:
            if tokens_len < self.max_len:
                padding = [False] * (self.max_len - tokens_len)
                tokens = tokens + padding
                token_mask = token_mask + padding
            else:
                tokens = tokens[:self.max_len]
                token_mask = token_mask[:self.max_len]
        labels = list(tokens)
        inputs['lang_tokens'] = np.array(tokens)
        inputs['lang_masks'] = np.array(token_mask)
        if self.with_labels:
            assert 'actions' in inputs, "Data must contain 'actions' key."
            actions = inputs['actions']
            inputs['labels'] = np.array(labels)
            inputs['labels'][:-(len(actions[0]) + 1)] = self.ignore_index
        return inputs


@TRANSFORMS.register_module()
class ProcessPromptsWithImage:
    """Process and tokenize prompts with image context for language models.
    This class handles the tokenization of prompts that include image context
    using a specified tokenizer. It constructs a structured text format that
    incorporates system prompts, user prompts, and image placeholders. The
    tokenized output adheres to a maximum length, and it can optionally prepare
    labels for language modeling tasks.
    The text format is inspired by the GR00T model's input structure.

    Args:
        tokenizer (Dict): Configuration for building the tokenizer.
        max_len (int, optional): Maximum length for tokenized prompts.
            Defaults to 180.
        with_labels (bool, optional): Whether to include labels in
            the output. Defaults to False.
        # ===== GR00T text template options =====
        add_system (bool, optional): Whether to add a system prompt section.
            Defaults to True.
        system_prompt (str, optional): The system prompt text.
            Defaults to "You are a helpful assistant.".
        add_assistant_stub (bool, optional): Whether to append an assistant
            stub at the end. Defaults to True.
        task_pos (str, optional): Position of the task description relative
            to images ('after_images' or 'before_images').
            Defaults to 'after_images'.
        front_eos_repeat (int, optional): Number of <|endoftext|>
            tokens to prepend.
            Defaults to 0.
        eos_token_str (str, optional): String representation of the
            end-of-text token.
            Defaults to "<|endoftext|>".
        im_start (str, optional): String marking the start of an
            image section.
            Defaults to "<|im_start|>".
        im_end (str, optional): String marking the end of an
            image section.
            Defaults to "<|im_end|>".
        image_tag_template (str, optional): Template for image tags.
            Defaults to "<image {i}>".
        img_start (str, optional): String marking the start of
            image context.
            Defaults to "<img>".
        img_end (str, optional): String marking the end of
            image context.
            Defaults to "</img>".
        img_context_token (str, optional): Token representing
            image context.
            Defaults to "<IMG_CONTEXT>".
        # ===== image token expansion options =====
        img_tokens_source (str, optional): Source for determining
            the number of image tokens ('from_inputs' or 'fixed').
            Defaults to 'fixed'.
        fixed_img_tokens (int, optional): Fixed number of tokens per image
            if img_tokens_source is 'fixed'. Defaults to 256.
        num_images (int, optional): Number of images to include.
            Defaults to 2.
        # ===== tokenization/padding options =====
        pad_to_max_len (bool, optional): Whether to pad
            sequences to max_len. Defaults to True.
        padding_side (str, optional): Side to apply padding
            ('left' or 'right'). Defaults to 'left'.
        use_eos_as_pad (bool, optional): Whether to use the
            EOS token as the padding token if no pad token
            is defined. Defaults to True.
        return_text (bool, optional): Whether to include
            the constructed text in the output for debugging.
            Defaults to False.
    """

    def __init__(
            self,
            tokenizer: Dict,
            max_len: int = 180,
            with_labels: bool = False,
            # ===== GR00T text template options =====
            add_system: bool = True,
            system_prompt: str = 'You are a helpful assistant.',
            add_assistant_stub: bool = True,
            task_pos: str = 'after_images',
            front_eos_repeat: int = 0,
            # tag definitions
            eos_token_str: str = '<|endoftext|>',
            im_start: str = '<|im_start|>',
            im_end: str = '<|im_end|>',
            image_tag_template: str = '<image {i}>',
            img_start: str = '<img>',
            img_end: str = '</img>',
            img_context_token: str = '<IMG_CONTEXT>',
            # ===== image token expansion options =====
            # 'from_inputs': read inputs['num_image_tokens']
            # 'fixed': always use fixed_img_tokens
            img_tokens_source: str = 'fixed',
            fixed_img_tokens: Optional[int] = 256,
            num_images: Optional[int] = 3,
            # ===== tokenization/padding options =====
            pad_to_max_len: bool = True,
            padding_side: str = 'left',  # 'left' or 'right'
            use_eos_as_pad: bool = True,
            return_text: bool = False,
            model_path=None):  # noqa: E129
        from fluxvla.engines import build_tokenizer_from_cfg
        if model_path is not None:
            tokenizer['model_path'] = os.path.join(model_path, 'tokenizer')
        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.max_len = max_len
        self.with_labels = with_labels

        self.add_system = add_system
        self.system_prompt = system_prompt
        self.add_assistant_stub = add_assistant_stub
        self.task_pos = task_pos

        self.front_eos_repeat = front_eos_repeat
        self.eos_token_str = eos_token_str
        self.im_start = im_start
        self.im_end = im_end
        self.image_tag_template = image_tag_template
        self.img_start = img_start
        self.img_end = img_end
        self.img_context_token = img_context_token

        self.img_tokens_source = img_tokens_source
        self.num_images = num_images
        self.fixed_img_tokens = fixed_img_tokens

        self.pad_to_max_len = pad_to_max_len
        self.padding_side = padding_side
        self.use_eos_as_pad = use_eos_as_pad
        self.return_text = return_text

        # If tokenizer has no pad_token, use eos as pad (common trick)
        if self.tokenizer.pad_token_id is None and self.use_eos_as_pad:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # === Build GR00T-style text string ===
    def _build_text(self, task_desc: str,
                    num_tokens_per_image: List[int]) -> str:
        parts = []
        # (0) prepend eos repeats
        if self.front_eos_repeat > 0:
            parts.append(self.eos_token_str * self.front_eos_repeat)

        # (1) system section
        if self.add_system:
            parts.append(
                f'{self.im_start}system\n{self.system_prompt}{self.im_end}\n')

        # (2) user section
        parts.append(f'{self.im_start}user\n')

        if self.task_pos == 'before_images' and task_desc:
            parts.append(task_desc + '\n')

        # insert images: <image i><img> + repeated <IMG_CONTEXT> + </img>
        for i, n_tok in enumerate(num_tokens_per_image, start=1):
            parts.append(self.image_tag_template.format(i=i))
            parts.append(self.img_start)
            parts.append(self.img_context_token * int(n_tok))
            parts.append(self.img_end)

        if self.task_pos == 'after_images' and task_desc:
            parts.append(task_desc)

        parts.append(f'{self.im_end}\n')

        # (3) assistant stub
        if self.add_assistant_stub:
            parts.append(f'{self.im_start}assistant\n')

        return ''.join(parts)

    def __call__(self, inputs: Dict):
        """Tokenize and process the prompt with image
        context in the input data.The method constructs a
        structured text format incorporating system
        prompts, user prompts, and image placeholders.
        It then tokenizes this text, applies padding or
        truncation to meet the maximum length
        requirement, and optionally creates a 'labels'
        field for language modeling tasks.
        """
        assert 'task_description' in inputs, "inputs must contain 'task_description'"  # noqa: E501

        # (1) resolve per-image token counts
        per_img = [int(self.fixed_img_tokens)] * self.num_images

        # (2) build GR00T-style text
        text = self._build_text(inputs['task_description'], per_img)

        # (3) tokenize
        encoded = self.tokenizer(text, add_special_tokens=True)
        tokens = encoded.input_ids
        mask = [1] * len(tokens)
        labels = list(tokens)

        # (4) pad/truncate
        if self.max_len is not None and self.pad_to_max_len:
            L = len(tokens)
            pad_id = self.tokenizer.pad_token_id if \
                self.tokenizer.pad_token_id is not None \
                else self.tokenizer.eos_token_id
            if L < self.max_len:
                pad_len = self.max_len - L
                if self.padding_side == 'left':
                    tokens = [pad_id] * pad_len + tokens
                    mask = [0] * pad_len + mask
                    if self.with_labels:
                        labels = [-100] * pad_len + labels
                else:
                    tokens = tokens + [pad_id] * pad_len
                    mask = mask + [0] * pad_len
                    if self.with_labels:
                        labels = labels + [-100] * pad_len
            else:
                if self.padding_side == 'left':
                    tokens = tokens[-self.max_len:]
                    mask = mask[-self.max_len:]
                    if self.with_labels:
                        labels = labels[-self.max_len:]
                else:
                    tokens = tokens[:self.max_len]
                    mask = mask[:self.max_len]
                    if self.with_labels:
                        labels = labels[:self.max_len]

        inputs['lang_tokens'] = np.asarray(tokens, dtype=np.int64)
        inputs['lang_masks'] = np.asarray(mask, dtype=np.int32)
        if self.with_labels:
            inputs['labels'] = np.asarray(labels, dtype=np.int64)
        if self.return_text:
            inputs['text'] = text
        return inputs


@TRANSFORMS.register_module()
class LiberoPromptFromInputs:
    """Build and tokenize Libero evaluation prompt.

    Constructs: "In: What action should the robot
    take to {task} ?\nOut:" + suffix
    Pads/truncates to max_len and outputs
    'lang_tokens' and 'lang_masks'.

    Args:
        tokenizer (Dict): Tokenizer config for build_tokenizer_from_cfg.
        max_len (int): Maximum token length.
        pad_token_id (int): Pad id to use if padding needed.
        prompt_suffix (str): Suffix appended after 'Out:'.
    """

    def __init__(self,
                 tokenizer: Dict,
                 max_len: int = 180,
                 pad_token_id: int = 0,
                 prompt_suffix: str = '',
                 use_conversation: bool = True,
                 add_new_line: bool = False) -> None:
        from fluxvla.engines import build_tokenizer_from_cfg
        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.prompt_suffix = prompt_suffix
        self.use_conversation = use_conversation
        self.add_new_line = add_new_line

    def __call__(self, inputs: Dict) -> Dict:
        assert 'task_description' in inputs, "inputs must contain 'task_description'"  # noqa: E501
        task_description = inputs['task_description']
        if self.use_conversation:
            prompt = (f'In: What action should the robot take to '
                      f'{str(task_description).lower()}?\nOut:' +
                      self.prompt_suffix)
        else:
            prompt = task_description
        if self.add_new_line:
            prompt += '\n'
        token_ids = self.tokenizer(prompt)['input_ids']
        mask = [True] * len(token_ids)

        if self.max_len is not None:
            if len(token_ids) < self.max_len:
                pad_len = self.max_len - len(token_ids)
                token_ids = token_ids + [self.pad_token_id] * pad_len
                mask = mask + [False] * pad_len
            else:
                token_ids = token_ids[:self.max_len]
                mask = mask[:self.max_len]

        inputs['lang_tokens'] = np.asarray(token_ids, dtype=np.int64)
        inputs['lang_masks'] = np.asarray(mask, dtype=np.bool_)
        return inputs
