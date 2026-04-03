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

import json
from pathlib import Path

import numpy as np
import sentencepiece

from fluxvla.engines import TOKENIZERS
from fluxvla.engines.utils.download_utils import maybe_download


@TOKENIZERS.register_module()
class PaligemmaTokenizer:
    """PaligemmaTokenizer for PaliGemma.
    This tokenizer is designed to work with the PaliGemma model,
    which uses a combination of a SentencePiece tokenizer for the
    prefix and a FAST tokenizer for the actions. The tokenizer
    handles the tokenization of prompts, states,
    and actions, and it also provides functionality to
    extract actions from the tokenized output.

    Args:
        model_path (str): Path to the PaliGemma tokenizer model.
            Defaults to "gs://big_vision/paligemma_tokenizer.model".
    """

    def __init__(
            self,
            download_path: str = 'gs://big_vision/paligemma_tokenizer.model',
            *args,
            **kwargs):

        path = maybe_download(download_path, gs={'token': 'anon'})
        with path.open('rb') as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read())

    def tokenize(self,
                 prompt: str,
                 state: np.ndarray | None = None,
                 **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize the input prompt, state, and actions.

        Args:
            prompt (str): The task prompt to be tokenized.
            state (np.ndarray | None): The state to be
                tokenized, expected to be a
                1D numpy array of continuous values.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - tokens (np.ndarray): The tokenized sequence.
        """
        cleaned_text = prompt.strip().replace('_', ' ').replace('\n', ' ')
        tokens = self._tokenizer.encode(
            cleaned_text, add_bos=True) + self._tokenizer.encode('\n')

        token_mask = [True] * len(tokens)
        return dict(input_ids=tokens, attention_mask=token_mask)

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def save_pretrained(self, save_directory: str):
        """Save the tokenizer to a directory.

        Args:
            save_directory (str): The directory to save the tokenizer.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. save sentencepiece model file
        spm_file = save_dir / 'tokenizer.model'
        # use serialized_model_proto is the most reliable way,
        # not dependent on "loading from file or bytes"
        with spm_file.open('wb') as f:
            f.write(self._tokenizer.serialized_model_proto())

        # 2. optional: save some meta information (for later loading)
        config = {
            'tokenizer_class': 'PaligemmaTokenizer',
            'spm_file': 'tokenizer.model'
        }
        with (save_dir / 'tokenizer_config.json').open(
                'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
