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
"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic \
    to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import AutoTokenizer

from fluxvla.engines.utils import TOKENIZERS


@TOKENIZERS.register_module()
class ActionTokenizer:

    def __init__(self,
                 model_path: str,
                 bins: int = 256,
                 min_action: int = -1,
                 max_action: int = 1) -> None:
        """Discretizes continuous robot actions into N bins
        per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to
        the LlamaTokenizer, where *the least used tokens* appear at
        the end of the vocabulary!

        Args:
            model_path (str): Path of the model to load the tokenizer from.
            bins (int, optional): Number of bins to use. Defaults to 256.
            min_action (int, optional): Minimum number of actions.
                Defaults to -1.
            max_action (int, optional): Maximum number of actions.
                Defaults to 1.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.n_bins, self.min_action, self.max_action = bins, \
            min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_idx" based on
        # `self.tokenizer.vocab_size - (self.n_bins + 1)` =>> Assumes
        # we're always overwriting the final `n_bins`
        # tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size -
                                               (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary
        (e.g., tokenizer.vocab[-256:]).

        Args:
            action (np.ndarray): Action to be tokenized. Can be a single
                action or a batch of actions.

        Returns:
            Union[str, List[str]]: Returns the tokenized action(s) as a
                string or a list of strings.
        """
        action = np.clip(
            action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(
                list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode(
                (self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(
            self, action_token_ids: np.ndarray) -> np.ndarray:
        """Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the
        bins (and not the bin centers), the digitization returns bin
        indices between [1, # bins], inclusive, when there are actually only
        (# bins - 1) bin intervals.

        Therefore, if the digitization returns the last possible index, we
        map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers
        has 255 values. Digitization returns indices between [1, 256]. We
        subtract 1 from all indices so that they are between [0, 255]. There
        is still one index (i==255) that would cause an out-of-bounds error if
        used to index into self._bin_centers. Therefore, if i==255, we subtract
        1 from it so that it just becomes the index of the last bin center. We
        implement this simply via clipping between [0, 255 - 1].

        Args:
            action_token_ids (np.ndarray): Input action token IDs to be
                decoded.

        Returns:
            np.ndarray: Returns the decoded continuous actions.
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
