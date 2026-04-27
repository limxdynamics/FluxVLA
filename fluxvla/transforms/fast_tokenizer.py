# Origin: Modified from
# Upstream-Repo: Physical-Intelligence/openpi
# Upstream-Path: src/openpi/models/tokenizer.py
# Upstream-Ref: main
# SPDX-License-Identifier: Apache-2.0
# Notes: Attribution normalized; no functional change.

import logging
from typing import Dict

import numpy as np
import sentencepiece
import torch
from transformers import AutoProcessor

from fluxvla.engines import TRANSFORMS
from fluxvla.engines.utils.download_utils import maybe_download


@TRANSFORMS.register_module()
class FASTTokenizer:
    """FASTTokenizer for PaliGemma.
    This tokenizer is designed to work with the PaliGemma model,
    which uses a combination of a SentencePiece tokenizer for the
    prefix and a FAST tokenizer for the actions. The tokenizer
    handles the tokenization of prompts, states,
    and actions, and it also provides functionality to
    extract actions from the tokenized output.

    Args:
        max_len (int): Maximum length of the tokenized sequence.
            Defaults to 256.
        fast_tokenizer_path (str): Path to the FAST tokenizer model.
            Defaults to "physical-intelligence/fast".
        paligemma_tokenizer_path (str): Path to the PaliGemma
            tokenizer model. Defaults to
            "gs://big_vision/paligemma_tokenizer.model".
        fast_skip_tokens (int): Number of tokens to skip
            in the FAST tokenizer. Defaults to 128, which
            corresponds to the last 128 tokens in the
            PaliGemma vocabulary that are reserved for
            special tokens.
    """

    def __init__(self,
                 max_len: int = 256,
                 fast_tokenizer_path: str = 'physical-intelligence/fast',
                 paligemma_tokenizer_path:
                 str = 'gs://big_vision/paligemma_tokenizer.model',
                 fast_skip_tokens: int = 128):
        self._max_len = max_len
        # Download base PaliGemma tokenizer
        path = maybe_download(paligemma_tokenizer_path, gs={'token': 'anon'})
        with path.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, trust_remote_code=True)
        # Skip last 128 tokens in PaliGemma vocab since they are special tokens
        self._fast_skip_tokens = fast_skip_tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenizes the input prompt, state, and actions.

        Args:
            prompt (str): The task prompt to be tokenized.
            state (np.ndarray): The state to be tokenized, expected to be a
                1D numpy array of continuous values.
            actions (np.ndarray | None): The actions to be tokenized, expected
                to be a 2D numpy array where each row corresponds to an action.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple
                containing:
                - tokens (np.ndarray): The tokenized sequence.
                - token_mask (np.ndarray): A mask indicating which tokens
                    are valid.
                - ar_mask (np.ndarray): An autoregressive mask for the
                    tokens.
                - loss_mask (np.ndarray): A mask indicating which tokens
                    should be included in the loss computation.
        """
        cleaned_text = prompt.lower().strip().replace('_', ' ')

        # Convention: state gets discretized into 256 discrete bins
        # (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(
            state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation
        # of state, followed by ';'
        state_str = ' '.join(map(str, discretized_state))
        prefix = 'Task: {}, State: {};\n'.format(cleaned_text, state_str)
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens
            # in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(
                action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens,
            # followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode('Action: ') +
                action_tokens_in_pg.tolist() +
                self._paligemma_tokenizer.encode('|'))
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix
        # (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. '  # noqa: E501
                    'Consider increasing the `max_token_len` in your model config if this happens frequently.'  # noqa: E501
                )
            tokens = tokens[:self._max_len]
            token_mask = token_mask[:self._max_len]
            ar_mask = ar_mask[:self._max_len]

        return torch.tensor(tokens), torch.tensor(token_mask), torch.tensor(
            ar_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int,
                        action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if 'Action: ' not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(
                decoded_tokens.split('Action: ')[1].split('|')[0].strip()))
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode([action_tokens.tolist()],
                                           time_horizon=action_horizon,
                                           action_dim=action_dim)[0]

    def _act_tokens_to_paligemma_tokens(
            self, tokens: np.ndarray | list[int]) -> np.ndarray:
        """Convert FAST tokenizer tokens to PaliGemma tokens.
        This function maps the FAST tokenizer tokens to the corresponding
        PaliGemma tokens by subtracting the number of fast skip tokens from the
        PaliGemma vocabulary size. This is necessary because the FAST tokenizer
        uses the last `fast_skip_tokens` tokens in the PaliGemma vocabulary for
        action tokens, which are not used in the PaliGemma tokenizer.

        Args:
            tokens (np.ndarray | list[int]): The tokens to be converted, can be
                a numpy array or a list of integers.

        Returns:
            np.ndarray: The converted tokens as a numpy array.
        """
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size(
        ) - 1 - self._fast_skip_tokens - tokens

    def __call__(self, data: Dict):
        """Tokenizes the input data.

        Args:
            data (Dict): A dictionary containing the input data.
                Expected keys are 'prompt', 'state', and optionally 'actions'.

        Returns:
            Dict: A dictionary containing the tokenized output with keys:
                - 'tokens': The tokenized sequence.
                - 'token_mask': A mask indicating which tokens are valid.
                - 'ar_mask': An autoregressive mask for the tokens.
        """
        prompt = data['prompt']
        states = data['states']
        actions = data.get('actions', None)

        tokens, token_mask, ar_mask = self.tokenize(prompt, states, actions)

        data['lang_tokens'] = tokens
        data['lang_masks'] = token_mask
        data['ar_mask'] = ar_mask
        return data
