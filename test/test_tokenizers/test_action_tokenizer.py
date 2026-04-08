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
import unittest

import numpy as np

from fluxvla.engines.utils import build_tokenizer_from_cfg

MODEL_PATH = 'checkpoints/openvla-7b-finetuned-libero-10'


@unittest.skipUnless(
    os.path.exists(MODEL_PATH), f'Checkpoint not found: {MODEL_PATH}')
class TestActionTokenizer(unittest.TestCase):

    def setUp(self):
        self.cfg = {
            'type': 'ActionTokenizer',
            'model_path': MODEL_PATH,
            'bins': 256,
            'min_action': -1,
            'max_action': 1,
        }
        self.tokenizer = build_tokenizer_from_cfg(self.cfg)

    def test_action_tokenizer_discretization_single(self):
        action = np.array([0.5])
        tokenized_action = self.tokenizer(action)
        self.assertIsInstance(tokenized_action, str,
                              'Tokenized action should be a string.')

    def test_action_tokenizer_discretization_batch(self):
        actions = np.array([[0.5, -0.5], [0.1, -0.1]])
        tokenized_actions = self.tokenizer(actions)
        self.assertIsInstance(
            tokenized_actions, list,
            'Tokenized actions should be a list of strings.')
        self.assertEqual(
            len(tokenized_actions), actions.shape[0],
            'Batch size mismatch in tokenized actions.')

    def test_action_tokenizer_decoding_single(self):
        action_token_ids = np.array(
            [self.tokenizer.tokenizer.vocab_size - 128])
        decoded_action = self.tokenizer.decode_token_ids_to_actions(
            action_token_ids)
        self.assertEqual(decoded_action.shape, (1, ),
                         'Decoded action should have shape (1,)')

    def test_action_tokenizer_decoding_batch(self):
        action_token_ids_batch = np.array([
            [
                self.tokenizer.tokenizer.vocab_size - 128,
                self.tokenizer.tokenizer.vocab_size - 64
            ],
            [
                self.tokenizer.tokenizer.vocab_size - 32,
                self.tokenizer.tokenizer.vocab_size - 16
            ],
        ])
        decoded_actions = self.tokenizer.decode_token_ids_to_actions(
            action_token_ids_batch)
        self.assertEqual(decoded_actions.shape, (2, 2),
                         'Decoded actions shape mismatch')

    def test_action_tokenizer_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, 256, 'Vocab size mismatch')
