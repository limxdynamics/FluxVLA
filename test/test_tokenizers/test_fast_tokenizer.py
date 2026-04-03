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

import unittest

import torch

from fluxvla.engines.utils import build_transform_from_cfg


class TestFastTokenizer(unittest.TestCase):

    def setUp(self):
        self.cfg = dict(type='FASTTokenizer', max_len=180)
        self.tokenizer = build_transform_from_cfg(self.cfg)

    def test_tokenize(self):
        prompt = 'pick up the black bowl from table center and place it on the plate'  # noqa: E501
        state = torch.tensor([
            -1.4757, -0.1029, 1.0837, 0.4882, 0.2471, 0.1194, 0.8376, -0.8249
        ])
        actions = torch.tensor(
            [[0.9477, 0.1393, -0.4881, -0.5573, -0.2796, -0.1774, -0.9554],
             [1.3449, 0.3441, -0.4408, -0.5880, -0.2548, -0.1743, -0.9554],
             [1.7132, 0.4960, -0.4107, -0.6095, -0.2525, -0.1619, -0.9554],
             [2.0310, 0.6017, -0.3677, -0.6464, -0.2525, -0.1712, -0.9554],
             [2.2765, 0.5885, -0.3720, -0.6864, -0.2525, -0.1867, -0.9554],
             [2.5076, 0.5356, -0.4752, -0.7232, -0.2525, -0.2054, -0.9554],
             [2.5220, 0.3375, -0.5182, -0.6956, -0.2525, -0.2116, -0.9554],
             [2.4643, -0.0258, -0.6687, -0.6064, -0.2619, -0.2054, -0.9554],
             [2.3704, -0.1777, -0.9008, -0.5296, -0.2819, -0.2147, -0.9554],
             [2.2837, -0.2900, -1.0341, -0.4958, -0.3019, -0.2240, -0.9554]])
        tokens, token_mask, ar_mask = self.tokenizer.tokenize(
            prompt, state, actions)
        self.assertEquals(
            torch.mean(tokens[:73].float()), torch.tensor(177167.49315))
        self.assertEqual(torch.sum(token_mask).item(), 74)
        self.assertEqual(torch.sum(ar_mask).item(), 23)
