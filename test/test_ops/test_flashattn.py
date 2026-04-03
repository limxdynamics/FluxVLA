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

import gc
import unittest

import numpy as np
import pytest
import torch
from flash_attn.flash_attn_interface import (flash_attn_func,
                                             flash_attn_kvpacked_func)


class TestFlashAttn(unittest.TestCase):

    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_flash_attn(self):
        np.random.seed(0)
        torch.manual_seed(0)

        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 8

        # Generate random input tensors
        qkv = torch.rand(
            batch_size,
            seq_len,
            3 * num_heads * head_dim,
            device='cuda',
            dtype=torch.bfloat16) - 0.5

        # Split qkv into q, k, v tensors (each of shape [batch_size,
        # seq_len, num_heads, head_dim])
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, num_heads, head_dim)
        v = v.view(batch_size, seq_len, num_heads, head_dim)

        # Compute attention using Flash Attention
        output = flash_attn_kvpacked_func(q, torch.stack([k, v], dim=2))

        # Validate the output shape
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

        # Check numerical properties (e.g., no NaNs or Infs)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        self.assertAlmostEqual(
            output.mean(),
            torch.tensor(-0.0005, device='cuda', dtype=torch.bfloat16),
            delta=1e-3)
        self.assertAlmostEqual(
            output.std(),
            torch.tensor(0.0752, device='cuda', dtype=torch.bfloat16),
            delta=1e-3)

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_flash_attn_with_causal(self):
        np.random.seed(0)
        torch.manual_seed(0)

        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 8

        # Generate random input tensors
        qkv = torch.rand(
            batch_size,
            seq_len,
            3 * num_heads * head_dim,
            device='cuda',
            dtype=torch.bfloat16) - 0.5

        # Split qkv into q, k, v tensors (each of shape [batch_size,
        # seq_len, num_heads, head_dim])
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, num_heads, head_dim)
        v = v.view(batch_size, seq_len, num_heads, head_dim)

        # Compute attention using Flash Attention, with causal=True
        output = flash_attn_kvpacked_func(
            q, torch.stack([k, v], dim=2), causal=True)

        # Validate the output shape
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

        # Check numerical properties (e.g., no NaNs or Infs)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        self.assertAlmostEqual(
            output.mean(),
            torch.tensor(0.0032, device='cuda', dtype=torch.bfloat16),
            delta=1e-3)
        self.assertAlmostEqual(
            output.std(),
            torch.tensor(0.1328, device='cuda', dtype=torch.bfloat16),
            delta=1e-3)

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_flash_attn_different_batch_sizes(self):
        np.random.seed(0)
        torch.manual_seed(0)

        batch_sizes = [1, 2, 4]  # Test different batch sizes
        seq_len = 16
        num_heads = 4
        head_dim = 8

        for batch_size in batch_sizes:
            qkv = torch.rand(
                batch_size,
                seq_len,
                3 * num_heads * head_dim,
                device='cuda',
                dtype=torch.bfloat16) - 0.5

            # Split qkv into q, k, v tensors (each of shape [batch_size,
            # seq_len, num_heads, head_dim])
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, num_heads, head_dim)
            v = v.view(batch_size, seq_len, num_heads, head_dim)

            # Compute attention using Flash Attention
            output = flash_attn_kvpacked_func(
                q, torch.stack([k, v], dim=2), causal=False)

            # Validate the output shape
            assert output.shape == (batch_size, seq_len, num_heads, head_dim)

            # Check numerical properties (e.g., no NaNs or Infs)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_flash_attn_different_seq_lens(self):
        np.random.seed(0)
        torch.manual_seed(0)

        batch_size = 2
        seq_lens = [8, 16, 32]  # Test different sequence lengths
        num_heads = 4
        head_dim = 8

        for seq_len in seq_lens:
            # Generate random input tensors for each seq_len
            qkv = torch.rand(
                batch_size,
                seq_len,
                3 * num_heads * head_dim,
                device='cuda',
                dtype=torch.bfloat16) - 0.5

            # Split qkv into q, k, v tensors (each of shape
            # [batch_size, seq_len, num_heads, head_dim])
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, num_heads, head_dim)
            v = v.view(batch_size, seq_len, num_heads, head_dim)

            # Compute attention using Flash Attention
            output = flash_attn_kvpacked_func(
                q, torch.stack([k, v], dim=2), causal=False)

            # Validate the output shape
            assert output.shape == (batch_size, seq_len, num_heads, head_dim)

            # Check numerical properties (e.g., no NaNs or Infs)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    @pytest.mark.skipif(
        condition=torch.cuda.is_available() is False,
        reason='No GPU available.')
    def test_flash_attn_func(self):
        np.random.seed(0)
        torch.manual_seed(0)

        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 8

        # Generate random input tensors
        qkv = torch.rand(
            batch_size,
            seq_len,
            3,
            num_heads,
            head_dim,
            device='cuda',
            dtype=torch.bfloat16) - 0.5

        # Split qkv into q, k, v tensors (each of shape [batch_size, seq_len,
        # num_heads, head_dim])
        q, k, v = qkv.chunk(3, dim=2)
        q = q.view(batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, num_heads, head_dim)
        v = v.view(batch_size, seq_len, num_heads, head_dim)

        # Compute attention using flash_attn_func
        output = flash_attn_func(q, k, v)

        # Validate the output shape
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

        # Check numerical properties (e.g., no NaNs or Infs)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        self.assertAlmostEqual(
            output.mean(),
            torch.tensor(-0.0012, device='cuda', dtype=torch.bfloat16),
            delta=1e-3)
        self.assertAlmostEqual(
            output.std(),
            torch.tensor(0.0752, device='cuda', dtype=torch.bfloat16),
            delta=1e-3)
