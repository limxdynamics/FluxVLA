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
import random
from typing import Callable, Optional

import numpy as np
import torch

# === Randomness ===


def set_global_seed(
        seed: int,
        get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets the global seed for random number generation in various libraries
    (NumPy, PyTorch, and Python's built-in random module). This is crucial for
    ensuring reproducibility in experiments, especially when using multiple
    threads or processes. The seed is set as an environment variable, which can
    be useful for tracking and debugging purposes.

    Args:
        seed (int): The seed value to set for random number generation.
        get_worker_init_fn (bool, optional): If True, returns a worker
            initialization function for PyTorch's DataLoader. Defaults to
            False.

    Returns:
        Optional[Callable[[int], None]]: If `get_worker_init_fn` is True, a
            worker initialization function that sets the seed for each worker.
            Otherwise, returns None.

    Raises:
        AssertionError: If the seed is outside the bounds of np.uint32.
    """
    assert np.iinfo(np.uint32).min < seed < np.iinfo(
        np.uint32).max, 'Seed outside the np.uint32 bounds!'

    # Set Seed as an Environment Variable
    os.environ['EXPERIMENT_GLOBAL_SEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return worker_init_function if get_worker_init_fn else None


def set_seed_everywhere(seed: int):
    """Sets the global seed for random number generation
    in various libraries (NumPy, PyTorch, and Python's
    built-in random module) to ensure reproducibility across
    different runs and environments. This function also
    configures PyTorch's cuDNN backend for deterministic
    behavior, which is important for models that
    rely on GPU computations. The seed is set as an
    environment variable for tracking purposes.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_function(worker_id: int) -> None:
    """Sets the seed for each worker in a distributed setting. This is called
    by PyTorch's DataLoader for each worker process. The seed is derived from
    the global seed, the worker ID, and the global rank (if applicable). This
    ensures that each worker has a unique seed, allowing for reproducible
    randomness across different runs and environments.

    Args:
        worker_id (int): The ID of the worker process.

    Raises:
        AssertionError: If the worker ID is not a non-negative integer.
    """
    # Get current `rank` (if running distributed) and `process_seed`
    global_rank, process_seed = int(
        os.environ['LOCAL_RANK']), torch.initial_seed()

    # Back out the "base" (original) seed - the per-worker seed is
    # set in PyTorch: > https://pytorch.org/docs/stable/data.html#data-loading-randomness  # noqa: E501
    base_seed = process_seed - worker_id

    # "Magic" code --> basically creates a seed sequence that
    # mixes different "sources" and seeds every library...
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed --> generate_state(k)
    # produces a `k` element array!
    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)

    # Torch Manual seed takes 64 bits (so just specify a dtype of uint64
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (
        random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) *
        [1 << 64, 1]).sum()
    random.seed(random_seed)


# === BFloat16 Support ===


def check_bloat16_supported() -> bool:
    try:
        import packaging.version
        import torch.cuda.nccl as nccl
        import torch.distributed as dist

        return ((torch.version.cuda is not None)
                and torch.cuda.is_bf16_supported()
                and (packaging.version.parse(torch.version.cuda).release >=
                     (11, 0)) and dist.is_nccl_available()
                and (nccl.version() >= (2, 10)))

    except Exception:
        return False
