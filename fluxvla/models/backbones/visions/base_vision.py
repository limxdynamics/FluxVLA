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

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn


class VisionBackbone(nn.Module, ABC):
    """
    Abstract base class for a vision backbone module.

    Subclasses should implement methods for forward pass, embedding dimension,
    number of patches, and FSDP (Fully Sharded Data Parallel) wrapping policy.

    Args:
        vision_backbone_id (str): A unique identifier for the backbone.
    """

    def __init__(self, vision_backbone_id: str) -> None:
        super().__init__()
        self.identifier: str = vision_backbone_id

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Returns the FSDP wrapping policy for model parallelism.

        Returns:
            Callable: A function that defines how layers should be wrapped for
            sharded training.
        """
        ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the backbone.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output feature tensor (e.g., patch or grid features).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """
        The dimension of the output feature embeddings.

        Returns:
            int: Embedding dimension.
        """
        ...

    @property
    @abstractmethod
    def num_patches(self) -> int:
        """
        The number of patches (or spatial locations) output by the backbone.

        Returns:
            int: Number of patches or tokens.
        """
        ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype:
        """
        The preferred dtype to use for half-precision inference/training.

        Returns:
            torch.dtype: Either torch.float16 or torch.bfloat16.
        """
        ...
