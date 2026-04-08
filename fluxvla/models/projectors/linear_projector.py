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

import torch
import torch.nn as nn

from fluxvla.engines import PROJECTORS


# === Definitions for Various Projection Modules, with Signature:
#     [..., in_dim] --> [..., out_dim] ===
@PROJECTORS.register_module()
class LinearProjector(nn.Module):
    """
    LinearProjector performs a single linear transformation to map vision
    features into the language model (LLM) feature space.

    This is the simplest form of projection with no non-linearity or
    intermediate layers.

    Args:
        in_dim (int): Dimension of the input vision features.
        out_dim (int): Target output dimension to align with LLM input.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for projecting vision features using a linear layer.

        Args:
            img_patches (Tensor): Input tensor of shape [..., in_dim],
                typically image patch features.

        Returns:
            Tensor: Projected tensor of shape [..., out_dim].
        """
        return self.projector(input_features)
