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
class MLPProjector(nn.Module):
    """
    MLPProjector projects vision features into a language model (LLM) feature
    space using a lightweight MLP.

    Args:
        vision_dim (int): Input feature dimension from vision backbone.
        llm_dim (int): Target output dimension to match the LLM feature space.
        mlp_type (str): Type of MLP to use. Currently supports only
            'gelu-mlp'.

    Raises:
        ValueError: If an unsupported `mlp_type` is specified.
    """

    def __init__(self,
                 vision_dim: int,
                 llm_dim: int,
                 mlp_type: str = 'gelu-mlp') -> None:
        super().__init__()
        if mlp_type == 'gelu-mlp':
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(
                f'Projector with mlp_type={mlp_type!r} is not supported!')

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for projecting vision features.

        Args:
            img_patches (Tensor): Input tensor of shape [..., vision_dim],
                typically extracted image patch embeddings.

        Returns:
            Tensor: Projected tensor of shape [..., llm_dim].
        """
        return self.projector(img_patches)
