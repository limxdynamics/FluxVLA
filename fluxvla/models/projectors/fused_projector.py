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
class FusedMLPProjector(nn.Module):
    """
    FusedMLPProjector projects fused vision-language features to the LLM space
    via a multi-layer perceptron (MLP). This module is typically used to align
    the dimensionality of visual features with that of language models.

    Args:
        fused_vision_dim (int): Dimension of the fused vision input features.
        llm_dim (int): Target output dimension matching the LLM feature space.
        mlp_type (str): MLP variant to use. Currently supports only
            'fused-gelu-mlp'.

    Raises:
        ValueError: If an unsupported `mlp_type` is specified.
    """

    def __init__(self,
                 fused_vision_dim: int,
                 llm_dim: int,
                 mlp_type: str = 'fused-gelu-mlp') -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == 'fused-gelu-mlp':
            self.projector = nn.Sequential(
                nn.Linear(
                    fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(
                f'Fused Projector with `{mlp_type = }` is not supported!')

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projector.

        Args:
            fused_img_patches (Tensor): Input tensor of shape [..., in_dim],
                typically the output from fused vision-language modules.

        Returns:
            Tensor: Projected tensor of shape [..., llm_dim].
        """
        return self.projector(fused_img_patches)
