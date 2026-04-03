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
dinosiglip_vit.py

Vision backbone that returns concatenated features from both DINOv2 and SigLIP.
"""

from functools import partial
from typing import Any, Callable, Dict, Tuple

import timm
import torch
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import (_module_wrap_policy, _or_policy,
                                         transformer_auto_wrap_policy)

from fluxvla.engines import VISION_BACKBONES
from .base_vision import VisionBackbone
from .configs import VISION_BACKBONE_CONFIGS


def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    """
    Unwraps the first element of a tuple returned by a function if it exists.

    Args:
        fn (Callable): A function returning a tuple or single value.

    Returns:
        Callable: A wrapper that returns only the first element of a tuple.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


@VISION_BACKBONES.register_module()
class DinoSigLIPViTBackbone(VisionBackbone):
    """
    A vision backbone that fuses features from DINOv2 and SigLIP ViT backbones.

    Args:
        vision_backbone_id (str): Identifier string for the backbone.
        dino_config (Dict): Config for DINOv2 TIMM model loading.
        siglip_config (Dict): Config for SigLIP TIMM model loading.
    """

    def __init__(self,
                 vision_backbone_id: str,
                 dino_config: Dict,
                 siglip_config: Dict,
                 pretrained: bool = True,
                 img_size: int = 224,
                 *args,
                 **kwargs) -> None:
        super().__init__(vision_backbone_id)
        dino_timm_path_or_url = VISION_BACKBONE_CONFIGS[dino_config.pop(
            'model_id')]['model_id']
        siglip_timm_path_or_url = VISION_BACKBONE_CONFIGS[siglip_config.pop(
            'model_id')]['model_id']

        self.dino_featurizer: VisionTransformer = timm.create_model(
            dino_timm_path_or_url,
            pretrained=pretrained,
            num_classes=0,
            pretrained_cfg=dino_config,
            img_size=img_size,
        )
        self.dino_featurizer.eval()

        self.siglip_featurizer: VisionTransformer = timm.create_model(
            siglip_timm_path_or_url,
            pretrained=pretrained,
            num_classes=0,
            pretrained_cfg=siglip_config,
            img_size=img_size,
        )
        self.siglip_featurizer.eval()

        # Patch forward() to extract the second-to-last block layer output
        self.dino_featurizer.forward = unpack_tuple(
            partial(
                self.dino_featurizer.get_intermediate_layers,
                n={len(self.dino_featurizer.blocks) - 2},
            ))
        self.siglip_featurizer.forward = unpack_tuple(
            partial(
                self.siglip_featurizer.get_intermediate_layers,
                n={len(self.siglip_featurizer.blocks) - 2},
            ))

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Define FSDP wrapping policy that includes full ViTs and
            transformer blocks.

        Returns:
            Callable: A composite policy for FSDP module wrapping.
        """
        vit_wrap_policy = partial(
            _module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(
            _or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a forward pass using both DINO and SigLIP backbones.

        Args:
            pixel_values (Dict[str, torch.Tensor]): Dictionary containing:
                - "dino": tensor input for DINOv2 model
                - "siglip": tensor input for SigLIP model

        Returns:
            torch.Tensor: Concatenated patch features from both models.
        """
        num_images = pixel_values.shape[1] // 3
        dino_patches_list = list()
        siglip_patches_list = list()
        for i in range(num_images // 2):
            dino_patches = self.dino_featurizer(pixel_values[:, i * 3:i * 3 +
                                                             3, :, :])
            siglip_patches = self.siglip_featurizer(
                pixel_values[:, num_images * 3 // 2 +
                             i * 3:num_images * 3 // 2 + i * 3 + 3, :, :])
            dino_patches_list.append(dino_patches)
            siglip_patches_list.append(siglip_patches)
        dino_patches_list = torch.cat(dino_patches_list, dim=2)
        siglip_patches_list = torch.cat(siglip_patches_list, dim=2)
        return torch.cat([dino_patches_list, siglip_patches_list], dim=2)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        """
        Get the default input resolution for the DINO backbone.

        Returns:
            Tuple[int, int, int]: Image shape as (C, H, W).
        """
        return self.dino_data_cfg['input_size']

    @property
    def embed_dim(self) -> int:
        """
        Get the combined embedding dimension from both backbones.

        Returns:
            int: Sum of DINO and SigLIP embedding dimensions.
        """
        return (self.dino_featurizer.embed_dim +
                self.siglip_featurizer.embed_dim)

    @property
    def num_patches(self) -> int:
        """
        Get the number of spatial patches output by the backbones.

        Returns:
            int: Number of patches (same for both models).
        """
        assert (self.dino_featurizer.patch_embed.num_patches ==
                self.siglip_featurizer.patch_embed.num_patches)
        return self.dino_featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """
        Preferred half-precision dtype for mixed-precision training.

        Returns:
            torch.dtype: Typically bfloat16.
        """
        return torch.bfloat16
