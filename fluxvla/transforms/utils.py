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
This module defines the abstract interface and default
implementation for Vision Backbones
(visual feature extractors). It includes:

- `VisionBackbone`: An abstract base class for visual featurizers.
- `TimmViTBackbone`: A base class for feature extractors built
    using TIMM Vision Transformer (ViT) models.
- Supporting transform utilities such as `LetterboxPad` and
    `unpack_tuple`.

These classes are designed to be used as plug-and-play modules in
multi-modal systems such as Visual Language Models (VLMs) or image
captioning architectures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import einops
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from PIL import Image as PILImage
from PIL.Image import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import (_module_wrap_policy, _or_policy,
                                         transformer_auto_wrap_policy)
from torchvision.transforms import Compose, Resize


def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    """
    Utility decorator to convert a function returning a single-element tuple
    into returning just the value.

    Useful for monkey-patching model forward functions that return tuples
    but are expected to be compatible with FSDP and PyTorch Trainer pipelines.

    Parameters
    ----------
    fn : Callable
        Function that returns a tuple.

    Returns
    -------
    Callable
        Wrapped function that returns the first element of the tuple.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


class ImageTransform(Protocol):
    """
    Protocol interface for any image transform.

    Implementations should accept a PIL image and return either a torch.Tensor
    or a dictionary of tensors.
    """

    def __call__(
            self, img: Image,
            **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        ...


@dataclass
class LetterboxPad:
    """
    Custom image padding transform to pad a PIL image to a square shape with
    a constant fill value.

    Parameters
    ----------
    padding_fill_value : Tuple[int, int, int]
        RGB value used for padding.
    """
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """
        Pad the input image to make it square by adding symmetric borders.

        Parameters
        ----------
        image : PIL.Image
            Input image.

        Returns
        -------
        PIL.Image
            Padded square image.
        """
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int(
            (max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(
            image,
            padding,
            fill=self.padding_fill_value,
            padding_mode='constant')


class VisionBackbone(nn.Module, ABC):
    """
    Abstract base class for a vision backbone (feature extractor).

    Subclasses should implement logic for forward inference and FSDP policy
    definition.

    Parameters
    ----------
    vision_backbone_id : str
        Identifier name for the backbone model.
    image_resize_strategy : str
        Image resizing strategy ('resize-naive', 'resize-crop', 'letterbox').
    default_image_size : int, optional
        Default input image size (width/height), by default 224.
    """

    def __init__(self,
                 vision_backbone_id: str,
                 image_resize_strategy: str,
                 default_image_size: int = 224) -> None:
        super().__init__()
        self.identifier = vision_backbone_id
        self.image_resize_strategy = image_resize_strategy
        self.default_image_size = default_image_size

        self.featurizer: nn.Module = None
        self.image_transform: ImageTransform = None

    def get_image_transform(self) -> ImageTransform:
        """
        Return the transform applied to input images before passing to
        the backbone.

        Returns
        -------
        ImageTransform
            Preprocessing pipeline.
        """
        return self.image_transform

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Return a callable that specifies the FSDP wrapping policy.

        Returns
        -------
        Callable
            FSDP wrapping policy function.
        """
        ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision backbone.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Preprocessed image tensors.

        Returns
        -------
        torch.Tensor
            Extracted features (e.g., patch embeddings).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def default_image_resolution(self) -> Tuple[int, int, int]:
        """Return the default input resolution for the backbone."""
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Return the dimensionality of the output feature embeddings."""
        ...

    @property
    @abstractmethod
    def num_patches(self) -> int:
        """Return the number of patch tokens in the output."""
        ...

    @property
    @abstractmethod
    def half_precision_dtype(self) -> torch.dtype:
        """Return the half-precision dtype used in this
        model (e.g., torch.bfloat16)."""
        ...


class TimmViTBackbone(VisionBackbone, ABC):
    """
    A base class for Vision Transformers using the TIMM library.

    Automatically handles model loading, resizing logic, image
    transform construction, and monkey-patching for intermediate
    layer extraction.

    Parameters
    ----------
    vision_backbone_id : str
        Identifier name for the backbone.
    timm_path_or_url : str
        Model name or checkpoint path recognized by TIMM.
    image_resize_strategy : str
        Resize strategy: 'resize-naive', 'resize-crop', or 'letterbox'.
    default_image_size : int, optional
        Input image resolution, by default 224.
    override_act_layer : Optional[str], optional
        Optional override for the activation function layer.
    """

    def __init__(
        self,
        vision_backbone_id: str,
        timm_path_or_url: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        override_act_layer: Optional[str] = None,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy,
                         default_image_size)
        self.timm_path_or_url = timm_path_or_url
        self.override_act_layer = override_act_layer
        self.dtype = torch.bfloat16

        if self.override_act_layer is None:
            self.featurizer = timm.create_model(
                self.timm_path_or_url,
                pretrained=True,
                num_classes=0,
                img_size=self.default_image_size)
        else:
            self.featurizer = timm.create_model(
                self.timm_path_or_url,
                pretrained=True,
                num_classes=0,
                img_size=self.default_image_size,
                act_layer=self.override_act_layer,
            )
        self.featurizer.eval()

        self.featurizer.forward = unpack_tuple(
            partial(
                self.featurizer.get_intermediate_layers,
                n={len(self.featurizer.blocks) - 2}))

        assert isinstance(self.featurizer, VisionTransformer), (
            'Only TIMM VisionTransformers are supported. Please extend this '
            'class for other backbones.')

        self.data_cfg = timm.data.resolve_model_data_config(self.featurizer)
        self.data_cfg['input_size'] = (3, self.default_image_size,
                                       self.default_image_size)
        default_image_transform = timm.data.create_transform(
            **self.data_cfg, is_training=False)

        if 'siglip' in self.timm_path_or_url or 'in1k' in self.timm_path_or_url:  # noqa: E501

            assert isinstance(default_image_transform, Compose)
            assert isinstance(default_image_transform.transforms[0], Resize)
            default_image_transform = Compose([
                Resize(
                    self.default_image_size,
                    interpolation=default_image_transform.transforms[0].
                    interpolation),
                *default_image_transform.transforms[1:],
            ])

        if self.image_resize_strategy == 'resize-naive':
            assert isinstance(default_image_transform, Compose)
            assert isinstance(default_image_transform.transforms[0], Resize)
            target_size = (self.default_image_size, self.default_image_size)
            self.image_transform = Compose([
                Resize(
                    target_size,
                    interpolation=default_image_transform.transforms[0].
                    interpolation)
            ] + default_image_transform.transforms[1:])
        elif self.image_resize_strategy == 'resize-crop':
            self.image_transform = default_image_transform
        elif self.image_resize_strategy == 'letterbox':
            assert isinstance(default_image_transform, Compose)
            assert 'mean' in self.data_cfg
            fill = tuple([int(x * 255) for x in self.data_cfg['mean']])
            self.image_transform = Compose(
                [LetterboxPad(fill), *default_image_transform.transforms])
        else:
            raise ValueError(
                f'Image Resize Strategy {self.image_resize_strategy} \
                    is not supported!')

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Return FSDP wrapping policy that wraps each transformer block and the
        entire featurizer.

        Returns
        -------
        Callable
            Policy function used in `FullyShardedDataParallel`.
        """
        vit_wrap_policy = partial(
            _module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(
            _or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(
        self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Forward input through ViT model to extract patch features.

        Parameters
        ----------
        pixel_values : torch.Tensor or dict
            Preprocessed input tensor(s).

        Returns
        -------
        torch.Tensor
            Feature tensor containing patch-level embeddings.
        """
        return self.featurizer(pixel_values)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        """Return the expected input resolution (C, H, W) for
        the featurizer."""
        return self.data_cfg['input_size']

    @property
    def embed_dim(self) -> int:
        """Return embedding dimension of the vision backbone."""
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        """Return the number of patch tokens in the output sequence."""
        return self.featurizer.patch_embed.num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Return the half precision dtype (e.g., bfloat16) used
        for inference."""
        return self.dtype


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros
        along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x


def parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, 'c h w -> h w c')
    return image


def resize_with_pad(images: np.ndarray,
                    height: int,
                    width: int,
                    method: int = PILImage.BILINEAR) -> np.ndarray:
    """
    Resize a batch of images to the target size with
        padding to preserve aspect ratio.

    Args:
        images: Input images in shape [..., height, width, channels].
        height: Target height.
        width: Target width.
        method: Interpolation method (default: Image.BILINEAR).

    Returns:
        Resized and padded images of shape [..., height, width, channels].
    """
    if images.shape[-3:-1] == (height, width):
        return images  # Already target shape

    flat_images = images.reshape(-1, *images.shape[-3:])
    resized = np.stack([
        _resize_single_with_pad(im, height, width, method)
        for im in flat_images
    ])
    return resized.reshape(*images.shape[:-3], *resized.shape[-3:])


def _resize_single_with_pad(image: np.ndarray, target_height: int,
                            target_width: int, method: int) -> np.ndarray:
    """
    Resize a single image with padding using PIL.

    Args:
        image: Single image as ndarray (H, W, C).
        target_height: Desired output height.
        target_width: Desired output width.
        method: Interpolation method.

    Returns:
        Resized and padded image as ndarray.
    """
    pil_image = PILImage.fromarray(image)
    orig_w, orig_h = pil_image.size

    if (orig_w, orig_h) == (target_width, target_height):
        return np.array(pil_image)

    # Compute resize scale
    scale = max(orig_w / target_width, orig_h / target_height)
    new_w = int(orig_w / scale)
    new_h = int(orig_h / scale)

    resized_image = pil_image.resize((new_w, new_h), resample=method)

    # Create black background and paste resized image at center
    canvas = PILImage.new(resized_image.mode, (target_width, target_height), 0)
    offset_x = (target_width - new_w) // 2
    offset_y = (target_height - new_h) // 2
    canvas.paste(resized_image, (offset_x, offset_y))

    return np.array(canvas)
