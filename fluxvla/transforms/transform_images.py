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

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize
from transformers import AutoImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import \
    PILImageResampling
from transformers.models.qwen2_vl.image_processing_qwen2_vl import \
    Qwen2VLImageProcessor as Qwen2VLImageProcessorHF

from fluxvla.engines import TRANSFORMS


@TRANSFORMS.register_module()
class ResizeImages:
    """Resize images in the dataset to a specified
    height and width. This transform resizes all images
    in the 'image' dictionary of the input data
    to the specified dimensions while maintaining the
    aspect ratio by padding if necessary.

    Args:
        height (int): The target height for the images.
        width (int): The target width for the images.
    """

    def __init__(self, height, width, *args, **kwargs):
        self.height = height
        self.width = width

    def __call__(self, data: dict):
        assert 'images' in data, "Input data must contain 'images' key"
        if isinstance(data['images'], np.ndarray):
            assert data['images'].ndim == 3, \
                "Input 'images' must be a 4D numpy array"
            images = data['images'].reshape(-1, 3, data['images'].shape[-2],
                                            data['images'].shape[-1])

        else:
            images = data['images']
        resized_images = list()
        for image in images:
            resized_images.append(
                cv2.resize(
                    image.transpose(1, 2, 0), (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1))

        resized_images = np.concatenate(resized_images, axis=0)
        data['images'] = resized_images
        return data


@TRANSFORMS.register_module()
class ResizeImageSequence:

    def __init__(self, height: int, width: int, *args, **kwargs):
        self.height = height
        self.width = width

    def __call__(self, data: Dict) -> Dict:
        assert 'images' in data, "Input data must contain 'images' key"
        images = np.asarray(data['images'])
        original_shape = images.shape
        if images.ndim < 4:
            raise ValueError(
                'Input image sequence must have at least 4 dimensions')
        flat_images = images.reshape(-1, original_shape[-3],
                                     original_shape[-2], original_shape[-1])

        resized_images = []
        for image in flat_images:
            resized_images.append(
                cv2.resize(
                    image.transpose(1, 2, 0), (self.width, self.height),
                    interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1))

        data['images'] = np.stack(
            resized_images, axis=0).reshape(*original_shape[:-2], self.height,
                                            self.width)
        return data


@TRANSFORMS.register_module()
class AugImage:
    """Augment images with random transformations including
    rotation, brightness/contrast adjustment, and random cropping.
    This transform applies various augmentations to all images
    in the 'images' dictionary of the input data.

    Args:
        rotation_range (float): Maximum rotation angle in degrees.
            The image will be rotated by a random angle in
            [-rotation_range, rotation_range]. Default: 15.0.
        brightness_range (Tuple[float, float]): Range for brightness
            adjustment as (min, max) multipliers. Default: (0.8, 1.2).
        contrast_range (Tuple[float, float]): Range for contrast
            adjustment as (min, max) multipliers. Default: (0.8, 1.2).
        crop_scale (Tuple[float, float]): Range for random crop scale
            as (min, max) fractions of original size. Default: (0.8, 1.0).
        prob (float): Probability of applying each augmentation.
            Default: 0.5.
    """

    def __init__(self,
                 rotation_range: float = 15.0,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 crop_scale: Tuple[float, float] = (0.8, 1.0),
                 prob: float = 0.5,
                 *args,
                 **kwargs):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.crop_scale = crop_scale
        self.prob = prob

    def _random_rotate(self, image: np.ndarray) -> np.ndarray:
        """Apply random rotation to the image."""
        if np.random.random() > self.prob:
            return image
        # image shape: (C, H, W)
        h, w = image.shape[1], image.shape[2]
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Transpose to (H, W, C) for cv2
        img_hwc = image.transpose(1, 2, 0)
        rotated = cv2.warpAffine(
            img_hwc, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return rotated.transpose(2, 0, 1)

    def _random_brightness(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness adjustment to the image."""
        if np.random.random() > self.prob:
            return image
        factor = np.random.uniform(self.brightness_range[0],
                                   self.brightness_range[1])
        return np.clip(image * factor, 0, 255).astype(image.dtype)

    def _random_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply random contrast adjustment to the image."""
        if np.random.random() > self.prob:
            return image
        factor = np.random.uniform(self.contrast_range[0],
                                   self.contrast_range[1])
        mean = np.mean(image, axis=(1, 2), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0,
                       255).astype(image.dtype)

    def _random_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply random crop and resize back to original size."""
        if np.random.random() > self.prob:
            return image
        # image shape: (C, H, W)
        c, h, w = image.shape
        scale = np.random.uniform(self.crop_scale[0], self.crop_scale[1])
        new_h, new_w = int(h * scale), int(w * scale)

        # Random crop position
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        # Crop
        cropped = image[:, top:top + new_h, left:left + new_w]

        # Resize back to original size
        img_hwc = cropped.transpose(1, 2, 0)
        resized = cv2.resize(img_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized.transpose(2, 0, 1)

    def __call__(self, data: dict):
        assert 'images' in data, "Input data must contain 'images' key"
        if isinstance(data['images'], np.ndarray):
            if data['images'].ndim == 3:
                images = data['images'].reshape(-1, 3,
                                                data['images'].shape[-2],
                                                data['images'].shape[-1])
            else:
                images = data['images']
        else:
            images = data['images']

        augmented_images = list()
        for image in images:
            aug_image = image.copy()
            aug_image = self._random_rotate(aug_image)
            aug_image = self._random_brightness(aug_image)
            aug_image = self._random_contrast(aug_image)
            aug_image = self._random_crop(aug_image)
            augmented_images.append(aug_image)

        augmented_images = np.stack(augmented_images, axis=0)
        # Reshape back to original shape if needed
        if data['images'].ndim == 3:
            augmented_images = augmented_images.reshape(data['images'].shape)
        data['images'] = augmented_images
        return data


@TRANSFORMS.register_module()
class NormalizeImages:
    """Normalize images in the dataset using specified
    means and standard deviations. This transform normalizes
    all images in the 'image' dictionary of the input data
    using the provided means and standard deviations for each
    image.

    Args:
        means (List): List of means for normalization,
            where each element is a list of means for each channel.
        stds (List): List of standard deviations for normalization,
            where each element is a list of stds for each channel.
    """

    def __init__(self, means: List, stds: List, *args, **kwargs):
        self.means = np.array(means)
        self.stds = np.array(stds)

    def __call__(self, data: dict):
        assert 'images' in data, "Input data must contain 'images' key"
        images = data['images'].reshape(-1, 3, data['images'].shape[-2],
                                        data['images'].shape[-1])

        normalized_images = list()
        for idx, image in enumerate(images):
            normalized_image = (image - self.means[idx][:, None, None]) / (
                self.stds[idx][:, None, None] + 1e-8)
            normalized_images.append(normalized_image)

        normalized_images = np.concatenate(normalized_images, axis=0)
        data['images'] = normalized_images
        return data


@TRANSFORMS.register_module()
class NormalizeImageSequence:

    def __init__(self,
                 means: List,
                 stds: List,
                 scale_to_unit_interval: bool = False,
                 *args,
                 **kwargs):
        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)
        self.scale_to_unit_interval = scale_to_unit_interval

    def __call__(self, data: Dict) -> Dict:
        assert 'images' in data, "Input data must contain 'images' key"
        images = np.asarray(data['images'])
        original_shape = images.shape
        flat_images = images.reshape(-1, original_shape[-3],
                                     original_shape[-2],
                                     original_shape[-1]).astype(np.float32)
        if self.scale_to_unit_interval:
            flat_images = flat_images / 255.0

        means = self.means
        stds = self.stds
        if means.ndim == 1:
            means = np.broadcast_to(means[None, :], (flat_images.shape[0], 3))
        if stds.ndim == 1:
            stds = np.broadcast_to(stds[None, :], (flat_images.shape[0], 3))
        if means.shape[0] == 1:
            means = np.broadcast_to(means, (flat_images.shape[0], 3))
        if stds.shape[0] == 1:
            stds = np.broadcast_to(stds, (flat_images.shape[0], 3))
        if (means.shape[0] != flat_images.shape[0]
                or stds.shape[0] != flat_images.shape[0]):
            raise ValueError(
                'Means/stds must have length 1 or match the number '
                'of images after flattening.')

        normalized_images = []
        for idx, image in enumerate(flat_images):
            normalized_images.append((image - means[idx][:, None, None]) /
                                     (stds[idx][:, None, None] + 1e-8))

        data['images'] = np.stack(
            normalized_images, axis=0).reshape(original_shape)
        return data


@TRANSFORMS.register_module()
class SimpleNormalizeImages:
    """Simple normalization of images in the dataset.
    This transform normalizes all images in the 'images' dictionary
    by dividing by 255 and then mapping to the range [-1, 1].

    Args:
        None: This transform does not require any parameters.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data: dict):
        assert 'images' in data, "Input data must contain 'images' key"
        images = data['images'].reshape(-1, 3, data['images'].shape[-2],
                                        data['images'].shape[-1])

        normalized_images = list()
        for image in images:
            # Divide by 255 to get [0, 1], then map to [-1, 1]
            normalized_image = (image / 255.0) * 2.0 - 1.0
            normalized_images.append(normalized_image)

        normalized_images = np.concatenate(normalized_images, axis=0)
        data['images'] = normalized_images
        return data


@TRANSFORMS.register_module()
class TransformImage:
    """Image processor for Prismatic models.
    This class applies a series of transformations to images,
    including resizing, cropping, normalization, and padding.
    It supports different image resize strategies and can handle
    multiple input sizes, means, and standard deviations for
    normalization.

    Args:
        use_fused_vision_backbone (bool): Whether to use a
            fused vision backbone.
        image_resize_strategy (str): The strategy for
            resizing images. Options are 'resize-naive',
            'letterbox', and 'resize-crop'.
        input_sizes (Optional[List[Tuple[int, int, int]]]): List
            of input sizes for the images, where each size is
            a tuple of (channels, height, width).
        means (Optional[List[Tuple[float, float, float]]]): List
            of means for normalization,
            where each mean is a tuple of (mean_r, mean_g, mean_b).
        stds (Optional[List[Tuple[float, float, float]]]): List of
            standard deviations for normalization,
            where each std is a tuple of (std_r, std_g, std_b).
    """

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = 'letterbox',
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: str,
    ) -> None:
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        # Handle `None` default values
        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds
        assert len(input_sizes) == len(means) == len(stds), \
            'Input sizes, means, and stds must have the same length.'
        # Set parameters
        self.input_sizes, self.means, self.stds = input_sizes, means, stds  # noqa: E501

        # Initialize the parameters for transformations
        self.resize_params = list()
        self.crop_params = list()
        self.normalize_params = list()
        self.do_letterbox, self.letterbox_fill = False, None

        for idx in range(len(input_sizes)):
            self.resize_params.append({
                'size': input_sizes[idx][-2:],
                'interpolation': 'bilinear'
            })
            self.crop_params.append({'output_size': input_sizes[idx][-2:]})
            self.normalize_params.append({
                'mean': np.array(means[idx]),
                'std': np.array(stds[idx]),
                'inplace': False
            })
            self.do_letterbox, self.letterbox_fill = False, None

            # Handle Prismatic `image_resize_strategy`
            if self.image_resize_strategy == 'resize-naive':
                self.resize_params[idx]['size'] = (input_sizes[idx][-1],
                                                   input_sizes[idx][-1])
            elif self.image_resize_strategy == 'letterbox':
                self.do_letterbox, self.letterbox_fill = True, tuple(
                    [int(x * 255) for x in self.means[idx]])
            elif self.image_resize_strategy == 'resize-crop':
                pass
            else:
                raise ValueError(
                    f"Image resize strategy '{self.image_resize_strategy}' is not supported!"  # noqa: E501
                )

    def apply_transform(self, img: Image.Image, resize_param: Dict,
                        crop_param: Dict, normalize_param: Dict) -> np.ndarray:
        """Apply the image transformations to a single image.
        This method resizes the image, crops it to the specified
        output size, normalizes it, and returns the pixel values
        as a numpy array. It supports multiple transformations
        based on the `resize_params`, `crop_params`, and
        `normalize_params` defined during initialization.

        Args:
            img (Image.Image): The input image to be transformed.

        Returns:
            np.ndarray: The transformed pixel values as a numpy array."""
        if self.image_resize_strategy == 'resize-naive':
            # Resize without keeping the aspect ratio (naive resize)
            img_resized = img.resize(resize_param['size'],
                                     Image.Resampling.BILINEAR)
        else:
            if self.do_letterbox:
                img = self.letterbox_pad_transform(img, self.letterbox_fill)
            # Resize the image
            img_resized = img.resize(resize_param['size'],
                                     Image.Resampling.BILINEAR)

        # Center crop
        left = (img_resized.width - crop_param['output_size'][0]) // 2
        top = (img_resized.height - crop_param['output_size'][1]) // 2
        img_cropped = img_resized.crop(
            (left, top, left + crop_param['output_size'][0],
             top + crop_param['output_size'][1]))

        # Convert to numpy array (ToTensor equivalent)
        img_np = np.array(img_cropped).astype(np.float32)

        # Normalize
        mean, std = normalize_param['mean'], normalize_param['std']
        img_np = (img_np - mean) / std
        return img_np.transpose(2, 0, 1)  # Convert to (C, H, W) format

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[str] = None,
        **_: str,
    ) -> dict:
        """Preprocess images by applying transformations and returning
        pixel values.

        Args:
            images (Union[Image.Image, List[Image.Image]]): Single image
                or list of images to preprocess.
            return_tensors (Optional[str]): If specified, returns the
                pixel values as a tensor of the specified type.

        Returns:
            dict: A dictionary containing the pixel values of the
                processed images.
        """
        if not isinstance(images, list):
            images = [images]
        assert len(images) == len(self.input_sizes), \
            f'Expected {len(self.input_sizes)} images, but got {len(images)}.'
        # Apply transformation to each image.

        pixel_values = list()
        for idx, img in enumerate(images):
            if not isinstance(img, Image.Image):
                raise TypeError(
                    f'Expected PIL Image, but got {type(img)} instead.')
            pixel_values.append(
                self.apply_transform(
                    img.convert('RGB'), self.resize_params[idx],
                    self.crop_params[idx], self.normalize_params[idx]))

        return np.concatenate(pixel_values)

    def __call__(self, inputs: Dict, **kwargs) -> dict:
        images = inputs['pixel_values']
        inputs['pixel_values'] = torch.from_numpy(
            self.preprocess(images, **kwargs)).float()
        return inputs

    def letterbox_pad_transform(
            self, img: Image.Image, fill_color: Tuple[int, int,
                                                      int]) -> Image.Image:
        """Apply letterbox padding to the image to fit the target size.
        This method resizes the image to fit within the target dimensions
        while maintaining the aspect ratio, and pads the remaining
        area with a specified fill color.

        Args:
            img (Image.Image): The input image to be padded.
            fill_color (Tuple[int, int, int]): The RGB color to use
                for padding.

        Returns:
            Image.Image: The padded image with the target dimensions.
        """
        target_width, target_height = self.resize_params[0]['size']
        img.thumbnail((target_width, target_height), Image.Resampling.BILINEAR)
        new_img = Image.new('RGB', (target_width, target_height), fill_color)
        new_img.paste(img, ((target_width - img.width) // 2,
                            (target_height - img.height) // 2))

        return new_img


@TRANSFORMS.register_module()
class DinoSigLIPImageTransform:
    """
    Image transform for Dino and SigLIP datasets.
    This class applies the same image transformation to both
    Dino and SigLIP datasets. It uses the default image transformation
    configurations for both datasets. The images are resized to a specified
    target size and then the default transformations are applied.

    Args:
        dino_data_cfg (dict): Configuration for Dino dataset
            transformations.
        siglip_data_cfg (dict): Configuration for SigLIP dataset
            transformations.
        default_image_size (int): The target size to resize the images to.
    """

    def __init__(self, dino_data_cfg, siglip_data_cfg, default_image_size=224):
        self.dino_data_cfg = dino_data_cfg
        self.siglip_data_cfg = siglip_data_cfg
        default_dino_transform = timm.data.create_transform(
            **self.dino_data_cfg, is_training=False)
        default_siglip_transform = timm.data.create_transform(
            **self.siglip_data_cfg, is_training=False)
        assert isinstance(default_siglip_transform,
                          Compose), 'Unexpected `default_image_transform`!'
        assert isinstance(default_siglip_transform.transforms[0], Resize)
        default_siglip_transform = Compose([
            Resize(
                default_image_size,
                interpolation=default_siglip_transform.transforms[0].
                interpolation),
            *default_siglip_transform.transforms[1:],
        ])
        assert isinstance(
            default_dino_transform,
            Compose), 'Unexpected `default_dino_image_transform`!'
        assert isinstance(
            default_siglip_transform,
            Compose), 'Unexpected `default_siglip_image_transform`!'
        assert isinstance(default_dino_transform.transforms[0], Resize)
        assert isinstance(default_siglip_transform.transforms[0], Resize)

        target_size = (default_image_size, default_image_size)
        self.dino_image_transform = Compose([
            Resize(
                target_size,
                interpolation=default_dino_transform.transforms[0].
                interpolation),
            *default_dino_transform.transforms[1:],
        ])
        self.siglip_image_transform = Compose([
            Resize(
                target_size,
                interpolation=default_siglip_transform.transforms[0].
                interpolation),
            *default_siglip_transform.transforms[1:],
        ])

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        return {
            'dino': self.dino_image_transform(img, **kwargs),
            'siglip': self.siglip_image_transform(img, **kwargs)
        }


@TRANSFORMS.register_module()
class PretrainedImageTransform:
    """ Pretrained image transform class that uses an image processor
    from a pretrained model.
    This class wraps the `AutoImageProcessor` from the `transformers`
    library to apply the image transformations defined in the
    pretrained model. It can be used to preprocess images for models
    that require specific image transformations, such as resizing,
    normalization, and other augmentations.

    Args:
        model_path (str): Path to the pretrained model.
        trust_remote_code (bool): Whether to trust remote code when loading
            the model. Defaults to True.
    """

    def __init__(self, model_path: str, trust_remote_code: bool = True):
        self.img_transform = AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code).apply_transform

    def __call__(self, *args, **kwds):
        return self.img_transform(*args, **kwds)


@TRANSFORMS.register_module()
class QWenPretrainedImageTransform:
    """ Pretrained image transform class that uses an image processor
    from a pretrained model.
    This class wraps the `AutoImageProcessor` from the `transformers`
    library to apply the image transformations defined in the
    pretrained model. It can be used to preprocess images for models
    that require specific image transformations, such as resizing,
    normalization, and other augmentations.

    Args:
        model_path (str): Path to the pretrained model.
        trust_remote_code (bool): Whether to trust remote code when loading
            the model. Defaults to True.
    """

    def __init__(self, model_path: str, trust_remote_code: bool = True):
        self.img_transform = AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)

    def __call__(self, inputs):
        ret_dict = self.img_transform(inputs['images'])
        inputs['images'] = ret_dict['pixel_values']
        inputs['image_grid_thw'] = ret_dict['image_grid_thw']
        return inputs


@TRANSFORMS.register_module()
class ConvertPILImageToNumpyArray:
    """ Convert PIL image to numpy array.
    """

    def __init__(self, img_key: str = 'pixel_values'):
        self.img_key = img_key

    def __call__(self, inputs):
        inputs[self.img_key] = np.array(inputs[self.img_key]).transpose(
            0, 3, 1, 2)
        return inputs


@TRANSFORMS.register_module()
class PrepareVideo:
    """Reshape multi-view / temporal image arrays into the video layout
    expected by DreamZero: ``[C, T, H_tiled, W]`` (per sample, no batch dim).

    Camera views are tiled *vertically* (view-0 on top, view-1 on bottom).

    This transform should be placed **after** ``SimpleNormalizeImages`` (or any
    other pixel-level transform) so that the spatial content is final before
    rearrangement.

    Args:
        num_views (int): Number of camera views. Default: 2.
        frame_window_size (int): Number of temporal frames. Default: 1.
    """

    def __init__(self,
                 num_views: int = 2,
                 frame_window_size: int = 1,
                 *args,
                 **kwargs):
        self.num_views = num_views
        self.frame_window_size = frame_window_size

    def __call__(self, data: dict):
        # Support both 'images' (training) and 'pixel_values' (eval) keys
        if 'images' in data:
            img_key = 'images'
        elif 'pixel_values' in data:
            img_key = 'pixel_values'
        else:
            raise KeyError(
                "Input data must contain 'images' or 'pixel_values' key")
        images = data[img_key]
        V = self.num_views
        T = self.frame_window_size

        # Handle both numpy arrays and torch tensors
        is_tensor = isinstance(images, torch.Tensor)

        if images.ndim == 3:
            # [V*T*C, H, W] or [C, H, W]
            channels, h, w = images.shape
            if channels > 3 and channels % 3 == 0:
                n_items = channels // 3
                if T > 1 and n_items == V * T:
                    # [V*T*C, H, W] -> [V, T, 3, H, W] -> [3, T, V, H, W]
                    #                                    -> [3, T, V*H, W]
                    images = images.reshape(V, T, 3, h, w)
                    if is_tensor:
                        images = images.permute(2, 1, 0, 3, 4)
                    else:
                        images = images.transpose(2, 1, 0, 3, 4)
                    images = images.reshape(3, T, V * h, w)
                    data[img_key] = images
                    return data
                # [V*C, H, W] single timestep multi-view -> [3, 1, V*H, W]
                images = images.reshape(n_items, 3, h, w)
                # tile vertically: concat along H
                if is_tensor:
                    tiled = torch.cat([images[i] for i in range(n_items)],
                                      dim=1)  # [3, n*H, W]
                    data[img_key] = tiled.unsqueeze(1)  # [3, 1, n*H, W]
                else:
                    tiled = np.concatenate([images[i] for i in range(n_items)],
                                           axis=1)  # [3, n*H, W]
                    data[img_key] = tiled[:, np.newaxis, :, :]
                return data
            # [C, H, W] single view, single timestep -> [C, 1, H, W]
            if is_tensor:
                data[img_key] = images.unsqueeze(1)
            else:
                data[img_key] = images[:, np.newaxis, :, :]
            return data

        raise ValueError(f'Unsupported image shape: {images.shape}')


@TRANSFORMS.register_module()
class QWen2VLImageTransform:
    """ QWen2VL image transform class that uses an image processor
    from a pretrained model.
    This class wraps the `Qwen2VLImageProcessor` from the `transformers`
    library to apply the image transformations defined in the
    pretrained model. It can be used to preprocess images for models
    that require specific image transformations, such as resizing,
    normalization, and other augmentations.
    """
    """
    Constructs a Qwen2-VL image processor that dynamically resizes
        images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        size (`dict[str, int]`, *optional*, defaults to `{"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}`):  # noqa: E501
            Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):  # noqa: E501
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):  # noqa: E501
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.  # noqa: E501
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    model_input_names = [
        'pixel_values', 'image_grid_thw', 'pixel_values_videos',
        'video_grid_thw'
    ]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        img_key: str = 'images',
        to_tensor: bool = False,
        **kwargs,
    ) -> None:
        self.img_transform = Qwen2VLImageProcessorHF(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            **kwargs)
        self.img_key = img_key
        self.to_tensor = to_tensor

    def __call__(self, inputs):
        ret_dict = self.img_transform(inputs[self.img_key].reshape(
            -1, 3, inputs[self.img_key].shape[-2],
            inputs[self.img_key].shape[-1]))
        if self.to_tensor:
            inputs[self.img_key] = torch.from_numpy(ret_dict['pixel_values'])
            inputs['image_grid_thw'] = torch.from_numpy(
                ret_dict['image_grid_thw'])
        else:
            inputs[self.img_key] = ret_dict['pixel_values']
            inputs['image_grid_thw'] = ret_dict['image_grid_thw']
        return inputs
