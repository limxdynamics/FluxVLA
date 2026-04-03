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

import copy
import logging
import os
from pathlib import Path
from typing import Dict, List

import av
import numpy as np
import tensorflow as tf
import torch
import torchvision
from PIL import Image

from fluxvla.engines import TRANSFORMS
from fluxvla.engines.utils.eval_utils import crop_and_resize, get_libero_image
from .utils import pad_to_dim, parse_image


@TRANSFORMS.register_module()
class ProcessLiberoInputs():
    """Process inputs for Libero dataset.
    This transform processes the inputs from the Libero
    dataset to match the expected format for the model.
    It pads the state and action dimensions to the specified
    action dimension and parses the images from the input data.
    The processed inputs are returned in a dictionary format
    that includes the state, images, image masks, and
    actions (if available). The prompt is also included
    if it exists in the input data.

    Args:
        action_dim (int): The dimension to which the state and
            actions will be padded.
        model_type (str): The type of model being used, which
            may affect how images are masked.
    """

    def __init__(self, action_dim: int, model_type: str):
        self.action_dim = action_dim
        self.model_type = model_type

    def __call__(self, data):
        state = pad_to_dim(data['state'], self.action_dim)
        # TODO: Change to opencv
        base_image = parse_image(data['image'])
        wrist_image = parse_image(data['wrist_image'])

        # Create inputs dict. Do not change the keys
        # in the dict below.
        inputs = {
            'states': state,
            'images': [base_image, wrist_image],
            'img_masks': torch.tensor(([True, True]))
        }
        if 'actions' in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = pad_to_dim(data['actions'], self.action_dim)
            inputs['actions'] = actions

        # Pass the prompt (aka language instruction)
        # to the model.
        # Keep this for your own dataset (but modify
        # the key if the instruction is not
        # stored in "prompt"; the output dict always
        # needs to have the key "prompt").
        if 'prompt' in data:
            inputs['prompt'] = data['prompt']

        return inputs


@TRANSFORMS.register_module()
class ProcessParquetInputs():
    """Process inputs for Parquet dataset.
    This transform processes the inputs from the Parquet
    dataset to match the expected format for the model.
    It pads the state and action dimensions to the specified
    action dimension and parses the images from the input data.
    The processed inputs are returned in a dictionary format
    that includes the state, images, image masks, and
    actions (if available). The prompt is also included
    if it exists in the input data.

    Args:
        parquet_keys (List[str]): List of keys to extract
            from the parquet data.
        video_keys (List[str]): List of keys corresponding
            to video data.
        data_root (str): Root directory for the video files.
        name_mappings (Dict, optional): Optional dictionary
            to map original keys to new keys.
            Defaults to None.
    """

    def __init__(self,
                 parquet_keys: List[str],
                 video_keys: List[str],
                 name_mappings: Dict = None,
                 embodiment_id: int = None,
                 embodiment_dim: int = None,
                 num_padding_imgs: int = 0):
        self.parquet_keys = parquet_keys
        self.video_keys = video_keys
        self.name_mappings = name_mappings
        self.embodiment_id = embodiment_id
        self.embodiment_dim = embodiment_dim
        self.num_padding_imgs = num_padding_imgs

    def decode_video_frames_torchvision(
        self,
        video_path: Path | str,
        timestamps: list[float],
        tolerance_s: float,
        backend: str = 'pyav',
        log_loaded_timestamps: bool = False,
    ) -> torch.Tensor:
        """ Decode video frames using torchvision.

        Args:
            video_path (Path | str): Path to the video file.
            timestamps (list[float]): List of timestamps to decode.
            tolerance_s (float): Tolerance in seconds for the timestamps.
            backend (str): Backend to use for video decoding.
                Defaults to 'pyav'.
            log_loaded_timestamps (bool): Whether to log the loaded timestamps.
                Defaults to False.

        Returns:
            torch.Tensor: Tensor of decoded video frames.
        """
        video_path = str(video_path)

        # set backend
        keyframes_only = False
        torchvision.set_video_backend(backend)
        if backend == 'pyav':
            keyframes_only = True  # pyav doesn't support accuracte seek

        # set a video stream reader
        # TODO(rcadene): also load audio stream at the same time
        reader = torchvision.io.VideoReader(video_path, 'video')

        # set the first and last requested timestamps
        # Note: previous timestamps are usually loaded,
        # since we need to access the previous key frame
        first_ts = timestamps[0]
        last_ts = timestamps[-1]

        # access closest key frame of the first requested frame
        # Note: closest key frame timestamp is usually smaller than
        # `first_ts` (e.g. key frame can be the first frame of the video)
        # for details on what `seek` is doing see:
        # https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek  # noqa: E501
        reader.seek(first_ts, keyframes_only=keyframes_only)

        # load all frames until last requested frame
        loaded_frames = []
        loaded_ts = []
        for frame in reader:
            current_ts = frame['pts']
            if log_loaded_timestamps:
                logging.info(f'frame loaded at timestamp={current_ts:.4f}')
            loaded_frames.append(frame['data'])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

        if backend == 'pyav':
            reader.container.close()

        reader = None

        query_ts = torch.tensor(timestamps)
        loaded_ts = torch.tensor(loaded_ts)

        # compute distances between each query timestamp
        # and timestamps of all loaded frames
        dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
        min_, argmin_ = dist.min(1)

        is_within_tol = min_ < tolerance_s
        assert is_within_tol.all(), (
            f'One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=}).'  # noqa: E501
            'It means that the closest frame that can be loaded from the video is too far away in time.'  # noqa: E501
            'This might be due to synchronization issues with timestamps during data collection.'  # noqa: E501
            'To be safe, we advise to ignore this item during training.'
            f'\nqueried timestamps: {query_ts}'
            f'\nloaded timestamps: {loaded_ts}'
            f'\nvideo: {video_path}'
            f'\nbackend: {backend}')

        # get closest frames to the query timestamps
        closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
        closest_ts = loaded_ts[argmin_]

        if log_loaded_timestamps:
            logging.info(f'{closest_ts=}')

        # convert to the pytorch format which is float32 in
        # [0,1] range (and channel first)
        closest_frames = closest_frames

        assert len(timestamps) == len(closest_frames)
        return closest_frames

    def __call__(self, data):
        assert 'info' in data, "Input data must contain 'info' key"
        info = data['info']
        inputs = dict()
        # Check if the video path is provided in the info
        assert 'video_path' in info, "Input data must contain 'video_path' key"
        video_root_path = info['video_path']
        for key in self.parquet_keys:
            assert key in data, f'Key {key} not found in input data'
            if self.name_mappings is not None and key in self.name_mappings:
                if isinstance(self.name_mappings[key], str):
                    if isinstance(data[key], list) or isinstance(
                            data[key], float):
                        inputs[self.name_mappings[key]] = np.array(data[key])
                    else:
                        inputs[self.name_mappings[key]] = data[key]
                else:
                    for mapped_key in self.name_mappings[key]:
                        if isinstance(data[key], list) or isinstance(
                                data[key], float):
                            inputs[mapped_key] = np.array(data[key])
                        else:
                            inputs[mapped_key] = data[key]
            else:
                if isinstance(data[key], list) or isinstance(data[key], float):
                    inputs[key] = np.array(data[key])
                else:
                    inputs[key] = data[key]
        images = list()
        img_masks = list()
        for video_key in self.video_keys:
            episode_chunk = data['episode_index'] // data['info'][
                'chunks_size']  # noqa: E501
            video_path = os.path.join(
                data['data_root'],
                video_root_path.format(
                    episode_chunk=episode_chunk,
                    video_key=video_key,
                    episode_index=data['episode_index']))
            assert os.path.exists(
                video_path), f'Video file not found: {video_path}'
            frame = self.decode_video_frames_torchvision(
                video_path, [data['timestamp']],
                0.1)[0]  # Convert to HWC format
            images.append(frame.numpy())
            img_masks.append(True)
        # Add padding images with zero values and False masks
        if self.num_padding_imgs > 0 and len(images) > 0:
            padding_img = np.zeros_like(images[0])
            for _ in range(self.num_padding_imgs):
                images.append(padding_img)
                img_masks.append(False)
        inputs['images'] = images
        inputs['img_masks'] = np.array(img_masks)
        inputs['task_description'] = data.get('task_description', '')
        if self.embodiment_id is not None:
            inputs['embodiment_ids'] = np.array(self.embodiment_id)

        return inputs

    def read_video_frame(self, video_path: str, frame_idx: int):
        container = av.open(video_path)
        for i, frame in enumerate(container.decode(video=0)):
            if i == frame_idx:
                return frame.to_ndarray(format='rgb24')


@TRANSFORMS.register_module()
class ProcessOBSInputs():
    """Process inputs for OBS dataset.
    This transform processes the inputs from the OBS dataset
    to match the expected format for the model.
    It pads the state and action dimensions to the specified
    action dimension and parses the images from the input data.
    The processed inputs are returned in a dictionary format
    that includes the state, images, image masks, and
    actions (if available). The prompt is also included
    if it exists in the input data.

    Args:
        action_dim (int): The dimension to which the state and
            actions will be padded.
        model_type (str): The type of model being used, which
            may affect how images are masked.
    """

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def __call__(self, inputs):
        inputs['states'] = torch.from_numpy(
            pad_to_dim(inputs['states'], self.action_dim))

        return inputs


# === Libero-specific Image Loader Transform ===
@TRANSFORMS.register_module()
class ProcessLiberoEvalInputs:
    """ Process Libero eval inputs.
    This class processes the Libero eval inputs by loading the images,
    applying the center crop, and returning the processed inputs.

    Args:
        img_keys (List[str]): Image keys to fetch from inputs.
            Default to ['agentview_image'].
        center_crop (bool): If True, center crop at 0.9 area before resize.
            Default to False.
        use_pil (bool): If True, use PIL to load the images.
            Default to True.
    """

    def __init__(self,
                 img_keys: List[str] = ['agentview_image'],
                 resize_size: int = 224,
                 center_crop: bool = False,
                 use_pil: bool = True,
                 embodiment_id: int = None) -> None:
        self.img_keys = img_keys
        self.resize_size = resize_size
        self.center_crop = center_crop
        self.use_pil = use_pil
        self.embodiment_id = embodiment_id

    def __call__(self, inputs: Dict) -> Dict:
        # Load raw images
        imgs = list()
        for img_key in self.img_keys:
            if img_key not in inputs:
                raise KeyError(f'Image key `{img_key}` not found in inputs!')
            imgs.append(get_libero_image(inputs, self.resize_size, img_key))
        replay_img = copy.deepcopy(imgs[0])
        images = list()
        img_masks = list()
        if self.use_pil:
            for img in imgs:
                image = Image.fromarray(img)
                image = image.convert('RGB')

                # (If trained with image augmentations)
                # Center crop image and then
                # resize back up to original size.
                # IMPORTANT: Let's say crop scale == 0.9. To get the new height
                # and width (post-crop), multiply
                # the original height and width by sqrt(0.9) -- not 0.9!
                if self.center_crop:
                    batch_size = 1
                    crop_scale = 0.9

                    # Convert to TF Tensor and record original data type
                    # (should be tf.uint8)
                    image = tf.convert_to_tensor(np.array(image))
                    orig_dtype = image.dtype

                    # Convert to data type tf.float32 and values between [0,1]
                    image = tf.image.convert_image_dtype(image, tf.float32)

                    # Crop and then resize back to original size
                    image = crop_and_resize(image, crop_scale, batch_size)

                    # Convert back to original data type
                    image = tf.clip_by_value(image, 0, 1)
                    image = tf.image.convert_image_dtype(
                        image, orig_dtype, saturate=True)

                    # Convert back to PIL Image
                    image = Image.fromarray(image.numpy())
                    image = image.convert('RGB')
                images.append(image)
                img_masks.append(True)
        else:
            images = imgs
            img_masks = [True] * len(imgs)
        inputs['pixel_values'] = images
        inputs['img_masks'] = img_masks
        inputs['replay_img'] = replay_img
        if self.embodiment_id is not None:
            inputs['embodiment_ids'] = np.array(
                self.embodiment_id, dtype=np.int32)
        return inputs


@TRANSFORMS.register_module()
class PadKeyToDim():
    """
    Pad the tensor of the specified keys in the input to an integer
        multiple of its current length, and fill the target dimension
        by copying the original tensor.

    Args:
        keys (List[str]): List of keys in the input dictionary
            to be padded.
        dim (int): The target dimension should be an integer
            multiple of the current length.
    """

    def __init__(self, keys: List[str], dim: int):
        self.keys = keys
        self.dim = dim

    def __call__(self, inputs):
        for key in self.keys:
            if key in inputs:
                tensor = inputs[key]
                orig_shape = tensor.shape
                orig_len = orig_shape[-1]
                target_len = ((orig_len + self.dim - 1) // self.dim) * self.dim
                if target_len == orig_len:
                    inputs[key] = tensor
                    continue
                # Pad by copying the entire original tensor to reach the
                # target length
                repeat_times = (target_len + orig_len - 1) // orig_len
                repeat_target = [1] * len(orig_shape)
                repeat_target[-1] = repeat_times
                tensor_padded = np.tile(tensor, repeat_target)
                inputs[key] = tensor_padded
        return inputs
