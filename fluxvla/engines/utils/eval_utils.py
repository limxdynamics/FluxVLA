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

import math
import os
import time

import imageio
import numpy as np
import tensorflow as tf
import torch
from PIL import Image

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

OPENVLA_V01_SYSTEM_PROMPT = (
    'A chat between a curious user and an artificial intelligence assistant. '
    "The assistant gives helpful, detailed, and polite answers to the user's questions."  # noqa: E501
)

# TODO: Change to data pipeline.


def resize_image(img, resize_size):
    """
    Resizes an image to the specified size using TensorFlow.

    Args:
        img (np.ndarray): The input image to resize, expected
            to be in HWC format.
        resize_size (int or tuple): The target size for resizing.
            If an int is provided, the image will be resized
            to (resize_size, resize_size). If a tuple is provided,
            it should be in the format (height, width).

    Returns:
        np.ndarray: The resized image, clipped to the range [0, 255]
            and converted to uint8.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(
        img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(
        img, expand_animations=False,
        dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method='lanczos3', antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_env(task, resolution=256, controller='OSC_POSE'):
    """Initializes a Libero environment for a given task.

    Args:
        task: The task object containing the problem folder and BDDL file.
        resolution (int): The resolution for the camera images.

    Returns:
        env: The initialized Libero environment.
        task_description (str): The language description of the task.
    """
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    env_args = {
        'bddl_file_name': task_bddl_file,
        'camera_heights': resolution,
        'camera_widths': resolution,
        'controller': controller,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        0
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state  # noqa: E501
    return env, task_description


def get_libero_image(obs, resize_size, img_key='agentview_image'):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs[img_key]
    img = img[::-1, ::
              -1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def get_libero_dummy_action():
    """Returns a dummy action for the Libero environment.

    Returns:
        list: A dummy action consisting of zeros, which is suitable
            for the Libero environment.
    """
    return [0, 0, 0, 0, 0, 0, -1]


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55  # noqa: E501

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def crop_and_resize(image, crop_scale, batch_size):
    """Center-crops an image to have area `crop_scale` *
    (original image area), and then resizes back to original size.
    We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and
            datatype tf.float32 with values between [0,1].
        crop_scale: The area of the center crop with respect to the
            original image.
        batch_size: Batch size.

    Returns:
        image: TF Tensor of shape (batch_size, 224, 224, C) or (224, 224, C)
            and datatype tf.float32 with values between [0,1].
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(
        tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size, ))
    new_widths = tf.reshape(
        tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size, ))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes,
                                     tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla,
                   processor,
                   base_vla_name,
                   obs,
                   task_label,
                   unnorm_key,
                   device,
                   center_crop=False):
    """Predicts an action using the VLA model based on the provided
        observations and task label.

    Args:
        vla: The VLA model instance used for action prediction.
        processor: The processor used to prepare inputs for the VLA model.
        base_vla_name (str): The base name of the VLA model, used to determine
            the prompt format.
        obs (dict): Observations containing the full image and other
            relevant data.
        task_label (str): The label describing the task to be performed.
        unnorm_key (str): Key for unnormalizing actions.
        device: The device on which the model is
            running (e.g., 'cuda' or 'cpu').
        center_crop (bool): Whether to apply center cropping to the image.

    Returns:
        action: The predicted action from the VLA model.
    """
    image = Image.fromarray(obs['full_image'])
    image = image.convert('RGB')

    # (If trained with image augmentations) Center crop image and then
    # resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height
    # and width (post-crop), multiply
    # the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
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
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert('RGB')

    # Build VLA prompt
    if 'openvla-v01' in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f'{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:'  # noqa: E501
        )
    else:  # OpenVLA
        prompt = f'In: What action should the robot take to {task_label.lower()}?\nOut:'  # noqa: E501

    # Process inputs.
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    # Get action.
    action = vla.predict_action(
        **inputs, unnorm_key=unnorm_key, do_sample=False)
    return action


def save_rollout_video(rollout_images,
                       idx,
                       success,
                       task_description,
                       work_dir,
                       log_file=None):
    """Saves a video of the rollout images to a file.

    Args:
        rollout_images (list): List of images representing the rollout.
        idx (int): Episode index for naming the video file.
        success (bool): Whether the task was successful.
        task_description (str): Description of the task,
            used in the filename.
        work_dir (str): Directory where the video will be saved.
        log_file (file object, optional): File to log the save path.
            Defaults to None.

    Returns:
        str: The path to the saved video file.
    """
    date = time.strftime('%Y_%m_%d')
    date_time = time.strftime('%Y_%m_%d-%H_%M_%S')
    rollout_dir = os.path.join(work_dir, 'rollouts', date)
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(
        ' ', '_').replace('\n', '_').replace('.', '_')[:50]
    mp4_path = f'{rollout_dir}/{date_time}--episode={idx}--success={success}--task={processed_task_description}.mp4'  # noqa: E501
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f'Saved rollout MP4 at path {mp4_path}')
    if log_file is not None:
        log_file.write(f'Saved rollout MP4 at path {mp4_path}\n')
    return mp4_path
