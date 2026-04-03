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
"""Episode transforms for DROID dataset."""

from typing import Any, Dict

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg


def rmat_to_euler(rot_mat):
    """
    Converts a rotation matrix to Euler angles.

    Args:
        rot_mat: A rotation matrix of shape [..., 3, 3].

    Returns:
        tf.Tensor: Euler angles (yaw, pitch, roll) of shape [..., 3].
    """
    return tfg.euler.from_rotation_matrix(rot_mat)


def euler_to_rmat(euler):
    """
    Converts Euler angles to a rotation matrix.

    Args:
        euler: A tensor of Euler angles of shape [..., 3].

    Returns:
        tf.Tensor: Rotation matrix of shape [..., 3, 3].
    """
    return tfg.rotation_matrix_3d.from_euler(euler)


def invert_rmat(rot_mat):
    """
    Inverts a rotation matrix.

    Args:
        rot_mat: A rotation matrix of shape [..., 3, 3].

    Returns:
        tf.Tensor: The inverse of the rotation matrix.
    """
    return tfg.rotation_matrix_3d.inverse(rot_mat)


def rotmat_to_rot6d(mat):
    """
    Converts a rotation matrix to 6D representation (first 2 rows).

    Args:
        mat: A rotation matrix of shape [..., 3, 3].

    Returns:
        tf.Tensor: 6D rotation vector of shape [..., 6].
    """
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def velocity_act_to_wrist_frame(velocity, wrist_in_robot_frame):
    """
    Converts velocity action from robot base frame to wrist frame.

    Args:
        velocity: A tensor of shape [N, 6] representing base-frame velocity.
        wrist_in_robot_frame: A tensor of shape [N, 6] for wrist pose in base.

    Returns:
        tf.Tensor: A tensor of shape [N, 9] with velocity in wrist frame.
    """
    R_frame = euler_to_rmat(wrist_in_robot_frame[:, 3:6])
    R_frame_inv = invert_rmat(R_frame)

    vel_t = (R_frame_inv @ velocity[:, :3][..., None])[..., 0]

    dR = euler_to_rmat(velocity[:, 3:6])
    dR = R_frame_inv @ (dR @ R_frame)
    dR_r6 = rotmat_to_rot6d(dR)
    return tf.concat([vel_t, dR_r6], axis=-1)


def rand_swap_exterior_images(img1, img2):
    """
    Randomly swaps exterior image pairs for training.

    Args:
        img1: First exterior image tensor.
        img2: Second exterior image tensor.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Either (img1, img2) or (img2, img1).
    """
    return tf.cond(
        tf.random.uniform(shape=[]) > 0.5, lambda: (img1, img2), lambda:
        (img2, img1))


def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms DROID action from base frame. Adds proprio and swaps images.

    Args:
        trajectory: A trajectory dict with 'action_dict' and 'observation'.

    Returns:
        Dict[str, Any]: The modified trajectory dict.
    """
    dt = trajectory['action_dict']['cartesian_velocity'][:, :3]
    dR = trajectory['action_dict']['cartesian_velocity'][:, 3:6]

    trajectory['action'] = tf.concat(
        (dt, dR, 1 - trajectory['action_dict']['gripper_position']), axis=-1)

    obs = trajectory['observation']
    obs['exterior_image_1_left'], obs['exterior_image_2_left'] = \
        rand_swap_exterior_images(obs['exterior_image_1_left'],
                                  obs['exterior_image_2_left'])

    obs['proprio'] = tf.concat(
        (obs['cartesian_position'], obs['gripper_position']), axis=-1)

    return trajectory


def droid_wristact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms DROID action from base to wrist frame. Adds proprio and swaps
    images.

    Args:
        trajectory: A trajectory dict with 'action_dict' and 'observation'.

    Returns:
        Dict[str, Any]: The modified trajectory dict.
    """
    wrist_act = velocity_act_to_wrist_frame(
        trajectory['action_dict']['cartesian_velocity'],
        trajectory['observation']['cartesian_position'],
    )

    trajectory['action'] = tf.concat(
        (wrist_act, trajectory['action_dict']['gripper_position']), axis=-1)

    obs = trajectory['observation']
    obs['exterior_image_1_left'], obs['exterior_image_2_left'] = \
        rand_swap_exterior_images(obs['exterior_image_1_left'],
                                  obs['exterior_image_2_left'])

    obs['proprio'] = tf.concat(
        (obs['cartesian_position'], obs['gripper_position']), axis=-1)

    return trajectory


def droid_finetuning_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms action for fine-tuning in base frame. Adds proprio only.

    Args:
        trajectory: A trajectory dict with 'action_dict' and 'observation'.

    Returns:
        Dict[str, Any]: The modified trajectory dict.
    """
    dt = trajectory['action_dict']['cartesian_velocity'][:, :3]
    dR = trajectory['action_dict']['cartesian_velocity'][:, 3:6]

    trajectory['action'] = tf.concat(
        (dt, dR, 1 - trajectory['action_dict']['gripper_position']), axis=-1)

    obs = trajectory['observation']
    obs['proprio'] = tf.concat(
        (obs['cartesian_position'], obs['gripper_position']), axis=-1)

    return trajectory


def zero_action_filter(traj: Dict) -> bool:
    """
    Filters out actions that are near-zero in all dimensions except gripper.

    This function compares to normalized zero-action after bounds_q99
    normalization.

    Args:
        traj: A trajectory dict with normalized 'action' field.

    Returns:
        bool: True if action has non-zero movement (ignoring gripper).
    """
    DROID_Q01 = tf.convert_to_tensor(
        [-0.7776, -0.5803, -0.5795, -0.6464, -0.7041, -0.8895])
    DROID_Q99 = tf.convert_to_tensor(
        [0.7598, 0.5726, 0.7351, 0.6706, 0.6465, 0.8898])

    DROID_NORM_0_ACT = 2 * (tf.zeros_like(traj['action'][:, :6]) -
                            DROID_Q01) / (DROID_Q99 - DROID_Q01 + 1e-8) - 1

    return tf.reduce_any(
        tf.math.abs(traj['action'][:, :6] - DROID_NORM_0_ACT) > 1e-5)
