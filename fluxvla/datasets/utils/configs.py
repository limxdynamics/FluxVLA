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
configs.py

Defines per-dataset configuration (kwargs) for each
    dataset in Open-X Embodiment.

Configuration adopts the following structure:
    image_obs_keys:
        primary: primary external RGB
        secondary: secondary external RGB
        wrist: wrist RGB

    depth_obs_keys:
        primary: primary external depth
        secondary: secondary external depth
        wrist: wrist depth

    # Always 8-dim =>> changes based on `StateEncoding`
    state_obs_keys:
        StateEncoding.POS_EULER:    EEF XYZ (3) +
            Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
        StateEncoding.POS_QUAT:     EEF XYZ (3) +
            Quaternion (4) + Gripper Open/Close (1)
        StateEncoding.JOINT:        Joint Angles
            (7, <PAD> if fewer) + Gripper Open/Close (1)

    state_encoding: Type of `StateEncoding`
    action_encoding: Type of action encoding (e.g.,
        EEF Position vs. Joint Position)
"""
import logging
from enum import IntEnum
from typing import Any, Dict, Tuple, Union

import dlimp as dl
import tensorflow as tf

from .data_utils import (binarize_gripper_actions, invert_gripper_actions,
                         rel2abs_gripper_actions, relabel_bridge_actions)
from .droid_utils import droid_baseact_transform, droid_finetuning_transform


def zero_action_filter(traj: Dict) -> bool:
    """Filters transitions whose actions are all-0 (only
        relative actions, no gripper action).
    Note: this filter is applied *after* action normalization,
        so need to compare to "normalized 0".

    Args:
        traj: A dictionary containing trajectory data.
            Should have an "action" key.
    """
    DROID_Q01 = tf.convert_to_tensor([
        -0.7776297926902771,
        -0.5803514122962952,
        -0.5795090794563293,
        -0.6464047729969025,
        -0.7041108310222626,
        -0.8895104378461838,
    ])
    DROID_Q99 = tf.convert_to_tensor([
        0.7597932070493698,
        0.5726242214441299,
        0.7351000607013702,
        0.6705610305070877,
        0.6464948207139969,
        0.8897542208433151,
    ])
    DROID_NORM_0_ACT = 2 * (tf.zeros_like(traj['action'][:, :6]) -
                            DROID_Q01) / (DROID_Q99 - DROID_Q01 + 1e-8) - 1

    return tf.reduce_any(
        tf.math.abs(traj['action'][:, :6] - DROID_NORM_0_ACT) > 1e-5)


# Defines Proprioceptive State Encoding Schemes
class StateEncoding(IntEnum):
    # fmt: off
    NONE = -1  # No Proprioceptive State
    # EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) +
    # Gripper Open/Close (1)
    POS_EULER = 1
    # EEF XYZ (3) + Quaternion (4) +
    # Gripper Open/Close (1)
    POS_QUAT = 2
    # Joint Angles (7, <PAD> if fewer) +
    # Gripper Open/Close (1)
    JOINT = 3
    # Joint Angles (2 x [ Joint Angles (6) +
    # Gripper Open/Close (1) ])
    JOINT_BIMANUAL = 4
    # fmt: on


# Defines Action Encoding Schemes
class ActionEncoding(IntEnum):
    # fmt: off
    # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3)
    # + Gripper Open/Close (1)
    EEF_POS = 1
    # Joint Delta Position (7) +
    # Gripper Open/Close (1)
    JOINT_POS = 2
    # Joint Delta Position (2 x [ Joint
    # Delta Position (6) + Gripper Open/Close (1) ])
    JOINT_POS_BIMANUAL = 3
    # EEF Delta XYZ (3) + R6 (6) +
    # Gripper Open/Close (1)
    EEF_R6 = 4
    # fmt: on


# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
    'fractal20220817_data': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['base_pose_tool_reached',
                           'gripper_closed'],  # noqa: E128
        'state_encoding': StateEncoding.POS_QUAT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'kuka': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': [
            'clip_function_input/base_pose_tool_reached',
            'gripper_closed',
        ],
        'state_encoding':
        StateEncoding.POS_QUAT,
        'action_encoding':
        ActionEncoding.EEF_POS,
    },
    'bridge_oxe': {
        # Version of Bridge V2 in Open X-Embodiment mixture
        'image_obs_keys': {
            'primary': 'image',
            'secondary': 'image_1',
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', None, 'gripper_state'],  # noqa: E128
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'bridge_orig': {
        # Original version of Bridge V2 from project website
        'image_obs_keys': {
            'primary': 'image_0',
            'secondary': 'image_1',
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', None, 'gripper_state'],  # noqa: E128
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'bridge_dataset': {
        # Original version of Bridge V2 from project website
        'image_obs_keys': {
            'primary': 'image_0',
            'secondary': 'image_1',
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'taco_play': {
        'image_obs_keys': {
            'primary': 'rgb_static',
            'secondary': None,
            'wrist': 'rgb_gripper',
        },
        'depth_obs_keys': {
            'primary': 'depth_static',
            'secondary': None,
            'wrist': 'depth_gripper',
        },
        'state_obs_keys': ['state_eef', None, 'state_gripper'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'jaco_play': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'image_wrist',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },
        'state_obs_keys': ['state_eef', None, 'state_gripper'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'berkeley_cable_routing': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': 'top_image',
            'wrist': 'wrist45_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },
        'state_obs_keys': ['robot_state', None],
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'roboturk': {
        'image_obs_keys': {
            'primary': 'front_rgb',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': [None, None, None, None, None, None, None,
                           None],  # noqa: E128
        'state_encoding': StateEncoding.NONE,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'nyu_door_opening_surprising_effectiveness': {
        'image_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': 'image'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': [None, None, None, None, None, None, None,
                           None],  # noqa: E128
        'state_encoding': StateEncoding.NONE,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'viola': {
        'image_obs_keys': {
            'primary': 'agentview_rgb',
            'secondary': None,
            'wrist': 'eye_in_hand_rgb',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['joint_states', 'gripper_states'],
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'berkeley_autolab_ur5': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'hand_image',
        },
        'depth_obs_keys': {
            'primary': 'depth',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state'],
        'state_encoding': StateEncoding.POS_QUAT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'toto': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state', None],
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'language_table': {
        'image_obs_keys': {
            'primary': 'rgb',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys':
        ['effector_translation', None, None, None, None, None,
         None],  # noqa: E128
        'state_encoding':
        StateEncoding.POS_EULER,
        'action_encoding':
        ActionEncoding.EEF_POS,
    },
    'columbia_cairlab_pusht_real': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['robot_state', None, None, None, None, None,
                           None],  # noqa: E128
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'stanford_kuka_multimodal_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': 'depth_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['ee_position', 'ee_orientation', None],
        'state_encoding': StateEncoding.POS_QUAT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'nyu_rot_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],  # noqa: E128
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'stanford_hydra_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'austin_buds_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state'],
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'nyu_franka_play_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': 'image_additional_view',
            'wrist': None,
        },
        'depth_obs_keys': {
            'primary': 'depth',
            'secondary': 'depth_additional_view',
            'wrist': None,
        },
        'state_obs_keys': ['eef_state', None, None],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'maniskill_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': 'depth',
            'secondary': None,
            'wrist': 'wrist_depth',
        },
        'state_obs_keys': ['tcp_pose', 'gripper_state'],
        'state_encoding': StateEncoding.POS_QUAT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'furniture_bench_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state'],
        'state_encoding': StateEncoding.POS_QUAT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'cmu_franka_exploration_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'highres_image',
            'secondary': None,
            'wrist': None,
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': [None, None, None, None, None, None, None,
                           None],  # noqa: E128
        'state_encoding': StateEncoding.NONE,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'ucsd_kitchen_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['joint_state', None],
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],  # noqa: E128
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'austin_sailor_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },
        'state_obs_keys': ['state'],
        'state_encoding': StateEncoding.POS_QUAT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'austin_sirius_dataset_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },
        'state_obs_keys': ['state'],
        'state_encoding': StateEncoding.POS_QUAT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'bc_z': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': [
            'present/xyz',
            'present/axis_angle',
            None,
            'present/sensed_close',
        ],
        'state_encoding':
        StateEncoding.POS_EULER,
        'action_encoding':
        ActionEncoding.EEF_POS,
    },
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': 'image2',
            'wrist': 'hand_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['end_effector_pose', None, None],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'utokyo_xarm_bimanual_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['pose_r', None, None],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'robo_net': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': 'image1',
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'berkeley_mvp_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': 'hand_image'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['pose', 'gripper'],
        'state_encoding': StateEncoding.POS_QUAT,
        'action_encoding': ActionEncoding.JOINT_POS,
    },
    'berkeley_rpt_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': 'hand_image'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['joint_pos', 'gripper'],
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.JOINT_POS,
    },
    'kaist_nonprehensile_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state', None],
        'state_encoding': StateEncoding.POS_QUAT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'stanford_mask_vit_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],  # noqa: E128
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'tokyo_u_lsmo_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'dlr_sara_pour_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state', None, None],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'dlr_sara_grid_clamp_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state', None, None],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'dlr_edan_shared_control_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state', None],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'asu_table_top_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],  # noqa: E128
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'stanford_robocook_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image_1',
            'secondary': 'image_2',
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': 'depth_1',
            'secondary': 'depth_2',
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],  # noqa: E128
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'imperialcollege_sawyer_wrist_cam': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': [None, None, None, None, None, None, None,
                           'state'],  # noqa: E128
        'state_encoding': StateEncoding.NONE,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['joint_state', 'gripper_state'],
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'uiuc_d3field': {
        'image_obs_keys': {
            'primary': 'image_1',
            'secondary': 'image_2',
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': 'depth_1',
            'secondary': 'depth_2',
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': [None, None, None, None, None, None, None,
                           None],  # noqa: E128
        'state_encoding': StateEncoding.NONE,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'utaustin_mutex': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state'],
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'berkeley_fanuc_manipulation': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['joint_state', None, 'gripper_state'],  # noqa: E128
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'cmu_playing_with_food': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'finger_vision_1',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state', None, None],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'cmu_play_fusion': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state'],
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'cmu_stretch': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['eef_state', None, 'gripper_state'],  # noqa: E128
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'berkeley_gnm_recon': {
        'image_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': 'image'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state', None, None],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'berkeley_gnm_cory_hall': {
        'image_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': 'image'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state', None, None],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'berkeley_gnm_sac_son': {
        'image_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': 'image'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['state', None, None],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'droid': {
        'image_obs_keys': {
            'primary': 'exterior_image_1_left',
            'secondary': 'exterior_image_2_left',
            'wrist': 'wrist_image_left',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['proprio'],
        'state_encoding': StateEncoding.POS_QUAT,
        'action_encoding': ActionEncoding.EEF_POS,
        'aux_kwargs': {
            'dataset_frame_transform_kwargs': {
                'chunk_filter_fn': zero_action_filter,
            },
        },
    },
    'fmb_dataset': {
        'image_obs_keys': {
            'primary': 'image_side_1',
            'secondary': 'image_side_2',
            'wrist': 'image_wrist_1',
        },
        'depth_obs_keys': {
            'primary': 'image_side_1_depth',
            'secondary': 'image_side_2_depth',
            'wrist': 'image_wrist_1_depth',
        },
        'state_obs_keys': ['proprio'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'dobbe': {
        'image_obs_keys': {
            'primary': 'wrist_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['proprio'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'roboset': {
        'image_obs_keys': {
            'primary': 'image_left',
            'secondary': 'image_right',
            'wrist': 'image_wrist',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['proprio'],
        'state_encoding': StateEncoding.JOINT,
        'action_encoding': ActionEncoding.JOINT_POS,
    },
    'rh20t': {
        'image_obs_keys': {
            'primary': 'image_front',
            'secondary': 'image_side_right',
            'wrist': 'image_wrist',
        },
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['proprio'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    # T-DROID datasets
    'tdroid_carrot_in_bowl': {
        # "put carrot in bowl" task,
        # 50 demos @ 5 Hz control
        'image_obs_keys': {
            'primary': 'static_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': 'static_depth_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'tdroid_pour_corn_in_pot': {
        # "pour corn from red bowl into steel pot" task,
        # 50 demos @ 5 Hz control
        'image_obs_keys': {
            'primary': 'static_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': 'static_depth_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'tdroid_flip_pot_upright': {
        # "flip pot upright" task, 10 demos @ 5 Hz control
        'image_obs_keys': {
            'primary': 'static_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': 'static_depth_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'tdroid_move_object_onto_plate': {
        # "move <object> onto plate" task, 150 demos @ 5 Hz control
        'image_obs_keys': {
            'primary': 'static_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': 'static_depth_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'tdroid_knock_object_over': {
        # "knock <object> over" task, 70 demos @ 5 Hz control
        'image_obs_keys': {
            'primary': 'static_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': 'static_depth_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', None, 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'tdroid_cover_object_with_towel': {
        # "cover <object> with towel" task, 45 demos @ 5 Hz control
        'image_obs_keys': {
            'primary': 'static_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': 'static_depth_image',
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', None, 'gripper_state'],  # noqa: E128
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    # DROID Finetuning datasets
    'droid_wipe': {
        'image_obs_keys': {
            'primary': 'exterior_image_2_left',
            'secondary': None,
            'wrist':  # noqa: E128
            'wrist_image_left'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['proprio'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    # LIBERO datasets (modified versions)
    'libero_spatial_no_noops': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'libero_object_no_noops': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'libero_goal_no_noops': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
    'libero_10_no_noops': {
        'image_obs_keys': {
            'primary': 'image',
            'secondary': None,
            'wrist': 'wrist_image'
        },  # noqa: E128
        'depth_obs_keys': {
            'primary': None,
            'secondary': None,
            'wrist': None
        },  # noqa: E128
        'state_obs_keys': ['EEF_state', 'gripper_state'],
        'state_encoding': StateEncoding.POS_EULER,
        'action_encoding': ActionEncoding.EEF_POS,
    },
}


def augment(obs: Dict, seed: tf.Tensor,
            augment_kwargs: Union[Dict, Dict[str, Dict]]) -> Dict:
    """Augments images, skipping padding images.

    Args:
        obs: Dictionary of observations.
        seed: Random seed for augmentation.
        augment_kwargs: Dictionary of augmentation parameters.
            If a single dictionary is provided, it will
            be used for all images. If a dictionary of dictionaries
            is provided, each image will be augmented with the
            corresponding dictionary.

    Returns:
        Dict: Dictionary of observations with augmented images.
    """
    image_names = {key[6:] for key in obs if key.startswith('image_')}

    # "augment_order" is required in augment_kwargs, so if it's there,
    # we can assume that the user has passed
    # in a single augmentation dict (otherwise, we assume that the
    # user has passed in a mapping from image
    # name to augmentation dict)
    if 'augment_order' in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    for i, name in enumerate(image_names):
        if name not in augment_kwargs:
            continue
        kwargs = augment_kwargs[name]
        logging.debug(f'Augmenting image_{name} with kwargs {kwargs}')
        obs[f'image_{name}'] = tf.cond(
            obs['pad_mask_dict'][f'image_{name}'],
            lambda: dl.transforms.augment_image(
                obs[f'image_{name}'],
                **kwargs,
                seed=seed + i,  # augment each image differently
            ),
            lambda: obs[f'image_{name}'],  # skip padding images
        )

    return obs


def decode_and_resize(
    obs: Dict,
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],
) -> Dict:
    """Decodes images and depth images, and then optionally resizes them.

    Args:
        obs: Dictionary of observations.
        resize_size: Size to resize images to. If a tuple, all images will
            be resized to this size. If a dict, each image will be resized
            to the size specified in the dict.
        depth_resize_size: Size to resize depth images to. If a tuple, all
            depth images will be resized to this size. If a dict, each depth
            image will be resized to the size specified in the dict.
    """
    image_names = {key[6:] for key in obs if key.startswith('image_')}
    depth_names = {key[6:] for key in obs if key.startswith('depth_')}

    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    if isinstance(depth_resize_size, tuple):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name in image_names:
        if name not in resize_size:
            logging.warning(f'No resize_size was provided for image_{name}. \
                    This will result in 1x1 '
                            'padding images, which may cause errors if you \
                    mix padding and non-padding images.')
        image = obs[f'image_{name}']
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                # this is a padding image
                image = tf.zeros((*resize_size.get(name, (1, 1)), 3),
                                 dtype=tf.uint8)
            else:
                image = tf.io.decode_image(
                    image, expand_animations=False, dtype=tf.uint8)
        elif image.dtype != tf.uint8:
            raise ValueError(f'Unsupported image dtype: found image_{name} '
                             f'with dtype {image.dtype}')
        if name in resize_size:
            image = dl.transforms.resize_image(image, size=resize_size[name])
        obs[f'image_{name}'] = image

    for name in depth_names:
        if name not in depth_resize_size:
            logging.warning(
                f'No depth_resize_size was provided for depth_{name}. \
                    This will result in 1x1 '
                'padding depth images, which may cause errors if you \
                    mix padding and non-padding images.')
        depth = obs[f'depth_{name}']

        if depth.dtype == tf.string:
            if tf.strings.length(depth) == 0:
                depth = tf.zeros((*depth_resize_size.get(name, (1, 1)), 1),
                                 dtype=tf.float32)
            else:
                depth = tf.io.decode_image(
                    depth, expand_animations=False, dtype=tf.float32)[..., 0]
        elif depth.dtype != tf.float32:
            raise ValueError(
                f'Unsupported depth dtype: found depth_{name} with \
                    dtype {depth.dtype}')

        if name in depth_resize_size:
            depth = dl.transforms.resize_depth_image(
                depth, size=depth_resize_size[name])

        obs[f'depth_{name}'] = depth

    return obs


def bridge_oxe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to version of Bridge V2 in Open X-Embodiment mixture.
    Note =>> In original Bridge V2 dataset, the first timestep has
    an all-zero action, so we remove it!

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the \
            trajectory data.
    Returns:
        Dict: Transformed trajectory data with the first \
            timestep removed.
    """
    for key in trajectory.keys():
        if key == 'traj_metadata':
            continue
        elif key in ['observation', 'action']:
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            trajectory['action']['rotation_delta'],
            tf.cast(trajectory['action']['open_gripper'][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    trajectory = relabel_bridge_actions(trajectory)
    trajectory['observation']['EEF_state'] = trajectory['observation'][
        'state'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -1:]
    return trajectory


def bridge_orig_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to original version of Bridge V2 from the official
    project website.
    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero
    action, so we remove it!

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    for key in trajectory.keys():
        if key == 'traj_metadata':
            continue
        elif key == 'observation':
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory['action'] = tf.concat(
        [
            trajectory['action'][:, :6],
            binarize_gripper_actions(trajectory['action'][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_bridge_actions(trajectory)
    trajectory['observation']['EEF_state'] = trajectory['observation'][
        'state'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -1:]
    return trajectory


def ppgm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to ppgm.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = tf.concat(
        [
            trajectory['action'][:, :6],
            binarize_gripper_actions(trajectory['action'][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory['observation']['EEF_state'] = trajectory['observation'][
        'cartesian_position'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'gripper_position'][:, -1:]
    return trajectory


def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to rt1.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory['action']['gripper_closedness_action'][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            trajectory['action']['rotation_delta'],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def kuka_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to kuka.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory['action']['gripper_closedness_action'][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            trajectory['action']['rotation_delta'],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory['observation']
        ['clip_function_input/base_pose_tool_reached'],
        compression_type='ZLIB',
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    trajectory['observation'][
        'clip_function_input/base_pose_tool_reached'] = tf.reshape(
            eef_value, (-1, 7))
    gripper_value = tf.io.decode_compressed(
        trajectory['observation']['gripper_closed'], compression_type='ZLIB')
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)
    trajectory['observation']['gripper_closed'] = tf.reshape(
        gripper_value, (-1, 1))
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def taco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to taco.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['state_eef'] = trajectory['observation'][
        'robot_obs'][:, :6]
    trajectory['observation']['state_gripper'] = trajectory['observation'][
        'robot_obs'][:, 7:8]
    trajectory['action'] = trajectory['action']['rel_actions_world']

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :6],
            tf.clip_by_value(trajectory['action'][:, -1:], 0, 1),
        ),
        axis=-1,
    )

    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def jaco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to jaco.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['state_eef'] = trajectory['observation'][
        'end_effector_cartesian_pos'][:, :6]
    trajectory['observation']['state_gripper'] = trajectory['observation'][
        'end_effector_cartesian_pos'][:, -1:]

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory['action']['gripper_closedness_action'][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            tf.zeros_like(trajectory['action']['world_vector']),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def berkeley_cable_routing_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to berkeley cable routing dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            trajectory['action']['rotation_delta'],
            tf.zeros_like(trajectory['action']['world_vector'][:, :1]),
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def roboturk_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to roboturk dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(
        tf.clip_by_value(
            trajectory['action']['gripper_closedness_action'],
            0,  # noqa: E501
            1))

    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            trajectory['action']['rotation_delta'],
            gripper_action,
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def nyu_door_opening_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to nyu door opening dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the
            trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory['action']['gripper_closedness_action'][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            trajectory['action']['rotation_delta'],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def viola_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to viola dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary
            containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the
            first timestep removed.
    """
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory['action'][
        'gripper_closedness_action'][:, None]  # noqa: E501
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            trajectory['action']['rotation_delta'],
            gripper_action,
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def berkeley_autolab_ur5_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to berkeley autolab dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the
            trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['state'] = trajectory['observation'][
        'robot_state'][:, 6:14]
    trajectory['observation']['depth'] = trajectory['observation'].pop(
        'image_with_depth')

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory['action']['gripper_closedness_action']
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            trajectory['action']['rotation_delta'],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def toto_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to toto dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing
            the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            trajectory['action']['rotation_delta'],
            tf.cast(trajectory['action']['open_gripper'][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def language_table_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to language table dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing
            the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first
            timestep removed.
    """
    # default to "open" gripper
    trajectory['action'] = tf.concat(
        (
            trajectory['action'],
            tf.zeros_like(trajectory['action']),
            tf.zeros_like(trajectory['action']),
            tf.ones_like(trajectory['action'][:, :1]),
        ),
        axis=-1,
    )

    # decode language instruction
    instruction_bytes = trajectory['observation']['instruction']
    instruction_encoded = tf.strings.unicode_encode(
        instruction_bytes, output_encoding='UTF-8')
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory['language_instruction'] = tf.strings.split(
        instruction_encoded, '\x00')[:, :1].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to pusht dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the
            trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first
            timestep removed.
    """
    trajectory['action'] = tf.concat(
        (
            trajectory['action']['world_vector'],
            trajectory['action']['rotation_delta'],
            trajectory['action']['gripper_closedness_action'][:, None],
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def stanford_kuka_multimodal_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to stanford kuka multimodal dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the
            trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first
            timestep removed.
    """
    trajectory['observation']['depth_image'] = trajectory['observation'][
        'depth_image'][..., 0]
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :3],
            tf.zeros_like(trajectory['action'][:, :3]),
            trajectory['action'][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory['observation']['eef_state'] = trajectory['observation'][
        'state'][..., :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][..., -1:]
    trajectory['action'] = trajectory['action'][..., :7]
    return trajectory


def stanford_hydra_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to stanford hydra dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the
            trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # invert gripper action, +1 = open, 0 = close
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :6],
            invert_gripper_actions(trajectory['action'][:, -1:]),
        ),
        axis=-1,
    )

    trajectory['observation']['eef_state'] = tf.concat(
        (
            trajectory['observation']['state'][:, :3],
            trajectory['observation']['state'][:, 7:10],
        ),
        axis=-1,
    )
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -3:-2]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_buds_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to austin buds dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the
            trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory['action'][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    trajectory['observation']['state'] = trajectory['observation'][
        'state'][:, :8]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def nyu_franka_play_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to nyu franka play dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['depth'] = tf.cast(
        trajectory['observation']['depth'][..., 0], tf.float32)
    trajectory['observation']['depth_additional_view'] = tf.cast(
        trajectory['observation']['depth_additional_view'][..., 0], tf.float32)
    trajectory['observation']['eef_state'] = trajectory['observation'][
        'state'][:, -6:]

    # clip gripper action, +1 = open, 0 = close
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, -8:-2],
            tf.clip_by_value(trajectory['action'][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def maniskill_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to maniskill dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][..., 7:8]
    return trajectory


def furniture_bench_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to funiture bench dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    import tensorflow_graphics.geometry.transformation as tft

    trajectory['observation']['state'] = tf.concat(
        (
            trajectory['observation']['state'][:, :7],
            trajectory['observation']['state'][:, -1:],
        ),
        axis=-1,
    )

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :3],
            tft.euler.from_quaternion(trajectory['action'][:, 3:7]),
            invert_gripper_actions(
                tf.clip_by_value(trajectory['action'][:, -1:], 0, 1)),
        ),
        axis=-1,
    )
    return trajectory


def cmu_franka_exploration_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to cmu franka dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = trajectory['action'][..., :-1]
    return trajectory


def ucsd_kitchen_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to ucsd kitchen dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['joint_state'] = trajectory['observation'][
        'state'][:, :7]
    trajectory['action'] = trajectory['action'][..., :-1]
    return trajectory


def ucsd_pick_place_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to ucsd pick place dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['eef_state'] = trajectory['observation'][
        'state'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -1:]
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :3],
            tf.zeros_like(trajectory['action'][:, :3]),
            trajectory['action'][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def austin_sailor_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to austin sailor dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory['action'][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_sirius_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to austin sirius dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory['action'][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def bc_z_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to bc_z dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = tf.concat(
        (
            trajectory['action']['future/xyz_residual'][:, :3],
            trajectory['action']['future/axis_angle_residual'][:, :3],
            invert_gripper_actions(
                tf.cast(trajectory['action']['future/target_close'][:, :1],
                        tf.float32)),
        ),
        axis=-1,
    )
    trajectory['language_instruction'] = trajectory['observation'][
        'natural_language_instruction']
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to tokyo pr2 dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['eef_state'] = trajectory['observation'][
        'state'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -1:]
    trajectory['action'] = trajectory['action'][..., :-1]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to tokyo pr2 tabletop dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['eef_state'] = trajectory['observation'][
        'state'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -1:]
    trajectory['action'] = trajectory['action'][..., :-1]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to utokyo xarm pick dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to utokuo xarm bimanual dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = trajectory['action'][..., -7:]
    return trajectory


def robo_net_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to robo net dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['eef_state'] = tf.concat(
        (
            trajectory['observation']['state'][:, :4],
            tf.zeros_like(trajectory['observation']['state'][:, :2]),
        ),
        axis=-1,
    )
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -1:]
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :4],
            tf.zeros_like(trajectory['action'][:, :2]),
            trajectory['action'][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def berkeley_mvp_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to berkeley mvp dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    return trajectory


def berkeley_rpt_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to berkeley rpt dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    return trajectory


def kaist_nonprehensible_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to kaist nonprehensible dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['state'] = trajectory['observation'][
        'state'][:, -7:]
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :6],
            tf.zeros_like(trajectory['action'][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def stanford_mask_vit_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to stanford mask vit dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['eef_state'] = tf.concat(
        (
            trajectory['observation']['end_effector_pose'][:, :4],
            tf.zeros_like(
                trajectory['observation']['end_effector_pose'][:, :2]),
        ),
        axis=-1,
    )
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'end_effector_pose'][:, -1:]
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :4],
            tf.zeros_like(trajectory['action'][:, :2]),
            trajectory['action'][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to tokyo lsmo dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['eef_state'] = trajectory['observation'][
        'state'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -1:]
    return trajectory


def dlr_sara_pour_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to dlr sara pour dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to dlr sara grid clamp dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['state'] = trajectory['observation'][
        'state'][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to dlr edan shared control dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # invert gripper action, +1 = open, 0 = close
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :6],
            invert_gripper_actions(trajectory['action'][:, -1:]),
        ),
        axis=-1,
    )
    return trajectory


def asu_table_top_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to asu table dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['eef_state'] = trajectory['ground_truth_states'][
        'EE']
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -1:]
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to robocook dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['eef_state'] = trajectory['observation'][
        'state'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -1:]
    return trajectory


def imperial_wristcam_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to imperial wristcam dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = trajectory['action'][..., :-1]
    return trajectory


def iamlab_pick_insert_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to iamlab pick insert dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    import tensorflow_graphics.geometry.transformation as tft

    trajectory['observation']['joint_state'] = trajectory['observation'][
        'state'][:, :7]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, 7:8]
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :3],
            tft.euler.from_quaternion(trajectory['action'][:, 3:7]),
            trajectory['action'][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to uiuc d3field dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = tf.concat(
        (
            trajectory['action'],
            tf.zeros_like(trajectory['action']),
            tf.zeros_like(trajectory['action'][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def utaustin_mutex_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to utaustin mutex dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['state'] = trajectory['observation'][
        'state'][:, :8]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory['action'][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def berkeley_fanuc_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to berkeley fanuc dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['joint_state'] = trajectory['observation'][
        'state'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, 6:7]

    # dataset does not store gripper actions, so use gripper state info,
    # invert so +1 = open, 0 = close
    trajectory['action'] = tf.concat(
        (
            trajectory['action'],
            invert_gripper_actions(trajectory['observation']['gripper_state']),
        ),
        axis=-1,
    )
    return trajectory


def cmu_playing_with_food_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to cmu playing with food dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    import tensorflow_graphics.geometry.transformation as tft

    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :3],
            tft.euler.from_quaternion(trajectory['action'][:, 3:7]),
            trajectory['action'][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def playfusion_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to playfusion dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :3],
            trajectory['action'][:, -4:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_stretch_dataset_transform(
        trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to cmu stretch dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['eef_state'] = tf.concat(
        (
            trajectory['observation']['state'][:, :3],
            tf.zeros_like(trajectory['observation']['state'][:, :3]),
        ),
        axis=-1,
    )
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -1:]
    trajectory['action'] = trajectory['action'][..., :-1]
    return trajectory


def gnm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to gnm dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['observation']['state'] = tf.concat(
        (
            trajectory['observation']['position'],
            tf.zeros_like(trajectory['observation']['state'][:, :3]),
            trajectory['observation']['yaw'],
        ),
        axis=-1,
    )
    trajectory['action'] = tf.concat(
        (
            trajectory['action'],
            tf.zeros_like(trajectory['action']),
            tf.zeros_like(trajectory['action']),
            tf.zeros_like(trajectory['action'][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to fmb dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # every input feature is batched, ie has leading batch dimension
    trajectory['observation']['proprio'] = tf.concat(
        (
            trajectory['observation']['eef_pose'],
            trajectory['observation']['state_gripper_pose'][..., None],
        ),
        axis=-1,
    )
    return trajectory


def dobbe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to dobbe dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # every input feature is batched, ie has leading batch dimension
    trajectory['observation']['proprio'] = trajectory['observation']['state']
    return trajectory


def roboset_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to roboset dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    # every input feature is batched, ie has leading batch dimension
    trajectory['observation']['proprio'] = trajectory['observation']['state']

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory['action'][:, -1:]
    gripper_action = invert_gripper_actions(
        tf.clip_by_value(gripper_action, 0, 1))

    trajectory['action'] = tf.concat(
        (
            trajectory['action'][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory


def rh20t_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to rh20t dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = tf.concat(
        (
            trajectory['action']['tcp_base'],
            tf.cast(trajectory['action']['gripper'][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory['observation']['proprio'] = tf.concat(
        (
            trajectory['observation']['tcp_base'],
            trajectory['observation']['gripper_width'][..., None],
        ),
        axis=-1,
    )
    return trajectory


def tdroid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to tdroid dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    trajectory['action'] = tf.concat(
        [
            trajectory['action'][:, :6],
            binarize_gripper_actions(trajectory['action'][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory['observation']['EEF_state'] = trajectory['observation'][
        'cartesian_position'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'gripper_position'][:, -1:]
    return trajectory


def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Applies to libero dataset.

    Args:
        trajectory (Dict[str, Any]): Dictionary containing the
            trajectory data.

    Returns:
        Dict: Transformed trajectory data with the first timestep removed.
    """
    gripper_action = trajectory['action'][:, -1:]
    gripper_action = invert_gripper_actions(
        tf.clip_by_value(gripper_action, 0, 1))

    trajectory['action'] = tf.concat(
        [
            trajectory['action'][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory['observation']['EEF_state'] = trajectory['observation'][
        'state'][:, :6]
    trajectory['observation']['gripper_state'] = trajectory['observation'][
        'state'][:, -2:]  # 2D gripper state
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {
    'bridge_oxe': bridge_oxe_dataset_transform,
    'bridge_orig': bridge_orig_dataset_transform,
    'bridge_dataset': bridge_orig_dataset_transform,
    'ppgm': ppgm_dataset_transform,
    'ppgm_static': ppgm_dataset_transform,
    'ppgm_wrist': ppgm_dataset_transform,
    'fractal20220817_data': rt1_dataset_transform,
    'kuka': kuka_dataset_transform,
    'taco_play': taco_play_dataset_transform,
    'jaco_play': jaco_play_dataset_transform,
    'berkeley_cable_routing': berkeley_cable_routing_dataset_transform,
    'roboturk': roboturk_dataset_transform,
    'nyu_door_opening_surprising_effectiveness':
    nyu_door_opening_dataset_transform,
    'viola': viola_dataset_transform,
    'berkeley_autolab_ur5': berkeley_autolab_ur5_dataset_transform,
    'toto': toto_dataset_transform,
    'language_table': language_table_dataset_transform,
    'columbia_cairlab_pusht_real': pusht_dataset_transform,
    'stanford_kuka_multimodal_dataset_converted_externally_to_rlds':
    stanford_kuka_multimodal_dataset_transform,
    'nyu_rot_dataset_converted_externally_to_rlds': nyu_rot_dataset_transform,
    'stanford_hydra_dataset_converted_externally_to_rlds':
    stanford_hydra_dataset_transform,
    'austin_buds_dataset_converted_externally_to_rlds':
    austin_buds_dataset_transform,
    'nyu_franka_play_dataset_converted_externally_to_rlds':
    nyu_franka_play_dataset_transform,
    'maniskill_dataset_converted_externally_to_rlds':
    maniskill_dataset_transform,
    'furniture_bench_dataset_converted_externally_to_rlds':
    furniture_bench_dataset_transform,
    'cmu_franka_exploration_dataset_converted_externally_to_rlds':
    cmu_franka_exploration_dataset_transform,
    'ucsd_kitchen_dataset_converted_externally_to_rlds':
    ucsd_kitchen_dataset_transform,
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds':
    ucsd_pick_place_dataset_transform,
    'austin_sailor_dataset_converted_externally_to_rlds':
    austin_sailor_dataset_transform,
    'austin_sirius_dataset_converted_externally_to_rlds':
    austin_sirius_dataset_transform,
    'bc_z': bc_z_dataset_transform,
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds':
    tokyo_pr2_opening_fridge_dataset_transform,
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds':
    tokyo_pr2_tabletop_manipulation_dataset_transform,
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds':
    utokyo_xarm_pick_place_dataset_transform,
    'utokyo_xarm_bimanual_converted_externally_to_rlds':
    utokyo_xarm_bimanual_dataset_transform,
    'robo_net': robo_net_dataset_transform,
    'berkeley_mvp_converted_externally_to_rlds':
    berkeley_mvp_dataset_transform,
    'berkeley_rpt_converted_externally_to_rlds':
    berkeley_rpt_dataset_transform,
    'kaist_nonprehensile_converted_externally_to_rlds':
    kaist_nonprehensible_dataset_transform,
    'stanford_mask_vit_converted_externally_to_rlds':
    stanford_mask_vit_dataset_transform,
    'tokyo_u_lsmo_converted_externally_to_rlds': tokyo_lsmo_dataset_transform,
    'dlr_sara_pour_converted_externally_to_rlds':
    dlr_sara_pour_dataset_transform,
    'dlr_sara_grid_clamp_converted_externally_to_rlds':
    dlr_sara_grid_clamp_dataset_transform,
    'dlr_edan_shared_control_converted_externally_to_rlds':
    dlr_edan_shared_control_dataset_transform,
    'asu_table_top_converted_externally_to_rlds':
    asu_table_top_dataset_transform,
    'stanford_robocook_converted_externally_to_rlds':
    robocook_dataset_transform,
    'imperialcollege_sawyer_wrist_cam': imperial_wristcam_dataset_transform,
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds':
    iamlab_pick_insert_dataset_transform,
    'uiuc_d3field': uiuc_d3field_dataset_transform,
    'utaustin_mutex': utaustin_mutex_dataset_transform,
    'berkeley_fanuc_manipulation': berkeley_fanuc_dataset_transform,
    'cmu_playing_with_food': cmu_playing_with_food_dataset_transform,
    'cmu_play_fusion': playfusion_dataset_transform,
    'cmu_stretch': cmu_stretch_dataset_transform,
    'berkeley_gnm_recon': gnm_dataset_transform,
    'berkeley_gnm_cory_hall': gnm_dataset_transform,
    'berkeley_gnm_sac_son': gnm_dataset_transform,
    'droid': droid_baseact_transform,
    'fmb_dataset': fmb_dataset_transform,
    'dobbe': dobbe_dataset_transform,
    'roboset': roboset_dataset_transform,
    'rh20t': rh20t_dataset_transform,
    # T-DROID datasets
    'tdroid_carrot_in_bowl': tdroid_dataset_transform,
    'tdroid_pour_corn_in_pot': tdroid_dataset_transform,
    'tdroid_flip_pot_upright': tdroid_dataset_transform,
    'tdroid_move_object_onto_plate': tdroid_dataset_transform,
    'tdroid_knock_object_over': tdroid_dataset_transform,
    'tdroid_cover_object_with_towel': tdroid_dataset_transform,
    # DROID Finetuning datasets
    'droid_wipe': droid_finetuning_transform,
    # LIBERO datasets (modified versions)
    'libero_spatial_no_noops': libero_dataset_transform,
    'libero_object_no_noops': libero_dataset_transform,
    'libero_goal_no_noops': libero_dataset_transform,
    'libero_10_no_noops': libero_dataset_transform,
}
