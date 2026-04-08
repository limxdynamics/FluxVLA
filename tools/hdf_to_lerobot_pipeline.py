import dataclasses
import gc
import json
import os
import resource
import shutil
from dataclasses import field
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import psutil
import torch
import tqdm
from hdf_to_lerobot_direct_video import enable_direct_video_patch
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DatasetConfig:
    robot_type: str
    use_videos: bool = True
    direct_video_encoding: bool = True
    direct_video_codec: str = 'libsvtav1'
    tolerance_s: float = 0.0001
    image_writer_processes: int = 2
    image_writer_threads: int = 6
    video_backend: str | None = None

    camera_names: list[str] = field(init=False)
    motors: list[str] = field(init=False)

    _camera_names_dict: dict[str, list[str]] = field(
        default_factory=lambda: {
            'aloha_sim': ['head_cam', 'left_cam', 'right_cam'],
            'aloha': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        },
        init=False,
        repr=False,
    )

    _motors_dict: dict[str, list[str]] = field(
        default_factory=lambda: {
            'aloha_sim': [
                'left_waist',
                'left_shoulder',
                'left_elbow',
                'left_forearm_roll',
                'left_wrist_angle',
                'left_wrist_rotate',
                'left_gripper',
                'right_waist',
                'right_shoulder',
                'right_elbow',
                'right_forearm_roll',
                'right_wrist_angle',
                'right_wrist_rotate',
                'right_gripper',
            ],
            'aloha': [
                'right_waist',
                'right_shoulder',
                'right_elbow',
                'right_forearm_roll',
                'right_wrist_angle',
                'right_wrist_rotate',
                'right_gripper',
                'left_waist',
                'left_shoulder',
                'left_elbow',
                'left_forearm_roll',
                'left_wrist_angle',
                'left_wrist_rotate',
                'left_gripper',
            ],
        },
        init=False,
        repr=False,
    )

    add_infos: list = field(default_factory=list)

    def __post_init__(self):
        if (self.robot_type not in self._camera_names_dict
                or self.robot_type not in self._motors_dict):
            raise ValueError(f"Unsupported robot_type '{self.robot_type}'")
        self.camera_names = self._camera_names_dict[self.robot_type]
        self.motors = self._motors_dict[self.robot_type]


# ---------------------------------------------------------------------------
# HDF5 I/O helpers
# ---------------------------------------------------------------------------


def has_cameras_depth(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], 'r') as ep:
        return '/observations/images_depth' in ep


def has_eepose(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], 'r') as ep:
        return '/observations/eepose' in ep


def has_action(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], 'r') as ep:
        return '/action' in ep


def load_raw_images_per_camera(ep: h5py.File,
                               cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f'/observations/images/{camera}'].ndim == 4
        if uncompressed:
            imgs_array = ep[f'/observations/images/{camera}'][:]
        else:
            import cv2

            imgs_array = []
            for data in ep[f'/observations/images/{camera}']:
                imgs_array.append(
                    cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)
        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_images_depth_per_camera(
        ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_depth_per_cam = {}
    for camera in cameras:
        uncompressed = ep[
            f'/observations/images_depth/{camera}_depth'].ndim == 3
        if uncompressed:
            imgs_array = ep[f'/observations/images_depth/{camera}_depth'][:]
        else:
            print('Skipping corrupted or invalid depth data.')
            continue
        imgs_depth_per_cam[camera] = imgs_array
    return imgs_depth_per_cam


def resize_dof_tensor(dof_tensor: torch.Tensor | None) -> torch.Tensor | None:
    if dof_tensor is None or dof_tensor.shape[-1] != 16:
        return dof_tensor
    scale = 0.1 / 0.07
    resized = torch.zeros(
        *dof_tensor.shape[:-1],
        14,
        dtype=dof_tensor.dtype,
        device=dof_tensor.device)
    resized[..., 0:6] = dof_tensor[..., 0:6]
    resized[..., 6] = (dof_tensor[..., 6] - dof_tensor[..., 7]) * scale
    resized[..., 7:13] = dof_tensor[..., 8:14]
    resized[..., 13] = (dof_tensor[..., 14] - dof_tensor[..., 15]) * scale
    return resized


def load_raw_episode_data(
    ep_path: Path,
    dataset_config: DatasetConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, torch.Tensor,
           torch.Tensor | None, torch.Tensor | None, ]:
    with h5py.File(ep_path, 'r') as ep:
        state = torch.from_numpy(ep['/observations/qpos'][:])
        state = resize_dof_tensor(state)

        action = None
        if '/action' in ep:
            action = torch.from_numpy(ep['/action'][:])
            action = resize_dof_tensor(action)

        eepose = None
        if '/observations/eepose' in ep:
            eepose = torch.from_numpy(ep['/observations/eepose'][:])

        imgs_per_cam_depth = None
        if ('/observations/images_depth' in ep
                and 'depth' in dataset_config.add_infos):
            try:
                imgs_per_cam_depth = load_raw_images_depth_per_camera(
                    ep, dataset_config.camera_names)
            except Exception as e:
                print(f'Error loading depth images: {e}')
                imgs_per_cam_depth = None

        imgs_per_cam = load_raw_images_per_camera(ep,
                                                  dataset_config.camera_names)

    return imgs_per_cam, imgs_per_cam_depth, state, action, eepose


# ---------------------------------------------------------------------------
# Dataset creation & conversion pipeline
# ---------------------------------------------------------------------------


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal['video', 'image'] = 'video',
    fps: int = 30,
    *,
    has_action: bool = False,
    has_eepose: bool = False,
    has_depth: bool = False,
    dataset_config: DatasetConfig,
    output_path: Path,
) -> LeRobotDataset:
    motors = dataset_config._motors_dict[robot_type]
    cameras = dataset_config._camera_names_dict[robot_type]
    features = {
        'observation.state': {
            'dtype': 'float32',
            'shape': (len(motors), ),
            'names': [motors],
        },
    }
    if has_action:
        features['action'] = {
            'dtype': 'float32',
            'shape': (len(motors), ),
            'names': [motors],
        }
    if has_eepose:
        features['observation.eepose'] = {
            'dtype':
            'float32',
            'shape': (14, ),
            'names': [
                'left_x',
                'left_y',
                'left_z',
                'left_qx',
                'left_qy',
                'left_qz',
                'left_qw',
                'right_x',
                'right_y',
                'right_z',
                'right_qx',
                'right_qy',
                'right_qz',
                'right_qw',
            ],
        }
    if has_depth and 'depth' in dataset_config.add_infos:
        for cam in cameras:
            features[f'observation.depth.{cam}'] = {
                'dtype': 'uint16',
                'shape': (480, 640),
                'names': ['height', 'width'],
            }
    for cam in cameras:
        features[f'observation.images.{cam}'] = {
            'dtype': mode,
            'shape': (480, 640, 3),
            'names': ['height', 'width', 'channels'],
        }

    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=output_path,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_task_annotations(task_path: Path | None,
                          init_task: list[str]) -> list[str]:
    if task_path is None:
        print('No task file path provided, using default task list')
        return init_task

    if not task_path.exists():
        raise FileNotFoundError(f'Task file {task_path} does not exist.')

    with open(task_path, 'r') as f:
        annotation = json.load(f)

    for segment in annotation[0]:
        if segment['label'] != 'empty':
            start_frame = segment['start_frame']
            end_frame = segment['end_frame']
            init_task[start_frame:end_frame] = [segment['label']] * (
                end_frame - start_frame)

    return init_task


def is_in_last_chunk(index: int, array_length: int, batch_size: int) -> bool:
    total_chunks = (array_length + batch_size - 1) // batch_size
    last_chunk_start = (total_chunks - 1) * batch_size
    last_chunk_end = min(last_chunk_start + batch_size, array_length)
    return last_chunk_start <= index < last_chunk_end


def oom_finder() -> None:
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f'************, RSS={mem.rss / 1024**2:.1f} MB, '  # noqa: E231
          f'VMS={mem.vms / 1024**2:.1f} MB, '  # noqa: E231
          f'peakRSS={peak / 1024:.1f} MB')  # noqa: E231


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: list[Path | None],
    episodes: list[int] | None = None,
    dataset_config: DatasetConfig | None = None,
    init_task: str | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        oom_finder()

        last_chunk_flag = is_in_last_chunk(ep_idx, len(hdf5_files),
                                           dataset.batch_encoding_size)
        if last_chunk_flag:
            dataset.batch_encoding_size = 1

        ep_path = hdf5_files[ep_idx]
        task_path = task[ep_idx]

        (imgs_per_cam, imgs_per_cam_depth, state, action,
         eepose) = load_raw_episode_data(ep_path, dataset_config)

        default_task = (
            init_task if init_task is not None else
            'pick up the yellow banana and put it on the pink plate')
        init_task_list = [default_task for _ in range(len(state))]
        tasks = load_task_annotations(task_path, init_task_list)
        if all(task_name == 'empty' for task_name in tasks):
            del imgs_per_cam, imgs_per_cam_depth, state, action, eepose
            gc.collect()
            continue

        num_frames = state.shape[0]
        for i in range(num_frames):
            frame = {
                'observation.state': state[i].float().numpy(),
            }
            for camera, img_array in imgs_per_cam.items():
                frame[f'observation.images.{camera}'] = img_array[i]
            if imgs_per_cam_depth is not None:
                for camera, img_depth_array in imgs_per_cam_depth.items():
                    frame[f'observation.depth.{camera}'] = img_depth_array[i]
            if action is not None:
                frame['action'] = action[i].float().numpy()
            if eepose is not None:
                frame['observation.eepose'] = eepose[i].float().numpy()
            dataset.add_frame(frame, tasks[i])

        dataset.save_episode()
        del imgs_per_cam, imgs_per_cam_depth, state, action, eepose
        gc.collect()

        dataset.hf_dataset = dataset.create_hf_dataset()
        gc.collect()

    return dataset


def collect_data_task(raw_dir: Path) -> tuple[list[Path], list[Path | None]]:
    hdf5_files = sorted([f for f in raw_dir.rglob('episode_*.hdf5')])
    task_files: list[Path | None] = [None] * len(hdf5_files)
    return hdf5_files, task_files


def port_hdf5(
    raw_dir: Path,
    repo_id: str,
    *,
    episodes: list[int] | None = None,
    robot_type: Literal['aloha_sim', 'aloha'] = 'aloha_sim',
    mode: Literal['video', 'image'] = 'video',
    debug_mode: bool = False,
    output_dir: Path | None = None,
    init_task: str | None = None,
    convert_depth: bool = False,
):
    if output_dir is None:
        output_path = HF_LEROBOT_HOME / repo_id
    else:
        output_path = Path(output_dir) / repo_id

    print(f'Dataset save path: {output_path.absolute()}')
    print(f'Dataset ID: {repo_id}')

    dataset_config = DatasetConfig(robot_type=robot_type)
    if convert_depth:
        dataset_config.add_infos.append('depth')
    if not raw_dir.exists():
        raise ValueError('raw_dir must be provided!')

    hdf5_files, task_files = collect_data_task(raw_dir=raw_dir)

    if debug_mode:
        hdf5_files = hdf5_files[:1]

    dataset = create_empty_dataset(
        repo_id,
        robot_type,
        mode=mode,
        has_eepose=has_eepose(hdf5_files),
        has_depth=has_cameras_depth(hdf5_files),
        has_action=has_action(hdf5_files),
        dataset_config=dataset_config,
        fps=30,
        output_path=output_path,
    )

    if mode == 'video' and dataset_config.direct_video_encoding:
        dataset = enable_direct_video_patch(
            dataset, codec=dataset_config.direct_video_codec)

    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task_files,
        episodes=episodes,
        dataset_config=dataset_config,
        init_task=init_task,
    )

    return dataset
