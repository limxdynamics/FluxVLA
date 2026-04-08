from pathlib import Path
from types import MethodType

import numpy as np
import PIL.Image
import torch
from lerobot.datasets.compute_stats import (compute_episode_stats,
                                            get_feature_stats, sample_indices)
from lerobot.datasets.image_writer import write_image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (check_timestamps_sync,
                                    get_episode_data_index,
                                    hf_transform_to_torch,
                                    validate_episode_buffer, validate_frame,
                                    write_info)
from lerobot.datasets.video_utils import decode_video_frames

import datasets


class DirectVideoWriter:
    """Writes video frames directly via PyAV.

    Bypasses the image-cache-then-encode pipeline.
    """

    def __init__(
        self,
        video_path: Path,
        fps: int,
        *,
        vcodec: str = 'libsvtav1',
        pix_fmt: str = 'yuv420p',
        gop_size: int = 2,
        crf: int = 30,
    ):
        self.video_path = Path(video_path)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.gop_size = gop_size
        self.crf = crf
        self.container = None
        self.stream = None

    def _to_rgb_uint8(
            self,
            image: torch.Tensor | np.ndarray | PIL.Image.Image) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, PIL.Image.Image):
            image = np.array(image.convert('RGB'))

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f'Unsupported image type for video encoding: {type(image)}')
        if image.ndim != 3:
            raise ValueError(
                f'Expected image with 3 dimensions, got shape {image.shape}')

        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        if image.dtype.kind == 'f':
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(image)

    def write(self,
              image: torch.Tensor | np.ndarray | PIL.Image.Image) -> None:
        image = self._to_rgb_uint8(image)
        height, width = image.shape[:2]

        if self.container is None:
            import av

            self.container = av.open(str(self.video_path), 'w')
            self.stream = self.container.add_stream(
                self.vcodec,
                rate=self.fps,
                options={
                    'crf': str(self.crf),
                    'g': str(self.gop_size)
                },
            )
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = self.pix_fmt

        import av

        frame = av.VideoFrame.from_ndarray(image, format='rgb24')
        packet = self.stream.encode(frame)
        if packet:
            self.container.mux(packet)

    def close(self) -> None:
        if self.container is None:
            return

        packet = self.stream.encode()
        if packet:
            self.container.mux(packet)
        self.container.close()
        self.container = None
        self.stream = None


def enable_direct_video_patch(dataset: LeRobotDataset,
                              codec: str = 'libsvtav1') -> LeRobotDataset:
    """Monkey-patch *dataset* so that video frames are encoded on-the-fly
    instead of first being cached as individual image files on disk."""

    if len(dataset.meta.video_keys) == 0:
        return dataset

    dataset._direct_video_writers = {}
    dataset._direct_video_codec = codec
    dataset.batch_encoding_size = 1
    dataset.episodes_since_last_encoding = 0

    if dataset.image_writer is not None and len(dataset.meta.image_keys) == 0:
        dataset.stop_image_writer()

    def _get_direct_video_writer(self: LeRobotDataset, episode_index: int,
                                 key: str) -> DirectVideoWriter:
        writer_key = (episode_index, key)
        if writer_key not in self._direct_video_writers:
            video_path = self.root / self.meta.get_video_file_path(
                episode_index, key)
            self._direct_video_writers[writer_key] = DirectVideoWriter(
                video_path=video_path,
                fps=self.fps,
                vcodec=self._direct_video_codec,
            )
        return self._direct_video_writers[writer_key]

    def _close_episode_video_writers(self: LeRobotDataset,
                                     episode_index: int) -> None:
        for writer_key, writer in list(self._direct_video_writers.items()):
            if writer_key[0] == episode_index:
                writer.close()
                del self._direct_video_writers[writer_key]

    def _compute_direct_video_stats(
        self: LeRobotDataset,
        episode_index: int,
        timestamps: list[float],
    ) -> dict[str, dict[str, np.ndarray]]:
        video_stats = {}
        sampled_indices = sample_indices(len(timestamps))
        sampled_timestamps = [
            float(timestamps[idx]) for idx in sampled_indices
        ]

        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(
                episode_index, key)
            frames = (
                decode_video_frames(
                    video_path,
                    sampled_timestamps,
                    self.tolerance_s,
                    self.video_backend,
                ).cpu().numpy())
            stats = get_feature_stats(frames, axis=(0, 2, 3), keepdims=True)
            video_stats[key] = {
                k: v if k == 'count' else np.squeeze(v / 255.0, axis=0)
                for k, v in stats.items()
            }

        return video_stats

    def _patched_save_image(
        self: LeRobotDataset,
        image: torch.Tensor | np.ndarray | PIL.Image.Image,
        fpath: Path,
    ) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def _patched_add_frame(
        self: LeRobotDataset,
        frame: dict,
        task: str,
        timestamp: float | None = None,
    ) -> None:
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer['size']
        if timestamp is None:
            timestamp = frame_index / self.fps

        self.episode_buffer['frame_index'].append(frame_index)
        self.episode_buffer['timestamp'].append(timestamp)
        self.episode_buffer['task'].append(task)

        episode_index = self.episode_buffer['episode_index']

        for key in frame:
            if key not in self.features:
                raise ValueError(f'An element of the frame is not in the '
                                 f"features. '{key}' not in "
                                 f"'{self.features.keys()}'.")

            feature_dtype = self.features[key]['dtype']
            if feature_dtype == 'image':
                img_path = self._get_image_file_path(
                    episode_index=episode_index,
                    image_key=key,
                    frame_index=frame_index,
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            elif feature_dtype == 'video':
                writer = self._get_direct_video_writer(episode_index, key)
                writer.write(frame[key])
                video_path = self.root / self.meta.get_video_file_path(
                    episode_index, key)
                self.episode_buffer[key].append(str(video_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer['size'] += 1

    def _patched_save_episode(self: LeRobotDataset,
                              episode_data: dict | None = None) -> None:
        episode_buffer = (
            self.episode_buffer if episode_data is None else episode_data)
        validate_episode_buffer(episode_buffer, self.meta.total_episodes,
                                self.features)

        episode_length = episode_buffer.pop('size')
        tasks = episode_buffer.pop('task')
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer['episode_index']

        episode_buffer['index'] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer['episode_index'] = np.full((episode_length, ),
                                                  episode_index)

        for episode_task in episode_tasks:
            task_index = self.meta.get_task_index(episode_task)
            if task_index is None:
                self.meta.add_task(episode_task)

        episode_buffer['task_index'] = np.array(
            [self.meta.get_task_index(episode_task) for episode_task in tasks])

        for key, ft in self.features.items():
            if key in ['index', 'episode_index', 'task_index'
                       ] or ft['dtype'] in [
                           'image',
                           'video',
                       ]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._close_episode_video_writers(episode_index)

        if len(self.meta.video_keys) > 0 and episode_index == 0:
            self.meta.update_video_info()
            write_info(self.meta.info, self.meta.root)

        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(
            episode_dict, features=self.hf_features, split='train')
        self.hf_dataset = datasets.concatenate_datasets(
            [self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(
            ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

        non_video_data = {
            key: value
            for key, value in episode_buffer.items()
            if key in self.features and self.features[key]['dtype'] != 'video'
        }
        episode_stats = compute_episode_stats(non_video_data, self.features)
        episode_stats.update(
            self._compute_direct_video_stats(episode_index,
                                             episode_buffer['timestamp']))

        self.meta.save_episode(episode_index, episode_length, episode_tasks,
                               episode_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes,
                                               [episode_index])
        ep_data_index_np = {
            key: value.numpy()
            for key, value in ep_data_index.items()
        }
        check_timestamps_sync(
            episode_buffer['timestamp'],
            episode_buffer['episode_index'],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        parquet_files = list(self.root.rglob('*.parquet'))
        assert len(parquet_files) == self.num_episodes
        video_files = list(self.root.rglob('*.mp4'))
        assert len(video_files) == self.num_episodes * len(
            self.meta.video_keys)

        if episode_data is None:
            self.episode_buffer = self.create_episode_buffer()

    dataset._save_image = MethodType(_patched_save_image, dataset)
    dataset._get_direct_video_writer = MethodType(_get_direct_video_writer,
                                                  dataset)
    dataset._close_episode_video_writers = MethodType(
        _close_episode_video_writers, dataset)
    dataset._compute_direct_video_stats = MethodType(
        _compute_direct_video_stats, dataset)
    dataset.add_frame = MethodType(_patched_add_frame, dataset)
    dataset.save_episode = MethodType(_patched_save_episode, dataset)
    return dataset
