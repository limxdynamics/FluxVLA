# DataConvert

Data conversion tool: Convert HDF5 format data collected by the ALOHA dual-arm robot to the LeRobot Dataset v2.1 format.

## Environment Setup

1. Create a virtual environment

   ```bash
   conda create -y -n lerobot python=3.10
   conda activate lerobot
   conda install -c conda-forge ffmpeg av
   ```

2. Install lerobot

   ```bash
   cd tools
   git clone https://github.com/huggingface/lerobot.git
   cd lerobot
   git checkout 55198de096f46a8e0447a8795129dd9ee84c088c
   pip install -e .
   ```

______________________________________________________________________

## Data Format Requirements

> **Robot-type naming conventions** — field names and ordering vary by `--robot-type`:
>
> |                 | `aloha_sim`                         | `aloha`                                         |
> | --------------- | ----------------------------------- | ----------------------------------------------- |
> | **Joint order** | Left arm (7) + Right arm (7)        | Right arm (7) + Left arm (7)                    |
> | **Cameras**     | `head_cam`, `left_cam`, `right_cam` | `cam_high`, `cam_left_wrist`, `cam_right_wrist` |

### Input Data Format (HDF5)

HDF5 files should follow the `episode_*.hdf5` naming convention. The conversion script recursively searches the specified directory for all matching files.

#### Required Fields

- **`/observations/qpos`** - Robot Joint Positions

  - Data type: `float32` or `float64`
  - Shape: `[num_frames, 14]` or `[num_frames, 16]` (16-dim is automatically converted to 14-dim)
  - Notes:
    - **16-dim format**: Gripper open/close is represented by the absolute positions of two gripper fingers (8 dims total)
    - **14-dim format**: Gripper open/close is represented by a relative position normalized to \[0, 0.1\]
    - Conversion formula: `gripper_value = (left_finger - right_finger) * (0.1 / 0.07)`

- **`/observations/images/<camera_name>`** - Camera Images

  - Format:
    - Uncompressed 4D numpy array `[num_frames, height, width, channels]` (`uint8`)
    - or JPEG-compressed byte stream `[num_frames]` (automatically decoded to RGB)

#### Optional Fields

- **`/action`** - Robot Desired Joint Positions

  - Data type: `float32` or `float64`
  - Shape: `[num_frames, 14]` or `[num_frames, 16]` (16-dim is automatically converted to 14-dim)

- **`/observations/eepose`** - End-effector Pose

  - Data type: `float32` or `float64`
  - Shape: `[num_frames, 14]`
  - Notes: Contains position (x, y, z) and quaternion (qx, qy, qz, qw) for both left and right end-effectors

- **`/observations/images_depth/<camera_name>_depth`** - Depth images

  - Data type: `uint16` (values in mm)
  - Shape: `[num_frames, height, width]`
  - Notes: Requires the `--depth` flag at runtime to be processed

______________________________________________________________________

### Output Data Format (LeRobot v2.1)

The converted dataset follows the LeRobot v2.1 format, stored in a HuggingFace Datasets-compatible directory structure:

```
<output_dir>/<repo_id>/
├── data/
│   ├── train/
│   │   ├── episode_0.parquet
│   │   └── ...
│   └── video/
│       ├── episode_0/
│       │   ├── observation.images.head_cam.mp4
│       │   └── ...
│       └── ...
├── info.json
└── meta.json
```

#### Data Field Descriptions

Each episode's parquet file contains the following fields:

- **`observation.state`**

  - Description: Robot joint positions
  - Type: `float32`
  - Shape: `(14,)`

- **`observation.images.<camera_name>`**

  - Description: Camera reference containing the video file path and frame timestamp
  - Type: `VideoFrame` object
  - Video specs: MP4, 30 FPS, `(480, 640, 3)`

- **`task`**

  - Description: Task label; customizable via the `--init-task` parameter
  - Type: `string`
  - Default: `"pick up the yellow banana and put it on the pink plate"`

- **`action`** (Optional)

  - Description: Robot desired joint positions; only generated when the input contains `/action`
  - Type: `float32`
  - Shape: `(14,)`

- **`observation.eepose`** (Optional)

  - Description: End-effector pose; only generated when the input contains `/observations/eepose`
  - Type: `float32`
  - Shape: `(14,)`

- **`observation.depth.<camera_name>`** (Optional)

  - Description: Depth image; only generated when the input contains depth images and the `--depth` flag is specified
  - Type: `uint16`
  - Shape: `(480, 640)`

______________________________________________________________________

## Usage

### Basic Usage

```bash
python convert_hdf_to_lerobot.py <raw_dir> [options]
```

**Arguments:**

| Argument       | Type                  | Default           | Description                                                                                   |
| -------------- | --------------------- | ----------------- | --------------------------------------------------------------------------------------------- |
| `raw_dir`      | Positional (required) | —                 | Path to the directory containing HDF5 files to convert                                        |
| `--repo-id`    | String                | `"0402"`          | Output dataset ID                                                                             |
| `--output-dir` | Path                  | `HF_LEROBOT_HOME` | Output directory                                                                              |
| `--init-task`  | String                | `None`            | Task description text, defaults to `"pick up the yellow banana and put it on the pink plate"` |
| `--robot-type` | String                | `"aloha"`         | Robot type, options: `"aloha"` or `"aloha_sim"`                                               |
| `--depth`      | Flag                  | `Off`             | Add this flag to convert depth images                                                         |

**Auto-detection Notes:**

- **`action`** and **`observation.eepose`** are automatically generated based on whether the HDF5 file contains `/action` and `/observations/eepose`, no manual configuration needed
- **Depth images** must be explicitly enabled via the `--depth` flag

______________________________________________________________________

## Configuration Options

Advanced conversion behavior can be adjusted by modifying the `DatasetConfig` class in the code:

- `use_videos`: Whether to use video mode (default: `True`)
- `direct_video_encoding`: Whether to use direct video encoding (default: `True`)
- `direct_video_codec`: Video codec (default: `"libsvtav1"`)
- `tolerance_s`: Timestamp tolerance (default: `0.0001`)
- `image_writer_processes`: Number of image writer processes (default: `2`)
- `image_writer_threads`: Number of threads per process (default: `6`)
