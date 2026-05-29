# FluxBisim

`FluxBisim` is a dual-arm collaborative manipulation simulation platform built on `Isaac Sim 4.5.0`. Integrated with `FluxVLA`, it supports simulation data collection, model training, deployment, and evaluation.

GitHub: [https://github.com/FluxVLA/FluxBisim](https://github.com/FluxVLA/FluxBisim)

## Installation

Install `FluxVLA` following the main [README](../README.md). For `FluxBisim` installation, including Isaac Sim, ROS Noetic, assets, and simulator dependencies, refer to the [FluxBisim README](https://github.com/FluxVLA/FluxBisim/blob/main/README.md).

## Data Preparation

Download the released `FluxBisim` dataset, which is already converted to the LeRobot Dataset v2.1 format:

```bash
cd /path/to/FluxVLA
hf download limxdynamics/FluxBisimData \
  --repo-type dataset \
  --local-dir datasets
```

You can also collect new HDF5 data in `FluxBisim`:

```bash
cd /path/to/FluxVLA/FluxBisim
benchmark_python <collect_script> \
  --env <env_name> \
  --config <config_name> \
  --data_path <output_hdf5_dir>
```

Available tasks:

| Task                    | Script                                                        | Environment             | Config                          |
| ----------------------- | ------------------------------------------------------------- | ----------------------- | ------------------------------- |
| Pick Fruit to Plate     | `data_collect/pick_place_fruit/fruit_pick_place_collect.py`   | `kitchen`, `apartment`  | `*_pick_place_config.yaml`      |
| Place Nut and Close Box | `data_collect/close_box/box_close_collect.py`                 | `industry`              | `box_close_config.yaml`         |
| Store Apple in Drawer   | `data_collect/pull_push_drawer/drawer_pull_push_collect.py`   | `apartment`             | `drawer_pull_push_config.yaml`  |
| Screw Pitcher Lid       | `data_collect/screw_pitcher_lid/pitcher_lid_screw_collect.py` | `apartmentshort`        | `pitcher_lid_screw_config.yaml` |
| Handover Book           | `data_collect/handover_book/book_handover_collect.py`         | `apartment`, `industry` | `book_handover_config.yaml`     |

Convert collected HDF5 data before training:

```bash
cd /path/to/FluxVLA
python tools/convert_hdf_to_lerobot.py <raw_hdf5_dir> \
  --repo-id <dataset_name> \
  --output-dir datasets \
  --init-task "<task_instruction>" \
  --robot-type aloha_sim
```

Set `--repo-id` to the converted dataset name and `--init-task` to the task instruction used for training.

For more details about the data format, see [Data Conversion](data_convert.md).

## Model Training

FluxBisim training configs are provided for `GR00T` and `PI0.5`:

| Task                    | GR00T Config                                                                | PI0.5 Config                                                               |
| ----------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Pick Banana to Plate    | `configs/gr00t/fluxbisim/gr00t_eagle_3b_pick_place_banana_full_finetune.py` | `configs/pi05/fluxbisim/pi05_paligemma_pick_place_banana_full_finetune.py` |
| Place Nut and Close Box | `configs/gr00t/fluxbisim/gr00t_eagle_3b_close_box_full_finetune.py`         | `configs/pi05/fluxbisim/pi05_paligemma_close_box_full_finetune.py`         |
| Store Apple in Drawer   | `configs/gr00t/fluxbisim/gr00t_eagle_3b_pull_push_drawer_full_finetune.py`  | `configs/pi05/fluxbisim/pi05_paligemma_pull_push_drawer_full_finetune.py`  |
| Screw Pitcher Lid       | `configs/gr00t/fluxbisim/gr00t_eagle_3b_screw_pitcher_lid_full_finetune.py` | `configs/pi05/fluxbisim/pi05_paligemma_screw_pitcher_lid_full_finetune.py` |
| Handover Book           | `configs/gr00t/fluxbisim/gr00t_eagle_3b_handover_book_full_finetune.py`     | `configs/pi05/fluxbisim/pi05_paligemma_handover_book_full_finetune.py`     |

Example training command:

```bash
cd /path/to/FluxVLA
export WANDB_MODE=disabled

bash scripts/train.sh \
  configs/pi05/fluxbisim/pi05_paligemma_pick_place_banana_full_finetune.py \
  ./work_dirs/pick_place_banana_pi05 \
  --cfg-options \
  train_dataloader.per_device_batch_size=8 \
  runner.max_keep_ckpts=5
```

If you use a custom dataset path, update `train_dataloader.dataset.datasets[*].data_root_path` in the config.

## Simulation Evaluation

Run simulation evaluation with two terminals: one for the `FluxBisim` benchmark environment and one for `FluxVLA` inference.

In the first terminal, start the target `FluxBisim` benchmark task:

```bash
cd /path/to/FluxVLA/FluxBisim
PYTHON_BIN=/path/to/install_dir/isaac-sim-4.5.0/python.sh \
  ./fluxbisim_benchmark.sh <benchmark_task>
```

In the second terminal, launch `FluxVLA` inference with the config and checkpoint for the same task:

```bash
cd /path/to/FluxVLA
./scripts/fluxbisim_eval.sh \
  <config_path> \
  <checkpoint_path>
```

Task mapping for simulation evaluation:

| Task                    | Benchmark Task      | Task ID |
| ----------------------- | ------------------- | ------- |
| Pick Banana to Plate    | `pick_place_banana` | `1`     |
| Place Nut and Close Box | `close_box`         | `2`     |
| Store Apple in Drawer   | `pull_push_drawer`  | `3`     |
| Screw Pitcher Lid       | `screw_pitcher_lid` | `4`     |
| Handover Book           | `handover_book`     | `5`     |

Make sure `<benchmark_task>`, `<config_path>`, and `<checkpoint_path>` all correspond to the same row in the table. During evaluation, `FluxBisim` reports the task success rate and completion statistics.
