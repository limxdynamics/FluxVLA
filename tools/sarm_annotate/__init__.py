# flake8: noqa
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""SARM subtask annotation pipeline for LeRobot datasets.

Ported from ``lerobot/data_processing/sarm_annotations`` in HuggingFace LeRobot.
Takes a standard LeRobot dataset and uses a local Qwen3-VL model to annotate
sparse / dense subtasks, writing them back as extra columns on the standard
``meta/episodes.jsonl`` / ``meta/episodes/*.parquet`` metadata. The columns
produced match what :mod:`fluxvla.datasets.utils.sarm_utils` reads.
"""
