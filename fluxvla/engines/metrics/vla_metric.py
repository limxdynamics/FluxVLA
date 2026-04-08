# Origin: Modified from
# Upstream-Repo: openvla/openvla
# Upstream-Path: prismatic/training/metrics.py
# Upstream-Ref: main
# SPDX-License-Identifier: MIT
# Notes: Attribution normalized; no functional change.

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union

import jsonlines
import numpy as np
import torch
import wandb

from fluxvla.engines.utils import initialize_overwatch
from ..utils.root import METRICS

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Define Tracker Interface ===
class Tracker(Protocol):

    def write_hyperparameters(self) -> None:
        ...

    def write(self, global_step: int,
              metrics: Dict[str, Union[int, float]]) -> None:
        ...

    def finalize(self) -> None:
        ...


# === Individual Tracker Definitions ===
class JSONLinesTracker:

    def __init__(self, run_id: str, run_dir: Path, hparams: Dict[str,
                                                                 Any]) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        with jsonlines.open(
                os.path.join(self.run_dir, 'run-metrics.jsonl'),
                mode='w',
                sort_keys=True) as js_tracker:
            js_tracker.write({
                'run_id': self.run_id,
                'hparams': str(self.hparams.to_dict())
            })

    @overwatch.rank_zero_only
    def write(self, _: int, metrics: Dict[str, Union[int, float]]) -> None:
        with jsonlines.open(
                os.path.join(self.run_dir, f'{self.run_id}.jsonl'),
                mode='a',
                sort_keys=True) as js_tracker:
            js_tracker.write(metrics)

    def finalize(self) -> None:
        return


class WeightsBiasesTracker:

    def __init__(
        self,
        run_id: str,
        run_dir: str,
        hparams: Dict[str, Any],
        group: str = 'align',
    ) -> None:
        """Initialize Weights & Biases Tracker.
        This class is used to log hyperparameters and metrics to
            Weights & Biases.

        Args:
            run_id (str): ID of the run.
            run_dir (str): Directory to save the run.
            hparams (Dict[str, Any]): parameters for the run.
            group (str, optional): Name of the group . Defaults to "align".

        Environment Variables:
            WANDB_PROJECT: Name of the project. Defaults to "fluxvla".
            WANDB_ENTITY: Name of the entity. Defaults to None.
        """
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

        # Get W&B-Specific Initialization Parameters from environment
        self.project = os.environ.get('WANDB_PROJECT', 'fluxvla')
        self.entity = os.environ.get('WANDB_ENTITY', None)
        self.group = group
        self.wandb_dir = self.run_dir

        # Call W&B.init()
        self.initialize()

    @overwatch.rank_zero_only
    def initialize(self) -> None:
        wandb.init(
            name=self.run_id,
            dir=self.wandb_dir,
            config=self.hparams,
            project=self.project,
            entity=self.entity,
            group=self.group,
        )

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        wandb.config = self.hparams

    @overwatch.rank_zero_only
    def write(self, global_step: int,
              metrics: Dict[str, Union[int, float]]) -> None:
        wandb.log(metrics, step=global_step)

    @staticmethod
    def finalize() -> None:
        if overwatch.is_rank_zero():
            wandb.finish()

        # A job gets 210 seconds to get its affairs in order
        time.sleep(210)


# === Core Metrics Container :: Initializes Trackers
# => Compiles/Pushes Metrics ===


@METRICS.register_module()
class VLAMetric:
    """VLAMetric is a class designed to track and log metrics during
        training or evaluation processes. It supports multiple tracking
        backends such as JSONLines and Weights & Biases (wandb) and
        provides functionality for logging, committing, and pushing metrics.

    Args:
        active_trackers (Tuple[str, ...]): List of active trackers to be used.
        run_id (str): ID of the run.
        run_dir (str): Directory to save the run.
        hparams (Dict[str, Any]): Hyperparameters for the run.
        grad_accumulation_steps (int, optional): Number of gradient
            accumulation steps. Defaults to 1.
        window_size (int, optional): Size of the window for smoothing.
            Defaults to 1.
        resume_step (Optional[int], optional): Step to resume from.
            Defaults to None.
        resume_epoch (Optional[int], optional): Epoch to resume from.
            Defaults to None.

    Environment Variables:
        WANDB_PROJECT: Name of the wandb project. Defaults to "fluxvla".
        WANDB_ENTITY: Name of the wandb entity. Defaults to None.
        WANDB_MODE: Mode for wandb (online/offline/disabled). Defaults to "online".  # noqa: E501
    """

    def __init__(
        self,
        active_trackers: Tuple[str, ...],
        run_id: str,
        run_dir: str,
        hparams: Dict[str, Any],
        grad_accumulation_steps: int = 1,
        window_size: int = 1,
        resume_step: Optional[int] = None,
        resume_epoch: Optional[int] = None,
        update_step_time: bool = True,
    ) -> None:
        self.update_step_time = update_step_time
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams
        os.makedirs(self.run_dir, exist_ok=True)
        # Initialize Trackers
        self.trackers = []
        for tracker_type in active_trackers:
            if tracker_type == 'jsonl':
                tracker = JSONLinesTracker(run_id, run_dir, hparams)
            elif tracker_type == 'wandb':
                tracker = WeightsBiasesTracker(
                    run_id, run_dir, hparams, group='vla-train')
            else:
                raise ValueError(
                    f'Tracker with type `{tracker_type} is not supported!')

            # Add Hyperparameters --> add to `self.trackers`
            tracker.write_hyperparameters()
            self.trackers.append(tracker)

        # Create Universal Metrics Buffers
        self.global_step = 0 if resume_step is None else resume_step
        self.epoch = 0 if resume_epoch is None else resume_epoch
        self.start_time, self.step_start_time = time.time(), time.time()
        self.state = {
            'loss_raw': deque(maxlen=grad_accumulation_steps),
            'loss': deque(maxlen=window_size),
            'l1_loss': deque(maxlen=window_size),
            'action_accuracy': deque(maxlen=window_size),
            'step_time': deque(maxlen=window_size),
            'lr': [],
        }

        # Created metrics buffers for individual tracked datasets
        self.dataset_trackers = defaultdict(lambda: VLAMetric([], '', '', {}))

    def log(self, global_step: int, metrics: Dict[str, Union[int,
                                                             float]]) -> None:
        for tracker in self.trackers:
            tracker.write(global_step, metrics)

    def get_status(self, loss: Optional[torch.Tensor] = None) -> str:
        lr = self.state['lr'][-1] if len(self.state['lr']) > 0 else 0
        if loss is None:
            return f'=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f}'  # noqa: E501

        # Otherwise, embed `loss` in status report!
        return f'=>> [Epoch {self.epoch:03d}] Global Step {self.global_step:06d} =>> LR :: {lr:.6f} - Loss :: {loss:.4f}'  # noqa: E501

    def commit(
        self,
        *,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
        lr: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Update all metrics in `self.state` by iterating
            through special positional arguments & kwargs."""
        if global_step is not None:
            self.global_step = global_step

        if epoch is not None:
            self.epoch = epoch

        # For all other variables --> only track on rank zero!
        if not overwatch.is_rank_zero():
            return

        # Special Positional Arguments
        if lr is not None:
            self.state['lr'].append(lr)

        if self.update_step_time:
            self.state['step_time'].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == 'loss':
                loss_val = value.detach()
                self.state['loss_raw'].append(loss_val)
                self.state['loss'].append(loss_val)
            else:
                self.state[key].append(value.detach())

    def commit_for_dataset(self, dataset_name: str, **kwargs) -> None:
        self.dataset_trackers[dataset_name].commit(**kwargs)

    @overwatch.rank_zero_only
    def push(self) -> str:
        # Note :: Raw Loss is an Average over
        # Gradient Accumulation Steps --> No Smoothing!
        loss_raw = torch.stack(list(self.state['loss_raw'])).mean().item()
        loss = torch.stack(list(self.state['loss'])).mean().item()
        if len(self.state['l1_loss']) > 0:
            l1_loss = torch.stack(list(self.state['l1_loss'])).mean().item()
        else:
            l1_loss = 0.0
        if len(self.state['action_accuracy']) > 0:
            action_accuracy = torch.stack(list(
                self.state['action_accuracy'])).mean().item()
        else:
            action_accuracy = 0.0
        if len(self.state['step_time']) > 0:
            step_time, lr = np.mean(list(
                self.state['step_time'])), self.state['lr'][-1]
        else:
            step_time, lr = 0.0, 0.0
        status = self.get_status(loss)

        # Get metrics per dataset
        dataset_metrics = {}
        for ds, tracker in self.dataset_trackers.items():
            dataset_metrics.update({
                f'{ds}/L1 Loss':
                torch.stack(list(tracker.state['l1_loss'])).mean().item(),
                f'{ds}/Action Token Accuracy':
                torch.stack(list(
                    tracker.state['action_accuracy'])).mean().item(),
            })

        # Fire to Trackers
        prefix = 'VLA Train'
        # Format lr to ensure at least 6 decimal places
        lr_formatted = f'{lr:.7f}'
        self.log(
            self.global_step,
            metrics={
                f'{prefix}/Step': self.global_step,
                f'{prefix}/Epoch': self.epoch,
                f'{prefix}/Loss': loss,
                f'{prefix}/L1 Loss': l1_loss,
                f'{prefix}/Action Token Accuracy': action_accuracy,
                f'{prefix}/Loss (Raw)': loss_raw,
                f'{prefix}/Learning Rate': lr_formatted,
                f'{prefix}/Step Time': step_time,
                **dataset_metrics,
            },
        )
        return status

    def finalize(self) -> str:
        for tracker in self.trackers:
            tracker.finalize()
