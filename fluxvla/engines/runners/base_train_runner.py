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
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from fluxvla.engines.utils import check_bloat16_supported
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import worker_init_function
from fluxvla.optimizers.schedulers import (get_constant_schedule,
                                           get_cosine_schedule_with_warmup,
                                           get_step_based_schedule)
from ..utils import build_tokenizer_from_cfg, initialize_overwatch

overwatch = initialize_overwatch(__name__)


class BaseTrainRunner(ABC):
    """Basic class for training VLA models.
    This class is designed to be subclassed and should not be used directly.

    Args:
        cfg (dict): Configuration dictionary containing model and training
            settings.
        stage (str): Stage of training (e.g., 'vla-train', 'vla-train').
        device_id (int): Device ID for training.
        epochs (int): Number of epochs to train.
        max_steps (int): Maximum number of training steps.
        learning_rate (int): Learning rate for the optimizer.
        collator (Dict): Collator configuration.
        save_iter_interval (int, optional): Interval for saving checkpoints
            based on iterations. Defaults to 10000.
        save_epoch_interval (int, optional): Interval for saving checkpoints
            based on epochs. Defaults to 10000.
        max_keep_ckpts (int, optional): Maximum number of checkpoints to keep.
            Defaults to 2.
        save_full_model (bool, optional): Whether to save the full model.
            Defaults to True.
        lr_scheduler_type (str, optional): Type of learning rate scheduler.
            Defaults to 'constant'.
        warmup_ratio (int, optional): Warm-up ratio for learning rate
            scheduler. Defaults to 0.
        enable_gradient_checkpointing (bool, optional): Enable gradient
            checkpointing. Defaults to True.
        enable_mixed_precision_training (bool, optional): Enable mixed
            precision training. Defaults to True.
        reduce_in_full_precision (bool, optional): Reduce in full precision.
            Defaults to True.
        mixed_precision_dtype (str, optional): Data type for mixed
            precision training. Defaults to 'bf16'.
        sharding_strategy (str, optional): Sharding strategy for
            distributed training. Defaults to 'full-shard'.
    """

    def __init__(self,
                 cfg: dict,
                 device_id: int,
                 learning_rate: int,
                 collator: Dict,
                 sampler: str,
                 metric: Dict,
                 max_epochs: int = None,
                 max_steps: Optional[int] = None,
                 save_epoch_interval: int = 1,
                 save_iter_interval: int = 10000,
                 max_keep_ckpts: int = 2,
                 save_full_model: bool = True,
                 lr_scheduler_type: str = 'constant',
                 lr_schedule: Optional[Dict[float, float]] = None,
                 warmup_ratio: int = 0,
                 enable_gradient_checkpointing: bool = True,
                 enable_mixed_precision_training: bool = True,
                 reduce_in_full_precision: bool = True,
                 mixed_precision_dtype: str = 'bf16',
                 tokenizer: Optional[Dict] = None,
                 resume_from: Optional[str] = None):
        from ..utils.builder import (build_collator_from_cfg,
                                     build_metric_from_cfg, build_vla_from_cfg)

        metric['hparams'] = cfg
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        metric[
            'run_id'] = f"{os.path.basename(cfg.filename).replace('.py', '')}_{timestamp}"  # noqa: E501
        self.metric = build_metric_from_cfg(metric)

        # Ensure only one training mode is set
        assert max_steps is None or max_epochs is None, \
            'Only one of `max_steps` or `max_epochs` can be set!'
        assert max_steps is not None or max_epochs is not None, \
            'One of `max_steps` or `max_epochs` must be set!'

        # Determine training mode
        self.training_mode = 'step_based' if max_steps is not None else 'epoch_based'  # noqa: E501

        self.vla = build_vla_from_cfg(cfg.model)
        self.all_module_keys = self.vla.all_module_keys
        self.trainable_module_keys = list()
        if self.vla.llm_backbone is not None:
            self.llm_transformer_layer_cls = self.vla.llm_backbone.transformer_layer_cls  # noqa: E501
        else:
            assert self.vla.vlm_backbone is not None, \
                'VLA model must have either an LLM or VLM backbone!'
            self.llm_transformer_layer_cls = self.vla.vlm_backbone.transformer_layer_cls  # noqa: E501

        self.device_id = device_id
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.collator = build_collator_from_cfg(collator)
        self.sampler = sampler
        self.save_iter_interval = save_iter_interval
        self.save_epoch_interval = save_epoch_interval
        self.max_keep_ckpts = max_keep_ckpts
        self.save_full_model = save_full_model
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.per_device_batch_size = cfg.train_dataloader.per_device_batch_size
        self.global_batch_size = self.per_device_batch_size * \
            overwatch.world_size()
        if hasattr(cfg.train_dataloader, 'per_device_num_workers'):
            self.per_device_num_workers = cfg.train_dataloader.per_device_num_workers  # noqa: E501
        else:
            self.per_device_num_workers = 0
        if tokenizer is not None:
            self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        else:
            self.tokenizer = None

        # Initialize training state
        self.current_epoch = 0
        self.steps_per_epoch = None  # Determined at runtime
        # Accumulate losses for checkpoint interval averaging
        self._loss_accumulator = []

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None
        self.wandb_mode = os.environ.get('WANDB_MODE', 'online')
        self.resume_from = resume_from
        # Track if optimizer state was successfully loaded
        self.optimizer_state_loaded = False
        # Store lr_schedule for step-based scheduler
        self.lr_schedule = lr_schedule

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), 'Per-device batch size must evenly divide global batch size!'
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size(  # noqa: E501
        )

        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, \
                'Only BF16 mixed precision training is supported!'
            assert check_bloat16_supported(), \
                'BFloat16 is not supported on this hardware; unset `mixed_precision`'  # noqa: E501

    def _convert_batch_to_dtype(self, batch: Dict, dtype: torch.dtype) -> Dict:
        """Convert floating point tensors in batch to specified dtype.

        This method automatically converts all floating point tensors in
        the batch to the target dtype (e.g., bfloat16), while preserving
        integer tensors and other data types.

        Args:
            batch (Dict): Input batch dictionary.
            dtype (torch.dtype): Target dtype (e.g., torch.bfloat16).

        Returns:
            Dict: Batch with converted dtypes.
        """
        converted_batch = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Convert floating point tensors to target dtype
                # Keep integer tensors (int, long, bool) as is
                if value.dtype.is_floating_point:
                    converted_batch[key] = value.to(dtype=dtype)
                else:
                    # Keep integer tensors unchanged
                    converted_batch[key] = value
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                converted_batch[key] = self._convert_batch_to_dtype(
                    value, dtype)
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples that may contain tensors
                converted_list = []
                for item in value:
                    if isinstance(
                            item,
                            torch.Tensor) and item.dtype.is_floating_point:
                        converted_list.append(item.to(dtype=dtype))
                    elif isinstance(item, dict):
                        converted_list.append(
                            self._convert_batch_to_dtype(item, dtype))
                    else:
                        converted_list.append(item)
                converted_batch[key] = (
                    tuple(converted_list)
                    if isinstance(value, tuple) else converted_list)
            else:
                # Keep non-tensor values as is
                converted_batch[key] = value

        return converted_batch

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save checkpoint including model, optimizer, and scheduler states.

        Subclasses should save:
        - Model state dict
        - Optimizer state dict
        - Scheduler state dict
        - Global step and epoch information
        """
        ...

    @abstractmethod
    def clip_grad_norm(self):
        """Clip gradient norm. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def _load_model_state(self, checkpoint_model_state: dict) -> None:
        """Load model state from checkpoint.

        Args:
            checkpoint_model_state (dict): Model state dict from checkpoint.
        """
        ...

    @abstractmethod
    def _load_optimizer_state(self, checkpoint_optimizer_state: dict) -> bool:
        """Load optimizer state from checkpoint.

        Args:
            checkpoint_optimizer_state (dict): Optimizer state dict from
                checkpoint.

        Returns:
            bool: True if optimizer state was successfully loaded,
                False otherwise.
        """
        ...

    def resume(self) -> None:
        """Resume training from checkpoint if specified.

        This method handles:
        - Loading training state (global_step, epoch, etc.)
        - Loading optimizer state (delegated to subclasses)
        - Loading scheduler state
        - Synchronizing all ranks after resume
        """
        if self.resume_from is None:
            return

        if overwatch.is_rank_zero():
            overwatch.info(
                f'Resuming training from checkpoint: {self.resume_from}')
        checkpoint_info = torch.load(self.resume_from)

        # Restore model state (delegated to subclasses for FSDP/DDP-specific
        # handling)
        if 'model' in checkpoint_info:
            self._load_model_state(checkpoint_info['model'])

        # Restore training state
        if 'global_step' in checkpoint_info:
            self.metric.global_step = checkpoint_info['global_step']
        if 'epoch' in checkpoint_info:
            self.current_epoch = checkpoint_info['epoch']

        # Restore optimizer state (delegated to subclasses)
        # Store checkpoint_info as instance variable for subclasses to access
        # additional information (e.g., parameter mappings)
        if ('optimizer_state_dict' in checkpoint_info
                and self.optimizer is not None):
            checkpoint_optimizer_state = checkpoint_info[
                'optimizer_state_dict']
            # Store checkpoint_info temporarily for _load_optimizer_state
            # to access
            self._current_checkpoint_info = checkpoint_info
            try:
                success = self._load_optimizer_state(
                    checkpoint_optimizer_state)
                if not success:
                    if overwatch.is_rank_zero():
                        overwatch.warning(
                            'Failed to load optimizer state. '
                            'Training will continue with fresh optimizer '
                            'state.')
            except Exception as e:
                if overwatch.is_rank_zero():
                    overwatch.warning(
                        f'Error loading optimizer state: {e}. '
                        f'Training will continue with fresh optimizer state.')
            finally:
                # Clean up temporary instance variable
                self._current_checkpoint_info = None
                # Ensure all ranks synchronize even if loading failed
                # This prevents deadlock if some ranks succeed and others fail
                dist.barrier()

        # Restore scheduler state
        if ('scheduler_state_dict' in checkpoint_info
                and self.lr_scheduler is not None):
            try:
                self.lr_scheduler.load_state_dict(
                    checkpoint_info['scheduler_state_dict'])
                if overwatch.is_rank_zero():
                    overwatch.info('Scheduler state restored from checkpoint')
            except Exception as e:
                overwatch.warning(f'Failed to load scheduler state: {e}')

        if overwatch.is_rank_zero():
            overwatch.info(
                f'Resumed training from step {self.metric.global_step}, '
                f'epoch {self.current_epoch}')
        dist.barrier()

    def _should_save_step_checkpoint(self) -> bool:
        """Check if checkpoint should be saved (step-based)."""
        return (self.metric.global_step % self.save_iter_interval) == 0

    def _should_save_epoch_checkpoint(self) -> bool:
        """Check if checkpoint should be saved (epoch-based)."""
        return (self.current_epoch % self.save_epoch_interval) == 0

    def _get_effective_dataset_size(self, dataset, sampler):
        """Get effective dataset size, handling RLDS datasets

        Args:
            dataset: The dataset object.
            sampler: The sampler used in DataLoader.
        """
        if sampler is not None:
            # Effective size after DistributedSampler processing
            return len(sampler)
        else:
            try:
                dataset_len = len(dataset)
                # If dataset has a finite length, use it
                return dataset_len // self.per_device_batch_size
            except (TypeError, AttributeError):
                return None

    def _estimate_steps_per_epoch(self, dataset, sampler):
        """Estimate steps per epoch, handling RLDS datasets"""
        if sampler is not None:
            # Effective size after DistributedSampler processing
            return len(sampler)
        else:
            dataset_len = len(dataset)
            # If dataset has a finite length, use it
            return math.ceil(dataset_len / self.global_batch_size)

    @staticmethod
    def _save_model_safetensors(model_state_dicts, safetensors_path):
        """Save model weights as safetensors alongside the .pt checkpoint.

        Handles both flat state dicts and nested dicts (FSDP with
        change_key_name) by flattening to a single-level {str: tensor} dict.
        """
        flat_dict = {}
        for key, value in model_state_dicts.items():
            if isinstance(value, dict):
                for sub_key, tensor in value.items():
                    flat_dict[f'{key}.{sub_key}'] = tensor
            elif isinstance(value, torch.Tensor):
                flat_dict[key] = value
        if flat_dict:
            save_file(flat_dict, safetensors_path)

    def _cleanup_old_checkpoints(self, checkpoint_dir: str):
        """Clean up old checkpoint files, keeping only the most recent ones."""
        ckpt_files = sorted(
            [
                f for f in os.listdir(checkpoint_dir)
                if f.endswith('.pt') and f != 'latest-checkpoint.pt'
            ],
            key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        if len(ckpt_files) > self.max_keep_ckpts:
            for old_ckpt in ckpt_files[:-self.max_keep_ckpts]:
                try:
                    os.remove(os.path.join(checkpoint_dir, old_ckpt))
                    overwatch.info(f'Removed old checkpoint: {old_ckpt}')
                    sf_file = old_ckpt.replace('.pt', '.safetensors')
                    sf_path = os.path.join(checkpoint_dir, sf_file)
                    if os.path.exists(sf_path):
                        os.remove(sf_path)
                        overwatch.info(f'Removed old safetensors: {sf_file}')
                except Exception as e:
                    overwatch.warning(
                        f'Failed to remove checkpoint {old_ckpt}: {e}')

    def _setup_optimizer_and_scheduler(
        self,
        n_train_examples: int,
        weight_decay: Optional[float] = None,
        lr_schedule: Optional[Dict[float, float]] = None,
    ) -> None:
        """Setup optimizer and learning rate scheduler.

        This method handles the creation of optimizer and scheduler based on
        the configured lr_scheduler_type. It supports parameter grouping
        with weight decay when weight_decay is provided.

        Args:
            n_train_examples: Number of training examples.
            weight_decay: Weight decay value for optimizer. If provided, will
                create parameter groups (decay/no_decay). If None, uses
                simple parameter list.
            lr_schedule: Dictionary mapping ratio (0-1) to learning rate for
                step-based scheduler. Required when lr_scheduler_type is
                'step-based'.
        """
        # Calculate number of training steps
        n_train_examples = math.ceil(
            n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples *
                                  self.max_epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == 'linear-warmup+cosine-decay':
            # Set warm-up steps (floor) based on `warmup_ratio`
            # (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            # Create Parameter Groups --> bias terms, normalization
            # layer parameters shouldn't be decayed!
            if weight_decay is not None:
                decay, no_decay = [], []
                for name, param in self.vla.named_parameters():
                    if not param.requires_grad:
                        continue

                    # Check on any parameters with fewer than 2 dimensions
                    # or with "bias" in the name
                    if param.ndim <= 1 or name.endswith('.bias'):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                # Build Parameter Groups
                groups = [{
                    'params': decay,
                    'weight_decay': weight_decay
                }, {
                    'params': no_decay,
                    'weight_decay': 0.0
                }]
            else:
                # Simple parameter list
                groups = [
                    param for param in self.vla.parameters()
                    if param.requires_grad
                ]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0

        elif self.lr_scheduler_type == 'constant':
            # Create Parameter Groups --> bias terms, normalization
            # layer parameters shouldn't be decayed!
            if weight_decay is not None:
                decay, no_decay = [], []
                for name, param in self.vla.named_parameters():
                    if not param.requires_grad:
                        continue

                    # Check on any parameters with fewer than 2 dimensions
                    # or with "bias" in the name
                    if param.ndim <= 1 or name.endswith('.bias'):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                # Build Parameter Groups
                groups = [{
                    'params': decay,
                    'weight_decay': weight_decay
                }, {
                    'params': no_decay,
                    'weight_decay': 0.0
                }]
            else:
                # Simple parameter list
                groups = [
                    param for param in self.vla.parameters()
                    if param.requires_grad
                ]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        elif self.lr_scheduler_type == 'step-based':
            if lr_schedule is None:
                raise ValueError('lr_schedule must be provided when using '
                                 'step-based scheduler')

            # Create Parameter Groups --> bias terms, normalization
            # layer parameters shouldn't be decayed!
            if weight_decay is not None:
                decay, no_decay = [], []
                for name, param in self.vla.named_parameters():
                    if not param.requires_grad:
                        continue

                    # Check on any parameters with fewer than 2 dimensions
                    # or with "bias" in the name
                    if param.ndim <= 1 or name.endswith('.bias'):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                # Build Parameter Groups
                groups = [{
                    'params': decay,
                    'weight_decay': weight_decay
                }, {
                    'params': no_decay,
                    'weight_decay': 0.0
                }]
            else:
                # Simple parameter list
                groups = [
                    param for param in self.vla.parameters()
                    if param.requires_grad
                ]

            # Create Optimizer & Step-based LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_step_based_schedule(
                self.optimizer, num_training_steps, lr_schedule)

        else:
            raise ValueError(f'Learning Rate Schedule with type '
                             f'`{self.lr_scheduler_type}` is not supported!')

    def run(self, vla_dataset) -> None:
        """Train the VLA model."""
        assert self.grad_accumulation_steps == 1, \
            'VLA training does not support gradient accumulation!'

        # Setup dataloader
        sampler = torch.utils.data.distributed.DistributedSampler(
            vla_dataset,
            num_replicas=overwatch.world_size(),
            rank=overwatch.rank(),
            shuffle=True,
            drop_last=False) if self.sampler == 'distributed' else None

        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=self.per_device_num_workers,
            worker_init_fn=worker_init_function)

        # Calculate steps per epoch
        self.steps_per_epoch = self._get_steps_per_epoch(vla_dataset)
        self._log_training_info(vla_dataset)
        self.resume()

        # Dispatch to training mode specific loop
        self.vla.train()
        self.optimizer.zero_grad()

        if self.training_mode == 'step_based':
            return self._run_step_based(dataloader, sampler)
        else:
            return self._run_epoch_based(dataloader, sampler)

    def _run_step_based(self, dataloader, sampler) -> str:
        """Step-based training loop. Handles infinite dataloaders."""
        with tqdm(
                total=self.max_steps,
                desc=self.metric.get_status(),
                leave=False,
                disable=not overwatch.is_rank_zero(),
                initial=self.metric.global_step) as progress:

            dataloader_iter = None
            epoch_step_count = 0

            while self.metric.global_step < self.max_steps:
                # Init/reset iterator at epoch start
                if dataloader_iter is None:
                    if sampler:
                        sampler.set_epoch(self.current_epoch)
                    dataloader_iter = iter(dataloader)
                    epoch_step_count = 0

                # Get next batch
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    # Finite dataloader exhausted, start new epoch
                    self.current_epoch += 1
                    dataloader_iter = None
                    continue

                loss = self._training_step(batch)
                self._loss_accumulator.append(
                    float(loss.detach().cpu().numpy().copy()))
                epoch_step_count += 1

                # Update metrics
                self.metric.commit(
                    global_step=self.metric.global_step + 1,
                    epoch=self.current_epoch,
                    lr=self.lr_scheduler.get_last_lr()[0])
                progress.set_description(self.metric.push())
                progress.update()

                # Save checkpoint
                if self._should_save_step_checkpoint():
                    self._save_and_sync()

                # For infinite dataloaders: check epoch boundary by step count
                if (self.steps_per_epoch
                        and epoch_step_count >= self.steps_per_epoch):
                    self.current_epoch += 1
                    dataloader_iter = None

        return self._get_checkpoint_path()

    def _run_epoch_based(self, dataloader, sampler) -> str:
        """Epoch-based training with nested progress bars. Handles
            infinite dataloaders.

        Args:
            dataloader: The dataloader object.
            sampler: The sampler used in DataLoader.

        Returns:
            str: The path to the latest checkpoint.
        """
        with tqdm(
                total=self.max_epochs,
                desc='Epochs',
                leave=False,
                disable=not overwatch.is_rank_zero(),
                initial=self.current_epoch) as epoch_pbar:

            while self.current_epoch < self.max_epochs:
                if sampler:
                    sampler.set_epoch(self.current_epoch)

                dataloader_iter = iter(dataloader)
                epoch_step_count = 0
                iter_total = self.steps_per_epoch or len(dataloader)

                with tqdm(
                        total=iter_total,
                        desc=f'Epoch {self.current_epoch}',
                        leave=False,
                        disable=not overwatch.is_rank_zero()) as iter_pbar:

                    while True:
                        # Get next batch
                        try:
                            batch = next(dataloader_iter)
                        except StopIteration:
                            # Finite dataloader exhausted
                            break

                        loss = self._training_step(batch)
                        self._loss_accumulator.append(
                            float(loss.detach().cpu().numpy().copy()))
                        epoch_step_count += 1

                        # Update metrics
                        self.metric.commit(
                            global_step=self.metric.global_step + 1,
                            epoch=self.current_epoch,
                            lr=self.lr_scheduler.get_last_lr()[0])
                        iter_pbar.set_description(self.metric.push())
                        iter_pbar.update()

                        # For infinite dataloaders: end epoch by step count
                        if (self.steps_per_epoch
                                and epoch_step_count >= self.steps_per_epoch):
                            break

                # Epoch completed
                self.current_epoch += 1
                epoch_pbar.update()

                # Save checkpoint at epoch end
                if self._should_save_epoch_checkpoint():
                    self._save_and_sync()

        return self._get_checkpoint_path()

    def _get_steps_per_epoch(self, vla_dataset) -> Optional[int]:
        """Calculate steps per epoch from dataset or estimate."""
        try:
            return math.ceil(len(vla_dataset) / self.global_batch_size)
        except (TypeError, AttributeError):
            return self._estimate_steps_per_epoch(vla_dataset, None)

    def _log_training_info(self, vla_dataset):
        """Log training configuration."""
        if not overwatch.is_rank_zero():
            return
        try:
            overwatch.info(f'Dataset length: {len(vla_dataset)}')
        except (TypeError, AttributeError):
            overwatch.info('Dataset length: unknown (infinite iterator)')
        overwatch.info(
            f'Training: mode={self.training_mode}, epochs={self.max_epochs}, '
            f'steps/epoch={self.steps_per_epoch}, '
            f'batch={self.global_batch_size} '
            f'({self.per_device_batch_size}x{overwatch.world_size()})')

    def _training_step(self, batch) -> torch.Tensor:
        """Execute single training step: forward, backward, optimize."""
        if self.enable_mixed_precision_training:
            batch = self._convert_batch_to_dtype(batch,
                                                 self.mixed_precision_dtype)
        with torch.autocast(
                'cuda',
                dtype=self.mixed_precision_dtype,
                enabled=self.enable_mixed_precision_training):
            output: CausalLMOutputWithPast = self.vla(**batch)
            loss = output['loss']

        self.metric.commit(loss=loss)
        loss.backward()

        # Commit per-dataset metrics
        if overwatch.is_rank_zero() and all(k in output for k in [
                'action_accuracy_ds', 'action_l1_loss_ds', 'ds_names'
        ]):  # noqa: E501
            for ds, acc, l1 in zip(output['ds_names'],
                                   output['action_accuracy_ds'],
                                   output['action_l1_loss_ds']):
                self.metric.commit_for_dataset(
                    dataset_name=ds.decode(), action_accuracy=acc, l1_loss=l1)

        # Gradient step with fallback on optimizer state mismatch
        self.clip_grad_norm()
        try:
            self.optimizer.step()
        except RuntimeError as e:
            if 'size' in str(e).lower() or 'shape' in str(e).lower():
                self._reinit_optimizer()
                self.optimizer.step()
            else:
                raise
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        # Custom hook for subclasses
        if hasattr(self, '_custom_training_step'):
            custom_loss = self._custom_training_step(batch, output, loss)
            if custom_loss is not None:
                loss = torch.tensor(custom_loss)

        return loss

    def _reinit_optimizer(self):
        """Reinitialize optimizer on state mismatch."""
        if overwatch.is_rank_zero():
            overwatch.warning('Optimizer state mismatch. Reinitializing.')
        trainable_params = [
            p for p in self.vla.parameters() if p.requires_grad
        ]
        current_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer = torch.optim.AdamW(trainable_params, lr=current_lr)
        self.optimizer_state_loaded = False
        if self.lr_scheduler and hasattr(self.lr_scheduler, 'optimizer'):
            self.lr_scheduler.optimizer = self.optimizer

    def _save_and_sync(self, loss_value: float = None):
        """Save checkpoint and synchronize.

        Uses averaged loss over the checkpoint interval if available.
        """
        # Use averaged loss if accumulated, otherwise use provided value
        if self._loss_accumulator:
            avg_loss = sum(self._loss_accumulator) / len(
                self._loss_accumulator)
            self._loss_accumulator.clear()
        else:
            avg_loss = loss_value

        self.save_checkpoint(
            self.metric.run_dir,
            self.metric.global_step,
            self.current_epoch,
            avg_loss,
            only_trainable=not self.save_full_model)
        dist.barrier()

    def _get_checkpoint_path(self) -> str:
        """Get latest checkpoint path."""
        return os.path.join(self.metric.run_dir, 'checkpoints',
                            'latest-checkpoint.pt')
