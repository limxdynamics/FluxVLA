# Origin: Modified from
# Upstream-Repo: openvla/openvla
# Upstream-Path: prismatic/training/strategies/ddp.py
# Upstream-Ref: main
# SPDX-License-Identifier: MIT
# Notes: Attribution normalized; no functional change.
import os
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import torch
import wandb
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

from fluxvla.engines.utils import build_vla_from_cfg
from fluxvla.engines.utils.overwatch import initialize_overwatch
from ..utils.root import RUNNERS
from .base_train_runner import BaseTrainRunner

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class DDPTrainRunner(BaseTrainRunner):
    """Distributed Data Parallel (DDP) Train Runner
    for Vision-Language Models. This runner is designed
    to fine-tune vision-language models using DDP
    across multiple GPUs. It handles the setup,
    training loop, and model checkpointing, while
    supporting features like mixed precision training,
    gradient accumulation, and LoRA (Low-Rank Adaptation)
    for efficient fine-tuning.

    Args:
        cfg (Dict): Configuration dictionary containing
            model, dataset, and training settings.
        args: Command-line arguments for the runner.
        learning_rate (float): Learning rate for the optimizer.
        collator (Dict): Collator configuration for batching.
        sampler (Dict): Sampler configuration for data loading.
        grad_accumulation_steps (int): Number of steps for gradient
            accumulation.
        max_epochs (int): Maximum number of training epochs.
        max_steps (int): Maximum number of training steps.
        save_epoch_interval (int): Interval for saving model checkpoints.
        save_iter_interval (int): Interval for saving model checkpoints.
        save_latest_checkpoint_only (bool): Whether to save only the
            latest checkpoint.
        max_keep_ckpts (int): Maximum number of checkpoints to keep.
        device_id (int): ID of the device to run the training on.
            Defaults to 2.
    """

    def __init__(self,
                 cfg: Dict,
                 args,
                 learning_rate: float,
                 weight_decay: Optional[float] = None,
                 max_grad_norm: float = 1.0,
                 collator: Dict = None,
                 sampler: str = 'distributed',
                 metric: Dict = None,
                 max_epochs: int = 10,
                 max_steps: Optional[int] = None,
                 save_epoch_interval: int = 1,
                 save_iter_interval: int = 10000,
                 max_keep_ckpts: int = 2,
                 save_full_model: bool = True,
                 lr_scheduler_type: str = 'constant',
                 lr_schedule: Optional[Dict[float, float]] = None,
                 warmup_ratio: int = 0,
                 freeze_steps: int = 0,
                 warmup_steps: int = 0,
                 lr_coef: float = 1.0,
                 betas: tuple = (0.9, 0.999),
                 use_cosine_decay: bool = False,
                 min_lr_ratio: float = 0.1,
                 enable_gradient_checkpointing: bool = True,
                 enable_mixed_precision_training: bool = True,
                 convert_batch_float_to_mixed_precision: bool = True,
                 reduce_in_full_precision: bool = True,
                 mixed_precision_dtype: str = 'bf16',
                 tokenizer: Optional[Dict] = None,
                 resume_from: Optional[str] = None,
                 **kwargs) -> None:

        device_id = overwatch.local_rank()
        super().__init__(
            cfg=cfg,
            device_id=device_id,
            learning_rate=learning_rate,
            collator=collator,
            sampler=sampler,
            metric=metric,
            max_epochs=max_epochs,
            max_steps=max_steps,
            save_epoch_interval=save_epoch_interval,
            save_iter_interval=save_iter_interval,
            max_keep_ckpts=max_keep_ckpts,
            save_full_model=save_full_model,
            lr_scheduler_type=lr_scheduler_type,
            lr_schedule=lr_schedule,
            warmup_ratio=warmup_ratio,
            freeze_steps=freeze_steps,
            warmup_steps=warmup_steps,
            lr_coef=lr_coef,
            betas=betas,
            use_cosine_decay=use_cosine_decay,
            min_lr_ratio=min_lr_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            convert_batch_float_to_mixed_precision=(
                convert_batch_float_to_mixed_precision),
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            tokenizer=tokenizer,
            resume_from=resume_from)

        self.cfg = cfg
        self.args = args
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.distributed_state = overwatch.distributed_state
        self.recent_losses = deque(maxlen=self.grad_accumulation_steps)

    def _load_model_state(self, checkpoint_model_state: dict) -> None:
        """Load DDP model state from checkpoint.

        Args:
            checkpoint_model_state (dict): Model state dict from checkpoint.
        """
        if overwatch.is_rank_zero():
            overwatch.info('Loading DDP model state')

        # Load model state dict (DDP-specific)
        if isinstance(self.vla, DDP):
            self.vla.module.load_state_dict(
                checkpoint_model_state, strict=False)
        else:
            self.vla.load_state_dict(checkpoint_model_state, strict=False)

        if overwatch.is_rank_zero():
            overwatch.info('DDP model state restored from checkpoint')

    def run_setup(self, n_train_examples: int) -> None:
        """Setup DDP-specific model configuration and distributed training."""
        torch.cuda.empty_cache()
        torch.cuda.set_device(device_id := self.device_id)

        self.vla.freeze_backbones()
        self.vla.from_pretrained()

        # Apply LoRA if specified
        if hasattr(self.cfg.model, 'use_lora') and self.cfg.model.use_lora:
            # Use configured lora_alpha, default to lora_rank if not specified
            lora_alpha = getattr(self.cfg.model, 'lora_alpha',
                                 self.cfg.model.lora_rank)
            # Get modules_to_save for full fine-tuning of specific modules
            modules_to_save = getattr(self.cfg.model, 'modules_to_save', None)
            lora_config = LoraConfig(
                r=self.cfg.model.lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=self.cfg.model.lora_dropout,
                target_modules=self.cfg.model.lora_target_modules,
                modules_to_save=modules_to_save,
                init_lora_weights='gaussian',
            )
            self.vla = get_peft_model(self.vla, lora_config)
            self.vla.print_trainable_parameters()

        # Setup optimizer and scheduler using base class method
        # Support optional weight_decay parameter grouping (if provided)
        self._setup_optimizer_and_scheduler(
            n_train_examples,
            weight_decay=self.weight_decay,
            lr_schedule=self.lr_schedule)

        # Move model to device and wrap with DDP
        torch.cuda.empty_cache()
        # Move to device and optionally convert to bf16
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:  # noqa: E501
            self.vla = self.vla.to(device=device_id, dtype=torch.bfloat16)
        else:
            self.vla = self.vla.to(device_id)

        # Apply Gradient Checkpointing (after moving to device)
        if self.enable_gradient_checkpointing:
            from functools import partial

            import torch.nn as nn
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # noqa: E501
                CheckpointImpl, apply_activation_checkpointing,
                checkpoint_wrapper)

            # Collect checkpoint layer classes (same as FSDP)
            checkpoint_layer_classes = set()

            # Add LLM backbone transformer layers
            if hasattr(self, 'llm_transformer_layer_cls'):
                checkpoint_layer_classes.add(self.llm_transformer_layer_cls)

            # Add Vision Transformer blocks (for timm models)
            try:
                from timm.models.vision_transformer import Block as VisionBlock
                checkpoint_layer_classes.add(VisionBlock)
            except ImportError:
                pass

            # Add LLM expert layers
            if hasattr(self.vla,
                       'llm_expert') and self.vla.llm_expert is not None:
                if hasattr(self.vla.llm_expert, 'transformer_layer_cls'):
                    checkpoint_layer_classes.add(
                        self.vla.llm_expert.transformer_layer_cls)

            # Apply checkpoint wrapper if we have layer classes
            if checkpoint_layer_classes:
                non_reentrant_wrapper = partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT)

                def check_fn(submodule: nn.Module) -> bool:
                    for layer_cls in checkpoint_layer_classes:
                        if isinstance(submodule, layer_cls):
                            return True
                    return False

                apply_activation_checkpointing(
                    self.vla,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=check_fn)

                if overwatch.is_rank_zero():
                    overwatch.info(
                        f'Applied gradient checkpointing to: '
                        f'{[cls.__name__ for cls in checkpoint_layer_classes]}'
                    )

        self.vla = DDP(
            self.vla,
            device_ids=[device_id],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=True)

        if overwatch.is_rank_zero():
            overwatch.info(
                'DDP =>> Finalized Training Setup:\n'
                f'|-> Training Mode = {self.training_mode}\n'
                f'|-> Max Epochs = {self.max_epochs}\n'
                f'|-> Max Steps = {self.max_steps}\n'
                f'|-> Global (Effective) Batch Size = {self.global_batch_size}\n'  # noqa: E501
                f'|-> Per-Device Batch Size = {self.per_device_batch_size}\n'
                f'|-> Distributed World Size = {overwatch.world_size()}\n'
                f'|-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n'  # noqa: E501
                f'|-> Gradient Checkpointing = {self.enable_gradient_checkpointing}\n'  # noqa: E501
                f'|-> Mixed Precision Training = {self.enable_mixed_precision_training}\n'  # noqa: E501
                f'     |-> Dtype = {self.mixed_precision_dtype}\n')

    def clip_grad_norm(self):
        """Clip gradient norm for DDP model."""
        torch.nn.utils.clip_grad_norm_(
            self.vla.parameters(), max_norm=self.max_grad_norm)

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save checkpoint with DDP and LoRA specific handling."""
        if overwatch.is_rank_zero():

            overwatch.info(f'Saving Model Checkpoint for Step {global_step}, '
                           f'Epoch {epoch}')

            # Save the model using PEFT save_pretrained if using LoRA
            save_dir = str(run_dir)

            # Save backbone configs if available
            if hasattr(self.vla.module, 'llm_backbone') and hasattr(
                    self.vla.module.llm_backbone, 'config'):
                self.vla.module.llm_backbone.config.to_json_file(
                    os.path.join(save_dir, 'llm_backbone_config.json'))
            if hasattr(self.vla.module, 'vlm_backbone') and hasattr(
                    self.vla.module.vlm_backbone, 'config'):
                self.vla.module.vlm_backbone.config.to_json_file(
                    os.path.join(save_dir, 'vlm_backbone_config.json'))

            # Handle LoRA merging and checkpoint creation
            if hasattr(self.cfg.model, 'use_lora') and self.cfg.model.use_lora:
                # First, save the current LoRA adapter to save_dir
                # This is necessary before loading it with
                # PeftModel.from_pretrained
                self.vla.module.save_pretrained(save_dir)

                base_vla = build_vla_from_cfg(self.cfg.model)
                # Load pretrained weights before merging LoRA
                base_vla.from_pretrained()
                merged_vla = PeftModel.from_pretrained(base_vla, save_dir)
                merged_vla = merged_vla.merge_and_unload()
                model_state_dict = merged_vla.state_dict()
            else:
                model_state_dict = self.vla.module.state_dict()

            checkpoint_dir = os.path.join(run_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Create checkpoint filename (unified format)
            checkpoint_name = f'step-{global_step:06d}-epoch-{epoch:03d}'

            if train_loss is not None:
                checkpoint_name += f'-loss={train_loss:.4f}'
            checkpoint_name += '.pt'

            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

            # Prepare checkpoint dictionary
            checkpoint_dict = {
                'model': model_state_dict,
                'global_step': global_step,
                'epoch': epoch,
            }

            # Save optimizer state with parameter name mapping
            # Fix: Directly save state_index -> param_name mapping dictionary
            if self.optimizer is not None:
                optimizer_state = self.optimizer.state_dict()
                optimizer_state_dict_actual = optimizer_state.get('state', {})

                # Get optimizer parameters in order
                optimizer_params = []
                for param_group in self.optimizer.param_groups:
                    optimizer_params.extend(param_group['params'])

                # Build mapping from parameter data_ptr to name
                model_state_dict_current = self.vla.module.state_dict()
                param_ptr_to_name = {}
                for name, model_param in model_state_dict_current.items():
                    param_ptr_to_name[model_param.data_ptr()] = name

                # Fix: Build state_index -> param_name mapping dictionary
                # This allows direct lookup of parameter names by
                # state_index during loading
                state_index_to_param_name = {}
                for state_idx in optimizer_state_dict_actual.keys():
                    if state_idx < len(optimizer_params):
                        param = optimizer_params[state_idx]
                        param_ptr = param.data_ptr()
                        if param_ptr in param_ptr_to_name:
                            state_index_to_param_name[
                                state_idx] = param_ptr_to_name[param_ptr]
                        else:
                            overwatch.warning(
                                f'Could not find name for optimizer '
                                f'parameter at index {state_idx}')
                    else:
                        overwatch.warning(
                            f'State index {state_idx} exceeds optimizer '
                            f'parameter count')

                overwatch.info(
                    f'Saving optimizer state: '
                    f'{len(optimizer_params)} total parameters, '
                    f'{len(optimizer_state_dict_actual)} states with '
                    f'gradients, {len(state_index_to_param_name)} names '
                    f'mapped')

                checkpoint_dict['optimizer_state_dict'] = optimizer_state
                # Fix: Save mapping dictionary instead of list
                checkpoint_dict['optimizer_state_index_to_name'] = (
                    state_index_to_param_name)

            # Save scheduler state
            if self.lr_scheduler is not None:
                checkpoint_dict[
                    'scheduler_state_dict'] = self.lr_scheduler.state_dict()

            torch.save(checkpoint_dict, checkpoint_path)
            overwatch.info(f'Saved Checkpoint at: {checkpoint_path}')

            # Save model weights as safetensors for fast loading
            safetensors_path = checkpoint_path.replace('.pt', '.safetensors')
            self._save_model_safetensors(model_state_dict, safetensors_path)
            overwatch.info(f'Saved safetensors at: {safetensors_path}')

            # Create/update latest checkpoint symlink
            latest_ckpt_link = os.path.join(checkpoint_dir,
                                            'latest-checkpoint.pt')
            if os.path.exists(latest_ckpt_link) or os.path.islink(
                    latest_ckpt_link):
                os.unlink(latest_ckpt_link)
            os.symlink(os.path.basename(checkpoint_path), latest_ckpt_link)

            latest_sf_link = os.path.join(checkpoint_dir,
                                          'latest-checkpoint.safetensors')
            if os.path.exists(latest_sf_link) or os.path.islink(
                    latest_sf_link):
                os.unlink(latest_sf_link)
            os.symlink(os.path.basename(safetensors_path), latest_sf_link)

            # Clean up old checkpoints
            self._cleanup_old_checkpoints(checkpoint_dir)

    def _load_optimizer_state(self, checkpoint_optimizer_state: dict) -> bool:
        """Load DDP optimizer state from checkpoint with robust parameter
        matching.

        Args:
            checkpoint_optimizer_state (dict): Optimizer state dict from
                checkpoint.

        Returns:
            bool: True if optimizer state was successfully loaded,
                False otherwise.

        Note:
            This method accesses checkpoint parameter mapping information
            from instance variables set by resume() method:
            - self._checkpoint_state_index_to_name: New format mapping
            - self._checkpoint_param_name_list: Old format list
              (backward compatibility)
        """
        # Get parameter mapping info from checkpoint_info
        # Try to get from _current_checkpoint_info (set by base class
        # resume method) or from instance variables (set by
        # load_checkpoint method)
        checkpoint_info = getattr(self, '_current_checkpoint_info', None)
        if checkpoint_info is None:
            # Fallback to instance variables (for load_checkpoint)
            checkpoint_state_index_to_name = getattr(
                self, '_checkpoint_state_index_to_name', None)
            checkpoint_param_name_list = getattr(
                self, '_checkpoint_param_name_list', None)
        else:
            # Get from checkpoint_info (for resume)
            checkpoint_state_index_to_name = checkpoint_info.get(
                'optimizer_state_index_to_name', None)
            checkpoint_param_name_list = checkpoint_info.get(
                'optimizer_param_name_list', None)
        # Check if optimizer is properly initialized
        if (not hasattr(self.optimizer, 'param_groups')
                or len(self.optimizer.param_groups) == 0):
            if overwatch.is_rank_zero():
                overwatch.warning('Optimizer is not properly initialized: '
                                  'param_groups is empty')
            return False

        # Get current parameters from optimizer
        current_params = []
        for param_group in self.optimizer.param_groups:
            current_params.extend(param_group['params'])

        if len(current_params) == 0:
            if overwatch.is_rank_zero():
                overwatch.warning('Current optimizer has no parameters.')
            return False

        checkpoint_state = checkpoint_optimizer_state.get('state', {})
        checkpoint_param_count = len(checkpoint_state)
        current_param_count = len(current_params)

        if overwatch.is_rank_zero():
            overwatch.info(
                f'Loading optimizer state: checkpoint has '
                f'{checkpoint_param_count} states, current optimizer has '
                f'{current_param_count} parameters')

        # Build current parameter name to index mapping
        current_model_state_dict = self.vla.module.state_dict()
        param_ptr_to_name = {}
        for name, model_param in current_model_state_dict.items():
            param_ptr_to_name[model_param.data_ptr()] = name

        current_name_to_idx = {}
        for idx, param in enumerate(current_params):
            param_ptr = param.data_ptr()
            if param_ptr in param_ptr_to_name:
                current_name_to_idx[param_ptr_to_name[param_ptr]] = idx

        # Prefer new format mapping dictionary
        if checkpoint_state_index_to_name is not None:
            return self._load_with_index_to_name_mapping(
                checkpoint_optimizer_state, checkpoint_state,
                checkpoint_state_index_to_name, current_params,
                current_name_to_idx)

        # Backward compatibility: use old format parameter name list
        if checkpoint_param_name_list is not None:
            return self._load_with_param_name_list(checkpoint_optimizer_state,
                                                   checkpoint_state,
                                                   checkpoint_param_name_list,
                                                   current_params,
                                                   current_name_to_idx)

        # No mapping information found, attempt direct load
        if overwatch.is_rank_zero():
            overwatch.warning(
                'No parameter mapping found. Attempting direct load...')

        if checkpoint_param_count != current_param_count:
            if overwatch.is_rank_zero():
                overwatch.warning(
                    f'Parameter count mismatch: {checkpoint_param_count} vs '
                    f'{current_param_count}. Skipping optimizer state loading.'
                )
            return False

        try:
            self.optimizer.load_state_dict(checkpoint_optimizer_state)
            if overwatch.is_rank_zero():
                overwatch.info('Optimizer state restored (direct load)')
            return True
        except Exception as e:
            if overwatch.is_rank_zero():
                overwatch.warning(f'Direct load failed: {e}')
            return False

    def _load_with_index_to_name_mapping(self,
                                         checkpoint_optimizer_state: dict,
                                         checkpoint_state: dict,
                                         state_index_to_name: dict,
                                         current_params: list,
                                         current_name_to_idx: dict) -> bool:
        """Load optimizer state using state_index -> param_name mapping
        (new format)."""

        matched_count = 0
        remapped_state = {}

        for ckpt_state_idx, param_name in state_index_to_name.items():
            # Ensure ckpt_state_idx is an integer (may be string when
            # loaded from JSON)
            ckpt_state_idx = int(ckpt_state_idx)

            if param_name not in current_name_to_idx:
                if overwatch.is_rank_zero():
                    overwatch.debug(
                        f'Parameter {param_name} not found in current model')
                continue

            if ckpt_state_idx not in checkpoint_state:
                if overwatch.is_rank_zero():
                    overwatch.debug(
                        f'State index {ckpt_state_idx} not found in checkpoint'
                    )
                continue

            current_idx = current_name_to_idx[param_name]
            ckpt_state = checkpoint_state[ckpt_state_idx]
            current_param = current_params[current_idx]

            # Verify shape compatibility
            state_compatible = True
            for key, value in ckpt_state.items():
                if isinstance(value, torch.Tensor):
                    if key in ['exp_avg', 'exp_avg_sq'
                               ] and value.shape != current_param.shape:
                        state_compatible = False
                        if overwatch.is_rank_zero():
                            overwatch.debug(
                                f'Shape mismatch for {param_name}.{key}: '
                                f'{value.shape} vs {current_param.shape}')
                        break

            if state_compatible:
                remapped_state[current_idx] = ckpt_state
                matched_count += 1

        if overwatch.is_rank_zero():
            overwatch.info(f'Matched {matched_count}/{len(current_params)} '
                           f'optimizer states')

        return self._apply_remapped_state(checkpoint_optimizer_state,
                                          remapped_state, matched_count,
                                          len(current_params))

    def _load_with_param_name_list(self, checkpoint_optimizer_state: dict,
                                   checkpoint_state: dict,
                                   param_name_list: list, current_params: list,
                                   current_name_to_idx: dict) -> bool:
        """Load optimizer state using parameter name list (old format,
        backward compatible)."""

        # Issue with old format: param_name_list is built in
        # state_indices order. So we need to get state_indices to
        # correctly map
        state_indices = sorted(checkpoint_state.keys())

        if len(param_name_list) != len(state_indices):
            if overwatch.is_rank_zero():
                overwatch.warning(
                    f'param_name_list length ({len(param_name_list)}) != '
                    f'state_indices length ({len(state_indices)})')
            return False

        matched_count = 0
        remapped_state = {}

        for list_idx, param_name in enumerate(param_name_list):
            if param_name.startswith('__unknown_param_'):
                continue

            if param_name not in current_name_to_idx:
                continue

            # Key fix: use state_indices[list_idx] instead of list_idx
            ckpt_state_idx = state_indices[list_idx]

            if ckpt_state_idx not in checkpoint_state:
                continue

            current_idx = current_name_to_idx[param_name]
            ckpt_state = checkpoint_state[ckpt_state_idx]
            current_param = current_params[current_idx]

            # Verify shape compatibility
            state_compatible = True
            for key, value in ckpt_state.items():
                if isinstance(value, torch.Tensor):
                    if key in ['exp_avg', 'exp_avg_sq'
                               ] and value.shape != current_param.shape:
                        state_compatible = False
                        break

            if state_compatible:
                remapped_state[current_idx] = ckpt_state
                matched_count += 1

        if overwatch.is_rank_zero():
            overwatch.info(f'Matched {matched_count}/{len(current_params)} '
                           f'optimizer states')

        return self._apply_remapped_state(checkpoint_optimizer_state,
                                          remapped_state, matched_count,
                                          len(current_params))

    def _apply_remapped_state(self, checkpoint_optimizer_state: dict,
                              remapped_state: dict, matched_count: int,
                              total_params: int) -> bool:
        """Apply remapped optimizer state."""

        # Skip loading if too few parameters matched
        min_match_ratio = 0.5  # At least 50% match required
        if matched_count < total_params * min_match_ratio:
            if overwatch.is_rank_zero():
                overwatch.warning(f'Too few parameters matched '
                                  f'({matched_count}/{total_params}, '
                                  f'< {min_match_ratio*100}%). '
                                  f'Skipping optimizer state loading.')
            return False

        remapped_optimizer_state = {
            'state': remapped_state,
            'param_groups': checkpoint_optimizer_state['param_groups']
        }

        try:
            self.optimizer.load_state_dict(remapped_optimizer_state)
            if overwatch.is_rank_zero():
                overwatch.info(
                    f'Optimizer state restored with {matched_count} '
                    f'matched parameters')
            return True
        except Exception as e:
            if overwatch.is_rank_zero():
                overwatch.warning(
                    f'Failed to load remapped optimizer state: {e}')
            return False

    def resume(self) -> None:
        """Resume training from checkpoint with DDP-specific optimizer state
        handling.

        Overrides base class to pass additional parameter mapping information
        to _load_optimizer_state.
        """
        import torch.distributed as dist

        if self.resume_from is None:
            return

        if overwatch.is_rank_zero():
            overwatch.info(
                f'Resuming training from checkpoint: {self.resume_from}')
        checkpoint_info = torch.load(self.resume_from)

        # Restore training state (reuse base class logic)
        if 'global_step' in checkpoint_info:
            self.metric.global_step = checkpoint_info['global_step']
        if 'epoch' in checkpoint_info:
            self.current_epoch = checkpoint_info['epoch']

        # Restore optimizer state with parameter mapping support
        # Store checkpoint info for _load_optimizer_state to access
        if ('optimizer_state_dict' in checkpoint_info
                and self.optimizer is not None):
            checkpoint_optimizer_state = checkpoint_info[
                'optimizer_state_dict']
            # Store additional mapping info as instance variables for
            # _load_optimizer_state
            self._checkpoint_state_index_to_name = checkpoint_info.get(
                'optimizer_state_index_to_name', None)
            self._checkpoint_param_name_list = checkpoint_info.get(
                'optimizer_param_name_list', None)

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
                        f'Training will continue with fresh optimizer '
                        f'state.')
            finally:
                # Clean up temporary instance variables
                self._checkpoint_state_index_to_name = None
                self._checkpoint_param_name_list = None
                # Ensure all ranks synchronize even if loading failed
                dist.barrier()

        # Restore scheduler state (reuse base class logic)
        if ('scheduler_state_dict' in checkpoint_info
                and self.lr_scheduler is not None):
            try:
                self.lr_scheduler.load_state_dict(
                    checkpoint_info['scheduler_state_dict'])
                if overwatch.is_rank_zero():
                    overwatch.info('Scheduler state restored from checkpoint')
            except Exception as e:
                if overwatch.is_rank_zero():
                    overwatch.warning(f'Failed to load scheduler state: {e}')

        if overwatch.is_rank_zero():
            overwatch.info(
                f'Resumed training from step {self.metric.global_step}, '
                f'epoch {self.current_epoch}')
        dist.barrier()

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint including model, optimizer, and scheduler states.

        Overrides base class to handle DDP-specific model loading and
        parameter mapping for optimizer state.
        """
        import torch.distributed as dist

        if not os.path.exists(checkpoint_path):
            if os.path.islink(checkpoint_path):
                checkpoint_path = os.readlink(checkpoint_path)
            else:
                raise FileNotFoundError(
                    f'Checkpoint not found: {checkpoint_path}')

        if overwatch.is_rank_zero():
            overwatch.info(f'Loading checkpoint from: {checkpoint_path}')

        checkpoint = torch.load(
            checkpoint_path, map_location=f'cuda:{self.device_id}')

        # Load model state dict (DDP-specific)
        if 'model' in checkpoint:
            if isinstance(self.vla, DDP):
                self.vla.module.load_state_dict(
                    checkpoint['model'], strict=False)
            else:
                self.vla.load_state_dict(checkpoint['model'], strict=False)
            if overwatch.is_rank_zero():
                overwatch.info('Model state dict loaded')

        # Prepare return dictionary (reuse base class structure)
        result = {
            'global_step': checkpoint.get('global_step', 0),
            'epoch': checkpoint.get('epoch', 0),
        }

        # Load optimizer state with parameter mapping support
        if ('optimizer_state_dict' in checkpoint
                and self.optimizer is not None):
            checkpoint_optimizer_state = checkpoint['optimizer_state_dict']
            # Store mapping info as instance variables for
            # _load_optimizer_state
            self._checkpoint_state_index_to_name = checkpoint.get(
                'optimizer_state_index_to_name', None)
            self._checkpoint_param_name_list = checkpoint.get(
                'optimizer_param_name_list', None)

            try:
                success = self._load_optimizer_state(
                    checkpoint_optimizer_state)
                if not success:
                    if overwatch.is_rank_zero():
                        overwatch.warning(
                            'Failed to load optimizer state. '
                            'Training will continue with fresh optimizer '
                            'state.')
            finally:
                # Clean up temporary instance variables
                self._checkpoint_state_index_to_name = None
                self._checkpoint_param_name_list = None

            result['optimizer_state_dict'] = checkpoint_optimizer_state

        # Load scheduler state (reuse base class logic)
        if 'scheduler_state_dict' in checkpoint:
            result['scheduler_state_dict'] = checkpoint.get(
                'scheduler_state_dict')

        dist.barrier()
        return result

    def _custom_training_step(self, batch, output, loss):
        """Custom training step for DDP-specific logging and metrics."""
        # Add loss to recent losses for smoothing
        self.recent_losses.append(loss.item())

        # Compute smoothed loss
        smoothened_loss = sum(self.recent_losses) / len(self.recent_losses)

        # Log to wandb if enabled
        if (overwatch.is_rank_zero() and self.metric.global_step % 10 == 0
                and self.wandb_mode != 'disabled'):
            wandb.log(
                {
                    'train_loss': smoothened_loss,
                    'epoch': self.current_epoch,
                    'lr': self.lr_scheduler.get_last_lr()[0]
                },
                step=self.metric.global_step)

        return smoothened_loss

    def run(self, vla_dataset):
        """Run training with DDP-specific enhancements while using BaseTrainRunner logic."""  # noqa: E501
        # Save dataset statistics if available
        if overwatch.is_rank_zero():
            if hasattr(vla_dataset, 'dataset_statistics'):
                from fluxvla.datasets.utils import save_dataset_statistics
                save_dataset_statistics(vla_dataset.dataset_statistics,
                                        self.args.work_dir)

        # Use parent's training logic
        return super().run(vla_dataset)
