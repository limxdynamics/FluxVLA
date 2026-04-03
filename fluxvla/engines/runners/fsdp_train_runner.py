# Origin: Modified from
# Upstream-Repo: openvla/openvla
# Upstream-Path: prismatic/training/strategies/fsdp.py
# Upstream-Ref: main
# SPDX-License-Identifier: MIT
# Notes: Attribution normalized; no functional change.

import math
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (MixedPrecision, ShardingStrategy,
                                    StateDictType)

from ..utils import initialize_overwatch
from ..utils.root import RUNNERS
from .base_train_runner import BaseTrainRunner

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class FSDPTrainRunner(BaseTrainRunner):
    """FSDP Runner for training VLMs with Fully Sharded Data Parallelism.
    This class extends the BaseTrainRunner and implements the
    setup and training process for FSDP.
    It initializes the FSDP strategy, sets up the optimizer and learning rate
    scheduler, and handles gradient checkpointing.
    It also provides a method to save checkpoints with only trainable
    parameters.

    Args:
        cfg (dict): Configuration dictionary for the runner.
        stage (str): Stage of training (e.g., 'vla-train', 'vla-train').
        epochs (int): Number of training epochs.
        max_steps (int): Maximum number of training steps.
        learning_rate (int): Learning rate for the optimizer.
        weight_decay (int): Weight decay for the optimizer.
        max_grad_norm (int): Maximum gradient norm for clipping.
        collator (Dict): Collator object for batching data.
        metric (Dict): Metric object for evaluation.
        save_iter_interval (int, optional): Interval for saving checkpoints
            based on iterations. Defaults to 10000.
        save_epoch_interval (int, optional): Interval for saving checkpoints
            based on epochs. Defaults to 1.
        max_keep_ckpts (int, optional): Maximum number of checkpoints to keep.
            Defaults to 2.
        save_full_model (bool, optional): Whether to save the full model.
            Defaults to True.
        lr_scheduler_type (str, optional): Type of learning rate scheduler.
            Supported types: 'constant', 'linear-warmup+cosine-decay',
            'step-based'. Defaults to 'constant'.
        warmup_ratio (int, optional): Ratio of warm-up steps.
            Defaults to 0.
        lr_schedule (Dict[float, float], optional): Dictionary mapping ratio
            (0-1) to learning rate for step-based scheduler. Required when
            lr_scheduler_type is 'step-based'. Format: {ratio: lr}, e.g.,
            {0: 1e-4, 0.8: 1e-5} means 0-80% of steps use 1e-4, 80%-100% use
            1e-5. Defaults to None.
        enable_gradient_checkpointing (bool, optional): Enable gradient
            checkpointing. Defaults to True.
        enable_mixed_precision_training (bool, optional): Enable mixed
            precision training. Defaults to True.
        reduce_in_full_precision (bool, optional): Reduce in full precision.
            Defaults to True.
        mixed_precision_dtype (str, optional): Data type for mixed precision
            training.  Defaults to 'bf16'.
        sharding_strategy (str, optional): Sharding strategy for FSDP.
            Defaults to 'full-shard'.
    """

    def __init__(self,
                 cfg: dict,
                 learning_rate: int,
                 weight_decay: int,
                 max_grad_norm: int,
                 collator: Dict,
                 sampler: str,
                 metric: Dict,
                 max_epochs: int = None,
                 max_steps: int = None,
                 save_epoch_interval: int = 1,
                 save_iter_interval: int = 10000,
                 max_keep_ckpts: int = 2,
                 save_full_model: bool = True,
                 lr_scheduler_type: str = 'constant',
                 warmup_ratio: int = 0,
                 lr_schedule: Optional[Dict[float, float]] = None,
                 enable_gradient_checkpointing: bool = True,
                 enable_mixed_precision_training: bool = True,
                 reduce_in_full_precision: bool = True,
                 mixed_precision_dtype: str = 'bf16',
                 sharding_strategy: str = 'hybrid-shard',
                 change_key_name: bool = False,
                 tokenizer: Optional[Dict] = None,
                 resume_from: Optional[str] = None,
                 *args,
                 **kwargs) -> None:
        device_id = overwatch.local_rank()
        super().__init__(cfg, device_id, learning_rate, collator, sampler,
                         metric, max_epochs, max_steps, save_epoch_interval,
                         save_iter_interval, max_keep_ckpts, save_full_model,
                         lr_scheduler_type, lr_schedule, warmup_ratio,
                         enable_gradient_checkpointing,
                         enable_mixed_precision_training,
                         reduce_in_full_precision, mixed_precision_dtype,
                         tokenizer, resume_from)
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.lr_schedule = lr_schedule
        self.sharding_strategy = sharding_strategy
        if self.sharding_strategy == 'shard-grad-op':
            self.fsdp_sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif self.sharding_strategy == 'full-shard':
            self.fsdp_sharding_strategy = ShardingStrategy.FULL_SHARD
        elif self.sharding_strategy == 'hybrid-shard':
            self.fsdp_sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise ValueError(
                f'FSDP Sharding Strategy {sharding_strategy} is not supported!'
            )
        self.change_key_name = change_key_name
        self.fsdp_state_dict_type = StateDictType.FULL_STATE_DICT
        self.fsdp_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True)

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Saves the checkpoint of the model.

        Args:
            run_dir (Path): Directory to save the checkpoint.
            global_step (int): Current global step.
            epoch (int): Current epoch.
            train_loss (Optional[float], optional): Training loss.
                Defaults to None.
            only_trainable (bool, optional): Whether to save only
                trainable parameters. Defaults to True.
        """
        assert isinstance(self.vla, FSDP), \
            'FSDPStrategy.save_checkpoint assumes VLA is \
                already wrapped in FSDP!'

        if hasattr(self.vla._fsdp_wrapped_module, 'llm_backbone'):
            if hasattr(self.vla._fsdp_wrapped_module.llm_backbone, 'config'):
                self.vla._fsdp_wrapped_module.llm_backbone.config.to_json_file(  # noqa: E501
                    os.path.join(run_dir, 'llm_backbone_config.json'))
        if hasattr(self.vla._fsdp_wrapped_module, 'vlm_backbone'):
            if hasattr(self.vla._fsdp_wrapped_module.vlm_backbone, 'config'):
                self.vla._fsdp_wrapped_module.vlm_backbone.config.to_json_file(  # noqa: E501
                    os.path.join(run_dir, 'vlm_backbone_config.json'))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(run_dir, 'tokenizer'))
        # Summon Full State Dictionary =>> Reconstitute from Shards
        with FSDP.state_dict_type(self.vla, self.fsdp_state_dict_type,
                                  self.fsdp_save_policy):
            full_vla_state_dict = self.vla.state_dict()
            model_state_dicts = {
                mkey: OrderedDict()
                for mkey in (self.trainable_module_keys
                             if only_trainable else self.all_module_keys)
            }

            # Iterate through `full_vlm_state_dict` and split
            # `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
            if self.change_key_name:
                for key, param in full_vla_state_dict.items():
                    for mkey in model_state_dicts:
                        if key.startswith(mprefix := f'{mkey}.'):
                            model_state_dicts[mkey][key.removeprefix(
                                mprefix)] = param
            else:
                model_state_dicts = full_vla_state_dict

            # Get full optimizer state dict for FSDP
            # FSDP shards optimizer states, so we need to gather the full state
            # IMPORTANT: Ensure all ranks are synchronized before gathering
            # optimizer state
            # This prevents AssertionError about different step values across
            # ranks
            # First barrier: ensure all ranks reach this point
            dist.barrier()

            # For FSDP, we need to ensure optimizer states are synchronized
            # before calling full_optim_state_dict
            # This is critical after resume, as different ranks might have
            # different optimizer states if loading failed on some ranks
            if self.optimizer is not None:
                # Ensure all ranks have completed the same number of optimizer
                # steps by synchronizing before gathering the full state
                dist.barrier()
                full_optimizer_state_dict = FSDP.full_optim_state_dict(
                    self.vla, self.optimizer)
            else:
                full_optimizer_state_dict = None

            # Save on rank zero *only*
            if overwatch.is_rank_zero():
                checkpoint_dir = os.path.join(run_dir, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                if train_loss is None:
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt'
                    )  # noqa: E501
                else:
                    checkpoint_path = (
                        os.path.join(
                            checkpoint_dir,
                            f'step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt'  # noqa: E501
                        )  # noqa: E501
                    )

                # Prepare checkpoint dictionary
                checkpoint_dict = {
                    'model': model_state_dicts,
                    'global_step': global_step,
                    'epoch': epoch,
                }

                # Save scheduler state
                if self.lr_scheduler is not None:
                    checkpoint_dict[
                        'scheduler_state_dict'] = self.lr_scheduler.state_dict(
                        )

                # Save full optimizer state dict (only on rank 0)
                if full_optimizer_state_dict is not None:
                    checkpoint_dict[
                        'optimizer_state_dict'] = full_optimizer_state_dict

                # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
                torch.save(checkpoint_dict, checkpoint_path)

                # Save model weights as safetensors for fast loading
                safetensors_path = checkpoint_path.replace(
                    '.pt', '.safetensors')
                self._save_model_safetensors(model_state_dicts,
                                             safetensors_path)
                overwatch.info(f'Saved safetensors at: {safetensors_path}')

                # Create symlink to latest checkpoint
                latest_ckpt_link = os.path.join(checkpoint_dir,
                                                'latest-checkpoint.pt')
                if os.path.islink(latest_ckpt_link) or os.path.exists(
                        latest_ckpt_link):
                    os.remove(latest_ckpt_link)
                os.symlink(os.path.abspath(checkpoint_path), latest_ckpt_link)

                latest_sf_link = os.path.join(checkpoint_dir,
                                              'latest-checkpoint.safetensors')
                if os.path.islink(latest_sf_link) or os.path.exists(
                        latest_sf_link):
                    os.remove(latest_sf_link)
                os.symlink(os.path.abspath(safetensors_path), latest_sf_link)

                self._cleanup_old_checkpoints(checkpoint_dir)

    def run_setup(self, n_train_examples: int) -> None:
        self.vla.from_pretrained()
        # Iteratively Assemble FSDP Wrapping Policy by fetching the wrapping
        # policies for each backbone/constituent
        torch.cuda.set_device(device_id := self.device_id)  # noqa: F841
        torch.cuda.empty_cache()
        vla_fsdp_wrapping_policy = self.vla.get_fsdp_wrapping_policy()

        # Assemble the Default FSDP Mixed Precision Policy
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:  # noqa: E501
            # MixedPrecision `param_dtype` specifies *compute*
            # dtype (for forward/backward only)
            #  => Reference: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision  # noqa: E501
            reduce_buffer_dtype = torch.bfloat16 if not \
                self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_buffer_dtype,
                buffer_dtype=reduce_buffer_dtype)
        else:
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32)

        self.vla.freeze_backbones()
        self.trainable_module_keys = self.vla.trainable_module_keys

        # Unify parameter dtypes before FSDP wrapping
        # FSDP requires all parameters in the same FSDP unit to float32
        for name, param in self.vla.named_parameters():
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
        overwatch.info(
            'Unified all model parameters to torch.float32', ctx_level=1)

        # Collect checkpoint layer classes BEFORE FSDP wrapping
        checkpoint_layer_classes = set()
        vlm_has_hf_checkpointing = False
        if self.enable_gradient_checkpointing:
            # Add LLM backbone transformer layers
            if hasattr(self, 'llm_transformer_layer_cls'):
                checkpoint_layer_classes.add(self.llm_transformer_layer_cls)

            # Add Vision Transformer blocks (for timm models)
            try:
                from timm.models.vision_transformer import Block as VisionBlock
                checkpoint_layer_classes.add(VisionBlock)
            except ImportError:
                pass

            # Check if VLM backbone has HuggingFace gradient checkpointing
            # We will enable it AFTER FSDP wrapping
            if hasattr(self.vla,
                       'vlm_backbone') and self.vla.vlm_backbone is not None:
                if hasattr(self.vla.vlm_backbone,
                           'enable_gradient_checkpointing'):
                    vlm_has_hf_checkpointing = True
                elif hasattr(self.vla.vlm_backbone, 'transformer_layer_cls'):
                    # Fallback: use PyTorch's apply_activation_checkpointing
                    checkpoint_layer_classes.add(
                        self.vla.vlm_backbone.transformer_layer_cls)

            # Add LLM expert layers
            if hasattr(self.vla,
                       'llm_expert') and self.vla.llm_expert is not None:
                if hasattr(self.vla.llm_expert, 'transformer_layer_cls'):
                    checkpoint_layer_classes.add(
                        self.vla.llm_expert.transformer_layer_cls)

        # <FSDP> => note that FSDP will automatically take care of
        # device placement (similar to `autocast`)
        self.vla = FSDP(
            self.vla,
            auto_wrap_policy=vla_fsdp_wrapping_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )

        # Apply Gradient Checkpointing AFTER FSDP wrapping
        if self.enable_gradient_checkpointing:
            # Enable HuggingFace gradient checkpointing for VLM backbone
            # This must be done AFTER FSDP wrapping
            if vlm_has_hf_checkpointing:
                # Find the vlm_backbone module within FSDP-wrapped model
                # Check for vlm_backbone attribute first (more specific)
                for name, module in self.vla.named_modules():
                    if 'vlm_backbone' in name and hasattr(
                            module, 'enable_gradient_checkpointing'):
                        module.enable_gradient_checkpointing()
                        overwatch.info(
                            f'VLM backbone ({name}) uses HuggingFace gradient '
                            'checkpointing (enabled after FSDP)',
                            ctx_level=1)
                        break

            # Apply PyTorch checkpoint wrapper for non-HF layers
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

        # Barrier =>> Sharding takes a minute?
        dist.barrier()
        # Create Optimizer and LR Scheduler
        # Use base class method to setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(
            n_train_examples,
            weight_decay=self.weight_decay,
            lr_schedule=self.lr_schedule)

        # Calculate values for logging
        n_train_examples_rounded = math.ceil(
            n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples_rounded *
                                  self.max_epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        # Finalize Setup =>> Log!
        overwatch.info(
            'FSDP Full-Shard Strategy =>> Finalized Training Setup:\n'  # noqa: E501
            f'         |-> Global (Effective) Batch Size = {self.global_batch_size}\n'  # noqa: E501
            f'         |-> Per-Device Batch Size = {self.per_device_batch_size}\n'  # noqa: E501
            f'         |-> Distributed World Size = {overwatch.world_size()}\n'  # noqa: E501
            f'         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n'  # noqa: E501
            f'         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n'  # noqa: E501
            f'         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n'  # noqa: E501
            f'                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n'  # noqa: E501
            f'                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n'  # noqa: E501
            f'                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n'  # noqa: E501
            f'         |-> Default AdamW LR = {self.learning_rate}\n'  # noqa: E501
            f'         |-> AdamW Weight Decay = {self.weight_decay}\n'  # noqa: E501
            f'         |-> LR Scheduler Type = {self.lr_scheduler_type}\n'  # noqa: E501
            f'         |-> LR Scheduler Warm-up Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n'  # noqa: E501
            f'         |-> Dataset Size = {n_train_examples} Examples\n'  # noqa: E501
            f'         |-> Max Steps = {num_training_steps}\n\n'  # noqa: E501
        )

    def clip_grad_norm(self) -> None:
        # Note =>> FSDP uses a custom `clip_grad_norm_` function; requires *uniform grad dtype*  # noqa: E501
        self.vla.clip_grad_norm_(max_norm=self.max_grad_norm)

    def _load_model_state(self, checkpoint_model_state: dict) -> None:
        """Load FSDP model state from checkpoint.

        Args:
            checkpoint_model_state (dict): Model state dict from checkpoint.
        """
        if overwatch.is_rank_zero():
            overwatch.info('Loading FSDP model state')

        # Synchronize all ranks before loading model state
        dist.barrier()

        # Load model state dict using FSDP state_dict_type
        with FSDP.state_dict_type(self.vla, self.fsdp_state_dict_type,
                                  self.fsdp_save_policy):
            # Handle both dict format (when change_key_name is True) and
            # direct state_dict format
            if self.change_key_name and isinstance(checkpoint_model_state,
                                                   dict):
                # Reconstruct full state dict from module keys
                full_state_dict = OrderedDict()
                for mkey, mstate_dict in checkpoint_model_state.items():
                    for key, param in mstate_dict.items():
                        full_state_dict[f'{mkey}.{key}'] = param
                checkpoint_model_state = full_state_dict

            # Load the state dict
            self.vla.load_state_dict(checkpoint_model_state, strict=False)

        # Synchronize after loading
        dist.barrier()

        if overwatch.is_rank_zero():
            overwatch.info('FSDP model state restored from checkpoint')

    def _load_optimizer_state(self, checkpoint_optimizer_state: dict) -> bool:
        """Load FSDP optimizer state from checkpoint.

        Args:
            checkpoint_optimizer_state (dict): Full optimizer state dict from
                checkpoint.

        Returns:
            bool: True if optimizer state was successfully loaded,
                False otherwise.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if overwatch.is_rank_zero():
            overwatch.info('Loading FSDP optimizer state')

        # Synchronize all ranks before loading optimizer state
        # This ensures all ranks are at the same point
        dist.barrier()

        # Load full optimizer state dict on rank 0, then shard it
        full_osd = checkpoint_optimizer_state

        # Use the new API if available, otherwise fall back to deprecated API
        try:
            # New API: optim_state_dict_to_load
            sharded_osd = FSDP.optim_state_dict_to_load(
                full_osd, self.vla, self.optimizer)
        except (AttributeError, TypeError):
            # Fall back to deprecated API for older PyTorch versions
            sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, self.vla)

        # Load the sharded optimizer state dict
        # All ranks must load the state to keep them synchronized
        self.optimizer.load_state_dict(sharded_osd)
        self.optimizer_state_loaded = True

        # Synchronize after loading to ensure all ranks have loaded
        dist.barrier()

        if overwatch.is_rank_zero():
            overwatch.info('FSDP optimizer state restored from checkpoint')

        return True
