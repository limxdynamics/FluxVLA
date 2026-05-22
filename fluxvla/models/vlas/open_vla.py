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

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from fluxvla.engines import (VLAS, build_tokenizer_from_cfg,
                             initialize_overwatch)
from .base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class OpenVLA(BaseVLA):
    """
    Implementation of `https://arxiv.org/abs/2406.09246`

    Args:
        model_family (str): Name of the model family.
        model_id (str): ID of the mode.
        vision_backbone (Dict): Config of the vision backbone.
        llm_backbone (Dict): Config of the llm backbone.
        projector (Dict): Config of the projector.
        vla_head (Dict): Config of the vla head.
        enable_mixed_precision_training (bool): Whether to use mixed precision
            for training. Defaults to True.
        freeze_vision_backbone (bool): Whether to freeze the weight of vision
            backbone. Defaults to False.
        freeze_llm_backbone (bool): Whether to freeze the weight of llm
            backbone. Defaults to False.
        freeze_projector (bool): Whether to freeze the weight of projector.
            defaults to False.
        freeze_vlm_backbone (bool): Whether to freeze the weight of vlm
            backbone. Defaults to False.
        vision_backbone_fp32 (bool): Whether to use fp32 to train vision
            backbone. Defaults to False.
        unfreeze_last_layer (bool): Whether to unfreeze the last layer.
            Defaults to False.
        ignore_index (int): The index to ignore. Defaults to -100.
        pretrained_name_or_path (Path): the path to the pretrained ckpt.
            Defaults to None.
        freeze_weights: (bool): Whether to freeze the weights of the model.
            Defaults to False.
        norm_stats (Dict): Normalization statistics for the model.
            Defaults to None.
        name_mapping (Dict): Mapping of model names for loading
            pretrained weights.

    """

    def __init__(self,
                 vla_head: Dict,
                 vision_backbone: Dict = None,
                 llm_backbone: Dict = None,
                 projector: Dict = None,
                 vlm_backbone: Dict = None,
                 tokenizer: Dict = None,
                 enable_mixed_precision_training: bool = True,
                 freeze_vision_backbone: bool = True,
                 freeze_llm_backbone: bool = True,
                 freeze_projector: bool = False,
                 freeze_vlm_backbone: bool = False,
                 vision_backbone_fp32: bool = False,
                 unfreeze_last_layer: bool = False,
                 ignore_index: int = -100,
                 freeze_weights: bool = False,
                 norm_stats: Dict = None,
                 pretrained_name_or_path: Path = None,
                 name_mapping: Dict = None,
                 strict_mapping: bool = False,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            vlm_backbone=vlm_backbone,
            projector=projector,
            vla_head=vla_head,
            enable_mixed_precision_training=enable_mixed_precision_training,
            freeze_vision_backbone=freeze_vision_backbone,
            freeze_llm_backbone=freeze_llm_backbone,
            freeze_projector=freeze_projector,
            freeze_vlm_backbone=freeze_vlm_backbone,
            vision_backbone_fp32=vision_backbone_fp32,
            unfreeze_last_layer=unfreeze_last_layer,
            ignore_index=ignore_index,
            norm_stats=norm_stats,
            pretrained_name_or_path=pretrained_name_or_path,
            name_mapping=name_mapping,
            strict_mapping=strict_mapping)
        self.all_module_keys = [
            'vision_backbone', 'llm_backbone', 'projector', 'head'
        ]
        if tokenizer is not None:
            # Build Tokenizer from Config
            self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.trainable_module_keys = []
        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens
        # corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        self.vision_backbone_requires_grad = not self.freeze_vision_backbone

    def forward_model(self,
                      input_ids: Optional[torch.LongTensor] = None,
                      attention_mask: Optional[torch.Tensor] = None,
                      pixel_values: Optional[torch.FloatTensor] = None,
                      labels: Optional[torch.LongTensor] = None,
                      inputs_embeds: Optional[torch.FloatTensor] = None,
                      past_key_values: Optional[List[
                          torch.FloatTensor]] = None,
                      use_cache: Optional[bool] = None,
                      output_attentions: Optional[bool] = None,
                      output_hidden_states: Optional[bool] = None,
                      return_dict: Optional[bool] = None,
                      multimodal_indices: Optional[torch.LongTensor] = None,
                      *args,
                      **kwargs) -> CausalLMOutputWithPast:
        """
        Forward pass of the OpenVLA model. Combines visual features and
        language embeddings into a fused sequence, processed by the LLM.

        Supports:
            - Autoregressive decoding with cached past key values
            - Fully multimodal batches (image + text)
            - Unimodal fallback (text only)

        Args:
            input_ids (LongTensor): Input token IDs [B, T].
            attention_mask (Tensor): Mask for input tokens [B, T].
            pixel_values (FloatTensor): Image tensor or dict for vision model.
            labels (LongTensor): Language modeling target tokens [B, T].
            inputs_embeds (FloatTensor): Optional precomputed input embeddings.
            past_key_values (List[FloatTensor]): LLM cache for fast decoding.
            use_cache (bool): Whether to return cache for next step.
            output_attentions (bool): Whether to return attention maps.
            output_hidden_states (bool): Whether to return hidden states.
            return_dict (bool): Whether to return a CausalLMOutputWithPast.
            multimodal_indices (LongTensor): Indices of samples using image +
                text.

        Returns:
            CausalLMOutputWithPast: Outputs including logits and optional
                cache.
        """
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cacdhe, so just redirect to
            # `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError('Invalid `forward()` call!')

        # Handle Multimodal Indices is None --> pretend like the batch is fully
        # multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(
                len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple
        # unimodal forward
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({
                    k: pixel_values[k][multimodal_indices]
                    for k in pixel_values
                })
            else:
                patch_features = self.vision_backbone(
                    pixel_values[multimodal_indices])

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>>
        # num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0],
                 projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # === Step 1: Get Input Embeddings from LLM ===
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

        # === Step 2: Build Multimodal Embeddings & Attention Mask ===
        multimodal_embeddings = torch.cat([
            input_embeddings[multimodal_indices, :1, :],
            projected_patch_embeddings, input_embeddings[multimodal_indices,
                                                         1:, :]
        ],
                                          dim=1)

        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat([
                attention_mask[multimodal_indices, :1],
                projected_patch_attention_mask,
                attention_mask[multimodal_indices, 1:]
            ],
                                                  dim=1)

        # === Step 3: Build Multimodal Labels (Ignore patch embeddings) ===
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0],
                 projected_patch_embeddings.shape[1]),
                self.ignore_index,
                dtype=labels.dtype,
                device=labels.device)
            multimodal_labels = torch.cat([
                labels[multimodal_indices, :1], projected_patch_labels,
                labels[multimodal_indices, 1:]
            ],
                                          dim=1)

        # === Step 4: Handle Unimodal Cases ===
        unimodal_indices = torch.tensor([
            idx
            for idx in range(len(input_ids)) if idx not in multimodal_indices
        ],
                                        dtype=torch.long,
                                        device=multimodal_indices.device)

        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels
        else:
            patch_len = projected_patch_embeddings.shape[1]
            embed_dim = input_embeddings.shape[2]

            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), patch_len, embed_dim),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device)
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), patch_len),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device)
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), patch_len),
                self.ignore_index,
                dtype=labels.dtype,
                device=labels.device)

            unimodal_embeddings = torch.cat(
                [input_embeddings[unimodal_indices], unimodal_embeddings_pad],
                dim=1)

            unimodal_attention_mask = torch.cat(
                [attention_mask[unimodal_indices], unimodal_attention_pad],
                dim=1)

            unimodal_labels = torch.cat(
                [labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # === Step 5: Merge Multimodal and Unimodal ===
            fused_embeddings = torch.vstack(
                [multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack(
                [multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # === Step 6: Final LLM Forward Pass ===
        output = self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        return output, fused_attention_mask

    def forward(self,
                lang_tokens: Optional[torch.LongTensor] = None,
                lang_masks: Optional[torch.Tensor] = None,
                images: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                multimodal_indices: Optional[torch.LongTensor] = None,
                dataset_names: Optional[List[str]] = None,
                *args,
                **kwargs) -> Dict:
        """Forward pass for training the OpenVLA model.

        Args:
            input_ids (LongTensor): Input token IDs [B, T].
            lang_masks (Tensor): Mask for input tokens [B, T].
            pixel_values (FloatTensor): Image tensor or dict for vision model.
            labels (LongTensor): Language modeling target tokens [B, T].
            inputs_embeds (FloatTensor): Optional precomputed input embeddings.
            past_key_values (List[FloatTensor]): LLM cache for fast decoding.
            use_cache (bool): Whether to return cache for next step.
            output_attentions (bool): Whether to return attention maps.
            output_hidden_states (bool): Whether to return hidden states.
            return_dict (bool): Whether to return a CausalLMOutputWithPast.
            multimodal_indices (LongTensor): Indices of samples using image +
                text.
        Returns:
            Dict: A dictionary containing the predictions and loss.
        """
        output, _ = self.forward_model(
            input_ids=lang_tokens,
            attention_mask=lang_masks,
            pixel_values=images,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            multimodal_indices=multimodal_indices,
        )
        action_preds = output.logits[:, self.vision_backbone.
                                     num_patches:-1].argmax(dim=2)
        action_gt = labels[:, 1:].to(action_preds.device)
        mask = action_gt > self.tokenizer.action_token_begin_idx

        # Compute Accuracy
        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # Compute L1 Loss on Predicted (Continuous) Actions
        continuous_actions_pred = torch.tensor(
            self.tokenizer.decode_token_ids_to_actions(
                action_preds[mask].cpu().numpy()))
        continuous_actions_gt = torch.tensor(
            self.tokenizer.decode_token_ids_to_actions(
                action_gt[mask].cpu().numpy()))
        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred,
                                                     continuous_actions_gt)
        datasets = set(dataset_names)
        ds_name_list = list()
        action_accuracy_list = list()
        action_l1_loss_list = list()
        if len(datasets) > 1:
            for ds in datasets:
                ds_mask = torch.tensor([elem == ds for elem in dataset_names])
                action_accuracy_ds = correct_preds[ds_mask].sum(  # noqa: E501
                ).float() / mask[ds_mask].sum().float()
                continuous_actions_pred_ds = torch.tensor(
                    self.tokenizer.decode_token_ids_to_actions(
                        action_preds[ds_mask][mask[ds_mask]].cpu().numpy()))
                continuous_actions_gt_ds = torch.tensor(
                    self.tokenizer.decode_token_ids_to_actions(
                        action_gt[ds_mask][mask[ds_mask]].cpu().numpy()))
                action_l1_loss_ds = torch.nn.functional.l1_loss(  # noqa: E501
                    continuous_actions_pred_ds, continuous_actions_gt_ds)
                ds_name_list.append(ds.decode())
                action_accuracy_list.append(action_accuracy_ds)
                action_l1_loss_list.append(action_l1_loss_ds)
        return_dict = dict(
            predictions=output.logits,
            loss=output.loss if output.loss is not None else None,
            action_accuracy=action_accuracy,
            action_l1_loss=action_l1_loss,
            action_accuracy_ds=action_accuracy_list,
            action_l1_loss_ds=action_l1_loss_list,
            ds_names=ds_name_list,
        )
        return return_dict

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Returns the FSDP wrapping policy for the model.

        Returns:
            Callable: The wrapping policy for FSDP.
        """
        fsdp_policy_list = list()
        if hasattr(self, 'vision_backbone') and hasattr(
                self.vision_backbone, 'get_fsdp_wrapping_policy'):
            # Get Vision Backbone FSDP Wrapping Policy
            # =>> just a module wrapping policy around `self.vision_backbone`
            vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy(  # noqa: E501
            )
            fsdp_policy_list.append(vision_fsdp_wrapping_policy)
        if hasattr(self, 'llm_backbone') and hasattr(
                self.llm_backbone, 'get_fsdp_wrapping_policy'):
            # Get LLM Backbone FSDP Wrapping Policy
            # =>> just a module wrapping policy around `self.llm_backbone`
            llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy(  # noqa: E501
            )
            fsdp_policy_list.append(llm_fsdp_wrapping_policy)
        if hasattr(self, 'vlm_backbone') and hasattr(
                self.vlm_backbone, 'get_fsdp_wrapping_policy'):
            # Get VLM Backbone FSDP Wrapping Policy
            # =>> just a module wrapping policy around `self.vlm_backbone`
            vlm_fsdp_wrapping_policy = self.vlm_backbone.get_fsdp_wrapping_policy(  # noqa: E501
            )
            fsdp_policy_list.append(vlm_fsdp_wrapping_policy)
        if hasattr(self, 'vla_head') and hasattr(self.vla_head,
                                                 'get_fsdp_wrapping_policy'):
            fsdp_policy_list.append(self.vla_head.get_fsdp_wrapping_policy())
        from fluxvla.engines import PROJECTORS

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy
        # around `self.projector`
        projector_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes=set(PROJECTORS._module_dict.values()),
        )
        fsdp_policy_list.append(projector_fsdp_wrapping_policy)
        # Return union (_or_) over constituent policies
        # => Note: there is *not* a fall-through policy; any module that isn't
        # covered by the above constituents will automatically be folded into
        # the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=fsdp_policy_list,
        )

    @torch.no_grad()
    def generate(self,
                 lang_tokens: torch.LongTensor,
                 images: Optional[torch.FloatTensor] = None,
                 multimodal_indices: Optional[torch.LongTensor] = None,
                 max_new_tokens: int = 20,
                 eos_token_id: Optional[int] = None,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 *args,
                 **kwargs) -> torch.LongTensor:
        """
        Generate response tokens autoregressively.

        Args:
            lang_tokens (LongTensor): Initial token sequence, shape [B, T].
            images (FloatTensor): Optional image tensor [B, C, H, W].
            multimodal_indices (LongTensor): Indices in batch using vision input. # noqa: E501
            max_new_tokens (int): Number of tokens to generate.
            eos_token_id (int): Optional early stopping token.
            temperature (float): Sampling temperature.
            top_k (int): Optional top-k sampling.

        Returns:
            output_ids (LongTensor): Generated tokens (including input),
            shape [B, T+max_new_tokens].
        """
        # device = lang_tokens.device
        generated = lang_tokens
        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                # First forward: image+text input
                outputs, _ = self.forward_model(
                    input_ids=generated,
                    pixel_values=images,
                    multimodal_indices=multimodal_indices,
                    past_key_values=None,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                # Subsequent forward: only new token
                outputs = self.forward_model(
                    input_ids=generated[:, -1:],  # last token only
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            logits = outputs.logits[:, -1, :] / temperature  # [B, vocab]
            if top_k is not None:
                topk = torch.topk(logits, top_k)
                probs = torch.full_like(logits, float('-inf'))
                probs.scatter_(1, topk.indices, topk.values)
                next_token = torch.argmax(probs, dim=-1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            # Append to sequence
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            past_key_values = outputs.past_key_values

        return generated

    def predict_action(self,
                       lang_tokens: Optional[torch.LongTensor] = None,
                       lang_masks: Optional[torch.Tensor] = None,
                       images: Optional[torch.FloatTensor] = None,
                       img_masks: Optional[torch.Tensor] = None,
                       unnorm_key: Optional[str] = None,
                       **kwargs: str) -> torch.FloatTensor:
        """Predicts the action logits based on the input
        IDs and other parameters.

        Args:
            lang_tokens (torch.LongTensor): Input token IDs.
            lang_masks (torch.Tensor): Mask for input tokens.
            images (torch.FloatTensor): Image tensor for vision model.
            img_masks (torch.Tensor): Mask for images.
            unnorm_key (str, optional): Key for unnormalization.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.FloatTensor: The predicted action logits.
        """
        output = self.generate(
            lang_tokens,
            images,
            max_new_tokens=self.get_action_dim(unnorm_key) + 1,
            **kwargs)

        action_preds = output[:, -self.get_action_dim(unnorm_key) - 1:-1]

        # action_preds = action_preds[:, ~lang_masks.squeeze(0)[1:]]
        # action_preds = action_preds[:, :7]
        continuous_actions_pred = torch.tensor(
            self.tokenizer.decode_token_ids_to_actions(
                action_preds.cpu().numpy()))

        return continuous_actions_pred

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]],
                          unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f'Your model was trained on more than one dataset, '
                f'please pass a `unnorm_key` from the following options to'
                f'choose the statistics'
                f'used for un-normalizing actions: {norm_stats.keys()}')
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f'The `unnorm_key` you chose is not in'
            f'the set of available dataset statistics,'
            f'please choose from: {norm_stats.keys()}')
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]['action']['q01'])

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Prepares the inputs for generation by handling the input IDs,
        past key values, and pixel values. This method ensures that the
        inputs are correctly formatted for the model's forward pass during
        generation.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            past_key_values (List[torch.FloatTensor], optional): Cached key
                values for fast decoding.
            inputs_embeds (torch.FloatTensor, optional): Precomputed input
                embeddings.
            pixel_values (torch.FloatTensor, optional): Image tensor for
                vision model.
            attention_mask (torch.Tensor, optional): Mask for input tokens.
            **kwargs (str): Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the prepared
                inputs for the model's forward pass.
        """
        if ((input_ids is not None) and
            (input_ids.shape[0] > 1)) or ((inputs_embeds is not None) and
                                          (inputs_embeds.shape[0] > 1)):
            raise ValueError(
                'Generation with batch size > 1 is not currently supported!')

        # Handle `past_key_values` (cache) =>> assume
        # `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them
        # in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'input_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update({
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
        })

        return model_inputs
