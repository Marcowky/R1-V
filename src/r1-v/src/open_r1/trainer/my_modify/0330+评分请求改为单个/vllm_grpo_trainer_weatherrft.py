# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import math
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object
from datetime import datetime

import torch
import torch.utils.data
import transformers
import warnings
from unittest.mock import patch
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.import_utils import is_vllm_available

from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl import GRPOTrainer

import copy

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb
import torch.nn as nn
from torch.utils.data import Sampler

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOVLLMTrainerWeatherRFT(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # qwen2-vl related params
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                model_init_kwargs["torch_dtype"] = torch.bfloat16 # 手动设置为 bfloat16
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_init_kwargs
                )
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.use_vllm = args.use_vllm

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == "cuda"
                    and int(vllm_device.split(":")[1]) >= torch.cuda.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {
                    f"cuda:{idx}" for idx in range(self.accelerator.num_processes)
                }:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                        "behavior. It is recommended to use a dedicated device for vLLM."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1
                )
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )
                with world_size_patch, profiling_patch:
                    print("vllm is running on: ", vllm_device)
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=torch.bfloat16,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        enforce_eager=True
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            raise ValueError(
                "GRPOVLLMTrainerModified only supports vllm generation, please set --use_vllm True"
            )

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        logits_to_keep,
    ):
        pixel_values = pixel_values.to(model.device)
        image_grid_thw = image_grid_thw.to(device=model.device)
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits  # (B, L, V)
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[
            :, -logits_to_keep:
        ]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        logits = logits[:, -logits_to_keep:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def cosine_lr_schedule(self, global_step, max_steps, base_lr=1, warmup_ratio=0.1, min_lr=0):
        """
        计算当前步骤的学习率，支持 warmup 和余弦衰减。

        参数:
            global_step (int): 当前步骤。
            max_steps (int): 总训练步骤。
            warmup_ratio (int): warmup 的比例。
            base_lr (float): 初始学习率。
            min_lr (float): 最小学习率。

        返回:
            float: 当前步骤的学习率。
        """
        
        # 计算 warmup_steps
        warmup_steps = max_steps * warmup_ratio
        if global_step < warmup_steps:
            # Warmup 阶段：线性增加学习率
            lr = base_lr * (global_step / warmup_steps)
        elif global_step >= max_steps:
            lr = min_lr
        else:
            # Cosine 衰减阶段
            progress = (global_step - warmup_steps) / (max_steps - warmup_steps)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        
        return lr


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        images = [x["image"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)
        
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=False,  # TODO: fix this, self.args.ds3_gather_for_generation,
                ) as unwrapped_model:
                    if is_compiled_module(unwrapped_model):
                        state_dict = unwrapped_model._orig_mod.state_dict()
                    else:
                        state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = (
                        self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    )
                    llm_model.load_weights(state_dict.items())
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            all_images = gather_object(images)
            # group into pairs
            all_multimodal_inputs = []

            use_naive_loop_sampling = False
            if use_naive_loop_sampling:
                # in this implementation, one sample will repeat `self.num_generations` times
                # it's not a efficient implementation, but safe to keep sampling diversity
                for prompt, image in zip(all_prompts_text, all_images):
                    for _ in range(self.num_generations):
                        all_multimodal_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                all_completion_ids = [None] * len(all_multimodal_inputs)
                for i in range(self.num_generations):
                    # Get the inputs for the current batch
                    batch_inputs = [all_multimodal_inputs[j] for j in range(i, len(all_multimodal_inputs), self.num_generations)]
                    if self.accelerator.is_main_process:
                        outputs = self.llm.generate(
                            batch_inputs,
                            sampling_params=self.sampling_params,
                            use_tqdm=False,
                        )
                        batch_completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                    else:
                        batch_completion_ids = [None] * len(batch_inputs)
                    # Place the results back into their original positions
                    for idx, completion_id in enumerate(batch_completion_ids):
                        all_completion_ids[i + idx * self.num_generations] = completion_id
                # Final completion IDs
                completion_ids = all_completion_ids

            # 2. Refer to TobiasLee's implementation suggestions
            # this is a better implementation for vLLM sampling.
            for prompt, image in zip(all_prompts_text, all_images):
                all_multimodal_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
            # Create sampling params with num_generations
            if self.accelerator.is_main_process:
                # Clone to avoid modifying original params
                sampling_params = copy.deepcopy(self.sampling_params)
                sampling_params.n = self.num_generations
                # Single generate call with all prompts
                if self.accelerator.is_main_process:
                    outputs = self.llm.generate(
                        all_multimodal_inputs,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                # Flatten outputs: [prompt1_gen1, prompt1_gen2, ..., prompt2_gen1, prompt2_gen2, ...]
                completion_ids = [out.token_ids for completion in outputs for out in completion.outputs]
            else:
                completion_ids = [None] * len(all_multimodal_inputs) * self.num_generations
            
            # broadcast and slice
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts) * self.num_generations,
                (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )
            prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
        else:
            raise ValueError("Only vLLM generation is supported in this version ")

        # below are the same with yifan's code
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        pixel_values = prompt_inputs["pixel_values"][None].repeat_interleave(self.num_generations, dim=0)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                        logits_to_keep,
                    )

        # Decode the generated completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]


        # 不同 reward 的分数比例
        # max_lr_steps = self.state.max_steps * 1
        # cur_lr = self.cosine_lr_schedule(global_step=self.state.global_step, max_steps=max_lr_steps)

        # reward_ratios = {
        #     "accuracy_reward": 10,
        #     "format_reward": 4,
        #     "length_reward": 2,
        #     "related_reward": 1,
        #     "fluency_logical_reward": 1
        # }
        reward_ratios = {
            "accuracy_reward": 10,
            "format_reward": 4,
            "related_reward": 1,
            "fluency_reward": 1,
            "think_depth_reward": 2,
        }

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [
                        apply_chat_template(x, reward_processing_class)["text"]
                        for x in messages
                    ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {
                    key: []
                    for key in inputs[0].keys()
                    if key not in ["prompt", "completion"]
                }
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs
                )

                # 对不同 reward 乘上不同的比例
                cur_ratio = reward_ratios[reward_func.__name__] if reward_func.__name__ in reward_ratios else 1
                output_reward_func = [x * cur_ratio for x in output_reward_func]

                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        # 计算 accuracy_reward 为 0 的奖励位置
        zero_accuracy_mask = torch.ones_like(rewards_per_func[:, 0], dtype=torch.bool)
        for i, reward_func in enumerate(self.reward_funcs):
            if reward_func.__name__ == 'accuracy_reward':
                # 获取 rewards_per_func[:, i] 为 0 的索引
                zero_accuracy_mask = rewards_per_func[:, i] != 0
                break

        # # 计算 formate reward 为 0 的奖励位置
        # zero_format_mask = torch.ones_like(rewards_per_func[:, 0], dtype=torch.bool)
        # for i, reward_func in enumerate(self.reward_funcs):
        #     if reward_func.__name__ == 'format_reward':
        #         # 获取 rewards_per_func[:, i] 为 0 的索引
        #         zero_format_mask = rewards_per_func[:, i] != 0
        #         break

        # 如果 accuracy_reward 为 0，将 length_reward, related_reward, fluency_logical_reward 也置为 0  
        for i, reward_func in enumerate(self.reward_funcs):
            if reward_func.__name__ not in ['accuracy_reward', 'format_reward']:
                # rewards_per_func[:, i] = rewards_per_func[:, i] * zero_accuracy_mask * zero_format_mask
                # rewards_per_func[:, i] = rewards_per_func[:, i] * zero_accuracy_mask
                for ii, mask in enumerate(zero_accuracy_mask):
                    if mask == 0:
                        rewards_per_func[ii][i] = rewards_per_func[ii][i] - reward_ratios[reward_func.__name__]

        # Log the completions and rewards
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a", encoding="utf-8") as f:
                for i, (completion, prompt) in enumerate(zip(completions, prompts)):
                    f.write(f"---------------------------{current_time} {self.state.global_step}/{self.state.max_steps}---------------------------\n")
                    f.write(f"[Info]: {reward_kwargs['id'][i]}, {reward_kwargs['category'][i]}\n")
                    f.write(f"[Prompt]: \n{prompt[0]['content'][1]['text']}\n")
                    f.write(f"[Content]: \n{completion[0]['content']}\n")
                    f.write(f"[Solution]: {reward_kwargs['solution'][i]}\n")
                    for reward_func, reward in zip(self.reward_funcs, rewards_per_func[i]):
                        f.write(f"[{reward_func.__name__}]: {reward}\n")
                    f.write("\n")

        rewards_per_func = gather(rewards_per_func)
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)


        ##### 去除最大最小 #####
        # 将奖励按提示分组
        rewards_grouped = rewards.view(-1, self.num_generations)
        # 获取最大最小值的索引
        sorted_indices = torch.argsort(rewards_grouped, dim=1)
        # 保留不是最大最小的
        mask = torch.ones_like(rewards_grouped, dtype=torch.bool) 
        # 计算剩余的提示数量
        rest_num_generations = self.num_generations

        # 如果生成次数大于 4，去掉最大最小值
        if self.num_generations >= 100: # 设置阈值
            min_indices = sorted_indices[:, 0]
            max_indices = sorted_indices[:, -1]
            mask.scatter_(1, min_indices.unsqueeze(1), 0)
            mask.scatter_(1, max_indices.unsqueeze(1), 0)
            rest_num_generations -= 2 # 去掉最大最小值后的生成次数

        # 将 mask 扩展，使其与原始奖励一一对应
        rest_mask = mask.view(-1)
        # 获取剩余的奖励
        rest_rewards = rewards[rest_mask]

        # 计算剩余分组的奖励统计量
        mean_grouped_rewards = rest_rewards.view(-1, rest_num_generations).mean(dim=1)  # 计算每个提示下所有生成结果的平均奖励
        std_grouped_rewards = rest_rewards.view(-1, rest_num_generations).std(dim=1)    # 计算每个提示下所有生成结果的标准差

        # 将平均奖励和标准差扩展，使其与每个生成结果一一对应
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            rest_num_generations, dim=0
        )  # 对每组的平均奖励进行重复，使其维度与原始奖励一致
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            rest_num_generations, dim=0
        )  # 对每组的标准差进行重复，使其维度与原始奖励一致

        # 计算优势函数：(剩余奖励 - 平均剩余奖励) / (剩余标准差 + 一个小常数防止除零)
        rest_advantages = (rest_rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # 填充优势值
        advantages = torch.zeros_like(rest_mask, dtype=torch.float)
        advantages[rest_mask] = rest_advantages

        # 只保留当前进程负责的数据部分
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )  # 根据当前进程索引计算需要处理的数据切片
        advantages = advantages[process_slice]  # 获取当前进程负责的优势值
        rest_mask = rest_mask[process_slice]


        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["rest_reward"].append(rest_rewards.mean().item())
        self._metrics["rest_reward_std"].append(std_grouped_rewards.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "rest_mask": rest_mask
        }

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        
        # 获取额外参数
        rest_mask = inputs["rest_mask"]
        advantages = inputs["advantages"]
        ref_per_token_logps = inputs["ref_per_token_logps"]

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            logits_to_keep,
        )
        
        # Compute the KL divergence between the model and the reference model
        
        # 添加 mask，过滤掉最大最小值
        per_token_logps = per_token_logps[rest_mask]
        ref_per_token_logps = ref_per_token_logps[rest_mask]
        advantages = advantages[rest_mask]
        completion_mask = completion_mask[rest_mask]

        per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps)- (ref_per_token_logps - per_token_logps)- 1)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) # x - x.detach() allows for preserving gradients from x
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # Log the metrics
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss

        
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()