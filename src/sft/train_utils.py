"""Model/tokenizer/trainer helpers for hybrid explainer SFT."""

from __future__ import annotations

import inspect
import random
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.sft.config import LoRAConfig, ModelConfig, TrainerConfig


TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(raw_dtype: str) -> torch.dtype:
    normalized = raw_dtype.lower()
    if normalized not in TORCH_DTYPE_MAP:
        raise ValueError(f"Unsupported torch dtype string: {raw_dtype}")
    return TORCH_DTYPE_MAP[normalized]


def load_tokenizer(model_config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_config: ModelConfig) -> AutoModelForCausalLM:
    dtype = resolve_torch_dtype(model_config.torch_dtype)
    model_name_normalized = model_config.model_name.lower()
    kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_config.model_name,
        "trust_remote_code": model_config.trust_remote_code,
        "dtype": dtype,
    }
    # Gemma 3 checkpoints currently instantiate a ConditionalGeneration class
    # whose constructor rejects use_cache as a from_pretrained kwarg. Keep the
    # old path for existing models and set use_cache on config post-load for
    # Gemma 3 only.
    if "gemma-3" not in model_name_normalized and "gemma3" not in model_name_normalized:
        kwargs["use_cache"] = model_config.use_cache
    if model_config.attn_implementation:
        kwargs["attn_implementation"] = model_config.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    model.config.use_cache = model_config.use_cache
    if getattr(model.config, "loss_type", None) in (None, "None"):
        model.config.loss_type = "ForCausalLM"

    if model_config.gradient_checkpointing:
        # The non-reentrant checkpoint path is more tolerant of dynamic forward
        # hooks than the legacy reentrant implementation and is the
        # recommended mode in modern PyTorch.
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    return model


def maybe_apply_lora(model: AutoModelForCausalLM, lora_config: LoRAConfig):
    if not lora_config.enabled:
        return model

    model_name = getattr(model.config, "_name_or_path", "").lower()
    fan_in_fan_out = "gpt2" in model_name

    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type="CAUSAL_LM",
        fan_in_fan_out=fan_in_fan_out,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def build_training_arguments(cfg: TrainerConfig) -> TrainingArguments:
    report_to = [] if cfg.report_to in (None, "none", "None") else [cfg.report_to]

    kwargs = dict(
        output_dir=str(cfg.output_dir),
        overwrite_output_dir=cfg.overwrite_output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        save_safetensors=False,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        max_grad_norm=cfg.max_grad_norm,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        remove_unused_columns=False,
        report_to=report_to,
        seed=cfg.seed,
        gradient_checkpointing=False,
    )

    signature = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = cfg.eval_strategy
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = cfg.eval_strategy

    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }

    return TrainingArguments(**filtered_kwargs)
