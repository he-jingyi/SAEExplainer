"""Configuration helpers for hybrid SAE-explainer SFT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(raw_path: str | None, project_root: Path) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    trust_remote_code: bool
    torch_dtype: str
    attn_implementation: str | None
    use_cache: bool
    gradient_checkpointing: bool


@dataclass(frozen=True)
class InjectionConfig:
    placeholder_text: str
    inject_layer: int
    steering_coefficient: float
    use_projector: bool
    vector_input_dim: int | None


@dataclass(frozen=True)
class LoRAConfig:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target_modules: str
    bias: str


@dataclass(frozen=True)
class DataConfig:
    train_jsonl: Path
    val_jsonl: Path | None
    max_length: int
    val_ratio: float
    seed: int
    vector_store_cache_dir: Path
    require_decoder_vector_path: bool


@dataclass(frozen=True)
class TrainerConfig:
    output_dir: Path
    overwrite_output_dir: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    save_total_limit: int
    keep_only_best_checkpoint: bool
    load_best_model_at_end: bool
    metric_for_best_model: str | None
    greater_is_better: bool | None
    early_stopping_patience: int | None
    bf16: bool
    fp16: bool
    max_grad_norm: float
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    report_to: str | None
    seed: int


@dataclass(frozen=True)
class EvalConfig:
    enabled: bool
    use_full_validation_set: bool
    max_new_tokens: int
    num_examples: int
    sample_randomly: bool
    sample_seed: int
    output_jsonl: Path | None


@dataclass(frozen=True)
class SFTConfig:
    model: ModelConfig
    injection: InjectionConfig
    lora: LoRAConfig
    data: DataConfig
    trainer: TrainerConfig
    eval: EvalConfig


def load_sft_config(config_path: Path, project_root: Path) -> SFTConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    model_raw = raw["model"]
    injection_raw = raw["injection"]
    lora_raw = raw["lora"]
    data_raw = raw["data"]
    trainer_raw = raw["trainer"]
    eval_raw = raw.get("eval", {})

    model = ModelConfig(
        model_name=str(model_raw["model_name"]),
        trust_remote_code=bool(model_raw.get("trust_remote_code", True)),
        torch_dtype=str(model_raw.get("torch_dtype", "bfloat16")),
        attn_implementation=model_raw.get("attn_implementation"),
        use_cache=bool(model_raw.get("use_cache", False)),
        gradient_checkpointing=bool(model_raw.get("gradient_checkpointing", True)),
    )

    injection = InjectionConfig(
        placeholder_text=str(injection_raw["placeholder_text"]),
        inject_layer=int(injection_raw.get("inject_layer", 1)),
        steering_coefficient=float(injection_raw.get("steering_coefficient", 1.0)),
        use_projector=bool(injection_raw.get("use_projector", False)),
        vector_input_dim=(
            int(injection_raw["vector_input_dim"])
            if injection_raw.get("vector_input_dim") is not None
            else None
        ),
    )

    lora = LoRAConfig(
        enabled=bool(lora_raw.get("enabled", True)),
        r=int(lora_raw.get("r", 32)),
        alpha=int(lora_raw.get("alpha", 64)),
        dropout=float(lora_raw.get("dropout", 0.05)),
        target_modules=str(lora_raw.get("target_modules", "all-linear")),
        bias=str(lora_raw.get("bias", "none")),
    )

    data = DataConfig(
        train_jsonl=_resolve_path(data_raw["train_jsonl"], project_root),
        val_jsonl=_resolve_path(data_raw.get("val_jsonl"), project_root),
        max_length=int(data_raw.get("max_length", 1536)),
        val_ratio=float(data_raw.get("val_ratio", 0.05)),
        seed=int(data_raw.get("seed", 42)),
        vector_store_cache_dir=_resolve_path(
            data_raw.get("vector_store_cache_dir", "data/cache/hf_saes"),
            project_root,
        ),
        require_decoder_vector_path=bool(data_raw.get("require_decoder_vector_path", False)),
    )

    trainer = TrainerConfig(
        output_dir=_resolve_path(trainer_raw["output_dir"], project_root),
        overwrite_output_dir=bool(trainer_raw.get("overwrite_output_dir", False)),
        per_device_train_batch_size=int(trainer_raw.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(trainer_raw.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(trainer_raw.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(trainer_raw.get("num_train_epochs", 2.0)),
        learning_rate=float(trainer_raw.get("learning_rate", 5e-5)),
        weight_decay=float(trainer_raw.get("weight_decay", 0.01)),
        warmup_ratio=float(trainer_raw.get("warmup_ratio", 0.03)),
        logging_steps=int(trainer_raw.get("logging_steps", 10)),
        eval_strategy=str(trainer_raw.get("eval_strategy", "steps")),
        eval_steps=int(trainer_raw.get("eval_steps", 100)),
        save_strategy=str(trainer_raw.get("save_strategy", "steps")),
        save_steps=int(trainer_raw.get("save_steps", 100)),
        save_total_limit=int(trainer_raw.get("save_total_limit", 2)),
        keep_only_best_checkpoint=bool(trainer_raw.get("keep_only_best_checkpoint", False)),
        load_best_model_at_end=bool(trainer_raw.get("load_best_model_at_end", False)),
        metric_for_best_model=trainer_raw.get("metric_for_best_model"),
        greater_is_better=trainer_raw.get("greater_is_better"),
        early_stopping_patience=(
            int(trainer_raw["early_stopping_patience"])
            if trainer_raw.get("early_stopping_patience") is not None
            else None
        ),
        bf16=bool(trainer_raw.get("bf16", True)),
        fp16=bool(trainer_raw.get("fp16", False)),
        max_grad_norm=float(trainer_raw.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(trainer_raw.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(trainer_raw.get("dataloader_pin_memory", False)),
        report_to=trainer_raw.get("report_to"),
        seed=int(trainer_raw.get("seed", data.seed)),
    )

    eval_config = EvalConfig(
        enabled=bool(eval_raw.get("enabled", True)),
        use_full_validation_set=bool(eval_raw.get("use_full_validation_set", True)),
        max_new_tokens=int(eval_raw.get("max_new_tokens", 64)),
        num_examples=int(eval_raw.get("num_examples", 8)),
        sample_randomly=bool(eval_raw.get("sample_randomly", True)),
        sample_seed=int(eval_raw.get("sample_seed", data.seed)),
        output_jsonl=_resolve_path(eval_raw.get("output_jsonl"), project_root),
    )

    return SFTConfig(
        model=model,
        injection=injection,
        lora=lora,
        data=data,
        trainer=trainer,
        eval=eval_config,
    )
