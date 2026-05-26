"""Configuration helpers for vec-only DPO training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def _resolve_path(raw_path: str | None, project_root: Path) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


@dataclass(frozen=True)
class DPOModelConfig:
    sft_config_path: Path
    init_checkpoint_dir: Path
    reference_checkpoint_dir: Path | None
    reference_device: str | None


@dataclass(frozen=True)
class DPODataConfig:
    preference_jsonl: Path
    val_preference_jsonl: Path | None
    val_ratio: float
    seed: int
    use_full_validation_set: bool
    eval_num_examples: int
    eval_sample_randomly: bool
    eval_sample_seed: int


@dataclass(frozen=True)
class DPOLossConfig:
    beta: float
    average_log_prob: bool


@dataclass(frozen=True)
class DPOTrainerConfig:
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
    secondary_metric_for_best_model: str | None
    secondary_greater_is_better: bool | None
    best_model_tie_epsilon: float
    early_stopping_patience: int | None
    bf16: bool
    fp16: bool
    max_grad_norm: float
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    report_to: str | None
    seed: int


@dataclass(frozen=True)
class DPOEvalConfig:
    enabled: bool
    test_jsonl: Path | None
    output_jsonl: Path | None


@dataclass(frozen=True)
class DPOConfig:
    model: DPOModelConfig
    data: DPODataConfig
    loss: DPOLossConfig
    trainer: DPOTrainerConfig
    eval: DPOEvalConfig


def load_dpo_config(config_path: Path, project_root: Path) -> DPOConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    model_raw = raw["model"]
    data_raw = raw["data"]
    loss_raw = raw["loss"]
    trainer_raw = raw["trainer"]
    eval_raw = raw.get("eval", {})

    return DPOConfig(
        model=DPOModelConfig(
            sft_config_path=_resolve_path(model_raw["sft_config_path"], project_root),
            init_checkpoint_dir=_resolve_path(model_raw["init_checkpoint_dir"], project_root),
            reference_checkpoint_dir=_resolve_path(model_raw.get("reference_checkpoint_dir"), project_root),
            reference_device=model_raw.get("reference_device"),
        ),
        data=DPODataConfig(
            preference_jsonl=_resolve_path(data_raw["preference_jsonl"], project_root),
            val_preference_jsonl=_resolve_path(data_raw.get("val_preference_jsonl"), project_root),
            val_ratio=float(data_raw.get("val_ratio", 0.1)),
            seed=int(data_raw.get("seed", 42)),
            use_full_validation_set=bool(data_raw.get("use_full_validation_set", True)),
            eval_num_examples=int(data_raw.get("eval_num_examples", 50)),
            eval_sample_randomly=bool(data_raw.get("eval_sample_randomly", False)),
            eval_sample_seed=int(data_raw.get("eval_sample_seed", 42)),
        ),
        loss=DPOLossConfig(
            beta=float(loss_raw.get("beta", 0.1)),
            average_log_prob=bool(loss_raw.get("average_log_prob", False)),
        ),
        trainer=DPOTrainerConfig(
            output_dir=_resolve_path(trainer_raw["output_dir"], project_root),
            overwrite_output_dir=bool(trainer_raw.get("overwrite_output_dir", False)),
            per_device_train_batch_size=int(trainer_raw.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(trainer_raw.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(trainer_raw.get("gradient_accumulation_steps", 8)),
            num_train_epochs=float(trainer_raw.get("num_train_epochs", 1.0)),
            learning_rate=float(trainer_raw.get("learning_rate", 5e-6)),
            weight_decay=float(trainer_raw.get("weight_decay", 0.0)),
            warmup_ratio=float(trainer_raw.get("warmup_ratio", 0.03)),
            logging_steps=int(trainer_raw.get("logging_steps", 10)),
            eval_strategy=str(trainer_raw.get("eval_strategy", "steps")),
            eval_steps=int(trainer_raw.get("eval_steps", 100)),
            save_strategy=str(trainer_raw.get("save_strategy", "steps")),
            save_steps=int(trainer_raw.get("save_steps", 100)),
            save_total_limit=int(trainer_raw.get("save_total_limit", 1)),
            keep_only_best_checkpoint=bool(trainer_raw.get("keep_only_best_checkpoint", True)),
            load_best_model_at_end=bool(trainer_raw.get("load_best_model_at_end", True)),
            metric_for_best_model=trainer_raw.get("metric_for_best_model", "eval_loss"),
            greater_is_better=trainer_raw.get("greater_is_better", False),
            secondary_metric_for_best_model=trainer_raw.get("secondary_metric_for_best_model"),
            secondary_greater_is_better=trainer_raw.get("secondary_greater_is_better"),
            best_model_tie_epsilon=float(trainer_raw.get("best_model_tie_epsilon", 0.0)),
            early_stopping_patience=(
                int(trainer_raw["early_stopping_patience"])
                if trainer_raw.get("early_stopping_patience") is not None
                else None
            ),
            bf16=bool(trainer_raw.get("bf16", True)),
            fp16=bool(trainer_raw.get("fp16", False)),
            max_grad_norm=float(trainer_raw.get("max_grad_norm", 1.0)),
            dataloader_num_workers=int(trainer_raw.get("dataloader_num_workers", 4)),
            dataloader_pin_memory=bool(trainer_raw.get("dataloader_pin_memory", True)),
            report_to=trainer_raw.get("report_to"),
            seed=int(trainer_raw.get("seed", 42)),
        ),
        eval=DPOEvalConfig(
            enabled=bool(eval_raw.get("enabled", True)),
            test_jsonl=_resolve_path(eval_raw.get("test_jsonl"), project_root),
            output_jsonl=_resolve_path(eval_raw.get("output_jsonl"), project_root),
        ),
    )
