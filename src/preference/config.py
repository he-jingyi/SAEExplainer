"""Configuration helpers for the vec-only preference pipeline."""

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
class ExplainerConfig:
    sft_config_path: Path
    checkpoint_dir: Path
    temperature: float
    top_p: float
    max_new_tokens: int
    num_candidates: int
    generation_batch_size: int


@dataclass(frozen=True)
class GeneratorConfig:
    backend: str
    model_name: str
    trust_remote_code: bool
    torch_dtype: str
    attn_implementation: str | None
    temperature: float
    top_p: float
    max_new_tokens: int
    num_texts_per_explanation: int
    api_base_url: str | None
    api_key_env: str | None
    api_model_name: str | None
    timeout_seconds: float
    max_retries: int
    max_concurrency: int


@dataclass(frozen=True)
class TargetModelConfig:
    model_name: str
    trust_remote_code: bool
    torch_dtype: str
    attn_implementation: str | None
    max_length: int
    score_chunk_size: int
    score_batch_size: int


@dataclass(frozen=True)
class PreferenceDataConfig:
    input_jsonl: Path
    output_dir: Path
    max_features: int | None
    random_seed: int
    rejected_strategy: str
    min_chosen_activation: float
    min_score_gap: float
    filter_min_chosen_activation: float
    filter_min_group_spread_ratio: float
    filter_min_relative_gap: float


@dataclass(frozen=True)
class PreferencePipelineConfig:
    explainer: ExplainerConfig
    generator: GeneratorConfig
    target_model: TargetModelConfig
    data: PreferenceDataConfig


def load_preference_config(config_path: Path, project_root: Path) -> PreferencePipelineConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    explainer_raw = raw["explainer"]
    generator_raw = raw["generator"]
    target_raw = raw["target_model"]
    data_raw = raw["data"]

    return PreferencePipelineConfig(
        explainer=ExplainerConfig(
            sft_config_path=_resolve_path(explainer_raw["sft_config_path"], project_root),
            checkpoint_dir=_resolve_path(explainer_raw["checkpoint_dir"], project_root),
            temperature=float(explainer_raw.get("temperature", 0.9)),
            top_p=float(explainer_raw.get("top_p", 0.9)),
            max_new_tokens=int(explainer_raw.get("max_new_tokens", 64)),
            num_candidates=int(explainer_raw.get("num_candidates", 10)),
            generation_batch_size=int(explainer_raw.get("generation_batch_size", 1)),
        ),
        generator=GeneratorConfig(
            backend=str(generator_raw.get("backend", "local")),
            model_name=str(generator_raw["model_name"]),
            trust_remote_code=bool(generator_raw.get("trust_remote_code", True)),
            torch_dtype=str(generator_raw.get("torch_dtype", "bfloat16")),
            attn_implementation=generator_raw.get("attn_implementation"),
            temperature=float(generator_raw.get("temperature", 0.8)),
            top_p=float(generator_raw.get("top_p", 0.95)),
            max_new_tokens=int(generator_raw.get("max_new_tokens", 128)),
            num_texts_per_explanation=int(generator_raw.get("num_texts_per_explanation", 4)),
            api_base_url=generator_raw.get("api_base_url"),
            api_key_env=generator_raw.get("api_key_env"),
            api_model_name=generator_raw.get("api_model_name"),
            timeout_seconds=float(generator_raw.get("timeout_seconds", 60.0)),
            max_retries=int(generator_raw.get("max_retries", 3)),
            max_concurrency=int(generator_raw.get("max_concurrency", 1)),
        ),
        target_model=TargetModelConfig(
            model_name=str(target_raw["model_name"]),
            trust_remote_code=bool(target_raw.get("trust_remote_code", True)),
            torch_dtype=str(target_raw.get("torch_dtype", "bfloat16")),
            attn_implementation=target_raw.get("attn_implementation"),
            max_length=int(target_raw.get("max_length", 512)),
            score_chunk_size=int(target_raw.get("score_chunk_size", 256)),
            score_batch_size=int(target_raw.get("score_batch_size", 8)),
        ),
        data=PreferenceDataConfig(
            input_jsonl=_resolve_path(data_raw["input_jsonl"], project_root),
            output_dir=_resolve_path(data_raw["output_dir"], project_root),
            max_features=(
                int(data_raw["max_features"])
                if data_raw.get("max_features") is not None
                else None
            ),
            random_seed=int(data_raw.get("random_seed", 42)),
            rejected_strategy=str(data_raw.get("rejected_strategy", "median")),
            min_chosen_activation=float(data_raw.get("min_chosen_activation", 0.0)),
            min_score_gap=float(data_raw.get("min_score_gap", 0.0)),
            filter_min_chosen_activation=float(data_raw.get("filter_min_chosen_activation", 10.0)),
            filter_min_group_spread_ratio=float(data_raw.get("filter_min_group_spread_ratio", 0.10)),
            filter_min_relative_gap=float(data_raw.get("filter_min_relative_gap", 0.15)),
        ),
    )
