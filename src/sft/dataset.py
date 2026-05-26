"""Dataset utilities for hybrid SAE explainer SFT."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from src.data_collection.sae_vectors import (
    SAEFeatureLocator,
    SAEVectorStore,
    load_jsonl,
    locator_from_record,
)
from src.sft.config import DataConfig, InjectionConfig
from src.sft.prompting import build_hybrid_explainer_prompt
from src.sft.tokenization import build_model_inputs, ensure_placeholder_is_single_token


@dataclass(frozen=True)
class PreparedExample:
    feature_id: str
    input_ids: list[int]
    labels: list[int]
    act_position: int
    assistant_start_idx: int
    target_text: str
    decoder_vector_path: Path | None
    vector_locator: SAEFeatureLocator | None


def _resolve_optional_path(raw_path: str | None, project_root: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def split_records_by_feature_id(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []

    feature_ids = sorted({str(record["feature_id"]) for record in records})
    rng = random.Random(seed)
    rng.shuffle(feature_ids)

    if val_ratio <= 0:
        val_feature_ids: set[str] = set()
    else:
        num_val_features = max(1, int(round(len(feature_ids) * val_ratio)))
        num_val_features = min(num_val_features, max(len(feature_ids) - 1, 1))
        val_feature_ids = set(feature_ids[:num_val_features])

    train_records = [record for record in records if record["feature_id"] not in val_feature_ids]
    val_records = [record for record in records if record["feature_id"] in val_feature_ids]
    return train_records, val_records


class SAEExplainerSFTDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        data_config: DataConfig,
        injection_config: InjectionConfig,
        project_root: Path,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.injection_config = injection_config
        self.project_root = project_root
        self.vector_store = SAEVectorStore(local_dir=data_config.vector_store_cache_dir)
        self._vector_cache: dict[Path, np.ndarray] = {}
        self.placeholder_token_id, self.placeholder_token = ensure_placeholder_is_single_token(
            tokenizer=tokenizer,
            placeholder_text=injection_config.placeholder_text,
        )
        self.examples = self._prepare_examples()

    def _prepare_examples(self) -> list[PreparedExample]:
        examples: list[PreparedExample] = []

        for record in self.records:
            explanation = str(record.get("explanation") or "").strip()
            if not explanation:
                continue

            user_prompt = build_hybrid_explainer_prompt(
                placeholder_text=self.injection_config.placeholder_text,
            )
            tokenized = build_model_inputs(
                tokenizer=self.tokenizer,
                user_prompt=user_prompt,
                assistant_response=explanation,
                placeholder_token_id=self.placeholder_token_id,
                max_length=self.data_config.max_length,
            )

            metadata = record.get("metadata") or {}
            decoder_vector_path = _resolve_optional_path(
                metadata.get("decoder_vector_path"),
                project_root=self.project_root,
            )
            vector_locator: SAEFeatureLocator | None = None
            if decoder_vector_path is None or not decoder_vector_path.exists():
                if self.data_config.require_decoder_vector_path:
                    raise FileNotFoundError(
                        f"decoder_vector_path missing or not found for feature {record.get('feature_id')}"
                    )
                vector_locator = locator_from_record(record)

            examples.append(
                PreparedExample(
                    feature_id=str(record["feature_id"]),
                    input_ids=tokenized["input_ids"],
                    labels=tokenized["labels"],
                    act_position=int(tokenized["act_position"]),
                    assistant_start_idx=int(tokenized["assistant_start_idx"]),
                    target_text=explanation,
                    decoder_vector_path=decoder_vector_path,
                    vector_locator=vector_locator,
                )
            )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _load_vector(self, example: PreparedExample) -> np.ndarray:
        if example.decoder_vector_path is not None and example.decoder_vector_path.exists():
            if example.decoder_vector_path not in self._vector_cache:
                self._vector_cache[example.decoder_vector_path] = np.load(example.decoder_vector_path).astype(
                    np.float32
                )
            return self._vector_cache[example.decoder_vector_path].copy()

        if example.vector_locator is None:
            raise ValueError(f"No vector locator available for feature {example.feature_id}")
        return self.vector_store.get_decoder_vector(example.vector_locator)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        vector = self._load_vector(example)
        return {
            "feature_id": example.feature_id,
            "input_ids": example.input_ids,
            "labels": example.labels,
            "act_position": example.act_position,
            "feature_vector": vector,
            "inject_mask": True,
        }

    def build_eval_sample(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        vector = self._load_vector(example)
        prompt_input_ids = example.input_ids[: example.assistant_start_idx]
        return {
            "feature_id": example.feature_id,
            "prompt_input_ids": prompt_input_ids,
            "target_text": example.target_text,
            "act_position": example.act_position,
            "feature_vector": vector,
            "inject_mask": True,
        }


def build_train_val_datasets(
    tokenizer: Any,
    data_config: DataConfig,
    injection_config: InjectionConfig,
    project_root: Path,
) -> tuple[SAEExplainerSFTDataset, SAEExplainerSFTDataset]:
    train_records = load_jsonl(data_config.train_jsonl)
    if data_config.val_jsonl is not None:
        val_records = load_jsonl(data_config.val_jsonl)
    else:
        train_records, val_records = split_records_by_feature_id(
            records=train_records,
            val_ratio=data_config.val_ratio,
            seed=data_config.seed,
        )

    train_dataset = SAEExplainerSFTDataset(
        records=train_records,
        tokenizer=tokenizer,
        data_config=data_config,
        injection_config=injection_config,
        project_root=project_root,
    )
    val_dataset = SAEExplainerSFTDataset(
        records=val_records,
        tokenizer=tokenizer,
        data_config=data_config,
        injection_config=injection_config,
        project_root=project_root,
    )
    return train_dataset, val_dataset
