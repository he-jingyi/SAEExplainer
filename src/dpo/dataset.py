"""Preference dataset and tokenization for vec-only DPO."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from src.data_collection.sae_vectors import SAEFeatureLocator, SAEVectorStore, locator_from_record, load_jsonl
from src.dpo.reference_cache import get_cached_reference_logps, load_reference_logp_cache
from src.sft.config import load_sft_config
from src.sft.prompting import build_hybrid_explainer_prompt
from src.sft.tokenization import build_model_inputs, ensure_placeholder_is_single_token


@dataclass(frozen=True)
class DPOSample:
    record_index: int
    feature_id: str
    metadata: dict[str, Any]
    chosen_input_ids: list[int]
    chosen_labels: list[int]
    rejected_input_ids: list[int]
    rejected_labels: list[int]
    act_position: int
    decoder_vector_path: Path | None
    vector_locator: SAEFeatureLocator | None
    chosen_text: str
    rejected_text: str
    chosen_score: float
    rejected_score: float
    score_gap: float
    cached_ref_chosen_logps: float | None
    cached_ref_rejected_logps: float | None


def _resolve_optional_path(raw_path: str | None, project_root: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def split_preferences_by_feature_id(
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


class VecOnlyDPODataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        sft_config_path: Path,
        project_root: Path,
        reference_logp_cache_path: Path | None = None,
    ) -> None:
        self.records = records
        self.project_root = project_root
        self.sft_cfg = load_sft_config(sft_config_path, project_root)
        self.reference_logp_cache_path = reference_logp_cache_path
        self.reference_logp_cache = (
            load_reference_logp_cache(reference_logp_cache_path)
            if reference_logp_cache_path is not None and reference_logp_cache_path.exists()
            else None
        )
        from src.sft.train_utils import load_tokenizer

        self.tokenizer = load_tokenizer(self.sft_cfg.model)
        self.vector_store = SAEVectorStore(local_dir=self.sft_cfg.data.vector_store_cache_dir)
        self._vector_cache: dict[Path, np.ndarray] = {}
        self.placeholder_token_id, self.placeholder_token = ensure_placeholder_is_single_token(
            self.tokenizer, self.sft_cfg.injection.placeholder_text
        )
        self.samples = self._prepare_samples()
        self.num_cached_reference_pairs = sum(
            1
            for sample in self.samples
            if sample.cached_ref_chosen_logps is not None and sample.cached_ref_rejected_logps is not None
        )
        self.all_have_cached_reference_logps = bool(self.samples) and (
            self.num_cached_reference_pairs == len(self.samples)
        )

    def _prepare_samples(self) -> list[DPOSample]:
        samples: list[DPOSample] = []
        user_prompt = build_hybrid_explainer_prompt(
            placeholder_text=self.sft_cfg.injection.placeholder_text,
        )

        for local_record_index, record in enumerate(self.records):
            record_index = int(record.get("__source_record_index__", local_record_index))
            chosen = str(record.get("chosen_explanation") or "").strip()
            rejected = str(record.get("rejected_explanation") or "").strip()
            if not chosen or not rejected:
                continue

            chosen_tokenized = build_model_inputs(
                tokenizer=self.tokenizer,
                user_prompt=user_prompt,
                assistant_response=chosen,
                placeholder_token_id=self.placeholder_token_id,
                max_length=self.sft_cfg.data.max_length,
            )
            rejected_tokenized = build_model_inputs(
                tokenizer=self.tokenizer,
                user_prompt=user_prompt,
                assistant_response=rejected,
                placeholder_token_id=self.placeholder_token_id,
                max_length=self.sft_cfg.data.max_length,
            )

            if chosen_tokenized["act_position"] != rejected_tokenized["act_position"]:
                raise ValueError("Chosen/rejected act positions do not match for the same prompt.")

            metadata = record.get("metadata") or {}
            decoder_vector_path = _resolve_optional_path(metadata.get("decoder_vector_path"), self.project_root)
            vector_locator: SAEFeatureLocator | None = None
            if decoder_vector_path is None or not decoder_vector_path.exists():
                vector_locator = locator_from_record(record)

            cached_ref_values = (
                get_cached_reference_logps(self.reference_logp_cache, record_index)
                if self.reference_logp_cache is not None
                else None
            )
            if cached_ref_values is not None:
                cached_ref_chosen_logps, cached_ref_rejected_logps = cached_ref_values
            else:
                cached_ref_chosen_logps = record.get("cached_ref_chosen_logps", record.get("ref_chosen_logps"))
                cached_ref_rejected_logps = record.get("cached_ref_rejected_logps", record.get("ref_rejected_logps"))

            samples.append(
                DPOSample(
                    record_index=record_index,
                    feature_id=str(record["feature_id"]),
                    metadata=dict(metadata),
                    chosen_input_ids=chosen_tokenized["input_ids"],
                    chosen_labels=chosen_tokenized["labels"],
                    rejected_input_ids=rejected_tokenized["input_ids"],
                    rejected_labels=rejected_tokenized["labels"],
                    act_position=int(chosen_tokenized["act_position"]),
                    decoder_vector_path=decoder_vector_path,
                    vector_locator=vector_locator,
                    chosen_text=chosen,
                    rejected_text=rejected,
                    chosen_score=float(record.get("chosen_score", 0.0)),
                    rejected_score=float(record.get("rejected_score", 0.0)),
                    score_gap=float(record.get("score_gap", 0.0)),
                    cached_ref_chosen_logps=(
                        float(cached_ref_chosen_logps) if cached_ref_chosen_logps is not None else None
                    ),
                    cached_ref_rejected_logps=(
                        float(cached_ref_rejected_logps) if cached_ref_rejected_logps is not None else None
                    ),
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_vector(self, sample: DPOSample) -> np.ndarray:
        if sample.decoder_vector_path is not None and sample.decoder_vector_path.exists():
            if sample.decoder_vector_path not in self._vector_cache:
                self._vector_cache[sample.decoder_vector_path] = np.load(sample.decoder_vector_path).astype(
                    np.float32
                )
            return self._vector_cache[sample.decoder_vector_path].copy()

        if sample.vector_locator is None:
            raise ValueError(f"No vector locator available for feature {sample.feature_id}")
        return self.vector_store.get_decoder_vector(sample.vector_locator)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        vector = self._load_vector(sample)
        return {
            "record_index": sample.record_index,
            "feature_id": sample.feature_id,
            "chosen_input_ids": sample.chosen_input_ids,
            "chosen_labels": sample.chosen_labels,
            "rejected_input_ids": sample.rejected_input_ids,
            "rejected_labels": sample.rejected_labels,
            "act_position": sample.act_position,
            "feature_vector": vector,
            "inject_mask": True,
            "chosen_text": sample.chosen_text,
            "rejected_text": sample.rejected_text,
            "chosen_score": sample.chosen_score,
            "rejected_score": sample.rejected_score,
            "score_gap": sample.score_gap,
            "has_cached_reference_logps": (
                sample.cached_ref_chosen_logps is not None and sample.cached_ref_rejected_logps is not None
            ),
            "cached_ref_chosen_logps": (
                float(sample.cached_ref_chosen_logps) if sample.cached_ref_chosen_logps is not None else 0.0
            ),
            "cached_ref_rejected_logps": (
                float(sample.cached_ref_rejected_logps) if sample.cached_ref_rejected_logps is not None else 0.0
            ),
        }


def build_dpo_train_val_datasets(
    preference_jsonl: Path,
    val_preference_jsonl: Path | None,
    sft_config_path: Path,
    project_root: Path,
    val_ratio: float,
    seed: int,
    train_reference_logp_cache_path: Path | None = None,
    val_reference_logp_cache_path: Path | None = None,
) -> tuple[VecOnlyDPODataset, VecOnlyDPODataset]:
    train_records = load_jsonl(preference_jsonl)
    for idx, record in enumerate(train_records):
        record["__source_record_index__"] = idx
    if val_preference_jsonl is not None:
        val_records = load_jsonl(val_preference_jsonl)
        for idx, record in enumerate(val_records):
            record["__source_record_index__"] = idx
    else:
        train_records, val_records = split_preferences_by_feature_id(train_records, val_ratio=val_ratio, seed=seed)
    return (
        VecOnlyDPODataset(
            train_records,
            sft_config_path=sft_config_path,
            project_root=project_root,
            reference_logp_cache_path=train_reference_logp_cache_path,
        ),
        VecOnlyDPODataset(
            val_records,
            sft_config_path=sft_config_path,
            project_root=project_root,
            reference_logp_cache_path=val_reference_logp_cache_path,
        ),
    )
