"""Reference log-prob cache helpers for vec-only DPO."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_collection.sae_vectors import load_jsonl
from src.dpo.collator import VecOnlyDPOCollator
from src.dpo.loss import sequence_log_probs_from_logits
from src.sft.config import load_sft_config
from src.sft.eval import load_checkpoint_model_for_eval


def build_reference_cache_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / "reference_cache" / f"{split_name}_reference_logps.pt"


def _build_reference_cache_metadata(
    *,
    input_jsonl: Path,
    reference_checkpoint_dir: Path,
    sft_config_path: Path,
    average_log_prob: bool,
    max_length: int,
) -> dict[str, Any]:
    input_jsonl = input_jsonl.resolve()
    reference_checkpoint_dir = reference_checkpoint_dir.resolve()
    sft_config_path = sft_config_path.resolve()
    stat = input_jsonl.stat()
    return {
        "input_jsonl": str(input_jsonl),
        "input_size": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "reference_checkpoint_dir": str(reference_checkpoint_dir),
        "sft_config_path": str(sft_config_path),
        "average_log_prob": bool(average_log_prob),
        "max_length": int(max_length),
    }


def load_reference_logp_cache(cache_path: Path) -> dict[str, Any]:
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Reference cache at {cache_path} is not a dict payload.")
    return payload


def get_cached_reference_logps(cache_payload: dict[str, Any], record_index: int) -> tuple[float, float] | None:
    valid_mask = cache_payload.get("valid_mask")
    chosen_logps = cache_payload.get("chosen_logps")
    rejected_logps = cache_payload.get("rejected_logps")
    if valid_mask is None or chosen_logps is None or rejected_logps is None:
        return None
    if record_index < 0 or record_index >= len(valid_mask):
        return None
    if not bool(valid_mask[record_index]):
        return None
    return float(chosen_logps[record_index]), float(rejected_logps[record_index])


def cache_matches_metadata(cache_payload: dict[str, Any], expected_metadata: dict[str, Any]) -> bool:
    return cache_payload.get("metadata") == expected_metadata


def _save_reference_logp_cache(
    *,
    cache_path: Path,
    metadata: dict[str, Any],
    num_records: int,
    cached_values: dict[int, tuple[float, float]],
) -> None:
    chosen_logps = torch.zeros(num_records, dtype=torch.float32)
    rejected_logps = torch.zeros(num_records, dtype=torch.float32)
    valid_mask = torch.zeros(num_records, dtype=torch.bool)
    for record_index, (chosen_logp, rejected_logp) in cached_values.items():
        chosen_logps[record_index] = float(chosen_logp)
        rejected_logps[record_index] = float(rejected_logp)
        valid_mask[record_index] = True

    payload = {
        "metadata": metadata,
        "chosen_logps": chosen_logps,
        "rejected_logps": rejected_logps,
        "valid_mask": valid_mask,
        "num_records": num_records,
        "num_cached_rows": int(valid_mask.sum().item()),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=str(cache_path.parent)) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, cache_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def compute_reference_logp_cache(
    *,
    input_jsonl: Path,
    cache_path: Path,
    dpo_cfg: Any,
    project_root: Path,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    from src.dpo.dataset import VecOnlyDPODataset

    raw_records = load_jsonl(input_jsonl)
    reference_dir = dpo_cfg.model.reference_checkpoint_dir or dpo_cfg.model.init_checkpoint_dir
    sft_cfg = load_sft_config(dpo_cfg.model.sft_config_path, project_root)
    metadata = _build_reference_cache_metadata(
        input_jsonl=input_jsonl,
        reference_checkpoint_dir=reference_dir,
        sft_config_path=dpo_cfg.model.sft_config_path,
        average_log_prob=dpo_cfg.loss.average_log_prob,
        max_length=sft_cfg.data.max_length,
    )

    _, tokenizer, reference_model = load_checkpoint_model_for_eval(
        config_path=dpo_cfg.model.sft_config_path,
        checkpoint_dir=reference_dir,
        project_root=project_root,
        device=device,
    )
    dataset = VecOnlyDPODataset(raw_records, sft_config_path=dpo_cfg.model.sft_config_path, project_root=project_root)
    collator = VecOnlyDPOCollator(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    device_obj = next(reference_model.parameters()).device
    cached_values: dict[int, tuple[float, float]] = {}

    try:
        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Caching ref logps: {input_jsonl.name}",
            dynamic_ncols=True,
            unit="batch",
        )
        for batch in iterator:
            ref_batch: dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    ref_batch[key] = value.to(device_obj)
                else:
                    ref_batch[key] = value

            shared_kwargs = {
                "feature_vectors": ref_batch["feature_vectors"],
                "act_positions": ref_batch["act_positions"],
                "inject_mask": ref_batch["inject_mask"],
            }

            with torch.no_grad():
                ref_chosen = reference_model(
                    input_ids=ref_batch["chosen_input_ids"],
                    attention_mask=ref_batch["chosen_attention_mask"],
                    **shared_kwargs,
                )
                ref_rejected = reference_model(
                    input_ids=ref_batch["rejected_input_ids"],
                    attention_mask=ref_batch["rejected_attention_mask"],
                    **shared_kwargs,
                )

            ref_chosen_logps = sequence_log_probs_from_logits(
                ref_chosen.logits,
                ref_batch["chosen_labels"],
                average_log_prob=dpo_cfg.loss.average_log_prob,
            )
            ref_rejected_logps = sequence_log_probs_from_logits(
                ref_rejected.logits,
                ref_batch["rejected_labels"],
                average_log_prob=dpo_cfg.loss.average_log_prob,
            )

            for record_index, chosen_logp, rejected_logp in zip(
                batch["record_indices"],
                ref_chosen_logps.detach().cpu().tolist(),
                ref_rejected_logps.detach().cpu().tolist(),
                strict=True,
            ):
                cached_values[int(record_index)] = (float(chosen_logp), float(rejected_logp))
    finally:
        del reference_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    _save_reference_logp_cache(
        cache_path=cache_path,
        metadata=metadata,
        num_records=len(raw_records),
        cached_values=cached_values,
    )

    return {
        "cache_path": str(cache_path),
        "input_jsonl": str(input_jsonl),
        "reference_checkpoint_dir": str(reference_dir),
        "device": str(device_obj),
        "num_records": len(raw_records),
        "num_dataset_rows": len(dataset),
        "num_cached_rows": len(cached_values),
        "all_rows_cached": len(cached_values) == len(dataset),
        "generated": True,
        "metadata": metadata,
    }


def ensure_reference_logp_cache(
    *,
    input_jsonl: Path,
    cache_path: Path,
    dpo_cfg: Any,
    project_root: Path,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    sft_cfg = load_sft_config(dpo_cfg.model.sft_config_path, project_root)
    reference_dir = dpo_cfg.model.reference_checkpoint_dir or dpo_cfg.model.init_checkpoint_dir
    expected_metadata = _build_reference_cache_metadata(
        input_jsonl=input_jsonl,
        reference_checkpoint_dir=reference_dir,
        sft_config_path=dpo_cfg.model.sft_config_path,
        average_log_prob=dpo_cfg.loss.average_log_prob,
        max_length=sft_cfg.data.max_length,
    )

    if cache_path.exists():
        try:
            payload = load_reference_logp_cache(cache_path)
            if cache_matches_metadata(payload, expected_metadata):
                return {
                    "cache_path": str(cache_path),
                    "input_jsonl": str(input_jsonl),
                    "generated": False,
                    "num_cached_rows": int(payload.get("num_cached_rows", 0)),
                    "num_records": int(payload.get("num_records", 0)),
                    "metadata": expected_metadata,
                }
        except Exception:
            pass

    return compute_reference_logp_cache(
        input_jsonl=input_jsonl,
        cache_path=cache_path,
        dpo_cfg=dpo_cfg,
        project_root=project_root,
        batch_size=batch_size,
        device=device,
    )
