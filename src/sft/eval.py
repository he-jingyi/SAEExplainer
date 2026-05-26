"""Basic eval preview utilities for hybrid explainer SFT."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any
import torch
from tqdm import tqdm


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _normalize_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = _normalize_tokens(prediction)
    target_tokens = _normalize_tokens(target)
    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    target_counts: dict[str, int] = {}
    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, target_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def run_basic_eval_preview(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    num_examples: int,
    max_new_tokens: int,
    use_full_validation_set: bool = True,
    sample_randomly: bool = True,
    sample_seed: int = 42,
    output_jsonl: Path | None = None,
) -> dict[str, Any]:
    if len(dataset) == 0:
        return {
            "num_examples": 0,
            "use_full_validation_set": use_full_validation_set,
            "exact_match": 0.0,
            "avg_token_f1": 0.0,
            "output_jsonl": str(output_jsonl) if output_jsonl is not None else None,
        }

    model.eval()
    device = next(model.parameters()).device

    results: list[dict[str, Any]] = []
    exact_match_count = 0
    token_f1_total = 0.0

    if use_full_validation_set:
        limit = len(dataset)
        indices = list(range(limit))
        sample_randomly = False
        sample_seed = -1
    else:
        if num_examples <= 0:
            return {
                "num_examples": 0,
                "use_full_validation_set": use_full_validation_set,
                "exact_match": 0.0,
                "avg_token_f1": 0.0,
                "output_jsonl": str(output_jsonl) if output_jsonl is not None else None,
            }
        limit = min(num_examples, len(dataset))
        if sample_randomly:
            rng = random.Random(sample_seed)
            indices = rng.sample(range(len(dataset)), limit)
        else:
            indices = list(range(limit))

    output_handle = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_jsonl.open("w", encoding="utf-8")

    try:
        iterator = tqdm(
            indices,
            total=limit,
            desc="Running eval preview",
            dynamic_ncols=True,
            unit="sample",
        )
        for index in iterator:
            sample = dataset.build_eval_sample(index)
            input_ids = torch.tensor([sample["prompt_input_ids"]], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            feature_vectors = torch.tensor(sample["feature_vector"][None, :], dtype=torch.float32, device=device)
            act_positions = torch.tensor([sample["act_position"]], dtype=torch.long, device=device)
            inject_mask = torch.tensor([sample["inject_mask"]], dtype=torch.bool, device=device)

            generated = model.generate_with_injection(
                input_ids=input_ids,
                attention_mask=attention_mask,
                feature_vectors=feature_vectors,
                act_positions=act_positions,
                inject_mask=inject_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            completion_ids = generated[0, input_ids.shape[1] :]
            prediction = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            target = sample["target_text"].strip()

            exact_match = prediction == target
            score = token_f1(prediction, target)
            exact_match_count += int(exact_match)
            token_f1_total += score

            item = {
                "feature_id": sample["feature_id"],
                "prediction": prediction,
                "target": target,
                "exact_match": exact_match,
                "token_f1": score,
            }
            results.append(item)

            if output_handle is not None:
                output_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()

    return {
        "num_examples": limit,
        "use_full_validation_set": use_full_validation_set,
        "sample_randomly": sample_randomly,
        "sample_seed": sample_seed,
        "exact_match": exact_match_count / limit,
        "avg_token_f1": token_f1_total / limit,
        "output_jsonl": str(output_jsonl) if output_jsonl is not None else None,
    }


def load_checkpoint_model_for_eval(
    config_path: Path,
    checkpoint_dir: Path,
    project_root: Path,
    device: str | torch.device = "cuda",
):
    from peft import PeftModel

    from src.sft.config import load_sft_config
    from src.sft.model import HybridExplainerModel
    from src.sft.train_utils import load_base_model, load_tokenizer, set_seed

    cfg = load_sft_config(config_path, project_root)
    set_seed(cfg.trainer.seed)

    tokenizer = load_tokenizer(cfg.model)

    base_model = load_base_model(cfg.model)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(base_model, "generation_config", None) is not None:
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    base_model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_dir),
        is_trainable=False,
    )
    if hasattr(base_model, "gradient_checkpointing_disable"):
        try:
            base_model.gradient_checkpointing_disable()
        except Exception:
            pass
    if hasattr(base_model, "config"):
        base_model.config.use_cache = True
    if getattr(base_model, "generation_config", None) is not None:
        base_model.generation_config.use_cache = True

    model = HybridExplainerModel(
        base_model=base_model,
        inject_layer=cfg.injection.inject_layer,
        steering_coefficient=cfg.injection.steering_coefficient,
        use_projector=cfg.injection.use_projector,
        vector_input_dim=cfg.injection.vector_input_dim,
    ).to(device).eval()
    model.load_projector_from_checkpoint(checkpoint_dir)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if getattr(model.base_model, "generation_config", None) is not None:
        model.base_model.generation_config.use_cache = True

    return cfg, tokenizer, model


def evaluate_checkpoint_preview(
    config_path: Path,
    checkpoint_dir: Path,
    project_root: Path,
    num_examples: int,
    output_jsonl: Path,
    use_full_validation_set: bool | None = None,
    sample_randomly: bool | None = None,
    sample_seed: int | None = None,
    max_new_tokens: int | None = None,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    from src.sft.dataset import build_train_val_datasets

    cfg, tokenizer, model = load_checkpoint_model_for_eval(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        project_root=project_root,
        device=device,
    )

    _, val_dataset = build_train_val_datasets(
        tokenizer=tokenizer,
        data_config=cfg.data,
        injection_config=cfg.injection,
        project_root=project_root,
    )

    summary = run_basic_eval_preview(
        model=model,
        tokenizer=tokenizer,
        dataset=val_dataset,
        num_examples=num_examples,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else cfg.eval.max_new_tokens,
        use_full_validation_set=(
            use_full_validation_set if use_full_validation_set is not None else cfg.eval.use_full_validation_set
        ),
        sample_randomly=(sample_randomly if sample_randomly is not None else cfg.eval.sample_randomly),
        sample_seed=(sample_seed if sample_seed is not None else cfg.eval.sample_seed),
        output_jsonl=output_jsonl,
    )
    summary["checkpoint_dir"] = str(checkpoint_dir)
    return summary


def evaluate_checkpoint_preview_on_jsonl(
    config_path: Path,
    checkpoint_dir: Path,
    project_root: Path,
    input_jsonl: Path,
    num_examples: int,
    output_jsonl: Path,
    use_full_validation_set: bool | None = None,
    sample_randomly: bool | None = None,
    sample_seed: int | None = None,
    max_new_tokens: int | None = None,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    from dataclasses import replace

    from src.data_collection.sae_vectors import load_jsonl
    from src.sft.dataset import SAEExplainerSFTDataset

    cfg, tokenizer, model = load_checkpoint_model_for_eval(
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        project_root=project_root,
        device=device,
    )

    records = load_jsonl(input_jsonl)
    data_cfg = replace(
        cfg.data,
        train_jsonl=input_jsonl,
        val_ratio=0.0,
    )
    dataset = SAEExplainerSFTDataset(
        records=records,
        tokenizer=tokenizer,
        data_config=data_cfg,
        injection_config=cfg.injection,
        project_root=project_root,
    )

    summary = run_basic_eval_preview(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_examples=num_examples,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else cfg.eval.max_new_tokens,
        use_full_validation_set=(
            use_full_validation_set if use_full_validation_set is not None else cfg.eval.use_full_validation_set
        ),
        sample_randomly=(sample_randomly if sample_randomly is not None else cfg.eval.sample_randomly),
        sample_seed=(sample_seed if sample_seed is not None else cfg.eval.sample_seed),
        output_jsonl=output_jsonl,
    )
    summary["checkpoint_dir"] = str(checkpoint_dir)
    summary["input_jsonl"] = str(input_jsonl)
    return summary
