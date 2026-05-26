"""DPO generation evaluation with local SAE activation scoring."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_collection.sae_vectors import (
    load_jsonl,
    locator_from_record,
    resolve_record_target_base_model_name,
)
from src.preference.prompts import build_generative_prompt
from src.preference.sae import JumpReluSAEStore
from src.sft.config import load_sft_config
from src.sft.dataset import SAEExplainerSFTDataset
from src.sft.eval import load_checkpoint_model_for_eval
from src.sft.train_utils import load_tokenizer, resolve_torch_dtype


THRESHOLD_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "cache" / "feature_thresholds.jsonl"
_THRESHOLD_CACHE: dict[str, dict[str, Any]] | None = None


def _completion_length_param(llm_model: str, max_output_tokens: int) -> dict[str, int]:
    if llm_model.startswith("gpt-5"):
        return {"max_completion_tokens": max_output_tokens}
    return {"max_tokens": max_output_tokens}


def _build_openai_client(
    *,
    api_key_env: str,
    base_url: str | None = None,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: openai. Install it first, for example:\n"
            "  pip install -U openai"
        ) from exc

    import os

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env!r} is not set.")
    api_key = api_key.strip()
    api_key = api_key.strip("\"'")
    api_key = api_key.strip("“”‘’")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"Environment variable {api_key_env!r} contains non-ASCII characters. "
            "This usually means the API key was exported with smart quotes. "
            "Re-export it using plain ASCII quotes, for example:\n"
            '  export OPENAI_API_KEY="sk-..."'
        ) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _merge_usage(total: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(total)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        merged[key] = int(merged.get(key, 0)) + int(update.get(key, 0))
    return merged


def _parse_generator_samples(raw_text: str) -> list[str]:
    raw_text = raw_text.strip()
    if not raw_text:
        return []

    try:
        payload = json.loads(raw_text)
        samples = payload.get("samples") or []
        cleaned = [str(item).strip() for item in samples if str(item).strip()]
        if cleaned:
            return cleaned
    except Exception:
        pass

    return [raw_text]


def _load_threshold_cache() -> dict[str, dict[str, Any]]:
    global _THRESHOLD_CACHE
    if _THRESHOLD_CACHE is not None:
        return _THRESHOLD_CACHE

    THRESHOLD_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not THRESHOLD_CACHE_PATH.exists():
        THRESHOLD_CACHE_PATH.touch()
        _THRESHOLD_CACHE = {}
        return _THRESHOLD_CACHE

    cache: dict[str, dict[str, Any]] = {}
    with THRESHOLD_CACHE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            feature_id = str(row.get("feature_id") or "").strip()
            threshold = row.get("threshold")
            if not feature_id or not isinstance(threshold, dict):
                continue
            cache[feature_id] = threshold
    _THRESHOLD_CACHE = cache
    return _THRESHOLD_CACHE


def _write_threshold_cache_entry(feature_id: str, threshold_info: dict[str, Any]) -> None:
    cache = _load_threshold_cache()
    cache[feature_id] = dict(threshold_info)
    THRESHOLD_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THRESHOLD_CACHE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "feature_id": feature_id,
                    "threshold": threshold_info,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _generate_texts_from_explanation(
    *,
    client: Any,
    explanation: str,
    num_samples: int,
    llm_model: str,
) -> tuple[list[str], dict[str, Any]]:
    prompt = build_generative_prompt(explanation, num_samples)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert test case generator for Sparse Autoencoder features. "
                "You generate diverse, high-quality texts that strongly activate one feature."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    params: dict[str, Any] = {
        "model": llm_model,
        "messages": messages,
    }
    params.update(_completion_length_param(llm_model, 4096))

    response = client.chat.completions.create(**params)
    content = response.choices[0].message.content or ""
    samples = _parse_generator_samples(content)
    return samples[:num_samples], _extract_usage(response)


def _generate_texts_batch_from_explanations(
    *,
    client: Any,
    jobs: list[tuple[str, str, int]],
    llm_model: str,
    max_concurrency: int,
    progress_desc: str | None = None,
) -> dict[str, tuple[list[str], dict[str, Any]]]:
    if not jobs:
        return {}

    if max_concurrency <= 1 or len(jobs) <= 1:
        results: dict[str, tuple[list[str], dict[str, Any]]] = {}
        iterator = jobs
        if progress_desc is not None:
            iterator = tqdm(jobs, desc=progress_desc, leave=False, dynamic_ncols=True, unit="req")
        for key, explanation, num_samples in iterator:
            results[key] = _generate_texts_from_explanation(
                client=client,
                explanation=explanation,
                num_samples=num_samples,
                llm_model=llm_model,
            )
        return results

    results: dict[str, tuple[list[str], dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_key = {
            executor.submit(
                _generate_texts_from_explanation,
                client=client,
                explanation=explanation,
                num_samples=num_samples,
                llm_model=llm_model,
            ): key
            for key, explanation, num_samples in jobs
        }
        progress = None
        if progress_desc is not None:
            progress = tqdm(total=len(future_to_key), desc=progress_desc, leave=False, dynamic_ncols=True, unit="req")
        try:
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                results[key] = future.result()
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()
    return results


def _build_eval_dataset(
    *,
    config_path: Path,
    input_jsonl: Path,
    project_root: Path,
    tokenizer: Any,
) -> tuple[Any, list[dict[str, Any]], SAEExplainerSFTDataset]:
    sft_cfg = load_sft_config(config_path, project_root)
    records = load_jsonl(input_jsonl)
    data_cfg = replace(
        sft_cfg.data,
        train_jsonl=input_jsonl,
        val_ratio=0.0,
    )
    dataset = SAEExplainerSFTDataset(
        records=records,
        tokenizer=tokenizer,
        data_config=data_cfg,
        injection_config=sft_cfg.injection,
        project_root=project_root,
    )
    records_by_feature_id = {str(record["feature_id"]): record for record in records}
    aligned_records = [
        records_by_feature_id[str(example.feature_id)]
        for example in dataset.examples
        if str(example.feature_id) in records_by_feature_id
    ]
    return sft_cfg, aligned_records, dataset


def _generate_one_explanation(
    *,
    model: Any,
    tokenizer: Any,
    prompt_input_ids: list[int],
    feature_vector: Any,
    act_position: int,
    max_new_tokens: int,
    device: str | torch.device,
) -> str:
    input_ids = torch.tensor([prompt_input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    feature_vectors = torch.tensor(feature_vector[None, :], dtype=torch.float32, device=device)
    act_positions = torch.tensor([act_position], dtype=torch.long, device=device)
    inject_mask = torch.tensor([True], dtype=torch.bool, device=device)

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
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _pad_prompt_sequences(sequences: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(seq) for seq in sequences)
    padded = [seq + [pad_token_id] * (max_length - len(seq)) for seq in sequences]
    input_ids = torch.tensor(padded, dtype=torch.long)
    attention_mask = (input_ids != pad_token_id).long()
    return input_ids, attention_mask


def _generate_explanations_batch(
    *,
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    max_new_tokens: int,
    device: str | torch.device,
) -> list[str]:
    if not samples:
        return []

    prompt_input_ids = [sample["prompt_input_ids"] for sample in samples]
    input_ids, attention_mask = _pad_prompt_sequences(prompt_input_ids, tokenizer.pad_token_id)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    feature_vectors = torch.tensor(
        np.stack([sample["feature_vector"] for sample in samples]),
        dtype=torch.float32,
        device=device,
    )
    act_positions = torch.tensor([sample["act_position"] for sample in samples], dtype=torch.long, device=device)
    inject_mask = torch.tensor([bool(sample["inject_mask"]) for sample in samples], dtype=torch.bool, device=device)

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

    prompt_width = input_ids.shape[1]
    results: list[str] = []
    for row in range(generated.shape[0]):
        completion_ids = generated[row, prompt_width:]
        results.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
    return results


def _resolve_target_scoring_model_name(
    records: list[dict[str, Any]],
    *,
    fallback_model_name: str,
) -> str:
    resolved_names = {
        resolve_record_target_base_model_name(
            record,
            fallback_model_name=fallback_model_name,
        )
        for record in records
    }
    if not resolved_names:
        return fallback_model_name
    if len(resolved_names) > 1:
        raise ValueError(
            "Expected a single target base model for evaluation, but found: "
            + ", ".join(sorted(resolved_names))
        )
    return next(iter(resolved_names))


def _compute_dynamic_threshold(
    record: dict[str, Any],
    *,
    fallback_threshold: float,
) -> tuple[float, dict[str, Any]]:
    feature_id = str(record.get("feature_id") or "").strip()
    if not feature_id:
        raise RuntimeError("Cannot compute threshold because record is missing feature_id.")

    cache = _load_threshold_cache()
    cached_threshold = cache.get(feature_id)
    if isinstance(cached_threshold, dict):
        raw_value = cached_threshold.get("value")
        if isinstance(raw_value, (int, float)):
            metadata = dict(cached_threshold)
            metadata.setdefault("source", "threshold_cache")
            metadata.setdefault("num_contexts", 0)
            metadata.setdefault("context_max_activations", [])
            return float(raw_value), metadata

    metadata = record.get("metadata") or {}
    approx = metadata.get("max_activation_approx")
    if approx is not None:
        threshold = float(approx) / 2.0
        metadata_out = {
            "source": "metadata_max_activation_approx_div2",
            "num_contexts": 0,
            "context_limit": 5,
            "context_max_activations": [],
            "max_activation_approx": float(approx),
            "value": threshold,
        }
        _write_threshold_cache_entry(feature_id, metadata_out)
        return threshold, metadata_out

    raise RuntimeError(
        "No cached threshold found and metadata.max_activation_approx is missing for "
        f"feature_id={feature_id}. Refusing to fall back to "
        f"fallback_activation_threshold={fallback_threshold!r}."
    )


def _build_valid_token_mask(
    input_ids_BL: torch.Tensor,
    attention_mask_BL: torch.Tensor,
    special_ids: set[int],
) -> torch.Tensor:
    valid_mask_BL = attention_mask_BL.bool().clone()
    if not special_ids:
        return valid_mask_BL

    special_mask_BL = torch.zeros_like(input_ids_BL, dtype=torch.bool)
    for token_id in special_ids:
        special_mask_BL |= input_ids_BL == token_id
    valid_mask_BL &= ~special_mask_BL
    return valid_mask_BL


class LocalActivationScorer:
    def __init__(
        self,
        *,
        model_name: str,
        trust_remote_code: bool,
        torch_dtype: str,
        attn_implementation: str | None,
        max_length: int,
        sae_cache_dir: Path,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": resolve_torch_dtype(torch_dtype),
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(**kwargs).to(self.device).eval()
        self.special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        self.sae_store = JumpReluSAEStore(local_dir=sae_cache_dir)

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def score_texts_batch(
        self,
        *,
        texts: list[str],
        record: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not texts:
            return []

        locator = locator_from_record(record)
        metadata = record.get("metadata") or {}
        raw_hook_layer = metadata.get("hook_layer")
        hook_layer = int(raw_hook_layer if raw_hook_layer is not None else metadata.get("layer", 0))

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
        if hook_layer + 1 >= len(outputs.hidden_states):
            raise IndexError(
                f"hook_layer={hook_layer} is out of bounds for hidden_states with "
                f"length {len(outputs.hidden_states)}"
            )

        valid_mask_BL = _build_valid_token_mask(
            input_ids_BL=encoded["input_ids"],
            attention_mask_BL=encoded["attention_mask"],
            special_ids=self.special_ids,
        )
        hidden_states_BLD = outputs.hidden_states[hook_layer + 1]

        results: list[dict[str, Any]] = []
        batch_size = len(texts)
        for batch_index in range(batch_size):
            input_ids_L = encoded["input_ids"][batch_index]
            attention_mask_L = encoded["attention_mask"][batch_index]
            sequence_length = int(attention_mask_L.sum().item())
            hidden_states_LD = hidden_states_BLD[batch_index, :sequence_length]
            valid_mask_L = valid_mask_BL[batch_index, :sequence_length]

            feature_score = self.sae_store.score_feature(
                hidden_states_LD=hidden_states_LD,
                locator=locator,
                valid_token_mask_L=valid_mask_L,
            )

            token_ids = input_ids_L[:sequence_length].detach().cpu().tolist()
            full_tokens = list(self.tokenizer.convert_ids_to_tokens(token_ids))
            valid_positions = [index for index, keep in enumerate(valid_mask_L.detach().cpu().tolist()) if keep]
            filtered_tokens = [full_tokens[index] for index in valid_positions]
            filtered_acts = [feature_score.per_token_activations[index] for index in valid_positions]
            filtered_argmax = int(np.argmax(filtered_acts)) if filtered_acts else 0
            max_token = (
                full_tokens[feature_score.argmax_token_index]
                if 0 <= feature_score.argmax_token_index < len(full_tokens)
                else "N/A"
            )

            results.append(
                {
                    "text": texts[batch_index],
                    "max_activation": feature_score.max_activation,
                    "mean_activation": feature_score.mean_activation,
                    "sum_activation": feature_score.sum_activation,
                    "argmax_token_index": filtered_argmax,
                    "max_token": max_token,
                    "tokens": filtered_tokens,
                    "per_token_activations": filtered_acts,
                    "hook_layer": hook_layer,
                }
            )

        del outputs, encoded
        return results


def evaluate_generation_ability(
    *,
    client: Any,
    explanation: str,
    record: dict[str, Any],
    scorer: LocalActivationScorer,
    threshold: float,
    llm_model: str,
    num_examples: int,
    progress_desc: str | None = None,
    generated_examples: list[str] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not explanation.strip():
        return {
            "metrics": {
                "total_examples": 0,
                "successful_examples": 0,
                "success_rate": 0.0,
                "avg_max_activation_all": 0.0,
                "avg_mean_activation_all": 0.0,
                "avg_max_activation_successful": 0.0,
                "avg_mean_activation_successful": 0.0,
            },
            "generated_examples": [],
            "detailed_results": [],
            "activation_threshold": threshold,
            "skipped": True,
            "error": "Empty explanation",
        }

    if generated_examples is None or token_usage is None:
        generated_examples, token_usage = _generate_texts_from_explanation(
            client=client,
            explanation=explanation,
            num_samples=num_examples,
            llm_model=llm_model,
        )
    detailed_results: list[dict[str, Any]] = []
    successful_count = 0
    total_max = 0.0
    total_mean = 0.0

    traces = scorer.score_texts_batch(texts=generated_examples, record=record)
    trace_iterator = traces
    if progress_desc is not None:
        trace_iterator = tqdm(
            traces,
            desc=progress_desc,
            leave=False,
            dynamic_ncols=True,
            unit="text",
        )

    for trace in trace_iterator:
        text = str(trace["text"])
        success = trace["max_activation"] >= threshold
        if success:
            successful_count += 1
        total_max += trace["max_activation"]
        total_mean += trace["mean_activation"]
        detailed_results.append(
            {
                "example": text,
                "max_activation": trace["max_activation"],
                "mean_activation": trace["mean_activation"],
                "sum_activation": trace["sum_activation"],
                "success": success,
                "max_token": trace["max_token"],
                "argmax_token_index": trace["argmax_token_index"],
                "tokens": trace["tokens"],
                "per_token_activations": trace["per_token_activations"],
            }
        )

    num_generated = len(generated_examples)
    success_rate = successful_count / num_generated if num_generated > 0 else 0.0
    avg_max = total_max / num_generated if num_generated > 0 else 0.0
    avg_mean = total_mean / num_generated if num_generated > 0 else 0.0
    successful_examples = [item for item in detailed_results if item["success"]]
    avg_max_success = (
        sum(item["max_activation"] for item in successful_examples) / len(successful_examples)
        if successful_examples
        else 0.0
    )
    avg_mean_success = (
        sum(item["mean_activation"] for item in successful_examples) / len(successful_examples)
        if successful_examples
        else 0.0
    )

    return {
        "metrics": {
            "total_examples": num_generated,
            "successful_examples": successful_count,
            "success_rate": success_rate,
            "avg_max_activation_all": avg_max,
            "avg_mean_activation_all": avg_mean,
            "avg_max_activation_successful": avg_max_success,
            "avg_mean_activation_successful": avg_mean_success,
        },
        "generated_examples": generated_examples,
        "detailed_results": detailed_results,
        "activation_threshold": threshold,
        "skipped": False,
        "token_usage": token_usage,
    }


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _json_dump_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_existing_feature_results(output_jsonl: Path | None) -> tuple[list[dict[str, Any]], set[str]]:
    if output_jsonl is None or not output_jsonl.exists():
        return [], set()

    latest_by_feature_id: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(output_jsonl):
        feature_id = str(row.get("feature_id") or "")
        if not feature_id:
            continue
        latest_by_feature_id[feature_id] = row
    existing_results = list(latest_by_feature_id.values())
    return existing_results, set(latest_by_feature_id.keys())


def _load_cached_generated_explanations(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    cached: dict[str, str] = {}
    for row in load_jsonl(path):
        feature_id = str(row.get("feature_id") or "")
        explanation = str(row.get("explanation") or row.get("dpo_explanation") or "").strip()
        if feature_id and explanation:
            cached[feature_id] = explanation
    return cached


def _load_feature_explanations(path: Path, field: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in load_jsonl(path):
        feature_id = str(row.get("feature_id") or "").strip()
        explanation = str(row.get(field) or "").strip()
        if feature_id and explanation:
            mapping[feature_id] = explanation
    return mapping


def _write_progress_snapshot(
    *,
    progress_json: Path | None,
    summary_payload: dict[str, Any],
) -> None:
    if progress_json is None:
        return
    progress_json.parent.mkdir(parents=True, exist_ok=True)
    progress_json.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_feature_result(
    *,
    record: dict[str, Any],
    dpo_explanation: str,
    threshold: float,
    threshold_info: dict[str, Any],
    dpo_generation_eval: dict[str, Any] | None,
    explanation_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "feature_id": record.get("feature_id"),
        "feature_url": record.get("feature_url"),
        "metadata": record.get("metadata") or {},
        "threshold": {
            "value": threshold,
            **threshold_info,
        },
        "dpo": {
            "explanation": dpo_explanation,
            "generation_evaluation": dpo_generation_eval,
        },
    }
    if explanation_source is not None:
        result["dpo"]["explanation_source"] = explanation_source
    return result


def _accumulate_feature_result_stats(
    feature_result: dict[str, Any],
    *,
    dpo_generation_rates: list[float],
    dpo_avg_max_activations: list[float],
    dpo_avg_mean_activations: list[float],
) -> None:
    dpo_generation_eval = feature_result.get("dpo", {}).get("generation_evaluation")
    if dpo_generation_eval is None:
        return

    metrics = dpo_generation_eval.get("metrics") or {}
    if "success_rate" in metrics:
        dpo_generation_rates.append(float(metrics["success_rate"]))
    if "avg_max_activation_all" in metrics:
        dpo_avg_max_activations.append(float(metrics["avg_max_activation_all"]))
    if "avg_mean_activation_all" in metrics:
        dpo_avg_mean_activations.append(float(metrics["avg_mean_activation_all"]))


def _build_summary_payload(
    *,
    sft_config_path: Path,
    checkpoint_dir: Path | None,
    input_jsonl: Path,
    output_jsonl: Path | None,
    generated_explanations_jsonl: Path | None,
    progress_json: Path | None,
    device: torch.device,
    num_generation_examples: int,
    generation_llm_model: str,
    fallback_activation_threshold: float,
    run_generation_eval: bool,
    generation_api_max_concurrency: int,
    sample_features: int | None,
    sample_seed: int,
    total_features: int,
    completed_features: int,
    dpo_generation_rates: list[float],
    dpo_avg_max_activations: list[float],
    dpo_avg_mean_activations: list[float],
    evaluation_source: str = "checkpoint",
    explanations_jsonl: Path | None = None,
    explanation_field: str | None = None,
) -> dict[str, Any]:
    return {
        "configuration": {
            "sft_config_path": str(sft_config_path),
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
            "input_jsonl": str(input_jsonl),
            "evaluation_source": evaluation_source,
            "explanations_jsonl": str(explanations_jsonl) if explanations_jsonl is not None else None,
            "explanation_field": explanation_field,
            "device": str(device),
            "num_generation_examples": num_generation_examples,
            "generation_llm_model": generation_llm_model,
            "fallback_activation_threshold": fallback_activation_threshold,
            "run_generation_eval": run_generation_eval,
            "generation_api_max_concurrency": generation_api_max_concurrency,
            "sample_features": sample_features,
            "sample_seed": sample_seed,
            "note": (
                "This evaluation measures only DPO explanations. It generates texts from each "
                "checkpoint-produced explanation, scores them with local hidden-state extraction "
                "plus JumpReLU SAE scoring, and reports activation-based generation metrics."
            ),
        },
        "summary_stats": {
            "total_features": total_features,
            "completed_features": completed_features,
            "remaining_features": max(total_features - completed_features, 0),
        },
        "average_metrics": {
            "dpo_generation_success_rate": _safe_mean(dpo_generation_rates),
            "dpo_avg_max_activation_all": _safe_mean(dpo_avg_max_activations),
            "dpo_avg_mean_activation_all": _safe_mean(dpo_avg_mean_activations),
        },
        "artifacts": {
            "output_jsonl": str(output_jsonl) if output_jsonl is not None else None,
            "generated_explanations_jsonl": (
                str(generated_explanations_jsonl) if generated_explanations_jsonl is not None else None
            ),
            "explanations_jsonl": str(explanations_jsonl) if explanations_jsonl is not None else None,
            "progress_json": str(progress_json) if progress_json is not None else None,
        },
    }


def evaluate_dpo_on_jsonl(
    *,
    sft_config_path: Path,
    checkpoint_dir: Path,
    input_jsonl: Path,
    project_root: Path,
    output_jsonl: Path | None = None,
    max_features: int | None = None,
    device: str | torch.device = "cuda",
    explainer_max_new_tokens: int | None = None,
    generation_llm_model: str = "gpt-5.4-mini",
    num_generation_examples: int = 10,
    fallback_activation_threshold: float = 8.0,
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_base_url: str | None = None,
    run_generation_eval: bool = True,
    generation_api_max_concurrency: int = 4,
    sample_features: int | None = None,
    sample_seed: int = 42,
    generated_explanations_jsonl: Path | None = None,
    progress_json: Path | None = None,
    explanation_batch_size: int = 8,
) -> dict[str, Any]:
    device = torch.device(device)

    sft_cfg = load_sft_config(sft_config_path, project_root)
    eval_tokenizer = load_tokenizer(sft_cfg.model)
    _, records, dataset = _build_eval_dataset(
        config_path=sft_config_path,
        input_jsonl=input_jsonl,
        project_root=project_root,
        tokenizer=eval_tokenizer,
    )
    del eval_tokenizer

    candidate_indices = list(range(len(records)))
    if max_features is not None:
        candidate_indices = candidate_indices[:max_features]
    if sample_features is not None:
        sample_size = min(int(sample_features), len(candidate_indices))
        rng = random.Random(sample_seed)
        candidate_indices = rng.sample(candidate_indices, k=sample_size)

    active_indices = candidate_indices
    active_records = [records[index] for index in active_indices]
    active_feature_ids = {str(record.get("feature_id") or "") for record in active_records}
    active_dataset_size = len(active_records)
    max_new_tokens = explainer_max_new_tokens or int(sft_cfg.eval.max_new_tokens)

    if generated_explanations_jsonl is None and output_jsonl is not None:
        generated_explanations_jsonl = output_jsonl.with_suffix(".generated_explanations.jsonl")
    if progress_json is None and output_jsonl is not None:
        progress_json = output_jsonl.with_suffix(".progress.json")
    if generated_explanations_jsonl is not None:
        generated_explanations_jsonl.parent.mkdir(parents=True, exist_ok=True)

    existing_feature_results, completed_feature_ids = _load_existing_feature_results(output_jsonl)
    cached_generated_explanations = _load_cached_generated_explanations(generated_explanations_jsonl)

    missing_generation_indices = [
        dataset_index
        for dataset_index in active_indices
        if str(records[dataset_index].get("feature_id") or "") not in completed_feature_ids
        and not cached_generated_explanations.get(str(records[dataset_index].get("feature_id") or ""), "").strip()
    ]

    generated_handle = (
        generated_explanations_jsonl.open("a", encoding="utf-8")
        if generated_explanations_jsonl is not None
        else None
    )

    if missing_generation_indices:
        _, explainer_tokenizer, explainer_model = load_checkpoint_model_for_eval(
            config_path=sft_config_path,
            checkpoint_dir=checkpoint_dir,
            project_root=project_root,
            device=device,
        )

        generation_progress = tqdm(
            total=len(missing_generation_indices),
            desc="Generating DPO explanations",
            dynamic_ncols=True,
            unit="feature",
        )
        try:
            batch_size = max(1, int(explanation_batch_size))
            for start in range(0, len(missing_generation_indices), batch_size):
                batch_indices = missing_generation_indices[start : start + batch_size]
                batch_records = [records[dataset_index] for dataset_index in batch_indices]
                batch_samples = [dataset.build_eval_sample(dataset_index) for dataset_index in batch_indices]
                dpo_explanations = _generate_explanations_batch(
                    model=explainer_model,
                    tokenizer=explainer_tokenizer,
                    samples=batch_samples,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                for record, dpo_explanation in zip(batch_records, dpo_explanations, strict=True):
                    feature_id = str(record.get("feature_id") or "")
                    cached_generated_explanations[feature_id] = dpo_explanation
                    if generated_handle is not None:
                        _json_dump_line(
                            generated_handle,
                            {
                                "feature_id": feature_id,
                                "explanation": dpo_explanation,
                                "checkpoint_dir": str(checkpoint_dir),
                            },
                        )
                if generated_handle is not None:
                    generated_handle.flush()
                generation_progress.update(len(batch_indices))
        finally:
            generation_progress.close()
            del explainer_model
            del explainer_tokenizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

    openai_client = None
    if run_generation_eval:
        openai_client = _build_openai_client(
            api_key_env=openai_api_key_env,
            base_url=openai_base_url,
        )

    scoring_model_name = _resolve_target_scoring_model_name(
        active_records,
        fallback_model_name=sft_cfg.model.model_name,
    )

    scorer = LocalActivationScorer(
        model_name=scoring_model_name,
        trust_remote_code=sft_cfg.model.trust_remote_code,
        torch_dtype=sft_cfg.model.torch_dtype,
        attn_implementation=sft_cfg.model.attn_implementation,
        max_length=sft_cfg.data.max_length,
        sae_cache_dir=project_root / "data" / "cache" / "hf_saes",
        device=device,
    )

    dpo_generation_rates: list[float] = []
    dpo_avg_max_activations: list[float] = []
    dpo_avg_mean_activations: list[float] = []
    for feature_result in existing_feature_results:
        _accumulate_feature_result_stats(
            feature_result,
            dpo_generation_rates=dpo_generation_rates,
            dpo_avg_max_activations=dpo_avg_max_activations,
            dpo_avg_mean_activations=dpo_avg_mean_activations,
        )

    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = output_jsonl.open("a", encoding="utf-8")
    else:
        jsonl_handle = None

    try:
        eval_progress = tqdm(
            total=active_dataset_size,
            desc="Running DPO evaluation",
            initial=len(completed_feature_ids & active_feature_ids),
            dynamic_ncols=True,
            unit="feature",
        )
        for index, dataset_index in enumerate(active_indices):
            record = active_records[index]
            feature_id = str(record.get("feature_id") or "")
            if feature_id in completed_feature_ids:
                continue

            feature_label = feature_id.split("/")[-1] or "unknown"
            eval_progress.set_postfix(feature=feature_label)

            dpo_explanation = cached_generated_explanations.get(feature_id, "").strip()
            threshold, threshold_info = _compute_dynamic_threshold(
                record,
                fallback_threshold=fallback_activation_threshold,
            )

            dpo_generation_eval = None
            if run_generation_eval:
                generation_outputs = _generate_texts_batch_from_explanations(
                    client=openai_client,
                    jobs=[("dpo", dpo_explanation, num_generation_examples)],
                    llm_model=generation_llm_model,
                    max_concurrency=max(1, generation_api_max_concurrency),
                    progress_desc=f"Gen API {feature_label}",
                )
                dpo_generation_eval = evaluate_generation_ability(
                    client=openai_client,
                    explanation=dpo_explanation,
                    record=record,
                    scorer=scorer,
                    threshold=threshold,
                    llm_model=generation_llm_model,
                    num_examples=num_generation_examples,
                    progress_desc=f"DPO gen-score {feature_label}",
                    generated_examples=generation_outputs.get("dpo", (None, None))[0],
                    token_usage=generation_outputs.get("dpo", (None, None))[1],
                )

            feature_result = _build_feature_result(
                record=record,
                dpo_explanation=dpo_explanation,
                threshold=threshold,
                threshold_info=threshold_info,
                dpo_generation_eval=dpo_generation_eval,
                explanation_source={
                    "mode": "checkpoint",
                    "checkpoint_dir": str(checkpoint_dir),
                },
            )

            _accumulate_feature_result_stats(
                feature_result,
                dpo_generation_rates=dpo_generation_rates,
                dpo_avg_max_activations=dpo_avg_max_activations,
                dpo_avg_mean_activations=dpo_avg_mean_activations,
            )
            completed_feature_ids.add(feature_id)

            if jsonl_handle is not None:
                _json_dump_line(jsonl_handle, feature_result)
                jsonl_handle.flush()

            summary_payload = _build_summary_payload(
                sft_config_path=sft_config_path,
                checkpoint_dir=checkpoint_dir,
                input_jsonl=input_jsonl,
                output_jsonl=output_jsonl,
                generated_explanations_jsonl=generated_explanations_jsonl,
                progress_json=progress_json,
                device=device,
                num_generation_examples=num_generation_examples,
                generation_llm_model=generation_llm_model,
                fallback_activation_threshold=fallback_activation_threshold,
                run_generation_eval=run_generation_eval,
                generation_api_max_concurrency=generation_api_max_concurrency,
                sample_features=sample_features,
                sample_seed=sample_seed,
                total_features=active_dataset_size,
                completed_features=len(completed_feature_ids & active_feature_ids),
                dpo_generation_rates=dpo_generation_rates,
                dpo_avg_max_activations=dpo_avg_max_activations,
                dpo_avg_mean_activations=dpo_avg_mean_activations,
            )
            _write_progress_snapshot(progress_json=progress_json, summary_payload=summary_payload)
            eval_progress.update(1)
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
        if generated_handle is not None:
            generated_handle.close()
        scorer.close()

    final_summary = _build_summary_payload(
        sft_config_path=sft_config_path,
        checkpoint_dir=checkpoint_dir,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        generated_explanations_jsonl=generated_explanations_jsonl,
        progress_json=progress_json,
        device=device,
        num_generation_examples=num_generation_examples,
        generation_llm_model=generation_llm_model,
        fallback_activation_threshold=fallback_activation_threshold,
        run_generation_eval=run_generation_eval,
        generation_api_max_concurrency=generation_api_max_concurrency,
        sample_features=sample_features,
        sample_seed=sample_seed,
        total_features=active_dataset_size,
        completed_features=len(completed_feature_ids & active_feature_ids),
        dpo_generation_rates=dpo_generation_rates,
        dpo_avg_max_activations=dpo_avg_max_activations,
        dpo_avg_mean_activations=dpo_avg_mean_activations,
    )
    _write_progress_snapshot(progress_json=progress_json, summary_payload=final_summary)
    return final_summary


def evaluate_explanations_on_jsonl(
    *,
    sft_config_path: Path,
    input_jsonl: Path,
    explanations_jsonl: Path,
    explanation_field: str,
    project_root: Path,
    output_jsonl: Path | None = None,
    max_features: int | None = None,
    device: str | torch.device = "cuda",
    generation_llm_model: str = "gpt-5.4-mini",
    num_generation_examples: int = 10,
    fallback_activation_threshold: float = 8.0,
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_base_url: str | None = None,
    run_generation_eval: bool = True,
    generation_api_max_concurrency: int = 4,
    sample_features: int | None = None,
    sample_seed: int = 42,
    progress_json: Path | None = None,
) -> dict[str, Any]:
    device = torch.device(device)
    sft_cfg = load_sft_config(sft_config_path, project_root)
    records = load_jsonl(input_jsonl)
    explanations_by_feature_id = _load_feature_explanations(explanations_jsonl, explanation_field)

    candidate_indices = list(range(len(records)))
    if max_features is not None:
        candidate_indices = candidate_indices[:max_features]
    if sample_features is not None:
        sample_size = min(int(sample_features), len(candidate_indices))
        rng = random.Random(sample_seed)
        candidate_indices = rng.sample(candidate_indices, k=sample_size)

    active_records = [records[index] for index in candidate_indices]
    active_feature_ids = {str(record.get("feature_id") or "") for record in active_records}
    active_dataset_size = len(active_records)

    existing_feature_results, completed_feature_ids = _load_existing_feature_results(output_jsonl)

    openai_client = None
    if run_generation_eval:
        openai_client = _build_openai_client(
            api_key_env=openai_api_key_env,
            base_url=openai_base_url,
        )

    scoring_model_name = _resolve_target_scoring_model_name(
        active_records,
        fallback_model_name=sft_cfg.model.model_name,
    )

    scorer = LocalActivationScorer(
        model_name=scoring_model_name,
        trust_remote_code=sft_cfg.model.trust_remote_code,
        torch_dtype=sft_cfg.model.torch_dtype,
        attn_implementation=sft_cfg.model.attn_implementation,
        max_length=sft_cfg.data.max_length,
        sae_cache_dir=project_root / "data" / "cache" / "hf_saes",
        device=device,
    )

    dpo_generation_rates: list[float] = []
    dpo_avg_max_activations: list[float] = []
    dpo_avg_mean_activations: list[float] = []
    for feature_result in existing_feature_results:
        _accumulate_feature_result_stats(
            feature_result,
            dpo_generation_rates=dpo_generation_rates,
            dpo_avg_max_activations=dpo_avg_max_activations,
            dpo_avg_mean_activations=dpo_avg_mean_activations,
        )

    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = output_jsonl.open("a", encoding="utf-8")
    else:
        jsonl_handle = None

    try:
        eval_progress = tqdm(
            total=active_dataset_size,
            desc="Running explanation-file evaluation",
            initial=len(completed_feature_ids & active_feature_ids),
            dynamic_ncols=True,
            unit="feature",
        )
        for record in active_records:
            feature_id = str(record.get("feature_id") or "")
            if feature_id in completed_feature_ids:
                continue

            feature_label = feature_id.split("/")[-1] or "unknown"
            eval_progress.set_postfix(feature=feature_label)

            dpo_explanation = explanations_by_feature_id.get(feature_id, "").strip()
            threshold, threshold_info = _compute_dynamic_threshold(
                record,
                fallback_threshold=fallback_activation_threshold,
            )

            dpo_generation_eval = None
            if run_generation_eval:
                generation_outputs = _generate_texts_batch_from_explanations(
                    client=openai_client,
                    jobs=[("dpo", dpo_explanation, num_generation_examples)],
                    llm_model=generation_llm_model,
                    max_concurrency=max(1, generation_api_max_concurrency),
                    progress_desc=f"Gen API {feature_label}",
                )
                dpo_generation_eval = evaluate_generation_ability(
                    client=openai_client,
                    explanation=dpo_explanation,
                    record=record,
                    scorer=scorer,
                    threshold=threshold,
                    llm_model=generation_llm_model,
                    num_examples=num_generation_examples,
                    progress_desc=f"DPO gen-score {feature_label}",
                    generated_examples=generation_outputs.get("dpo", (None, None))[0],
                    token_usage=generation_outputs.get("dpo", (None, None))[1],
                )

            feature_result = _build_feature_result(
                record=record,
                dpo_explanation=dpo_explanation,
                threshold=threshold,
                threshold_info=threshold_info,
                dpo_generation_eval=dpo_generation_eval,
                explanation_source={
                    "mode": "explanation_file",
                    "explanations_jsonl": str(explanations_jsonl),
                    "explanation_field": explanation_field,
                },
            )

            _accumulate_feature_result_stats(
                feature_result,
                dpo_generation_rates=dpo_generation_rates,
                dpo_avg_max_activations=dpo_avg_max_activations,
                dpo_avg_mean_activations=dpo_avg_mean_activations,
            )
            completed_feature_ids.add(feature_id)

            if jsonl_handle is not None:
                _json_dump_line(jsonl_handle, feature_result)
                jsonl_handle.flush()

            summary_payload = _build_summary_payload(
                sft_config_path=sft_config_path,
                checkpoint_dir=None,
                input_jsonl=input_jsonl,
                output_jsonl=output_jsonl,
                generated_explanations_jsonl=None,
                progress_json=progress_json,
                device=device,
                num_generation_examples=num_generation_examples,
                generation_llm_model=generation_llm_model,
                fallback_activation_threshold=fallback_activation_threshold,
                run_generation_eval=run_generation_eval,
                generation_api_max_concurrency=generation_api_max_concurrency,
                sample_features=sample_features,
                sample_seed=sample_seed,
                total_features=active_dataset_size,
                completed_features=len(completed_feature_ids & active_feature_ids),
                dpo_generation_rates=dpo_generation_rates,
                dpo_avg_max_activations=dpo_avg_max_activations,
                dpo_avg_mean_activations=dpo_avg_mean_activations,
                evaluation_source="explanation_file",
                explanations_jsonl=explanations_jsonl,
                explanation_field=explanation_field,
            )
            _write_progress_snapshot(progress_json=progress_json, summary_payload=summary_payload)
            eval_progress.update(1)
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
        scorer.close()

    final_summary = _build_summary_payload(
        sft_config_path=sft_config_path,
        checkpoint_dir=None,
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        generated_explanations_jsonl=None,
        progress_json=progress_json,
        device=device,
        num_generation_examples=num_generation_examples,
        generation_llm_model=generation_llm_model,
        fallback_activation_threshold=fallback_activation_threshold,
        run_generation_eval=run_generation_eval,
        generation_api_max_concurrency=generation_api_max_concurrency,
        sample_features=sample_features,
        sample_seed=sample_seed,
        total_features=active_dataset_size,
        completed_features=len(completed_feature_ids & active_feature_ids),
        dpo_generation_rates=dpo_generation_rates,
        dpo_avg_max_activations=dpo_avg_max_activations,
        dpo_avg_mean_activations=dpo_avg_mean_activations,
        evaluation_source="explanation_file",
        explanations_jsonl=explanations_jsonl,
        explanation_field=explanation_field,
    )
    _write_progress_snapshot(progress_json=progress_json, summary_payload=final_summary)
    return final_summary
