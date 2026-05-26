"""Vec-only candidate generation, activation scoring, and preference construction."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.data_collection.sae_vectors import SAEFeatureLocator, load_jsonl, locator_from_record
from src.preference.config import PreferencePipelineConfig
from src.preference.prompts import build_generator_prompt
from src.preference.sae import JumpReluSAEStore
from src.sft.config import load_sft_config
from src.sft.dataset import SAEExplainerSFTDataset
from src.sft.eval import load_checkpoint_model_for_eval
from src.sft.model import resolve_injection_submodule
from src.sft.train_utils import resolve_torch_dtype

PIPELINE_STAGES = (
    "candidates",
    "generated_texts",
    "scored_texts",
    "scored_explanations",
    "preferences",
)


def _configure_low_cpu_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("RAYON_NUM_THREADS", "1")

    torch_num_threads = int(os.environ.get("SAE_EXPLAINER_TORCH_NUM_THREADS", "1"))
    torch_num_interop_threads = int(os.environ.get("SAE_EXPLAINER_TORCH_INTEROP_THREADS", "1"))
    try:
        torch.set_num_threads(max(1, torch_num_threads))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, torch_num_interop_threads))
    except Exception:
        pass


@dataclass(frozen=True)
class CandidateExplanation:
    feature_id: str
    candidate_id: int
    explanation: str
    target_explanation: str
    layer: int
    feature_index: int


@dataclass(frozen=True)
class GeneratedText:
    feature_id: str
    candidate_id: int
    sample_id: int
    explanation: str
    generated_text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    request_duration_seconds: float | None = None
    request_sample_count: int | None = None


@dataclass(frozen=True)
class TextScore:
    feature_id: str
    candidate_id: int
    sample_id: int
    explanation: str
    generated_text: str
    max_token_activation: float
    mean_activation: float
    sum_activation: float
    argmax_token_index: int


@dataclass(frozen=True)
class ExplanationScore:
    feature_id: str
    candidate_id: int
    explanation: str
    score: float
    best_text: str
    best_text_score: float
    best_text_mean_activation: float
    best_text_sum_activation: float
    best_text_argmax_token_index: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _CachedScoredTextRow:
    generated_text_hash: str
    raw_line: str


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stream_jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", buffering=1)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_jsonl_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue
            yield raw_line


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _iter_jsonl_chunks(path: Path, chunk_size: int):
    chunk: list[dict[str, Any]] = []
    for row in _iter_jsonl(path):
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _candidate_key(feature_id: str, candidate_id: int) -> tuple[str, int]:
    return (feature_id, int(candidate_id))


def _generated_text_key(feature_id: str, candidate_id: int, sample_id: int) -> tuple[str, int, int]:
    return (feature_id, int(candidate_id), int(sample_id))


def _hash_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _stage_index(stage: str | None) -> int:
    if stage in (None, "start"):
        return -1
    if stage not in PIPELINE_STAGES:
        raise ValueError(f"Unknown pipeline stage: {stage}. Expected one of: {('start',) + PIPELINE_STAGES}")
    return PIPELINE_STAGES.index(stage)


def _ensure_exists(path: Path, stage_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot resume from stage '{stage_name}' because required file does not exist: {path}"
        )


def _select_records(input_jsonl: Path, max_features: int | None) -> list[dict[str, Any]]:
    records = load_jsonl(input_jsonl)
    if max_features is not None:
        return records[:max_features]
    return records


def _build_explainer_eval_dataset(
    cfg_path: Path,
    input_jsonl: Path,
    project_root: Path,
    tokenizer: Any,
):
    sft_cfg = load_sft_config(cfg_path, project_root)
    records = _select_records(input_jsonl, max_features=None)
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
    return sft_cfg, records, dataset


def _generate_one_completion(
    model: Any,
    tokenizer: Any,
    prompt_input_ids: list[int],
    feature_vector: Any,
    act_position: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str | torch.device = "cuda",
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
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion_ids = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def _generate_repeated_completions(
    model: Any,
    tokenizer: Any,
    prompt_input_ids: list[int],
    feature_vector: Any,
    act_position: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_completions: int,
    device: str | torch.device = "cuda",
) -> list[str]:
    if num_completions <= 0:
        return []
    if num_completions == 1:
        return [
            _generate_one_completion(
                model=model,
                tokenizer=tokenizer,
                prompt_input_ids=prompt_input_ids,
                feature_vector=feature_vector,
                act_position=act_position,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )
        ]

    prompt_tensor = torch.tensor(prompt_input_ids, dtype=torch.long, device=device)
    input_ids = prompt_tensor.unsqueeze(0).repeat(num_completions, 1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    feature_vector_tensor = torch.tensor(feature_vector, dtype=torch.float32, device=device)
    feature_vectors = feature_vector_tensor.unsqueeze(0).repeat(num_completions, 1)
    act_positions = torch.full((num_completions,), act_position, dtype=torch.long, device=device)
    inject_mask = torch.ones((num_completions,), dtype=torch.bool, device=device)

    generated = model.generate_with_injection(
        input_ids=input_ids,
        attention_mask=attention_mask,
        feature_vectors=feature_vectors,
        act_positions=act_positions,
        inject_mask=inject_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_length = input_ids.shape[1]
    completions: list[str] = []
    for row_index in range(num_completions):
        completion_ids = generated[row_index, prompt_length:]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
    return completions


def _generate_batched_completions(
    model: Any,
    tokenizer: Any,
    batch_samples: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str | torch.device = "cuda",
) -> list[list[str]]:
    if not batch_samples:
        return []

    expanded_samples: list[tuple[int, dict[str, Any]]] = []
    for sample_index, sample in enumerate(batch_samples):
        num_completions = int(sample["num_completions"])
        for _ in range(num_completions):
            expanded_samples.append((sample_index, sample))

    if len(expanded_samples) == 1:
        _, sample = expanded_samples[0]
        return [[
            _generate_one_completion(
                model=model,
                tokenizer=tokenizer,
                prompt_input_ids=sample["prompt_input_ids"],
                feature_vector=sample["feature_vector"],
                act_position=int(sample["act_position"]),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )
        ]]

    max_prompt_len = max(len(sample["prompt_input_ids"]) for _, sample in expanded_samples)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for batched generation.")

    input_ids_rows: list[list[int]] = []
    attention_mask_rows: list[list[int]] = []
    feature_vectors_rows: list[Any] = []
    act_positions_rows: list[int] = []
    inject_mask_rows: list[bool] = []

    for _, sample in expanded_samples:
        prompt_ids = list(sample["prompt_input_ids"])
        pad_len = max_prompt_len - len(prompt_ids)
        input_ids_rows.append([pad_token_id] * pad_len + prompt_ids)
        attention_mask_rows.append([0] * pad_len + [1] * len(prompt_ids))
        feature_vectors_rows.append(sample["feature_vector"])
        act_positions_rows.append(pad_len + int(sample["act_position"]))
        inject_mask_rows.append(True)

    input_ids = torch.tensor(input_ids_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask_rows, dtype=torch.long, device=device)
    feature_vectors_np = np.stack(feature_vectors_rows).astype(np.float32, copy=False)
    feature_vectors = torch.from_numpy(feature_vectors_np).to(device)
    act_positions = torch.tensor(act_positions_rows, dtype=torch.long, device=device)
    inject_mask = torch.tensor(inject_mask_rows, dtype=torch.bool, device=device)

    generated = model.generate_with_injection(
        input_ids=input_ids,
        attention_mask=attention_mask,
        feature_vectors=feature_vectors,
        act_positions=act_positions,
        inject_mask=inject_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    completions_by_sample: list[list[str]] = [[] for _ in batch_samples]
    for row_index, (sample_index, _) in enumerate(expanded_samples):
        completion_ids = generated[row_index, max_prompt_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        completions_by_sample[sample_index].append(completion)
    return completions_by_sample


def generate_candidate_explanations(
    cfg: PreferencePipelineConfig,
    project_root: Path,
    output_path: Path,
) -> int:
    _, tokenizer, model = load_checkpoint_model_for_eval(
        config_path=cfg.explainer.sft_config_path,
        checkpoint_dir=cfg.explainer.checkpoint_dir,
        project_root=project_root,
        device="cuda",
    )
    _, records, dataset = _build_explainer_eval_dataset(
        cfg_path=cfg.explainer.sft_config_path,
        input_jsonl=cfg.data.input_jsonl,
        project_root=project_root,
        tokenizer=tokenizer,
    )

    active_records = records[: cfg.data.max_features] if cfg.data.max_features else records
    existing_rows: dict[tuple[str, int], dict[str, Any]] = {}
    if output_path.exists():
        for row in _iter_jsonl(output_path):
            existing_rows[_candidate_key(str(row["feature_id"]), int(row["candidate_id"]))] = row

    candidate_count = 0
    generation_batch_size = max(1, int(cfg.explainer.generation_batch_size))
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with _stream_jsonl_writer(temp_path) as writer:
        pending_batch: list[dict[str, Any]] = []

        def flush_pending_batch() -> None:
            nonlocal candidate_count
            if not pending_batch:
                return

            batch_inputs = [
                {
                    "prompt_input_ids": item["sample"]["prompt_input_ids"],
                    "feature_vector": item["sample"]["feature_vector"],
                    "act_position": int(item["sample"]["act_position"]),
                    "num_completions": len(item["missing_candidate_ids"]),
                }
                for item in pending_batch
            ]
            completions_by_item = _generate_batched_completions(
                model=model,
                tokenizer=tokenizer,
                batch_samples=batch_inputs,
                max_new_tokens=cfg.explainer.max_new_tokens,
                temperature=cfg.explainer.temperature,
                top_p=cfg.explainer.top_p,
            )

            for item, explanations in zip(pending_batch, completions_by_item, strict=True):
                generated_rows_by_candidate: dict[int, dict[str, Any]] = {}
                metadata = item["record"]["metadata"]
                for candidate_id, explanation in zip(item["missing_candidate_ids"], explanations, strict=True):
                    candidate = CandidateExplanation(
                        feature_id=item["record"]["feature_id"],
                        candidate_id=candidate_id,
                        explanation=explanation,
                        target_explanation=item["record"]["explanation"],
                        layer=int(metadata["layer"]),
                        feature_index=int(metadata["feature_index"]),
                    )
                    generated_rows_by_candidate[candidate_id] = candidate.__dict__

                for candidate_id in range(cfg.explainer.num_candidates):
                    row = item["cached_rows_by_candidate"].get(candidate_id) or generated_rows_by_candidate.get(candidate_id)
                    if row is None:
                        raise RuntimeError(
                            "Missing generated candidate row for "
                            f"feature_id={item['record']['feature_id']} candidate_id={candidate_id}"
                        )
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    candidate_count += 1

            pending_batch.clear()

        for index, record in enumerate(tqdm(active_records, desc="Generating explanation candidates")):
            sample = dataset.build_eval_sample(index)
            metadata = record["metadata"]
            cached_rows_by_candidate: dict[int, dict[str, Any]] = {}
            missing_candidate_ids: list[int] = []
            for candidate_id in range(cfg.explainer.num_candidates):
                key = _candidate_key(record["feature_id"], candidate_id)
                existing = existing_rows.get(key)
                if existing is not None:
                    cached_rows_by_candidate[candidate_id] = existing
                    continue
                missing_candidate_ids.append(candidate_id)

            if not missing_candidate_ids:
                for candidate_id in range(cfg.explainer.num_candidates):
                    row = cached_rows_by_candidate.get(candidate_id)
                    if row is None:
                        raise RuntimeError(
                            f"Missing cached candidate row for feature_id={record['feature_id']} candidate_id={candidate_id}"
                        )
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    candidate_count += 1
                continue

            pending_batch.append(
                {
                    "record": record,
                    "sample": sample,
                    "cached_rows_by_candidate": cached_rows_by_candidate,
                    "missing_candidate_ids": missing_candidate_ids,
                }
            )
            if len(pending_batch) >= generation_batch_size:
                flush_pending_batch()

        flush_pending_batch()
    temp_path.replace(output_path)
    del model
    torch.cuda.empty_cache()
    return candidate_count


def _build_generator_client(
    cfg: PreferencePipelineConfig,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: openai. Install it first, for example:\n"
            "  pip install -U openai"
        ) from exc

    api_key_env = cfg.generator.api_key_env or "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env!r} is not set.")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if cfg.generator.api_base_url:
        client_kwargs["base_url"] = cfg.generator.api_base_url
    return OpenAI(**client_kwargs)


def _call_generator_api(
    client: Any,
    cfg: PreferencePipelineConfig,
    prompt: str,
) -> dict[str, Any]:

    model_name = cfg.generator.api_model_name or cfg.generator.model_name
    start_time = time.perf_counter()
    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=cfg.generator.temperature,
        top_p=cfg.generator.top_p,
        max_output_tokens=cfg.generator.max_new_tokens,
        store=False,
    )
    duration_seconds = time.perf_counter() - start_time
    prompt_tokens, completion_tokens, total_tokens = _extract_response_usage(response)
    return {
        "output_text": response.output_text.strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "request_duration_seconds": duration_seconds,
    }


def _extract_response_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None, None, None

    def _read(obj: Any, *keys: str) -> int | None:
        for key in keys:
            if isinstance(obj, dict) and obj.get(key) is not None:
                return int(obj[key])
            value = getattr(obj, key, None)
            if value is not None:
                return int(value)
        return None

    prompt_tokens = _read(usage, "input_tokens", "prompt_tokens")
    completion_tokens = _read(usage, "output_tokens", "completion_tokens")
    total_tokens = _read(usage, "total_tokens")
    return prompt_tokens, completion_tokens, total_tokens


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

    # Fallback: if the model ignored JSON instructions, keep the whole text as one sample.
    return [raw_text]


def generate_texts_from_explanations(
    cfg: PreferencePipelineConfig,
    candidates_path: Path,
    output_path: Path,
) -> int:
    results_count = 0
    client = _build_generator_client(cfg)
    max_workers = max(1, cfg.generator.max_concurrency)
    existing_rows: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    if output_path.exists():
        for row in _iter_jsonl(output_path):
            key = _candidate_key(str(row["feature_id"]), int(row["candidate_id"]))
            existing_rows.setdefault(key, {})[int(row["sample_id"])] = row

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with _stream_jsonl_writer(temp_path) as writer:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            iterator = tqdm(
                _iter_jsonl(candidates_path),
                total=_count_jsonl_rows(candidates_path),
                desc="Generating texts from explanations",
            )
            in_flight = []

            def submit_candidate(candidate: CandidateExplanation):
                prompt = build_generator_prompt(
                    candidate.explanation,
                    cfg.generator.num_texts_per_explanation,
                )
                future = executor.submit(_call_generator_api, client, cfg, prompt)
                in_flight.append((candidate, future))

            for row in iterator:
                candidate = CandidateExplanation(**row)
                key = _candidate_key(candidate.feature_id, candidate.candidate_id)
                existing = existing_rows.get(key, {})
                if all(sample_id in existing for sample_id in range(cfg.generator.num_texts_per_explanation)):
                    for sample_id in range(cfg.generator.num_texts_per_explanation):
                        writer.write(json.dumps(existing[sample_id], ensure_ascii=False) + "\n")
                        results_count += 1
                    continue
                submit_candidate(candidate)

                if len(in_flight) >= max_workers:
                    candidate_done, future_done = in_flight.pop(0)
                    response_payload = future_done.result()
                    samples = _parse_generator_samples(str(response_payload["output_text"]))
                    samples = samples[: cfg.generator.num_texts_per_explanation]
                    for sample_id, text in enumerate(samples):
                        result = GeneratedText(
                            feature_id=candidate_done.feature_id,
                            candidate_id=candidate_done.candidate_id,
                            sample_id=sample_id,
                            explanation=candidate_done.explanation,
                            generated_text=text,
                            prompt_tokens=response_payload["prompt_tokens"] if sample_id == 0 else None,
                            completion_tokens=response_payload["completion_tokens"] if sample_id == 0 else None,
                            total_tokens=response_payload["total_tokens"] if sample_id == 0 else None,
                            request_duration_seconds=(
                                response_payload["request_duration_seconds"] if sample_id == 0 else None
                            ),
                            request_sample_count=len(samples) if sample_id == 0 else None,
                        )
                        writer.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
                        results_count += 1

            while in_flight:
                candidate_done, future_done = in_flight.pop(0)
                response_payload = future_done.result()
                samples = _parse_generator_samples(str(response_payload["output_text"]))
                samples = samples[: cfg.generator.num_texts_per_explanation]
                for sample_id, text in enumerate(samples):
                    result = GeneratedText(
                        feature_id=candidate_done.feature_id,
                        candidate_id=candidate_done.candidate_id,
                        sample_id=sample_id,
                        explanation=candidate_done.explanation,
                        generated_text=text,
                        prompt_tokens=response_payload["prompt_tokens"] if sample_id == 0 else None,
                        completion_tokens=response_payload["completion_tokens"] if sample_id == 0 else None,
                        total_tokens=response_payload["total_tokens"] if sample_id == 0 else None,
                        request_duration_seconds=(
                            response_payload["request_duration_seconds"] if sample_id == 0 else None
                        ),
                        request_sample_count=len(samples) if sample_id == 0 else None,
                    )
                    writer.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
                    results_count += 1
    temp_path.replace(output_path)
    return results_count


def _load_target_model(cfg: PreferencePipelineConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.target_model.model_name,
        trust_remote_code=cfg.target_model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": cfg.target_model.model_name,
        "trust_remote_code": cfg.target_model.trust_remote_code,
        "torch_dtype": resolve_torch_dtype(cfg.target_model.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if cfg.target_model.attn_implementation:
        kwargs["attn_implementation"] = cfg.target_model.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(**kwargs).to("cuda").eval()
    return tokenizer, model


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


def _extract_hidden_states_for_hook_layer(
    model: Any,
    encoded: dict[str, torch.Tensor],
    hook_layer: int,
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden_states"] = hidden

    target_module = resolve_injection_submodule(model, hook_layer)
    handle = target_module.register_forward_hook(hook_fn)
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()

    hidden_states = captured.get("hidden_states")
    if hidden_states is None:
        raise RuntimeError(f"Failed to capture hidden states for hook_layer={hook_layer}")
    return hidden_states


def _score_generated_text_batch(
    cfg: PreferencePipelineConfig,
    *,
    tokenizer: Any,
    model: Any,
    sae_store: JumpReluSAEStore,
    special_ids: set[int],
    batch_items: list[tuple[int, GeneratedText, dict[str, Any], SAEFeatureLocator]],
) -> dict[int, TextScore]:
    texts = [item.generated_text for _, item, _, _ in batch_items]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.target_model.max_length,
        return_attention_mask=True,
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}

    input_ids_BL = encoded["input_ids"]
    valid_mask_BL = _build_valid_token_mask(
        input_ids_BL=input_ids_BL,
        attention_mask_BL=encoded["attention_mask"],
        special_ids=special_ids,
    )

    text_scores: dict[int, TextScore] = {}
    batch_indices_by_hook_layer: dict[int, list[int]] = defaultdict(list)
    for batch_index, (_, _, record, _) in enumerate(batch_items):
        hook_layer = int(record["metadata"]["hook_layer"])
        batch_indices_by_hook_layer[hook_layer].append(batch_index)

    hidden_states_by_hook_layer: dict[int, torch.Tensor] = {}
    if len(batch_indices_by_hook_layer) == 1:
        hook_layer = next(iter(batch_indices_by_hook_layer))
        hidden_states_by_hook_layer[hook_layer] = _extract_hidden_states_for_hook_layer(model, encoded, hook_layer)
    else:
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        for hook_layer in batch_indices_by_hook_layer:
            hidden_states_by_hook_layer[hook_layer] = outputs.hidden_states[hook_layer + 1]
        del outputs

    for batch_index, (chunk_index, item, record, locator) in enumerate(batch_items):
        hook_layer = int(record["metadata"]["hook_layer"])
        sequence_length = int(encoded["attention_mask"][batch_index].sum().item())
        hidden_states_LD = hidden_states_by_hook_layer[hook_layer][batch_index, :sequence_length]
        feature_score = sae_store.score_feature(
            hidden_states_LD=hidden_states_LD,
            locator=locator,
            valid_token_mask_L=valid_mask_BL[batch_index, :sequence_length],
        )
        text_scores[chunk_index] = TextScore(
            feature_id=item.feature_id,
            candidate_id=item.candidate_id,
            sample_id=item.sample_id,
            explanation=item.explanation,
            generated_text=item.generated_text,
            max_token_activation=feature_score.max_activation,
            mean_activation=feature_score.mean_activation,
            sum_activation=feature_score.sum_activation,
            argmax_token_index=feature_score.argmax_token_index,
        )

    del hidden_states_by_hook_layer, encoded, input_ids_BL, valid_mask_BL
    return text_scores


def _text_score_to_explanation_score(
    text_score: TextScore,
    records_by_feature_id: dict[str, dict[str, Any]] | None = None,
) -> ExplanationScore:
    return ExplanationScore(
        feature_id=text_score.feature_id,
        candidate_id=text_score.candidate_id,
        explanation=text_score.explanation,
        # Keep max_activation as the ranking score for preference construction.
        score=text_score.max_token_activation,
        best_text=text_score.generated_text,
        best_text_score=text_score.max_token_activation,
        best_text_mean_activation=text_score.mean_activation,
        best_text_sum_activation=text_score.sum_activation,
        best_text_argmax_token_index=text_score.argmax_token_index,
        metadata=dict((records_by_feature_id or {}).get(text_score.feature_id, {}).get("metadata", {})),
    )


def score_generated_texts(
    cfg: PreferencePipelineConfig,
    records_by_feature_id: dict[str, dict[str, Any]],
    generated_texts_path: Path,
    sae_cache_dir: Path,
    output_path: Path,
) -> int:
    tokenizer, model = _load_target_model(cfg)
    sae_store = JumpReluSAEStore(local_dir=sae_cache_dir)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    chunk_size = max(1, cfg.target_model.score_chunk_size)
    batch_size = max(1, cfg.target_model.score_batch_size)
    locator_by_feature_id = {
        feature_id: locator_from_record(record)
        for feature_id, record in records_by_feature_id.items()
    }

    score_count = 0
    existing_rows: dict[tuple[str, int, int], _CachedScoredTextRow] = {}
    if output_path.exists():
        for raw_line in _iter_jsonl_lines(output_path):
            row = json.loads(raw_line)
            key = _generated_text_key(
                str(row["feature_id"]),
                int(row["candidate_id"]),
                int(row["sample_id"]),
            )
            existing_rows[key] = _CachedScoredTextRow(
                generated_text_hash=_hash_text(str(row["generated_text"])),
                raw_line=raw_line,
            )

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_rows = _count_jsonl_rows(generated_texts_path)
    with _stream_jsonl_writer(temp_path) as writer:
        with tqdm(total=total_rows, desc="Scoring generated texts") as pbar:
            for chunk in _iter_jsonl_chunks(generated_texts_path, chunk_size):
                chunk_lines: list[str | None] = [None] * len(chunk)
                batch_items: list[tuple[int, GeneratedText, dict[str, Any], SAEFeatureLocator]] = []

                for chunk_index, row in enumerate(chunk):
                    item = GeneratedText(**row)
                    key = _generated_text_key(item.feature_id, item.candidate_id, item.sample_id)
                    existing = existing_rows.get(key)
                    if existing is not None and existing.generated_text_hash == _hash_text(item.generated_text):
                        chunk_lines[chunk_index] = existing.raw_line
                        continue

                    record = records_by_feature_id[item.feature_id]
                    locator = locator_by_feature_id[item.feature_id]
                    batch_items.append(
                        (
                            chunk_index,
                            item,
                            record,
                            locator,
                        )
                    )

                for batch_start in range(0, len(batch_items), batch_size):
                    batch_chunk = batch_items[batch_start : batch_start + batch_size]
                    chunk_scores = _score_generated_text_batch(
                        cfg,
                        tokenizer=tokenizer,
                        model=model,
                        sae_store=sae_store,
                        special_ids=special_ids,
                        batch_items=batch_chunk,
                    )
                    for chunk_index, text_score in chunk_scores.items():
                        chunk_lines[chunk_index] = json.dumps(text_score.__dict__, ensure_ascii=False)
                        score_count += 1
                    pbar.update(len(batch_chunk))

                reused_count = len(chunk) - len(batch_items)
                if reused_count:
                    pbar.update(reused_count)

                for serialized_row in chunk_lines:
                    if serialized_row is None:
                        raise RuntimeError("Missing serialized score row while writing scored texts.")
                    writer.write(serialized_row + "\n")
    temp_path.replace(output_path)

    del model
    torch.cuda.empty_cache()
    return score_count


def aggregate_explanation_scores(
    text_scores: list[TextScore],
    records_by_feature_id: dict[str, dict[str, Any]] | None = None,
) -> list[ExplanationScore]:
    aggregated: list[ExplanationScore] = []
    for best_score in text_scores:
        aggregated.append(_text_score_to_explanation_score(best_score, records_by_feature_id=records_by_feature_id))
    return aggregated


def build_preference_pairs(
    cfg: PreferencePipelineConfig,
    explanation_scores: list[ExplanationScore],
    records_by_feature_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[ExplanationScore]] = {}
    for item in explanation_scores:
        grouped.setdefault(item.feature_id, []).append(item)

    preferences: list[dict[str, Any]] = []
    for feature_id, items in grouped.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda item: item.score, reverse=True)
        chosen = items[0]

        # Hard filter: if even the best explanation cannot induce a sufficiently
        # strong activation, treat the whole feature as failed generation and skip it.
        if chosen.score < cfg.data.min_chosen_activation:
            continue

        if cfg.data.rejected_strategy == "median":
            rejected = items[len(items) // 2]
        elif cfg.data.rejected_strategy == "lowest":
            rejected = items[-1]
        else:
            raise ValueError(f"Unknown rejected_strategy: {cfg.data.rejected_strategy}")

        score_gap = chosen.score - rejected.score
        if score_gap <= cfg.data.min_score_gap:
            continue

        preferences.append(
            {
                "feature_id": feature_id,
                "chosen_explanation": chosen.explanation,
                "rejected_explanation": rejected.explanation,
                "chosen_score": chosen.score,
                "rejected_score": rejected.score,
                "score_gap": score_gap,
                "chosen_best_text": chosen.best_text,
                "rejected_best_text": rejected.best_text,
                "metadata": chosen.metadata or (records_by_feature_id or {}).get(feature_id, {}).get("metadata", {}),
            }
        )
    return preferences


def run_vec_only_preference_pipeline(
    cfg: PreferencePipelineConfig,
    *,
    resume_from: str = "start",
    stop_after: str | None = None,
) -> dict[str, Any]:
    _configure_low_cpu_runtime()

    project_root = cfg.data.input_jsonl.parents[2]
    records = _select_records(cfg.data.input_jsonl, cfg.data.max_features)
    records_by_feature_id = {record["feature_id"]: record for record in records}
    output_dir = cfg.data.output_dir
    sae_cache_dir = project_root / "data" / "cache" / "hf_saes"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidate_explanations.jsonl"
    generated_texts_path = output_dir / "generated_texts.jsonl"
    scored_texts_path = output_dir / "scored_texts.jsonl"
    scored_explanations_path = output_dir / "scored_explanations.jsonl"
    preferences_path = output_dir / "preferences.jsonl"
    preferences_filter_summary_path = output_dir / "preferences.filter_summary.json"

    resume_idx = _stage_index(resume_from)
    stop_idx = _stage_index(stop_after) if stop_after is not None else len(PIPELINE_STAGES) - 1
    if stop_idx < resume_idx:
        raise ValueError(
            f"stop_after={stop_after!r} is earlier than resume_from={resume_from!r}; nothing to run."
        )

    def _should_run(stage_name: str) -> bool:
        idx = _stage_index(stage_name)
        return resume_idx < idx <= stop_idx

    # Stage 1: candidate explanations
    if _should_run("candidates"):
        candidate_count = generate_candidate_explanations(cfg, project_root, candidates_path)
    else:
        _ensure_exists(candidates_path, "candidates")
        candidate_count = _count_jsonl_rows(candidates_path)

    # Stage 2: generated texts
    if _should_run("generated_texts"):
        generated_text_count = generate_texts_from_explanations(cfg, candidates_path, generated_texts_path)
    else:
        if stop_idx >= _stage_index("generated_texts"):
            _ensure_exists(generated_texts_path, "generated_texts")
            generated_text_count = _count_jsonl_rows(generated_texts_path)
        else:
            generated_text_count = 0

    # Stage 3: scored texts
    if _should_run("scored_texts"):
        scored_text_count = score_generated_texts(
            cfg,
            records_by_feature_id,
            generated_texts_path,
            sae_cache_dir,
            scored_texts_path,
        )
    else:
        if stop_idx >= _stage_index("scored_texts"):
            _ensure_exists(scored_texts_path, "scored_texts")
            scored_text_count = _count_jsonl_rows(scored_texts_path)
        else:
            scored_text_count = 0

    # Stage 4: aggregate explanation scores
    if _should_run("scored_explanations"):
        best_text_score_by_candidate: dict[tuple[str, int], TextScore] = {}
        for row in _iter_jsonl(scored_texts_path):
            text_score = TextScore(**row)
            key = (text_score.feature_id, text_score.candidate_id)
            previous = best_text_score_by_candidate.get(key)
            if previous is None or text_score.max_token_activation > previous.max_token_activation:
                best_text_score_by_candidate[key] = text_score

        explanation_scores = aggregate_explanation_scores(
            list(best_text_score_by_candidate.values()),
            records_by_feature_id=records_by_feature_id,
        )
        _write_jsonl(scored_explanations_path, [item.__dict__ for item in explanation_scores])
        scored_explanations_count = len(explanation_scores)
    else:
        if stop_idx >= _stage_index("scored_explanations"):
            _ensure_exists(scored_explanations_path, "scored_explanations")
            scored_explanations_count = _count_jsonl_rows(scored_explanations_path)
        else:
            scored_explanations_count = 0

    # Stage 5: preference pairs
    if _should_run("preferences"):
        from src.preference.filtering import filter_scored_explanations_file

        filter_summary = filter_scored_explanations_file(
            input_jsonl=scored_explanations_path,
            output_jsonl=preferences_path,
            summary_json=preferences_filter_summary_path,
            dropped_jsonl=None,
            min_chosen_activation=cfg.data.filter_min_chosen_activation,
            min_group_spread_ratio=cfg.data.filter_min_group_spread_ratio,
            min_relative_gap=cfg.data.filter_min_relative_gap,
        )
        preference_count = int(filter_summary["num_preferences_kept"])
    else:
        if stop_idx >= _stage_index("preferences"):
            _ensure_exists(preferences_path, "preferences")
            preference_count = _count_jsonl_rows(preferences_path)
        else:
            preference_count = 0

    summary = {
        "num_features": len(records),
        "num_candidates": candidate_count,
        "num_generated_texts": generated_text_count,
        "num_scored_texts": scored_text_count,
        "num_scored_explanations": scored_explanations_count,
        "num_preferences": preference_count,
        "resume_from": resume_from,
        "stop_after": stop_after,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
