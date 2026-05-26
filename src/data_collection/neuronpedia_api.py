"""Neuronpedia data collection for SAEExplainer SFT warm-up datasets."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import string
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml


TEXT_FIELD_CANDIDATES = (
    "text",
    "bufferText",
    "buffer_text",
    "fullText",
    "full_text",
)
ACTIVATION_FIELD_CANDIDATES = (
    "maxValue",
    "maxAct",
    "maxActApprox",
    "activation",
    "act",
    "score",
    "binMax",
    "dfaMaxValue",
)
EXPLANATION_FIELD_CANDIDATES = (
    "description",
    "explanation",
    "text",
)
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
WHITESPACE_RE = re.compile(r"[ \t]+")
EXPLANATION_MODEL_PRIORITY = {
    "gpt-5": 0,
    "gpt-4.1": 1,
    "gpt-4o": 2,
    "gpt-4o-mini": 3,
    "claude-4": 4,
    "claude-3.7": 5,
    "claude-3.5": 6,
}


@dataclass(frozen=True)
class LayerConfig:
    layer: int
    source_id: str
    ratio: float


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key_env: str
    timeout_seconds: float
    concurrency: int
    requests_per_second: float
    max_retries: int
    retry_statuses: tuple[int, ...]
    retry_backoff_seconds: float


@dataclass(frozen=True)
class CollectionConfig:
    target_model: str
    sae_family: str
    sae_variant: str
    dictionary_size: int
    total_samples: int
    top_k_activations: int
    random_seed: int
    enforce_english_only: bool
    english_ascii_ratio_threshold: float
    output_jsonl: Path
    filtered_jsonl: Path
    summary_json: Path
    layers: tuple[LayerConfig, ...]


@dataclass(frozen=True)
class DataCollectionConfig:
    api: ApiConfig
    data_collection: CollectionConfig


@dataclass
class LayerState:
    layer_config: LayerConfig
    quota_target: int
    remaining_indices: deque[int]
    attempted_count: int = 0
    valid_count: int = 0
    filtered_count: int = 0
    api_error_count: int = 0
    exhausted: bool = False


@dataclass(frozen=True)
class FetchOutcome:
    layer: int
    source_id: str
    feature_index: int
    payload: Optional[dict[str, Any]] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    final_url: Optional[str] = None
    redirect_count: int = 0


def _resolve_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _derive_filtered_jsonl_path(output_jsonl: Path) -> Path:
    if output_jsonl.suffix:
        return output_jsonl.with_suffix(f".filtered{output_jsonl.suffix}")
    return output_jsonl.with_name(output_jsonl.name + ".filtered.jsonl")


def load_config(config_path: Path, project_root: Path) -> DataCollectionConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    api_raw = raw["api"]
    collection_raw = raw["data_collection"]

    layers = tuple(
        LayerConfig(
            layer=int(item["layer"]),
            source_id=str(item["source_id"]),
            ratio=float(item["ratio"]),
        )
        for item in collection_raw["layers"]
    )

    api_config = ApiConfig(
        base_url=str(api_raw["base_url"]).rstrip("/"),
        api_key_env=str(api_raw["api_key_env"]),
        timeout_seconds=float(api_raw.get("timeout_seconds", 30)),
        concurrency=max(1, int(api_raw.get("concurrency", 8))),
        requests_per_second=max(0.1, float(api_raw.get("requests_per_second", 2.0))),
        max_retries=max(1, int(api_raw.get("max_retries", 3))),
        retry_statuses=tuple(int(code) for code in api_raw.get("retry_statuses", [429, 500])),
        retry_backoff_seconds=max(0.1, float(api_raw.get("retry_backoff_seconds", 2.0))),
    )

    collection_config = CollectionConfig(
        target_model=str(collection_raw["target_model"]),
        sae_family=str(collection_raw["sae_family"]),
        sae_variant=str(collection_raw["sae_variant"]),
        dictionary_size=int(collection_raw["dictionary_size"]),
        total_samples=int(collection_raw["total_samples"]),
        top_k_activations=int(collection_raw["top_k_activations"]),
        random_seed=int(collection_raw.get("random_seed", 42)),
        enforce_english_only=bool(collection_raw.get("enforce_english_only", True)),
        english_ascii_ratio_threshold=float(collection_raw.get("english_ascii_ratio_threshold", 0.85)),
        output_jsonl=_resolve_path(collection_raw["output_jsonl"], project_root),
        filtered_jsonl=(
            _resolve_path(collection_raw["filtered_jsonl"], project_root)
            if collection_raw.get("filtered_jsonl") is not None
            else _derive_filtered_jsonl_path(_resolve_path(collection_raw["output_jsonl"], project_root))
        ),
        summary_json=_resolve_path(collection_raw["summary_json"], project_root),
        layers=layers,
    )

    return DataCollectionConfig(api=api_config, data_collection=collection_config)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u0000", " ")
    return " ".join(text.split()).strip()


def _cleanup_rendered_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("▁", " ")
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    text = text.replace("<0x0A>", "\n")
    text = re.sub(r" *\n *", "\n", text)
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _detokenize_tokens(tokens: list[Any]) -> str:
    raw_text = "".join(str(token) for token in tokens if token is not None)
    return _cleanup_rendered_text(raw_text)


def _looks_english(text: str, ascii_ratio_threshold: float) -> bool:
    if not text:
        return False
    if CJK_PATTERN.search(text):
        return False

    printable_ascii = 0
    ascii_letters = 0
    allowed_ascii = set(string.ascii_letters + string.digits + string.punctuation + " \t\n\r")
    for character in text:
        if character in allowed_ascii:
            printable_ascii += 1
        if character.isascii() and character.isalpha():
            ascii_letters += 1

    ascii_ratio = printable_ascii / max(len(text), 1)
    return ascii_letters > 0 and ascii_ratio >= ascii_ratio_threshold


def _explanation_model_rank(model_name: str) -> int:
    normalized = model_name.lower()
    for prefix, rank in EXPLANATION_MODEL_PRIORITY.items():
        if normalized.startswith(prefix):
            return rank
    return 100


def _extract_explanation_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    explanations = payload.get("explanations") or []
    if not isinstance(explanations, list):
        return []

    ranked_candidates: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for position, explanation in enumerate(explanations):
        if not isinstance(explanation, dict):
            continue
        text = ""
        for field in EXPLANATION_FIELD_CANDIDATES:
            text = _normalize_text(explanation.get(field))
            if text:
                break
        if not text:
            continue
        model_name = _normalize_text(explanation.get("explanationModelName"))
        user = explanation.get("triggeredByUser") or {}
        user_name = _normalize_text(user.get("name"))
        priority = (
            0 if explanation.get("isFeatured") else 1,
            0 if user_name == "bot" else 1,
            _explanation_model_rank(model_name),
            position,
        )
        ranked_candidates.append(
            (
                priority,
                {
                    "description": text,
                    "explanation_model_name": model_name or None,
                    "type_name": _normalize_text(explanation.get("typeName")) or None,
                    "triggered_by_user": user_name or None,
                    "is_featured": bool(explanation.get("isFeatured")),
                },
            )
        )

    ranked_candidates.sort(key=lambda item: item[0])
    return [candidate for _, candidate in ranked_candidates]


def _extract_explanation(payload: dict[str, Any]) -> str:
    candidates = _extract_explanation_candidates(payload)
    if not candidates:
        return ""
    return candidates[0]["description"]


def _extract_activation_score(item: dict[str, Any]) -> float:
    for field in ACTIVATION_FIELD_CANDIDATES:
        value = item.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    return float("-inf")


def _extract_activation_text(item: dict[str, Any]) -> str:
    for field in TEXT_FIELD_CANDIDATES:
        text = _cleanup_rendered_text(str(item.get(field))) if item.get(field) is not None else ""
        if text:
            return text

    tokens = item.get("tokens")
    if isinstance(tokens, list):
        return _detokenize_tokens(tokens)

    return ""


def _extract_top_activating_records(payload: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    activations = payload.get("activations") or []
    if not isinstance(activations, list):
        return []

    enriched: list[tuple[float, int, dict[str, Any]]] = []
    for position, activation in enumerate(activations):
        if not isinstance(activation, dict):
            continue
        text = _extract_activation_text(activation)
        if not text:
            continue
        activation_score = _extract_activation_score(activation)
        tokens = activation.get("tokens")
        max_value_token_index = activation.get("maxValueTokenIndex")
        max_value_token = None
        if (
            isinstance(tokens, list)
            and isinstance(max_value_token_index, int)
            and 0 <= max_value_token_index < len(tokens)
        ):
            max_value_token = _cleanup_rendered_text(str(tokens[max_value_token_index])) or None

        enriched.append(
            (
                activation_score,
                position,
                {
                    "text": text,
                    "max_activation": (
                        float(activation_score) if math.isfinite(activation_score) else None
                    ),
                    "max_activation_token": max_value_token,
                    "max_activation_token_index": (
                        max_value_token_index if isinstance(max_value_token_index, int) else None
                    ),
                    "data_index": activation.get("dataIndex"),
                },
            )
        )

    if not enriched:
        return []

    has_numeric_scores = any(math.isfinite(item[0]) for item in enriched)
    if has_numeric_scores:
        enriched.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    else:
        enriched.sort(key=lambda item: item[1])

    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, activation_record in enriched:
        text = activation_record["text"]
        if text in seen:
            continue
        seen.add(text)
        deduplicated.append(activation_record)
        if len(deduplicated) >= top_k:
            break
    return deduplicated


def _extract_top_activating_texts(payload: dict[str, Any], top_k: int) -> list[str]:
    return [item["text"] for item in _extract_top_activating_records(payload, top_k)]


def _extract_logit_pairs(tokens: Any, values: Any, limit: int = 10) -> list[dict[str, Any]]:
    if not isinstance(tokens, list) or not isinstance(values, list):
        return []

    pairs: list[dict[str, Any]] = []
    for token, value in zip(tokens[:limit], values[:limit]):
        if token is None or not isinstance(value, (int, float)):
            continue
        pairs.append(
            {
                "token": _cleanup_rendered_text(str(token)),
                "value": float(value),
            }
        )
    return pairs


def _build_feature_url(model_id: str, source_id: str, feature_index: int) -> str:
    return f"https://www.neuronpedia.org/{model_id}/{source_id}/{feature_index}"


def _extract_record_metadata(
    payload: dict[str, Any],
    layer_config: LayerConfig,
    feature_index: int,
    collection_config: CollectionConfig,
) -> dict[str, Any]:
    source_payload = payload.get("source") or {}
    if not isinstance(source_payload, dict):
        source_payload = {}
    saelens_config = source_payload.get("saelensConfig") or {}
    if not isinstance(saelens_config, dict):
        saelens_config = {}

    return {
        "target_model": collection_config.target_model,
        "sae_id": layer_config.source_id,
        "feature_index": feature_index,
        "layer": layer_config.layer,
        "sae_family": collection_config.sae_family,
        "sae_variant": collection_config.sae_variant,
        "data_source": "neuronpedia",
        "activation_density": payload.get("frac_nonzero"),
        "max_activation_approx": payload.get("maxActApprox"),
        "source_set_name": payload.get("sourceSetName"),
        "hook_name": payload.get("hookName") or saelens_config.get("hook_name"),
        "hook_layer": saelens_config.get("hook_layer"),
        "d_in": saelens_config.get("d_in"),
        "d_sae": saelens_config.get("d_sae"),
        "hf_repo_id": source_payload.get("hfRepoId"),
        "hf_folder_id": source_payload.get("hfFolderId"),
        "saelens_release": source_payload.get("saelensRelease"),
        "saelens_sae_id": source_payload.get("saelensSaeId"),
        "inference_enabled": source_payload.get("inferenceEnabled"),
    }


def _build_base_record(
    payload: dict[str, Any],
    layer_config: LayerConfig,
    feature_index: int,
    collection_config: CollectionConfig,
) -> dict[str, Any]:
    explanation = _extract_explanation(payload) or None
    top_activation_records = _extract_top_activating_records(payload, collection_config.top_k_activations)
    activating_texts = [item["text"] for item in top_activation_records]
    return {
        "feature_id": f"{collection_config.target_model}/{layer_config.source_id}/{feature_index}",
        "feature_url": _build_feature_url(
            model_id=collection_config.target_model,
            source_id=layer_config.source_id,
            feature_index=feature_index,
        ),
        "explanation": explanation,
        "top_activating_texts": activating_texts,
        "top_activating_contexts": top_activation_records,
        "metadata": _extract_record_metadata(
            payload=payload,
            layer_config=layer_config,
            feature_index=feature_index,
            collection_config=collection_config,
        ),
    }


def _build_output_record(
    payload: dict[str, Any],
    layer_config: LayerConfig,
    feature_index: int,
    collection_config: CollectionConfig,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not payload:
        return None, "empty_payload"

    record = _build_base_record(
        payload=payload,
        layer_config=layer_config,
        feature_index=feature_index,
        collection_config=collection_config,
    )

    explanation = str(record.get("explanation") or "").strip()
    if not explanation:
        return None, "missing_explanation"

    activating_texts = record.get("top_activating_texts") or []
    if not activating_texts:
        return None, "missing_activations"

    if collection_config.enforce_english_only:
        if not _looks_english(explanation, collection_config.english_ascii_ratio_threshold):
            return None, "non_english_explanation"
        for snippet in activating_texts:
            if not _looks_english(str(snippet), collection_config.english_ascii_ratio_threshold):
                return None, "non_english_activation_text"

    return record, None


def _build_filtered_record(
    payload: dict[str, Any],
    layer_config: LayerConfig,
    feature_index: int,
    collection_config: CollectionConfig,
    filter_failure_reason: str,
) -> dict[str, Any]:
    record = _build_base_record(
        payload=payload,
        layer_config=layer_config,
        feature_index=feature_index,
        collection_config=collection_config,
    )
    record["filter_failure_reason"] = filter_failure_reason
    return record


class ProgressTracker:
    """Simple terminal progress bar without external dependencies."""

    def __init__(self, total: int, layer_states: list[LayerState]):
        self.total = max(total, 1)
        self.layer_states = layer_states
        self.enabled = sys.stderr.isatty()
        self._last_render = 0.0
        self._spinner = "|/-\\"
        self._spinner_index = 0

    def start(self) -> None:
        layer_summary = ", ".join(
            f"L{state.layer_config.layer}:{state.quota_target}"
            for state in self.layer_states
        )
        print(
            f"Collecting {self.total} valid features from Neuronpedia "
            f"({layer_summary}).",
            file=sys.stderr,
        )
        self.render(0, Counter(), force=True)

    def render(
        self,
        successful_records: int,
        skip_reasons: Counter[str],
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < 0.2:
            return
        self._last_render = now

        attempted = sum(state.attempted_count for state in self.layer_states)
        filtered = sum(state.filtered_count for state in self.layer_states)
        api_errors = sum(state.api_error_count for state in self.layer_states)
        layer_valids = " ".join(
            f"L{state.layer_config.layer}:{state.valid_count}/{state.quota_target}"
            for state in self.layer_states
        )
        top_skip = ""
        if skip_reasons:
            reason, count = skip_reasons.most_common(1)[0]
            top_skip = f" | top_skip={reason}:{count}"

        width = 28
        filled = min(width, int(width * successful_records / self.total))
        bar = "#" * filled + "-" * (width - filled)
        percent = 100.0 * successful_records / self.total
        spinner = self._spinner[self._spinner_index % len(self._spinner)]
        self._spinner_index += 1
        line = (
            f"\r{spinner} [{bar}] {successful_records}/{self.total} ({percent:5.1f}%)"
            f" | attempted={attempted} filtered={filtered} api_errors={api_errors}"
            f" | {layer_valids}"
            f"{top_skip}"
        )

        if self.enabled:
            print(line, end="", file=sys.stderr, flush=True)
        else:
            print(line.lstrip("\r"), file=sys.stderr, flush=True)

    def close(self, successful_records: int, skip_reasons: Counter[str]) -> None:
        self.render(successful_records, skip_reasons, force=True)
        if self.enabled:
            print(file=sys.stderr, flush=True)
        print(
            f"Finished collection: {successful_records}/{self.total} valid samples.",
            file=sys.stderr,
        )


def _allocate_layer_quotas(total_samples: int, layers: tuple[LayerConfig, ...]) -> dict[int, int]:
    total_ratio = sum(layer.ratio for layer in layers)
    if total_ratio <= 0:
        raise ValueError("Layer ratios must sum to a positive value.")

    normalized = [layer.ratio / total_ratio for layer in layers]
    raw_targets = [ratio * total_samples for ratio in normalized]
    floor_targets = [math.floor(value) for value in raw_targets]
    remainder = total_samples - sum(floor_targets)

    fractional_indices = sorted(
        range(len(layers)),
        key=lambda index: raw_targets[index] - floor_targets[index],
        reverse=True,
    )
    for index in fractional_indices[:remainder]:
        floor_targets[index] += 1

    return {layers[index].layer: floor_targets[index] for index in range(len(layers))}


def _build_layer_states(collection_config: CollectionConfig) -> list[LayerState]:
    quotas = _allocate_layer_quotas(collection_config.total_samples, collection_config.layers)
    states: list[LayerState] = []

    for layer_config in collection_config.layers:
        layer_rng = random.Random(collection_config.random_seed + layer_config.layer * 100_003)
        indices = list(range(collection_config.dictionary_size))
        layer_rng.shuffle(indices)
        states.append(
            LayerState(
                layer_config=layer_config,
                quota_target=quotas[layer_config.layer],
                remaining_indices=deque(indices),
            )
        )
    return states


def _select_next_layer(
    states: list[LayerState],
    excluded_layers: Optional[set[int]] = None,
) -> Optional[LayerState]:
    excluded_layers = excluded_layers or set()
    available_states = [
        state
        for state in states
        if state.remaining_indices and state.layer_config.layer not in excluded_layers
    ]
    if not available_states:
        return None

    unmet_states = [state for state in available_states if state.valid_count < state.quota_target]
    candidate_states = unmet_states or available_states

    def score(state: LayerState) -> tuple[int, int, float]:
        quota_gap = max(state.quota_target - state.valid_count, 0)
        return (
            quota_gap,
            len(state.remaining_indices),
            state.layer_config.ratio,
        )

    return max(candidate_states, key=score)


class NeuronpediaClient:
    def __init__(self, api_config: ApiConfig, api_key: Optional[str]):
        self.api_config = api_config
        self.api_key = api_key
        self._rate_lock = asyncio.Lock()
        self._next_request_ts = 0.0

    async def _wait_for_rate_limit_slot(self) -> None:
        interval = 1.0 / self.api_config.requests_per_second
        async with self._rate_lock:
            now = time.monotonic()
            if now < self._next_request_ts:
                await asyncio.sleep(self._next_request_ts - now)
                now = time.monotonic()
            self._next_request_ts = max(self._next_request_ts, now) + interval

    async def fetch_feature(
        self,
        client: httpx.AsyncClient,
        model_id: str,
        source_id: str,
        feature_index: int,
        layer: int,
    ) -> FetchOutcome:
        url = f"{self.api_config.base_url}/{model_id}/{source_id}/{feature_index}"
        headers = {"X-Api-Key": self.api_key} if self.api_key else {}
        backoff = self.api_config.retry_backoff_seconds
        last_error: Optional[str] = None

        for attempt in range(1, self.api_config.max_retries + 1):
            await self._wait_for_rate_limit_slot()
            try:
                response = await client.get(url, headers=headers, follow_redirects=True)
            except httpx.HTTPError as exc:
                last_error = f"http_error: {exc}"
                if attempt == self.api_config.max_retries:
                    break
                await asyncio.sleep(backoff * attempt)
                continue

            if response.status_code == 200:
                payload = response.json()
                return FetchOutcome(
                    layer=layer,
                    source_id=source_id,
                    feature_index=feature_index,
                    status_code=response.status_code,
                    payload=payload if isinstance(payload, dict) else {},
                    final_url=str(response.url),
                    redirect_count=len(response.history),
                )

            if response.status_code in self.api_config.retry_statuses and attempt < self.api_config.max_retries:
                await asyncio.sleep(backoff * attempt)
                continue

            try:
                error_payload = response.json()
                message = error_payload.get("message", response.text)
            except json.JSONDecodeError:
                message = response.text
            last_error = f"status_{response.status_code}: {message[:200]}"
            return FetchOutcome(
                layer=layer,
                source_id=source_id,
                feature_index=feature_index,
                status_code=response.status_code,
                error=last_error,
                final_url=str(response.url),
                redirect_count=len(response.history),
            )

        return FetchOutcome(
            layer=layer,
            source_id=source_id,
            feature_index=feature_index,
            error=last_error or "request_failed",
        )


async def _collect_dataset(config: DataCollectionConfig) -> dict[str, Any]:
    collection_config = config.data_collection
    api_key = os.environ.get(config.api.api_key_env)

    collection_config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    collection_config.filtered_jsonl.parent.mkdir(parents=True, exist_ok=True)
    collection_config.summary_json.parent.mkdir(parents=True, exist_ok=True)

    layer_states = _build_layer_states(collection_config)
    client = NeuronpediaClient(
        api_config=config.api,
        api_key=api_key,
    )
    skip_reasons: Counter[str] = Counter()
    successful_records = 0
    progress = ProgressTracker(
        total=collection_config.total_samples,
        layer_states=layer_states,
    )

    timeout = httpx.Timeout(config.api.timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        with collection_config.output_jsonl.open("w", encoding="utf-8") as writer:
            with collection_config.filtered_jsonl.open("w", encoding="utf-8") as filtered_writer:
                progress.start()
                while successful_records < collection_config.total_samples:
                    available_states = [state for state in layer_states if state.remaining_indices]
                    if not available_states:
                        break

                    pending_requests: list[tuple[LayerState, int]] = []
                    max_batch_size = min(
                        config.api.concurrency,
                        collection_config.total_samples - successful_records,
                        sum(len(state.remaining_indices) for state in available_states),
                    )

                    selected_layers: set[int] = set()
                    for _ in range(max_batch_size):
                        state = _select_next_layer(layer_states, excluded_layers=selected_layers)
                        if state is None and selected_layers:
                            selected_layers.clear()
                            state = _select_next_layer(layer_states)
                        if state is None:
                            break
                        feature_index = state.remaining_indices.popleft()
                        state.attempted_count += 1
                        if not state.remaining_indices:
                            state.exhausted = True
                        pending_requests.append((state, feature_index))
                        selected_layers.add(state.layer_config.layer)

                    if not pending_requests:
                        break

                    outcomes = await asyncio.gather(
                        *[
                            client.fetch_feature(
                                client=http_client,
                                model_id=collection_config.target_model,
                                source_id=state.layer_config.source_id,
                                feature_index=feature_index,
                                layer=state.layer_config.layer,
                            )
                            for state, feature_index in pending_requests
                        ]
                    )

                    for (state, feature_index), outcome in zip(pending_requests, outcomes):
                        if outcome.error or not outcome.payload:
                            state.api_error_count += 1
                            skip_reasons["api_error"] += 1
                            progress.render(successful_records, skip_reasons)
                            continue

                        record, reason = _build_output_record(
                            payload=outcome.payload,
                            layer_config=state.layer_config,
                            feature_index=feature_index,
                            collection_config=collection_config,
                        )
                        if record is None:
                            state.filtered_count += 1
                            filter_reason = reason or "filtered_unknown"
                            skip_reasons[filter_reason] += 1
                            filtered_record = _build_filtered_record(
                                payload=outcome.payload,
                                layer_config=state.layer_config,
                                feature_index=feature_index,
                                collection_config=collection_config,
                                filter_failure_reason=filter_reason,
                            )
                            filtered_writer.write(json.dumps(filtered_record, ensure_ascii=False) + "\n")
                            filtered_writer.flush()
                            progress.render(successful_records, skip_reasons)
                            continue

                        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                        writer.flush()
                        successful_records += 1
                        state.valid_count += 1
                        progress.render(successful_records, skip_reasons)
                        if successful_records >= collection_config.total_samples:
                            break
                progress.close(successful_records, skip_reasons)

    summary = {
        "target_model": collection_config.target_model,
        "sae_family": collection_config.sae_family,
        "sae_variant": collection_config.sae_variant,
        "used_api_key": bool(api_key),
        "requested_samples": collection_config.total_samples,
        "collected_samples": successful_records,
        "top_k_activations": collection_config.top_k_activations,
        "output_jsonl": str(collection_config.output_jsonl),
        "filtered_jsonl": str(collection_config.filtered_jsonl),
        "skip_reasons": dict(skip_reasons),
        "layers": [
            {
                "layer": state.layer_config.layer,
                "source_id": state.layer_config.source_id,
                "ratio": state.layer_config.ratio,
                "quota_target": state.quota_target,
                "attempted_count": state.attempted_count,
                "valid_count": state.valid_count,
                "filtered_count": state.filtered_count,
                "api_error_count": state.api_error_count,
                "remaining_candidates": len(state.remaining_indices),
            }
            for state in layer_states
        ],
    }
    collection_config.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


async def collect_dataset_from_config(
    config_path: Path,
    project_root: Path,
    total_samples_override: Optional[int] = None,
    random_seed_override: Optional[int] = None,
    output_jsonl_override: Optional[Path] = None,
    top_k_activations_override: Optional[int] = None,
) -> dict[str, Any]:
    config = load_config(config_path=config_path, project_root=project_root)
    collection_config = config.data_collection

    if total_samples_override is not None:
        collection_config = replace(collection_config, total_samples=total_samples_override)
    if random_seed_override is not None:
        collection_config = replace(collection_config, random_seed=random_seed_override)
    if output_jsonl_override is not None:
        summary_override = output_jsonl_override.with_suffix(".summary.json")
        filtered_override = _derive_filtered_jsonl_path(output_jsonl_override)
        collection_config = replace(
            collection_config,
            output_jsonl=output_jsonl_override,
            filtered_jsonl=filtered_override,
            summary_json=summary_override,
        )
    if top_k_activations_override is not None:
        collection_config = replace(
            collection_config,
            top_k_activations=top_k_activations_override,
        )

    config = replace(config, data_collection=collection_config)
    return await _collect_dataset(config)


async def probe_feature_from_config(
    config_path: Path,
    project_root: Path,
    feature_index: int = 0,
    layer: Optional[int] = None,
    source_id: Optional[str] = None,
    top_k_override: Optional[int] = None,
) -> dict[str, Any]:
    """Fetch and inspect a single feature endpoint for quick API diagnostics."""
    config = load_config(config_path=config_path, project_root=project_root)
    collection_config = config.data_collection
    api_key = os.environ.get(config.api.api_key_env)

    chosen_layer_config: Optional[LayerConfig] = None
    if source_id is None:
        if layer is None:
            chosen_layer_config = collection_config.layers[0]
        else:
            for layer_config in collection_config.layers:
                if layer_config.layer == layer:
                    chosen_layer_config = layer_config
                    break
            if chosen_layer_config is None:
                raise ValueError(
                    f"Layer {layer} is not defined in {config_path}. "
                    f"Configured layers: {[item.layer for item in collection_config.layers]}"
                )
        source_id = chosen_layer_config.source_id
    elif layer is None:
        for layer_config in collection_config.layers:
            if layer_config.source_id == source_id:
                chosen_layer_config = layer_config
                break
    else:
        chosen_layer_config = LayerConfig(layer=layer, source_id=source_id, ratio=1.0)

    if chosen_layer_config is None:
        chosen_layer_config = LayerConfig(layer=-1, source_id=source_id, ratio=1.0)

    top_k = top_k_override or collection_config.top_k_activations
    timeout = httpx.Timeout(config.api.timeout_seconds)

    client = NeuronpediaClient(
        api_config=config.api,
        api_key=api_key,
    )
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        outcome = await client.fetch_feature(
            client=http_client,
            model_id=collection_config.target_model,
            source_id=source_id,
            feature_index=feature_index,
            layer=chosen_layer_config.layer,
        )

    payload = outcome.payload or {}
    explanations = payload.get("explanations") or []
    activations = payload.get("activations") or []
    extracted_explanation = _extract_explanation(payload) if payload else ""
    extracted_top_text = _extract_top_activating_texts(payload, top_k) if payload else []
    record, filter_failure_reason = _build_output_record(
        payload=payload,
        layer_config=chosen_layer_config,
        feature_index=feature_index,
        collection_config=replace(collection_config, top_k_activations=top_k),
    ) if payload else (None, "api_error")

    diagnosis = "ok"
    if outcome.error:
        if outcome.status_code in (401, 403):
            diagnosis = "authentication_failed"
        elif outcome.status_code == 404:
            diagnosis = "endpoint_or_source_not_found"
        elif outcome.status_code == 429:
            diagnosis = "rate_limited"
        elif outcome.status_code and outcome.status_code >= 500:
            diagnosis = "server_error"
        else:
            diagnosis = "request_failed"
    elif record is None:
        diagnosis = f"payload_received_but_filtered:{filter_failure_reason}"

    return {
        "request": {
            "url": f"{config.api.base_url}/{collection_config.target_model}/{source_id}/{feature_index}",
            "target_model": collection_config.target_model,
            "source_id": source_id,
            "layer": chosen_layer_config.layer,
            "feature_index": feature_index,
            "top_k_activations": top_k,
        },
        "response": {
            "status_code": outcome.status_code,
            "error": outcome.error,
            "diagnosis": diagnosis,
            "final_url": outcome.final_url,
            "redirect_count": outcome.redirect_count,
            "payload_received": bool(payload),
            "top_level_keys": sorted(payload.keys()) if payload else [],
            "explanations_count": len(explanations) if isinstance(explanations, list) else 0,
            "activations_count": len(activations) if isinstance(activations, list) else 0,
        },
        "extraction": {
            "passes_current_filter": record is not None,
            "filter_failure_reason": filter_failure_reason,
            "extracted_explanation": extracted_explanation,
            "extracted_top_activating_text": extracted_top_text,
        },
    }
