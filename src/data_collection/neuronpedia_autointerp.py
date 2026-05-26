"""Helpers for fetching and generating Neuronpedia explanations by model."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
import sys
from typing import Any

import httpx
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None


DEFAULT_FEATURE_EXPLAINER_SYSTEM_PROMPT = (
    "You are an expert in mechanistic interpretability and sparse autoencoder feature analysis. "
    "Given feature top activating contexts, infer the narrowest robust explanation "
    "for what the feature detects. Return only the explanation text."
)
DEFAULT_NEURONPEDIA_HOSTED_BASE_URL = "https://www.neuronpedia.org"
DEFAULT_NEURONPEDIA_HOSTED_EXPLANATION_TYPE = "oai_token-act-pair"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_REQUEST_MAX_RETRIES = 4
DEFAULT_REQUEST_BACKOFF_SECONDS = 2.0
ANTHROPIC_HOSTED_TO_OFFICIAL_MODEL_MAP = {
    "claude-4-5-sonnet": "claude-sonnet-4-20250514",
}
NEURONPEDIA_HOSTED_MODEL_ALIASES = {
    "gpt-4o-mini": "o4-mini",
    "gpt4omini": "o4-mini",
    "gpt-4omini": "o4-mini",
    "o4mini": "o4-mini",
}
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>|<bos>|<eos>|</s>|<s>")
WHITESPACE_RE = re.compile(r"\s+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_rows(handle: Any, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _records_with_feature_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("feature_id") or "").strip()]


def _create_progress_bar(
    *,
    total: int,
    desc: str,
    unit: str = "req",
) -> Any:
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, dynamic_ncols=True, unit=unit)


def _merge_summary_counts(summary: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        summary[key] = summary.get(key, 0) + int(value)


def _request_key(
    feature_id: str,
    requested_model: str,
    requested_explanation_type: str | None = None,
) -> tuple[str, str, str | None]:
    return (
        str(feature_id).strip(),
        str(requested_model).strip(),
        str(requested_explanation_type).strip() if requested_explanation_type is not None else None,
    )


def _row_request_key(row: dict[str, Any]) -> tuple[str, str, str | None] | None:
    feature_id = str(row.get("feature_id") or "").strip()
    requested_model = str(row.get("requested_model") or "").strip()
    if not feature_id or not requested_model:
        return None
    if "explanation" in row and row.get("explanation") is None:
        return None

    requested_explanation_type = row.get("requested_explanation_type")
    if requested_explanation_type is None:
        metadata = row.get("metadata") or {}
        if isinstance(metadata, dict):
            np_metadata = metadata.get("neuronpedia_explanation_metadata") or {}
            if isinstance(np_metadata, dict):
                request_metadata = np_metadata.get("request") or {}
                if isinstance(request_metadata, dict):
                    requested_explanation_type = request_metadata.get("explanationType")

    return _request_key(feature_id, requested_model, requested_explanation_type)


def load_existing_request_keys(path: Path) -> set[tuple[str, str, str | None]]:
    if not path.exists():
        return set()

    keys: set[tuple[str, str, str | None]] = set()
    for row in load_jsonl(path):
        key = _row_request_key(row)
        if key is not None:
            keys.add(key)
    return keys


def build_error_output_row(
    *,
    record: dict[str, Any],
    requested_model: str,
    error: Exception,
    metadata_field: str,
    requested_explanation_type: str | None = None,
) -> dict[str, Any]:
    error_metadata = {
        "source": "error",
        "error_type": type(error).__name__,
        "error": str(error),
    }
    feature_metadata = dict(record.get("metadata") or {})
    feature_metadata[metadata_field] = error_metadata
    row: dict[str, Any] = {
        "feature_id": str(record.get("feature_id") or "").strip(),
        "feature_url": record.get("feature_url"),
        "requested_model": requested_model,
        "explanation": None,
        "metadata": feature_metadata,
    }
    if requested_explanation_type is not None:
        row["requested_explanation_type"] = requested_explanation_type
    return row


def _report_null_explanation_error(
    *,
    feature_id: str,
    requested_model: str,
    error: Exception,
    requested_explanation_type: str | None = None,
) -> None:
    type_suffix = (
        f" explanation_type={requested_explanation_type}"
        if requested_explanation_type is not None
        else ""
    )
    print(
        "[error] explanation=null"
        f" feature_id={feature_id}"
        f" model={requested_model}"
        f"{type_suffix}"
        f" error={type(error).__name__}: {error}",
        file=sys.stderr,
        flush=True,
    )


def _build_output_metadata(
    record: dict[str, Any],
    *,
    metadata_field: str,
    metadata_value: dict[str, Any],
) -> dict[str, Any]:
    merged_metadata = dict(record.get("metadata") or {})
    merged_metadata[metadata_field] = metadata_value
    return merged_metadata


def _resolve_direct_provider_model_name(model_name: str) -> tuple[str | None, str | None]:
    normalized = model_name.strip().lower()
    if not normalized:
        return None, None

    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai", model_name

    if normalized in ANTHROPIC_HOSTED_TO_OFFICIAL_MODEL_MAP:
        return "anthropic", ANTHROPIC_HOSTED_TO_OFFICIAL_MODEL_MAP[normalized]

    if normalized.startswith(("claude-opus-", "claude-sonnet-", "claude-3-", "claude-4-")):
        return "anthropic", model_name

    return None, None


def _resolve_neuronpedia_hosted_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    return NEURONPEDIA_HOSTED_MODEL_ALIASES.get(normalized, model_name)


def _request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    follow_redirects: bool = False,
    timeout: float | None = None,
    max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_REQUEST_BACKOFF_SECONDS,
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> httpx.Response:
    max_retries = max(1, int(max_retries))
    backoff_seconds = max(0.0, float(backoff_seconds))
    for attempt in range(max_retries):
        try:
            response = client.request(
                method,
                url,
                headers=headers,
                json=json_body,
                follow_redirects=follow_redirects,
                timeout=timeout,
            )
            if response.status_code in retry_status_codes and attempt < max_retries - 1:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            return response
        except (
            httpx.ConnectError,
            httpx.ProxyError,
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.WriteError,
            httpx.WriteTimeout,
        ):
            if attempt >= max_retries - 1:
                raise
            time.sleep(backoff_seconds * (2**attempt))

    raise RuntimeError(f"Unreachable request retry state for {method} {url}")


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

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def _extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _extract_anthropic_usage(response_json: dict[str, Any]) -> dict[str, Any]:
    usage = response_json.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _extract_chat_completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _extract_anthropic_text(response_json: Any) -> str:
    if not isinstance(response_json, dict):
        return ""
    content = response_json.get("content") or []
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = str(item.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _format_activation_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    return str(value or "unknown")


def _clean_context_text(value: Any) -> str:
    text = str(value or "")
    text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = text.replace("\u0000", " ")
    text = text.replace("Â", "")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _anchor_token_is_useful(token: str) -> bool:
    cleaned = str(token or "").strip()
    if len(cleaned) < 2:
        return False
    return any(char.isalnum() for char in cleaned)


def _extract_context_snippet(text: Any, *, anchor_token: str, window_chars: int) -> str:
    cleaned_text = _clean_context_text(text)
    if not cleaned_text:
        return ""

    if not _anchor_token_is_useful(anchor_token):
        return _truncate_text(cleaned_text, max(40, 2 * window_chars))

    match = re.search(re.escape(anchor_token), cleaned_text, flags=re.IGNORECASE)
    if match is None:
        return _truncate_text(cleaned_text, max(40, 2 * window_chars))

    start = max(0, match.start() - window_chars)
    end = min(len(cleaned_text), match.end() + window_chars)
    snippet = cleaned_text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(cleaned_text):
        snippet = snippet + "..."
    return snippet


def _build_feature_explanation_prompt(
    record: dict[str, Any],
    *,
    max_context_examples: int,
    max_chars_per_context: int,
    include_existing_explanation: bool,
) -> tuple[str, dict[str, Any]]:
    raw_contexts = record.get("top_activating_contexts") or []
    formatted_contexts: list[str] = []
    effective_max_examples = min(max_context_examples, 3)
    window_chars = min(180, max(80, max_chars_per_context // 4))
    for raw_context in raw_contexts:
        if len(formatted_contexts) >= effective_max_examples:
            break
        if not isinstance(raw_context, dict):
            continue
        anchor_token = str(raw_context.get("max_activation_token") or "").strip()
        snippet = _extract_context_snippet(
            raw_context.get("text"),
            anchor_token=anchor_token,
            window_chars=window_chars,
        )
        if not snippet:
            continue
        snippet_lines = [f"Example {len(formatted_contexts) + 1}"]
        if _anchor_token_is_useful(anchor_token):
            snippet_lines.append(f"anchor_token: {anchor_token}")
        snippet_lines.extend(["snippet:", snippet])
        formatted_contexts.append("\n".join(snippet_lines))

    lines = [
        "Analyze the sparse autoencoder feature below and write one concise explanation.",
        "Return only the explanation text. Do not return bullets, JSON, or extra commentary.",
        "",
        "activation_snippets:",
    ]
    if formatted_contexts:
        lines.extend(formatted_contexts)
    else:
        lines.append("No activation snippets were provided.")

    lines.extend(
        [
            "",
            "Requirements:",
            "- Infer the narrowest robust recurring pattern.",
            "- Prefer concrete lexical, semantic, syntactic, formatting, or domain-specific descriptions.",
            "- If the evidence is noisy, describe the strongest shared signal instead of over-generalizing.",
            "- Keep the explanation to one short sentence or noun phrase.",
        ]
    )

    prompt_metadata = {
        "num_context_examples": len(formatted_contexts),
        "max_chars_per_context": max_chars_per_context,
        "window_chars": window_chars,
        "include_existing_explanation": False,
    }
    return "\n".join(lines), prompt_metadata


def parse_feature_id(feature_id: str) -> tuple[str, str, int]:
    parts = str(feature_id).split("/")
    if len(parts) != 3:
        raise ValueError(f"Unexpected feature_id format: {feature_id!r}")
    model_id, source_id, feature_index_raw = parts
    return model_id, source_id, int(feature_index_raw)


def fetch_feature_payload(
    *,
    client: httpx.Client,
    feature_id: str,
    base_url: str,
    api_key: str | None,
    max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_REQUEST_BACKOFF_SECONDS,
) -> dict[str, Any]:
    model_id, source_id, feature_index = parse_feature_id(feature_id)
    url = f"{base_url.rstrip('/')}/{model_id}/{source_id}/{feature_index}"
    headers = {"X-Api-Key": api_key} if api_key else {}
    response = _request_with_retries(
        client,
        "GET",
        url,
        headers=headers,
        follow_redirects=True,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload for {feature_id}, got {type(payload)!r}")
    return payload


def normalize_explanation_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("explanations") or []
    if not isinstance(candidates, list):
        return []

    normalized: list[dict[str, Any]] = []
    for explanation in candidates:
        if not isinstance(explanation, dict):
            continue
        text = ""
        for field in ("description", "explanation", "text"):
            text = str(explanation.get(field) or "").strip()
            if text:
                break
        if not text:
            continue
        triggered_by_user = explanation.get("triggeredByUser") or {}
        normalized.append(
            {
                "description": text,
                "explanation_model_name": str(explanation.get("explanationModelName") or "").strip() or None,
                "type_name": str(explanation.get("typeName") or "").strip() or None,
                "is_featured": bool(explanation.get("isFeatured")),
                "triggered_by_user": str(triggered_by_user.get("name") or "").strip() or None,
            }
        )
    return normalized


def select_explanation_candidate(
    payload: dict[str, Any],
    preferred_model: str,
    preferred_type: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    preferred_model_normalized = preferred_model.strip().lower()
    preferred_type_normalized = preferred_type.strip().lower() if preferred_type else None
    candidates = normalize_explanation_candidates(payload)
    if not candidates:
        return "", None

    def _model_matches(candidate: dict[str, Any]) -> bool:
        model_name = str(candidate.get("explanation_model_name") or "").lower()
        return (
            model_name == preferred_model_normalized
            or model_name.startswith(preferred_model_normalized)
            or preferred_model_normalized in model_name
        )

    def _type_matches(candidate: dict[str, Any]) -> bool:
        if preferred_type_normalized is None:
            return True
        type_name = str(candidate.get("type_name") or "").lower()
        return (
            type_name == preferred_type_normalized
            or type_name.startswith(preferred_type_normalized)
            or preferred_type_normalized in type_name
        )

    if preferred_type_normalized is not None:
        for candidate in candidates:
            if _model_matches(candidate) and _type_matches(candidate):
                return str(candidate["description"]), candidate

    for candidate in candidates:
        if _model_matches(candidate):
            return str(candidate["description"]), candidate
    return "", None


def select_explanation_candidate_by_model(
    payload: dict[str, Any],
    preferred_model: str,
) -> tuple[str, dict[str, Any] | None]:
    return select_explanation_candidate(payload, preferred_model, preferred_type=None)


def extract_activation_values(activation: dict[str, Any]) -> list[float] | None:
    for field in ("values", "acts", "activations", "scores"):
        value = activation.get(field)
        if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
            return [float(item) for item in value]
    return None


def extract_autointerp_activations(
    payload: dict[str, Any],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    raw_activations = payload.get("activations") or []
    if not isinstance(raw_activations, list):
        return []

    formatted: list[dict[str, Any]] = []
    for activation in raw_activations:
        if not isinstance(activation, dict):
            continue
        tokens = activation.get("tokens")
        values = extract_activation_values(activation)
        if not isinstance(tokens, list) or not isinstance(values, list):
            continue
        if not tokens or len(tokens) != len(values):
            continue
        formatted.append(
            {
                "tokens": [str(token) for token in tokens],
                "values": values,
            }
        )
        if len(formatted) >= max_examples:
            break
    return formatted


def extract_generated_explanation(response_json: Any) -> str:
    queue: list[Any] = [response_json]
    seen_ids: set[int] = set()
    candidate_fields = (
        "explanation",
        "response",
        "result",
        "description",
        "interpretation",
        "content",
    )

    while queue:
        current = queue.pop(0)
        current_id = id(current)
        if current_id in seen_ids:
            continue
        seen_ids.add(current_id)

        if isinstance(current, dict):
            for field in candidate_fields:
                value = current.get(field)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in current.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            queue.extend(current)

    return ""


def generate_explanation_via_autointerp(
    *,
    client: httpx.Client,
    payload: dict[str, Any],
    feature_id: str,
    model_name: str,
    autointerp_base_url: str,
    autointerp_secret: str | None,
    openrouter_key: str,
    max_activation_examples: int,
    request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
    request_backoff_seconds: float = DEFAULT_REQUEST_BACKOFF_SECONDS,
) -> tuple[str, dict[str, Any]]:
    activations = extract_autointerp_activations(
        payload,
        max_examples=max_activation_examples,
    )
    if not activations:
        raise ValueError(
            f"Feature {feature_id} does not expose token/value activation traces required for autointerp."
        )

    url = f"{autointerp_base_url.rstrip('/')}/explain/default"
    headers = {"Content-Type": "application/json"}
    if autointerp_secret:
        headers["X-SECRET-KEY"] = autointerp_secret

    request_body = {
        "model": model_name,
        "openrouter_key": openrouter_key,
        "activations": activations,
    }
    response = _request_with_retries(
        client,
        "POST",
        url,
        headers=headers,
        json_body=request_body,
        timeout=300.0,
        max_retries=request_max_retries,
        backoff_seconds=request_backoff_seconds,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        response_body = _truncate_text(response.text, 4000) or "<empty response body>"
        raise RuntimeError(
            "Autointerp request failed "
            f"for feature {feature_id} using model {model_name} "
            f"with HTTP {response.status_code} at {url}. "
            f"Response body: {response_body}"
        ) from exc
    response_json = response.json()
    explanation = extract_generated_explanation(response_json)
    if not explanation:
        raise RuntimeError(
            f"Autointerp endpoint returned a response for {feature_id}, but no explanation text could be extracted."
        )

    metadata = {
        "source": "generated",
        "autointerp_model": model_name,
        "autointerp_url": url,
        "num_activation_examples": len(activations),
        "raw_response": response_json,
    }
    return explanation, metadata


def generate_explanation_via_neuronpedia_hosted_api(
    *,
    client: httpx.Client,
    feature_id: str,
    explanation_type: str,
    explanation_model_name: str,
    base_url: str,
    api_key: str,
    feature_api_base_url: str = "https://www.neuronpedia.org/api/feature",
    request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
    request_backoff_seconds: float = DEFAULT_REQUEST_BACKOFF_SECONDS,
) -> tuple[str, dict[str, Any]]:
    model_id, layer, feature_index = parse_feature_id(feature_id)
    url = f"{base_url.rstrip('/')}/api/explanation/generate"
    resolved_model_name = _resolve_neuronpedia_hosted_model_name(explanation_model_name)
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    request_body = {
        "modelId": model_id,
        "layer": layer,
        "index": feature_index,
        "explanationType": explanation_type,
        "explanationModelName": resolved_model_name,
    }

    # Avoid an unnecessary generate call when the exact hosted explanation
    # already exists for this feature/model/type combination.
    feature_payload = fetch_feature_payload(
        client=client,
        feature_id=feature_id,
        base_url=feature_api_base_url,
        api_key=api_key,
        max_retries=request_max_retries,
        backoff_seconds=request_backoff_seconds,
    )
    explanation, matched_candidate = select_explanation_candidate(
        feature_payload,
        resolved_model_name,
        preferred_type=explanation_type,
    )
    if explanation:
        metadata = {
            "source": "neuronpedia_hosted_existing_precheck",
            "endpoint_url": url,
            "request": request_body,
            "matched_candidate": matched_candidate,
        }
        return explanation, metadata

    response = _request_with_retries(
        client,
        "POST",
        url,
        headers=headers,
        json_body=request_body,
        timeout=300.0,
        max_retries=request_max_retries,
        backoff_seconds=request_backoff_seconds,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        response_body = _truncate_text(response.text, 4000) or "<empty response body>"
        response_json: Any
        try:
            response_json = response.json()
        except Exception:
            response_json = None
        response_message = ""
        if isinstance(response_json, dict):
            response_message = str(response_json.get("message") or "").strip()

        if response.status_code == 400 and "already exists" in response_message.lower():
            explanation, matched_candidate = select_explanation_candidate(
                feature_payload,
                resolved_model_name,
                preferred_type=explanation_type,
            )
            if explanation:
                metadata = {
                    "source": "neuronpedia_hosted_existing",
                    "endpoint_url": url,
                    "request": request_body,
                    "raw_response": response_json,
                    "matched_candidate": matched_candidate,
                }
                return explanation, metadata

        raise RuntimeError(
            "Neuronpedia hosted explanation generation failed "
            f"for feature {feature_id} using explanation type {explanation_type} "
            f"and model {explanation_model_name} with HTTP {response.status_code} at {url}. "
            f"Response body: {response_body}"
        ) from exc

    response_json = response.json()
    explanation = ""
    if isinstance(response_json, dict):
        explanation, _ = select_explanation_candidate(
            response_json,
            resolved_model_name,
            preferred_type=explanation_type,
        )
    if not explanation:
        explanation = extract_generated_explanation(response_json)

    # Fallback: if the hosted endpoint acknowledges generation but does not return the
    # explanation text inline, refetch the feature and read the saved explanation.
    if not explanation:
        refreshed_feature_payload = fetch_feature_payload(
            client=client,
            feature_id=feature_id,
            base_url=feature_api_base_url,
            api_key=api_key,
            max_retries=request_max_retries,
            backoff_seconds=request_backoff_seconds,
        )
        explanation, _ = select_explanation_candidate(
            refreshed_feature_payload,
            resolved_model_name,
            preferred_type=explanation_type,
        )

    if not explanation:
        raise RuntimeError(
            f"Neuronpedia hosted generation returned a response for {feature_id}, "
            "but no explanation text could be extracted."
        )

    metadata = {
        "source": "neuronpedia_hosted_generation",
        "endpoint_url": url,
        "request": request_body,
        "requested_model_name": explanation_model_name,
        "resolved_model_name": resolved_model_name,
        "raw_response": response_json,
    }
    return explanation, metadata


def generate_explanation_via_openai_from_record(
    *,
    client: Any,
    record: dict[str, Any],
    model_name: str,
    system_prompt: str = DEFAULT_FEATURE_EXPLAINER_SYSTEM_PROMPT,
    max_context_examples: int = 5,
    max_chars_per_context: int = 1200,
    include_existing_explanation: bool = False,
    temperature: float | None = None,
    max_output_tokens: int = 128,
) -> tuple[str, dict[str, Any]]:
    prompt, prompt_metadata = _build_feature_explanation_prompt(
        record,
        max_context_examples=max_context_examples,
        max_chars_per_context=max_chars_per_context,
        include_existing_explanation=include_existing_explanation,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    params: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    params.update(_completion_length_param(model_name, max_output_tokens))
    if temperature is not None:
        params["temperature"] = temperature

    response = client.chat.completions.create(**params)
    explanation = _extract_chat_completion_text(response)
    if not explanation:
        feature_id = str(record.get("feature_id") or "").strip()
        raise RuntimeError(
            f"OpenAI-compatible generator returned an empty explanation for {feature_id or '<unknown feature>'}."
        )

    metadata = {
        "source": "openai_custom_generation",
        "generator_model": model_name,
        "usage": _extract_usage(response),
        "prompt_metadata": prompt_metadata,
    }
    return explanation, metadata


def generate_explanation_via_anthropic_from_record(
    *,
    client: httpx.Client,
    record: dict[str, Any],
    model_name: str,
    anthropic_api_key: str,
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
    system_prompt: str = DEFAULT_FEATURE_EXPLAINER_SYSTEM_PROMPT,
    max_context_examples: int = 5,
    max_chars_per_context: int = 1200,
    include_existing_explanation: bool = False,
    temperature: float | None = None,
    max_output_tokens: int = 128,
    request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
    request_backoff_seconds: float = DEFAULT_REQUEST_BACKOFF_SECONDS,
) -> tuple[str, dict[str, Any]]:
    prompt, prompt_metadata = _build_feature_explanation_prompt(
        record,
        max_context_examples=max_context_examples,
        max_chars_per_context=max_chars_per_context,
        include_existing_explanation=include_existing_explanation,
    )

    url = f"{anthropic_base_url.rstrip('/')}/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": anthropic_api_key,
        "anthropic-version": anthropic_version,
    }
    request_body: dict[str, Any] = {
        "model": model_name,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
    }
    if temperature is not None:
        request_body["temperature"] = temperature

    response = _request_with_retries(
        client,
        "POST",
        url,
        headers=headers,
        json_body=request_body,
        timeout=300.0,
        max_retries=request_max_retries,
        backoff_seconds=request_backoff_seconds,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        response_body = _truncate_text(response.text, 4000) or "<empty response body>"
        feature_id = str(record.get("feature_id") or "").strip()
        raise RuntimeError(
            "Anthropic generation failed "
            f"for feature {feature_id or '<unknown feature>'} using model {model_name} "
            f"with HTTP {response.status_code} at {url}. "
            f"Response body: {response_body}"
        ) from exc

    response_json = response.json()
    explanation = _extract_anthropic_text(response_json)
    if not explanation:
        feature_id = str(record.get("feature_id") or "").strip()
        raise RuntimeError(
            f"Anthropic API returned an empty explanation for {feature_id or '<unknown feature>'}."
        )

    metadata = {
        "source": "anthropic_custom_generation",
        "generator_model": model_name,
        "usage": _extract_anthropic_usage(response_json),
        "prompt_metadata": prompt_metadata,
        "anthropic_version": anthropic_version,
    }
    return explanation, metadata


def load_neuronpedia_baseline_rows_by_feature_id(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_feature_id: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        feature_id = str(row.get("feature_id") or "").strip()
        if not feature_id:
            continue
        rows_by_feature_id.setdefault(feature_id, []).append(row)
    return rows_by_feature_id


def select_baseline_row_by_model(
    rows: list[dict[str, Any]],
    preferred_model: str | None,
) -> dict[str, Any] | None:
    if not rows:
        return None
    if preferred_model is None:
        return rows[0]

    preferred = preferred_model.strip().lower()
    for row in rows:
        requested_model = str(row.get("requested_model") or "").strip().lower()
        if requested_model == preferred:
            return row
    for row in rows:
        requested_model = str(row.get("requested_model") or "").strip().lower()
        if requested_model.startswith(preferred):
            return row
    for row in rows:
        requested_model = str(row.get("requested_model") or "").strip().lower()
        if preferred in requested_model:
            return row
    return None


def generate_openai_feature_explanations_on_jsonl(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    llm_model: str,
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_base_url: str | None = None,
    system_prompt: str = DEFAULT_FEATURE_EXPLAINER_SYSTEM_PROMPT,
    max_features: int | None = None,
    max_context_examples: int = 5,
    max_chars_per_context: int = 1200,
    include_existing_explanation: bool = False,
    temperature: float | None = None,
    max_output_tokens: int = 128,
    resume: bool = False,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    records = load_jsonl(input_jsonl)
    if max_features is not None:
        records = records[:max_features]
    records = _records_with_feature_id(records)
    existing_keys = load_existing_request_keys(output_jsonl) if resume else set()
    pending_records: list[dict[str, Any]] = []
    resumed_count = 0
    for record in records:
        feature_id = str(record.get("feature_id") or "").strip()
        key = _request_key(feature_id, llm_model, None)
        if key in existing_keys:
            resumed_count += 1
            continue
        pending_records.append(record)

    client = _build_openai_client(
        api_key_env=openai_api_key_env,
        base_url=openai_base_url,
    )

    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    progress = _create_progress_bar(
        total=len(pending_records),
        desc=f"Generating explanations via {llm_model}",
        unit="feature",
    )
    rows_written = 0
    try:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "a" if resume and output_jsonl.exists() else "w"
        with output_jsonl.open(file_mode, encoding="utf-8") as writer:
            for index, record in enumerate(pending_records):
                feature_id = str(record.get("feature_id") or "").strip()

                explanation, generation_metadata = generate_explanation_via_openai_from_record(
                    client=client,
                    record=record,
                    model_name=llm_model,
                    system_prompt=system_prompt,
                    max_context_examples=max_context_examples,
                    max_chars_per_context=max_chars_per_context,
                    include_existing_explanation=include_existing_explanation,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )

                usage = dict(generation_metadata.get("usage") or {})
                for key in usage_totals:
                    usage_totals[key] += int(usage.get(key, 0) or 0)

                append_jsonl_rows(
                    writer,
                    [
                        {
                            "feature_id": feature_id,
                            "feature_url": record.get("feature_url"),
                            "requested_model": llm_model,
                            "explanation": explanation,
                            "metadata": _build_output_metadata(
                                record,
                                metadata_field="explanation_generation_metadata",
                                metadata_value=generation_metadata,
                            ),
                        }
                    ],
                )
                rows_written += 1

                if progress is not None:
                    progress.update(1)

                if sleep_seconds > 0 and index < len(pending_records) - 1:
                    time.sleep(sleep_seconds)
    finally:
        if progress is not None:
            progress.close()

    total_rows = len(existing_keys) + rows_written
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "llm_model": llm_model,
        "num_rows": total_rows,
        "num_rows_written": rows_written,
        "num_rows_resumed": resumed_count,
        "max_context_examples": max_context_examples,
        "max_chars_per_context": max_chars_per_context,
        "include_existing_explanation": include_existing_explanation,
        "openai_base_url": openai_base_url,
        "resume": resume,
        "usage": usage_totals,
    }


def generate_anthropic_feature_explanations_on_jsonl(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    models: list[str],
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY",
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
    system_prompt: str = DEFAULT_FEATURE_EXPLAINER_SYSTEM_PROMPT,
    max_features: int | None = None,
    max_context_examples: int = 5,
    max_chars_per_context: int = 1200,
    include_existing_explanation: bool = False,
    temperature: float | None = None,
    max_output_tokens: int = 128,
    request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
    request_backoff_seconds: float = DEFAULT_REQUEST_BACKOFF_SECONDS,
    trust_env: bool = True,
    resume: bool = False,
    continue_on_error: bool = False,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    anthropic_api_key = os.environ.get(anthropic_api_key_env)
    if not anthropic_api_key:
        raise RuntimeError(f"Environment variable {anthropic_api_key_env!r} is not set.")

    records = load_jsonl(input_jsonl)
    if max_features is not None:
        records = records[:max_features]
    records = _records_with_feature_id(records)

    existing_keys = load_existing_request_keys(output_jsonl) if resume else set()
    jobs: list[tuple[int, dict[str, Any], list[str]]] = []
    summary: dict[str, int] = {}
    resumed_count = 0
    for index, record in enumerate(records):
        feature_id = str(record.get("feature_id") or "").strip()
        pending_models: list[str] = []
        for model_name in models:
            key = _request_key(feature_id, model_name, None)
            if key in existing_keys:
                summary[f"resumed::{model_name}"] = summary.get(f"resumed::{model_name}", 0) + 1
                resumed_count += 1
                continue
            pending_models.append(model_name)
        if pending_models:
            jobs.append((index, record, pending_models))

    total_requests = sum(len(pending_models) for _, _, pending_models in jobs)
    progress = _create_progress_bar(
        total=total_requests,
        desc="Generating Anthropic explanations",
        unit="req",
    )

    def process_record(
        record: dict[str, Any],
        pending_models: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, int], int]:
        feature_id = str(record.get("feature_id") or "").strip()
        local_rows: list[dict[str, Any]] = []
        local_summary: dict[str, int] = {}
        with httpx.Client(timeout=30.0, trust_env=trust_env) as client:
            for model_name in pending_models:
                try:
                    explanation, generation_metadata = generate_explanation_via_anthropic_from_record(
                        client=client,
                        record=record,
                        model_name=model_name,
                        anthropic_api_key=anthropic_api_key,
                        anthropic_base_url=anthropic_base_url,
                        anthropic_version=anthropic_version,
                        system_prompt=system_prompt,
                        max_context_examples=max_context_examples,
                        max_chars_per_context=max_chars_per_context,
                        include_existing_explanation=include_existing_explanation,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        request_max_retries=request_max_retries,
                        request_backoff_seconds=request_backoff_seconds,
                    )
                    local_summary[f"generated::{model_name}"] = local_summary.get(f"generated::{model_name}", 0) + 1
                    row = {
                        "feature_id": feature_id,
                        "feature_url": record.get("feature_url"),
                        "requested_model": model_name,
                        "explanation": explanation,
                        "metadata": _build_output_metadata(
                            record,
                            metadata_field="explanation_generation_metadata",
                            metadata_value=generation_metadata,
                        ),
                    }
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    local_summary[f"failed::{model_name}"] = local_summary.get(f"failed::{model_name}", 0) + 1
                    _report_null_explanation_error(
                        feature_id=feature_id,
                        requested_model=model_name,
                        error=exc,
                    )
                    row = build_error_output_row(
                        record=record,
                        requested_model=model_name,
                        metadata_field="explanation_generation_metadata",
                        error=exc,
                    )
                local_rows.append(row)
        return local_rows, local_summary, len(pending_models)

    rows_written = 0
    try:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "a" if resume and output_jsonl.exists() else "w"
        with output_jsonl.open(file_mode, encoding="utf-8") as writer:
            for job_index, (_, record, pending_models) in enumerate(jobs):
                local_rows, local_summary, processed_requests = process_record(record, pending_models)
                append_jsonl_rows(writer, local_rows)
                rows_written += len(local_rows)
                _merge_summary_counts(summary, local_summary)
                if progress is not None:
                    progress.update(processed_requests)

                if sleep_seconds > 0 and job_index < len(jobs) - 1:
                    time.sleep(sleep_seconds)
    finally:
        if progress is not None:
            progress.close()

    total_rows = len(existing_keys) + rows_written
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "models": list(models),
        "num_rows": total_rows,
        "num_rows_written": rows_written,
        "num_rows_resumed": resumed_count,
        "summary": summary,
        "anthropic_base_url": anthropic_base_url,
        "anthropic_version": anthropic_version,
        "request_max_retries": request_max_retries,
        "request_backoff_seconds": request_backoff_seconds,
        "trust_env": trust_env,
        "resume": resume,
        "continue_on_error": continue_on_error,
    }


def generate_explanation_via_direct_provider_fallback(
    *,
    record: dict[str, Any],
    requested_model_name: str,
    openai_api_key_env: str,
    openai_base_url: str | None,
    anthropic_api_key_env: str,
    anthropic_base_url: str,
    anthropic_version: str,
    system_prompt: str,
    max_context_examples: int,
    max_chars_per_context: int,
    include_existing_explanation: bool,
    temperature: float | None,
    max_output_tokens: int,
    request_max_retries: int,
    request_backoff_seconds: float,
    trust_env: bool,
) -> tuple[str, dict[str, Any]] | None:
    provider, provider_model_name = _resolve_direct_provider_model_name(requested_model_name)
    if provider is None or provider_model_name is None:
        return None

    if provider == "openai":
        try:
            client = _build_openai_client(
                api_key_env=openai_api_key_env,
                base_url=openai_base_url,
            )
        except Exception:
            return None

        explanation, metadata = generate_explanation_via_openai_from_record(
            client=client,
            record=record,
            model_name=provider_model_name,
            system_prompt=system_prompt,
            max_context_examples=max_context_examples,
            max_chars_per_context=max_chars_per_context,
            include_existing_explanation=include_existing_explanation,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        metadata = {
            **metadata,
            "source": "provider_fallback_openai",
            "requested_model_name": requested_model_name,
            "provider_model_name": provider_model_name,
        }
        return explanation, metadata

    anthropic_api_key = os.environ.get(anthropic_api_key_env)
    if not anthropic_api_key:
        return None

    with httpx.Client(timeout=30.0, trust_env=trust_env) as client:
        explanation, metadata = generate_explanation_via_anthropic_from_record(
            client=client,
            record=record,
            model_name=provider_model_name,
            anthropic_api_key=anthropic_api_key,
            anthropic_base_url=anthropic_base_url,
            anthropic_version=anthropic_version,
            system_prompt=system_prompt,
            max_context_examples=max_context_examples,
            max_chars_per_context=max_chars_per_context,
            include_existing_explanation=include_existing_explanation,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            request_max_retries=request_max_retries,
            request_backoff_seconds=request_backoff_seconds,
        )
    metadata = {
        **metadata,
        "source": "provider_fallback_anthropic",
        "requested_model_name": requested_model_name,
        "provider_model_name": provider_model_name,
    }
    return explanation, metadata


def generate_neuronpedia_hosted_explanations_on_jsonl(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    models: list[str],
    explanation_type: str = DEFAULT_NEURONPEDIA_HOSTED_EXPLANATION_TYPE,
    neuronpedia_base_url: str = DEFAULT_NEURONPEDIA_HOSTED_BASE_URL,
    neuronpedia_api_key_env: str = "NEURONPEDIA_API_KEY",
    feature_api_base_url: str = "https://www.neuronpedia.org/api/feature",
    max_features: int | None = None,
    max_concurrency: int = 1,
    request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
    request_backoff_seconds: float = DEFAULT_REQUEST_BACKOFF_SECONDS,
    trust_env: bool = True,
    resume: bool = False,
    continue_on_error: bool = False,
    fallback_to_direct_provider: bool = False,
    fallback_openai_api_key_env: str = "OPENAI_API_KEY",
    fallback_openai_base_url: str | None = None,
    fallback_anthropic_api_key_env: str = "ANTHROPIC_API_KEY",
    fallback_anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
    fallback_anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    api_key = os.environ.get(neuronpedia_api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {neuronpedia_api_key_env!r} is not set.")

    records = load_jsonl(input_jsonl)
    if max_features is not None:
        records = records[:max_features]
    records = _records_with_feature_id(records)

    existing_keys = load_existing_request_keys(output_jsonl) if resume else set()
    jobs: list[tuple[int, dict[str, Any], list[str]]] = []
    summary: dict[str, int] = {}
    resumed_count = 0
    for index, record in enumerate(records):
        feature_id = str(record.get("feature_id") or "").strip()
        pending_models: list[str] = []
        for model_name in models:
            key = _request_key(feature_id, model_name, explanation_type)
            if key in existing_keys:
                summary[f"resumed::{model_name}"] = summary.get(f"resumed::{model_name}", 0) + 1
                resumed_count += 1
                continue
            pending_models.append(model_name)
        if pending_models:
            jobs.append((index, record, pending_models))

    total_requests = sum(len(pending_models) for _, _, pending_models in jobs)
    progress = _create_progress_bar(
        total=total_requests,
        desc="Generating hosted Neuronpedia explanations",
        unit="req",
    )
    max_concurrency = max(1, int(max_concurrency))

    def process_record(
        record: dict[str, Any],
        pending_models: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, int], int]:
        feature_id = str(record.get("feature_id") or "").strip()
        local_rows: list[dict[str, Any]] = []
        local_summary: dict[str, int] = {}
        with httpx.Client(timeout=30.0, trust_env=trust_env) as client:
            for model_name in pending_models:
                try:
                    explanation, metadata = generate_explanation_via_neuronpedia_hosted_api(
                        client=client,
                        feature_id=feature_id,
                        explanation_type=explanation_type,
                        explanation_model_name=model_name,
                        base_url=neuronpedia_base_url,
                        api_key=api_key,
                        feature_api_base_url=feature_api_base_url,
                        request_max_retries=request_max_retries,
                        request_backoff_seconds=request_backoff_seconds,
                    )
                except Exception as exc:
                    fallback_result = None
                    if fallback_to_direct_provider:
                        try:
                            fallback_result = generate_explanation_via_direct_provider_fallback(
                                record=record,
                                requested_model_name=model_name,
                                openai_api_key_env=fallback_openai_api_key_env,
                                openai_base_url=fallback_openai_base_url,
                                anthropic_api_key_env=fallback_anthropic_api_key_env,
                                anthropic_base_url=fallback_anthropic_base_url,
                                anthropic_version=fallback_anthropic_version,
                                system_prompt=DEFAULT_FEATURE_EXPLAINER_SYSTEM_PROMPT,
                                max_context_examples=5,
                                max_chars_per_context=1200,
                                include_existing_explanation=False,
                                temperature=None,
                                max_output_tokens=128,
                                request_max_retries=request_max_retries,
                                request_backoff_seconds=request_backoff_seconds,
                                trust_env=trust_env,
                            )
                        except Exception:
                            fallback_result = None

                    if fallback_result is not None:
                        explanation, metadata = fallback_result
                    else:
                        if not continue_on_error:
                            raise
                        local_summary[f"failed::{model_name}"] = local_summary.get(f"failed::{model_name}", 0) + 1
                        _report_null_explanation_error(
                            feature_id=feature_id,
                            requested_model=model_name,
                            requested_explanation_type=explanation_type,
                            error=exc,
                        )
                        local_rows.append(
                            build_error_output_row(
                                record=record,
                                requested_model=model_name,
                                metadata_field="neuronpedia_explanation_metadata",
                                requested_explanation_type=explanation_type,
                                error=exc,
                            )
                        )
                        continue

                source = str(metadata.get("source") or "")
                if source in {"neuronpedia_hosted_existing", "neuronpedia_hosted_existing_precheck"}:
                    local_summary[f"existing::{model_name}"] = local_summary.get(f"existing::{model_name}", 0) + 1
                elif source in {"provider_fallback_openai", "provider_fallback_anthropic"}:
                    local_summary[f"fallback_generated::{model_name}"] = (
                        local_summary.get(f"fallback_generated::{model_name}", 0) + 1
                    )
                else:
                    local_summary[f"generated::{model_name}"] = local_summary.get(f"generated::{model_name}", 0) + 1
                local_rows.append(
                    {
                        "feature_id": feature_id,
                        "feature_url": record.get("feature_url"),
                        "requested_model": model_name,
                        "requested_explanation_type": explanation_type,
                        "explanation": explanation,
                        "metadata": _build_output_metadata(
                            record,
                            metadata_field="neuronpedia_explanation_metadata",
                            metadata_value=metadata,
                        ),
                    }
                )
        return local_rows, local_summary, len(pending_models)

    rows_written = 0
    try:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "a" if resume and output_jsonl.exists() else "w"
        with output_jsonl.open(file_mode, encoding="utf-8") as writer:
            if max_concurrency <= 1 or len(jobs) <= 1:
                for job_index, (_, record, pending_models) in enumerate(jobs):
                    local_rows, local_summary, processed_requests = process_record(record, pending_models)
                    append_jsonl_rows(writer, local_rows)
                    rows_written += len(local_rows)
                    _merge_summary_counts(summary, local_summary)
                    if progress is not None:
                        progress.update(processed_requests)

                    if sleep_seconds > 0 and job_index < len(jobs) - 1:
                        time.sleep(sleep_seconds)
            else:
                with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                    future_to_job = {
                        executor.submit(process_record, record, pending_models): (index, record)
                        for index, record, pending_models in jobs
                    }
                    for future in as_completed(future_to_job):
                        local_rows, local_summary, processed_requests = future.result()
                        append_jsonl_rows(writer, local_rows)
                        rows_written += len(local_rows)
                        _merge_summary_counts(summary, local_summary)
                        if progress is not None:
                            progress.update(processed_requests)
    finally:
        if progress is not None:
            progress.close()

    total_rows = len(existing_keys) + rows_written
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "models": list(models),
        "num_rows": total_rows,
        "num_rows_written": rows_written,
        "num_rows_resumed": resumed_count,
        "summary": summary,
        "explanation_type": explanation_type,
        "max_concurrency": max_concurrency,
        "request_max_retries": request_max_retries,
        "request_backoff_seconds": request_backoff_seconds,
        "trust_env": trust_env,
        "resume": resume,
        "continue_on_error": continue_on_error,
        "fallback_to_direct_provider": fallback_to_direct_provider,
        "neuronpedia_base_url": neuronpedia_base_url,
    }


def generate_neuronpedia_model_explanations_on_jsonl(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    models: list[str],
    feature_api_base_url: str = "https://www.neuronpedia.org/api/feature",
    feature_api_key_env: str = "NEURONPEDIA_API_KEY",
    autointerp_base_url: str | None = None,
    autointerp_secret_env: str = "AUTOINTERP_SERVER_SECRET",
    openrouter_api_key_env: str = "OPENROUTER_API_KEY",
    max_features: int | None = None,
    max_activation_examples: int = 20,
    max_concurrency: int = 1,
    request_max_retries: int = DEFAULT_REQUEST_MAX_RETRIES,
    request_backoff_seconds: float = DEFAULT_REQUEST_BACKOFF_SECONDS,
    trust_env: bool = True,
    resume: bool = False,
    continue_on_error: bool = False,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    feature_api_key = os.environ.get(feature_api_key_env)
    autointerp_secret = os.environ.get(autointerp_secret_env)
    openrouter_key = os.environ.get(openrouter_api_key_env)
    if autointerp_base_url is not None and not openrouter_key:
        raise RuntimeError(
            f"{openrouter_api_key_env} is required when autointerp_base_url is provided."
        )

    records = load_jsonl(input_jsonl)
    if max_features is not None:
        records = records[:max_features]
    records = _records_with_feature_id(records)

    existing_keys = load_existing_request_keys(output_jsonl) if resume else set()
    jobs: list[tuple[int, dict[str, Any], list[str]]] = []
    summary: dict[str, int] = {}
    resumed_count = 0
    for index, record in enumerate(records):
        feature_id = str(record.get("feature_id") or "").strip()
        pending_models: list[str] = []
        for model_name in models:
            key = _request_key(feature_id, model_name, None)
            if key in existing_keys:
                summary[f"resumed::{model_name}"] = summary.get(f"resumed::{model_name}", 0) + 1
                resumed_count += 1
                continue
            pending_models.append(model_name)
        if pending_models:
            jobs.append((index, record, pending_models))

    total_requests = sum(len(pending_models) for _, _, pending_models in jobs)
    progress = _create_progress_bar(
        total=total_requests,
        desc="Fetching/generating Neuronpedia explanations",
        unit="req",
    )
    max_concurrency = max(1, int(max_concurrency))

    def process_record(
        record: dict[str, Any],
        pending_models: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, int], int]:
        feature_id = str(record.get("feature_id") or "").strip()
        local_rows: list[dict[str, Any]] = []
        local_summary: dict[str, int] = {}
        with httpx.Client(timeout=30.0, trust_env=trust_env) as client:
            try:
                payload = fetch_feature_payload(
                    client=client,
                    feature_id=feature_id,
                    base_url=feature_api_base_url,
                    api_key=feature_api_key,
                    max_retries=request_max_retries,
                    backoff_seconds=request_backoff_seconds,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                for model_name in pending_models:
                    local_summary[f"failed::{model_name}"] = local_summary.get(f"failed::{model_name}", 0) + 1
                    _report_null_explanation_error(
                        feature_id=feature_id,
                        requested_model=model_name,
                        error=exc,
                    )
                    local_rows.append(
                        build_error_output_row(
                            record=record,
                            requested_model=model_name,
                            metadata_field="neuronpedia_explanation_metadata",
                            error=exc,
                        )
                    )
                return local_rows, local_summary, len(pending_models)

            for model_name in pending_models:
                try:
                    explanation, candidate = select_explanation_candidate_by_model(payload, model_name)
                    metadata: dict[str, Any]
                    if explanation:
                        metadata = {
                            "source": "existing",
                            "matched_candidate": candidate,
                        }
                        local_summary[f"existing::{model_name}"] = local_summary.get(f"existing::{model_name}", 0) + 1
                    else:
                        if autointerp_base_url is None:
                            local_summary[f"missing::{model_name}"] = local_summary.get(f"missing::{model_name}", 0) + 1
                            continue
                        explanation, metadata = generate_explanation_via_autointerp(
                            client=client,
                            payload=payload,
                            feature_id=feature_id,
                            model_name=model_name,
                            autointerp_base_url=autointerp_base_url,
                            autointerp_secret=autointerp_secret,
                            openrouter_key=str(openrouter_key),
                            max_activation_examples=max_activation_examples,
                            request_max_retries=request_max_retries,
                            request_backoff_seconds=request_backoff_seconds,
                        )
                        local_summary[f"generated::{model_name}"] = local_summary.get(f"generated::{model_name}", 0) + 1
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    local_summary[f"failed::{model_name}"] = local_summary.get(f"failed::{model_name}", 0) + 1
                    _report_null_explanation_error(
                        feature_id=feature_id,
                        requested_model=model_name,
                        error=exc,
                    )
                    local_rows.append(
                        build_error_output_row(
                            record=record,
                            requested_model=model_name,
                            metadata_field="neuronpedia_explanation_metadata",
                            error=exc,
                        )
                    )
                    continue

                local_rows.append(
                    {
                        "feature_id": feature_id,
                        "feature_url": record.get("feature_url"),
                        "requested_model": model_name,
                        "explanation": explanation,
                        "metadata": _build_output_metadata(
                            record,
                            metadata_field="neuronpedia_explanation_metadata",
                            metadata_value=metadata,
                        ),
                    }
                )
        return local_rows, local_summary, len(pending_models)

    rows_written = 0
    try:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "a" if resume and output_jsonl.exists() else "w"
        with output_jsonl.open(file_mode, encoding="utf-8") as writer:
            if max_concurrency <= 1 or len(jobs) <= 1:
                for job_index, (_, record, pending_models) in enumerate(jobs):
                    local_rows, local_summary, processed_requests = process_record(record, pending_models)
                    append_jsonl_rows(writer, local_rows)
                    rows_written += len(local_rows)
                    _merge_summary_counts(summary, local_summary)
                    if progress is not None:
                        progress.update(processed_requests)

                    if sleep_seconds > 0 and job_index < len(jobs) - 1:
                        time.sleep(sleep_seconds)
            else:
                with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                    future_to_job = {
                        executor.submit(process_record, record, pending_models): (index, record)
                        for index, record, pending_models in jobs
                    }
                    for future in as_completed(future_to_job):
                        local_rows, local_summary, processed_requests = future.result()
                        append_jsonl_rows(writer, local_rows)
                        rows_written += len(local_rows)
                        _merge_summary_counts(summary, local_summary)
                        if progress is not None:
                            progress.update(processed_requests)
    finally:
        if progress is not None:
            progress.close()

    total_rows = len(existing_keys) + rows_written
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "models": list(models),
        "num_rows": total_rows,
        "num_rows_written": rows_written,
        "num_rows_resumed": resumed_count,
        "summary": summary,
        "max_concurrency": max_concurrency,
        "request_max_retries": request_max_retries,
        "request_backoff_seconds": request_backoff_seconds,
        "trust_env": trust_env,
        "resume": resume,
        "continue_on_error": continue_on_error,
        "autointerp_base_url": autointerp_base_url,
    }
