"""Tokenization and label-masking helpers for hybrid explainer SFT."""

from __future__ import annotations

from typing import Any


def _normalize_token_ids(token_ids: Any, source_name: str) -> list[int]:
    if isinstance(token_ids, dict):
        if "input_ids" not in token_ids:
            raise TypeError(
                f"{source_name} returned a dict-like object without 'input_ids': {token_ids!r}"
            )
        token_ids = token_ids["input_ids"]

    if hasattr(token_ids, "keys") and hasattr(token_ids, "__getitem__"):
        try:
            keys = list(token_ids.keys())
        except Exception:
            keys = []
        if "input_ids" in keys:
            token_ids = token_ids["input_ids"]

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list):
        if not token_ids:
            return []

        if all(isinstance(item, int) for item in token_ids):
            return token_ids

        if len(token_ids) == 1 and isinstance(token_ids[0], list):
            inner = token_ids[0]
            if all(isinstance(item, int) for item in inner):
                return inner

    raise TypeError(
        f"Expected {source_name} to return token ids for a single sequence, "
        f"but got value of type {type(token_ids)!r}: {token_ids!r}"
    )


def ensure_placeholder_is_single_token(tokenizer: Any, placeholder_text: str) -> tuple[int, str]:
    token_ids = tokenizer.encode(placeholder_text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Placeholder {placeholder_text!r} is not a single token for this tokenizer: {token_ids}"
        )
    token_id = token_ids[0]
    token = tokenizer.convert_ids_to_tokens(token_ids)[0]
    return token_id, token


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> list[int]:
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        prompt_parts: list[str] = []
        for message in messages:
            role = message["role"].strip().capitalize()
            prompt_parts.append(f"{role}: {message['content'].strip()}")
        if add_generation_prompt:
            prompt_parts.append("Assistant:")
        text = "\n\n".join(prompt_parts)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        return _normalize_token_ids(token_ids, "tokenizer.encode(...)")

    kwargs = dict(
        tokenize=True,
        return_tensors=None,
        padding=False,
    )

    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    return _normalize_token_ids(token_ids, "tokenizer.apply_chat_template(...)")

def _resolve_assistant_start_idx(
    tokenizer: Any,
    assistant_response: str,
    prompt_only_ids: list[int],
    full_ids: list[int],
) -> tuple[int, str]:
    if len(prompt_only_ids) <= len(full_ids) and full_ids[: len(prompt_only_ids)] == prompt_only_ids:
        return len(prompt_only_ids), "prompt_only_prefix"

    prompt_preview = tokenizer.decode(prompt_only_ids[: min(len(prompt_only_ids), 64)], skip_special_tokens=False)
    full_preview = tokenizer.decode(full_ids[: min(len(full_ids), 64)], skip_special_tokens=False)
    raise ValueError(
        "Prompt-only chat template tokens are not a strict prefix of the full turn tokens. "
        f"assistant_response_preview={assistant_response[:80]!r}, "
        f"prompt_only_len={len(prompt_only_ids)}, full_len={len(full_ids)}, "
        f"prompt_preview={prompt_preview!r}, full_preview={full_preview!r}"
    )


def build_chat_turn_inputs(
    tokenizer: Any,
    user_prompt: str,
    assistant_response: str,
    max_length: int | None = None,
) -> dict[str, Any]:
    prompt_messages = [{"role": "user", "content": user_prompt}]
    prompt_only_ids = _apply_chat_template(
        tokenizer=tokenizer,
        messages=prompt_messages,
        add_generation_prompt=True,
    )

    full_messages = prompt_messages + [{"role": "assistant", "content": assistant_response}]
    full_ids = _apply_chat_template(
        tokenizer=tokenizer,
        messages=full_messages,
        add_generation_prompt=False,
    )

    if max_length is not None and len(full_ids) > max_length:
        raise ValueError(
            f"Tokenized sequence length {len(full_ids)} exceeds max_length={max_length}. "
            "Either raise max_length or reduce the prompt size."
        )

    assistant_start_idx, alignment_method = _resolve_assistant_start_idx(
        tokenizer=tokenizer,
        assistant_response=assistant_response,
        prompt_only_ids=prompt_only_ids,
        full_ids=full_ids,
    )

    return {
        "prompt_only_ids": prompt_only_ids,
        "full_ids": full_ids,
        "assistant_start_idx": assistant_start_idx,
        "assistant_alignment_method": alignment_method,
    }


def build_model_inputs(
    tokenizer: Any,
    user_prompt: str,
    assistant_response: str,
    placeholder_token_id: int,
    max_length: int,
) -> dict[str, Any]:
    turn_inputs = build_chat_turn_inputs(
        tokenizer=tokenizer,
        user_prompt=user_prompt,
        assistant_response=assistant_response,
        max_length=max_length,
    )
    full_ids = turn_inputs["full_ids"]
    assistant_start_idx = int(turn_inputs["assistant_start_idx"])
    labels = full_ids.copy()
    for index in range(assistant_start_idx):
        labels[index] = -100

    prompt_prefix_ids = full_ids[:assistant_start_idx]
    try:
        act_position = prompt_prefix_ids.index(placeholder_token_id)
    except ValueError as exc:
        preview = tokenizer.decode(prompt_prefix_ids[: min(len(prompt_prefix_ids), 128)], skip_special_tokens=False)
        raise ValueError(
            "Could not find the placeholder token inside the user prompt portion of the sequence. "
            f"placeholder_token_id={placeholder_token_id}, "
            f"prompt_prefix_preview={preview!r}"
        ) from exc

    return {
        "input_ids": full_ids,
        "labels": labels,
        "act_position": act_position,
        "assistant_start_idx": assistant_start_idx,
    }
