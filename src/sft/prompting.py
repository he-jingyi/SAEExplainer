"""Prompt builders for vector-conditioned explanation SFT."""

from __future__ import annotations


def build_hybrid_explainer_prompt(placeholder_text: str) -> str:
    return (
        "The internal neural feature represented by"
        f"{placeholder_text} is provided only through internal neural conditioning. "
        "Using the injected feature representation, provide one clear, accurate explanation "
        "of the feature."
    )
