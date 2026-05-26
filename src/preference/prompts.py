"""Prompt helpers for the vec-only preference pipeline."""

from __future__ import annotations

from typing import Any


def build_vec_only_explanation_prompt(placeholder_text: str) -> str:
    return (
        "The internal neural feature represented by"
        f"{placeholder_text} is provided only through internal neural conditioning. "
        "Using only the injected feature representation, provide one clear, accurate explanation "
        "of the feature."
    )


def build_generator_prompt(explanation: str, num_samples: int) -> str:
    return (
        "You are an expert corpus-construction specialist and careful semantic writer. "
        "Your task is to write short texts that faithfully and unmistakably express one given semantic concept.\n\n"
        "Please carefully read the following [Core Concept]. Treat it as the exact semantic target, not a loose theme:\n"
        f"[{explanation}]\n\n"
        f"Based on this concept, create {num_samples} completely independent short texts.\n\n"
        "[Strict Requirements]\n"
        "1. Semantic fidelity is the top priority: every text must clearly express the exact core concept above. "
        "Do not broaden it into neighboring concepts, adjacent domains, or vaguely related themes.\n"
        "2. Stay on-concept even when varying the surface form: diversity should come from wording, scenario details, or discourse style, "
        "not from changing the underlying semantic meaning.\n"
        "3. Hidden and natural (show, don't tell): do not define the concept explicitly as if writing a dictionary. "
        "Do not say things like 'this text illustrates...'. Instead, embed the concept naturally into a concrete situation.\n"
        "4. Extremely concrete: use specific scenes, actions, terminology, or dialogue so that the concept is expressed with high semantic density.\n"
        "5. Do not introduce strong domain cues unless they are genuinely implied by the concept itself. "
        "For example, if the concept is about statistics, stay with statistics-like content; if it is about sports competition, do not drift into law, programming, or medicine.\n"
        "6. Avoid generic topical similarity: a text is bad if it merely feels loosely related while failing to make a careful reader recover the same concept.\n"
        "7. Short and compact: Each text must be between 15 and 30 words.\n"
        f"8. Moderate diversity only: the {num_samples} texts should not be duplicates, but they should remain tightly centered on the same concept. "
        "Different texts may vary in tone or local situation, but they should still point to the same semantic explanation.\n"
        "Before answering, silently check each text: would an independent reader likely summarize it with the same core concept above? "
        "If not, rewrite it.\n\n"
        "You must return only valid JSON with a top-level field named \"samples\". "
        "Do not include Markdown code fences or any extra explanation.\n"
        "{\n"
        '  "samples": [\n'
        '    "Text 1...",\n'
        '    "Text 2...",\n'
        '    "..."\n'
        "  ]\n"
        "}"
    )


def build_generator_prompt_sage_style_json(explanation: str, num_samples: int) -> str:
    return (
        "You are an expert test case generator for Sparse Autoencoder (SAE) features. "
        "Your task is to generate diverse test sentences that strongly activate the given feature.\n\n"
        "FEATURE DESCRIPTION:\n"
        f"{explanation}\n\n"
        "TASK:\n"
        f"Generate exactly {num_samples} diverse test sentences that should activate this feature HIGHLY. "
        "Each sentence must clearly demonstrate the pattern or behavior described in the feature description.\n\n"
        "REQUIREMENTS:\n"
        "1. Sentence length: 15-30 words each\n"
        "2. Diversity: Vary contexts, phrasings, subjects, and scenarios\n"
        "3. Clarity: Each sentence should unambiguously match the feature description\n"
        "4. Naturalness: Use natural, grammatically correct English\n"
        "5. Uniqueness: Avoid repetitive or overly similar sentences\n"
        "6. Coverage: Test different aspects and variations of the described pattern\n\n"
        "OUTPUT FORMAT:\n"
        "Return only valid JSON with a top-level field named \"samples\".\n"
        "Do not include Markdown code fences or any extra explanation.\n"
        "{\n"
        '  "samples": [\n'
        '    "Sentence 1...",\n'
        '    "Sentence 2...",\n'
        '    "..."\n'
        "  ]\n"
        "}\n\n"
        "IMPORTANT:\n"
        "- Output ONLY the JSON object, with no additional commentary\n"
        "- Do not include quotes around the whole response\n"
        "- Ensure each sentence is distinct and tests different variations of the pattern\n"
        "- Focus on sentences that would produce strong activation values for this feature\n"
    )


def tokenize_generation_prompt(tokenizer: Any, user_prompt: str) -> list[int]:
    messages = [{"role": "user", "content": user_prompt}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        kwargs = dict(tokenize=True, return_tensors=None, padding=False)
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            token_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                **kwargs,
            )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if not isinstance(token_ids, list):
            raise TypeError("Expected tokenizer.apply_chat_template(...) to yield token ids.")
        return token_ids

    raw_text = f"User: {user_prompt}\n\nAssistant:"
    token_ids = tokenizer.encode(raw_text, add_special_tokens=True)
    if not isinstance(token_ids, list):
        raise TypeError("Expected tokenizer.encode(...) to yield token ids.")
    return token_ids
