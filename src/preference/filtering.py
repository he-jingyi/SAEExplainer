"""Reusable filtering logic from scored explanations to preference pairs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_scored_explanations_rows(
    rows: list[dict[str, Any]],
    *,
    input_jsonl: str | None = None,
    output_jsonl: str | None = None,
    dropped_jsonl: str | None = None,
    min_chosen_activation: float = 10.0,
    min_group_spread_ratio: float = 0.15,
    min_relative_gap: float = 0.10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["feature_id"])].append(row)

    preferences: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    dropped_reasons: Counter[str] = Counter()

    for feature_id, items in grouped.items():
        items = sorted(items, key=lambda item: float(item["score"]), reverse=True)
        chosen = items[0]
        chosen_score = float(chosen["score"])
        min_score = float(items[-1]["score"])

        if chosen_score == 0.0:
            reason = "all_zero_scores"
            dropped_reasons[reason] += 1
            dropped.append(
                {
                    "feature_id": feature_id,
                    "reason": reason,
                    "chosen_score": chosen_score,
                    "min_score": min_score,
                    "scores": [float(item["score"]) for item in items],
                }
            )
            continue

        if chosen_score < min_chosen_activation:
            reason = "low_chosen_activation"
            dropped_reasons[reason] += 1
            dropped.append(
                {
                    "feature_id": feature_id,
                    "reason": reason,
                    "chosen_score": chosen_score,
                    "min_score": min_score,
                    "scores": [float(item["score"]) for item in items],
                }
            )
            continue

        spread_ratio = (chosen_score - min_score) / chosen_score
        if spread_ratio <= min_group_spread_ratio:
            reason = "low_group_spread"
            dropped_reasons[reason] += 1
            dropped.append(
                {
                    "feature_id": feature_id,
                    "reason": reason,
                    "chosen_score": chosen_score,
                    "min_score": min_score,
                    "spread_ratio": spread_ratio,
                    "scores": [float(item["score"]) for item in items],
                }
            )
            continue

        lower_half = items[len(items) // 2 :]
        valid_rejected = []
        for item in lower_half:
            rel_gap = (chosen_score - float(item["score"])) / chosen_score
            if rel_gap > min_relative_gap:
                valid_rejected.append((item, rel_gap))

        if valid_rejected:
            rejected, relative_gap = max(valid_rejected, key=lambda pair: float(pair[0]["score"]))
            rejected_selection = "lower_half_gap_filtered"
        else:
            rejected = items[-1]
            relative_gap = (chosen_score - float(rejected["score"])) / chosen_score
            rejected_selection = "fallback_lowest"

        preferences.append(
            {
                "feature_id": feature_id,
                "chosen_explanation": chosen["explanation"],
                "rejected_explanation": rejected["explanation"],
                "chosen_score": chosen_score,
                "rejected_score": float(rejected["score"]),
                "score_gap": chosen_score - float(rejected["score"]),
                "relative_gap": relative_gap,
                "spread_ratio": spread_ratio,
                "rejected_selection": rejected_selection,
                "chosen_best_text": chosen["best_text"],
                "rejected_best_text": rejected["best_text"],
                "metadata": dict(chosen.get("metadata") or {}),
            }
        )

    summary = {
        "input_jsonl": input_jsonl,
        "output_jsonl": output_jsonl,
        "dropped_jsonl": dropped_jsonl,
        "num_features_total": len(grouped),
        "num_preferences_kept": len(preferences),
        "num_features_dropped": len(dropped),
        "min_chosen_activation": min_chosen_activation,
        "min_group_spread_ratio": min_group_spread_ratio,
        "min_relative_gap": min_relative_gap,
        "dropped_reasons": dict(dropped_reasons),
    }
    return preferences, dropped, summary


def filter_scored_explanations_file(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    summary_json: Path,
    dropped_jsonl: Path | None = None,
    min_chosen_activation: float = 10.0,
    min_group_spread_ratio: float = 0.15,
    min_relative_gap: float = 0.10,
) -> dict[str, Any]:
    rows = load_jsonl(input_jsonl)
    preferences, dropped, summary = filter_scored_explanations_rows(
        rows,
        input_jsonl=str(input_jsonl),
        output_jsonl=str(output_jsonl),
        dropped_jsonl=str(dropped_jsonl) if dropped_jsonl is not None else None,
        min_chosen_activation=min_chosen_activation,
        min_group_spread_ratio=min_group_spread_ratio,
        min_relative_gap=min_relative_gap,
    )
    write_jsonl(output_jsonl, preferences)
    if dropped_jsonl is not None:
        write_jsonl(dropped_jsonl, dropped)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
