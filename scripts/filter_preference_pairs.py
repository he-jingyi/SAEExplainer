#!/usr/bin/env python3
"""Filter scored explanations into preference pairs with explicit heuristics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter scored_explanations.jsonl into a cleaner preference dataset."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to scored_explanations.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Path to output filtered preferences JSONL. Defaults next to input.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to output summary JSON. Defaults next to input.",
    )
    parser.add_argument(
        "--dropped-jsonl",
        type=Path,
        default=None,
        help="Optional path to save dropped feature groups with reasons.",
    )
    parser.add_argument(
        "--min-chosen-activation",
        type=float,
        default=10.0,
        help="Drop a feature if the best explanation score is below this threshold.",
    )
    parser.add_argument(
        "--min-group-spread-ratio",
        type=float,
        default=0.10,
        help="Drop a feature if (max_score - min_score) / max_score <= this threshold.",
    )
    parser.add_argument(
        "--min-relative-gap",
        type=float,
        default=0.15,
        help="Prefer rejected explanations in the lower half whose relative gap to chosen exceeds this threshold.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_path = args.input_jsonl
    output_path = args.output_jsonl or input_path.with_name("filtered_preferences.jsonl")
    summary_path = args.summary_json or input_path.with_name("filtered_preferences.summary.json")
    dropped_path = args.dropped_jsonl or input_path.with_name("filtered_preferences.dropped.jsonl")

    from src.preference.filtering import filter_scored_explanations_file

    summary = filter_scored_explanations_file(
        input_jsonl=input_path,
        output_jsonl=output_path,
        summary_json=summary_path,
        dropped_jsonl=dropped_path,
        min_chosen_activation=args.min_chosen_activation,
        min_group_spread_ratio=args.min_group_spread_ratio,
        min_relative_gap=args.min_relative_gap,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
