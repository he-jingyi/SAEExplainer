#!/usr/bin/env python3
"""Run the vec-only explanation -> text -> activation -> preference pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the vec-only preference-construction pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "explamma_15" / "config_preference_explamma_15_dpo_train.yaml",
        help="Path to the vec-only preference pipeline config.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="start",
        choices=["start", "candidates", "generated_texts", "scored_texts", "scored_explanations", "preferences"],
        help=(
            "Resume from an existing intermediate stage. "
            "'candidates' means candidate_explanations.jsonl already exists and generation should continue from texts."
        ),
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default=None,
        choices=["candidates", "generated_texts", "scored_texts", "scored_explanations", "preferences"],
        help="Stop after completing the specified stage and writing its output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.preference.config import load_preference_config
    from src.preference.pipeline import run_vec_only_preference_pipeline

    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    cfg = load_preference_config(config_path, PROJECT_ROOT)
    summary = run_vec_only_preference_pipeline(
        cfg,
        resume_from=args.resume_from,
        stop_after=args.stop_after,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
