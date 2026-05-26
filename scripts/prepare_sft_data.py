#!/usr/bin/env python3
"""Build an SFT warm-up dataset from Neuronpedia feature pages."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Neuronpedia feature data and build a JSONL SFT warm-up dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "explamma_15" / "config_data_explamma_15.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=3000,
        help="Override the total number of valid samples to collect.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override the sampling seed.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Override the dataset output path.",
    )
    parser.add_argument(
        "--top-k-activations",
        type=int,
        default=5,
        help="Override how many top activating contexts to keep per feature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.data_collection.neuronpedia_api import collect_dataset_from_config

    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    output_jsonl = None
    if args.output_jsonl is not None:
        output_jsonl = (
            args.output_jsonl
            if args.output_jsonl.is_absolute()
            else (PROJECT_ROOT / args.output_jsonl).resolve()
        )

    summary = asyncio.run(
        collect_dataset_from_config(
            config_path=config_path,
            project_root=PROJECT_ROOT,
            total_samples_override=args.total_samples,
            random_seed_override=args.random_seed,
            output_jsonl_override=output_jsonl,
            top_k_activations_override=args.top_k_activations,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
