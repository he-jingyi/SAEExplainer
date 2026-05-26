#!/usr/bin/env python3
"""Build .pt reference log-prob caches for vec-only DPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute .pt reference log-prob caches for DPO.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the DPO YAML config.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "both"),
        default="both",
        help="Which split(s) to cache.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for reference forward passes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the reference model on. Defaults to config.model.reference_device or cuda.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the cache even if a compatible .pt cache already exists.",
    )
    return parser.parse_args()


def main() -> None:
    from src.dpo.config import load_dpo_config
    from src.dpo.reference_cache import (
        build_reference_cache_path,
        compute_reference_logp_cache,
        ensure_reference_logp_cache,
    )

    args = parse_args()
    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    cfg = load_dpo_config(config_path, PROJECT_ROOT)
    device = args.device or cfg.model.reference_device or "cuda"

    tasks: list[tuple[str, Path]] = []
    if args.split in {"train", "both"}:
        tasks.append(("train", cfg.data.preference_jsonl))
    if args.split in {"val", "both"} and cfg.data.val_preference_jsonl is not None:
        tasks.append(("val", cfg.data.val_preference_jsonl))

    summaries = []
    for split_name, input_jsonl in tasks:
        cache_path = build_reference_cache_path(cfg.trainer.output_dir, split_name)
        if args.force:
            summary = compute_reference_logp_cache(
                input_jsonl=input_jsonl,
                cache_path=cache_path,
                dpo_cfg=cfg,
                project_root=PROJECT_ROOT,
                batch_size=args.batch_size,
                device=device,
            )
        else:
            summary = ensure_reference_logp_cache(
                input_jsonl=input_jsonl,
                cache_path=cache_path,
                dpo_cfg=cfg,
                project_root=PROJECT_ROOT,
                batch_size=args.batch_size,
                device=device,
            )
        summary["split"] = split_name
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps({"completed": len(summaries), "tasks": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
