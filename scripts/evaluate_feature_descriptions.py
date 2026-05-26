#!/usr/bin/env python3
"""Evaluate feature descriptions with input/output metrics on SAEExplainer data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate feature descriptions using input and output metrics adapted "
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "exp2" / "test_small.jsonl",
        help="Feature dataset to evaluate. Defaults to data/exp2/test_small.jsonl.",
    )
    parser.add_argument(
        "--sft-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "exp2" / "config_sft_exp2.yaml",
        help="SFT config used to resolve the target model and SAE cache.",
    )
    parser.add_argument(
        "--explanation-source",
        type=str,
        choices=["dataset_field", "jsonl", "dpo_checkpoint"],
        default="dataset_field",
        help="Where explanations come from. Only this changes when comparing baselines.",
    )
    parser.add_argument(
        "--explanation-field",
        type=str,
        default="explanation",
        help="Field name to read from the dataset/external JSONL explanation rows.",
    )
    parser.add_argument(
        "--explanation-jsonl",
        type=Path,
        default=None,
        help="External explanation JSONL when --explanation-source=jsonl.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory when --explanation-source=dpo_checkpoint.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Where to write per-feature evaluation results.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        required=True,
        help="Where to write the aggregate summary JSON.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Optional progress snapshot JSON path for resume monitoring.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs instead of resuming from output-jsonl.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--sample-features", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--metric-llm-model",
        type=str,
        default="gpt-5.4-mini",
        help="LLM used for positive/negative list generation and output judging.",
    )
    parser.add_argument(
        "--openai-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key for the metric LLM.",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="Optional base URL for OpenAI-compatible endpoints.",
    )
    parser.add_argument("--steering-max-new-tokens", type=int, default=25)
    parser.add_argument("--steering-temperature", type=float, default=0.75)
    parser.add_argument(
        "--skip-input",
        action="store_true",
        help="Skip input metric evaluation and only run output metric.",
    )
    parser.add_argument(
        "--skip-output",
        action="store_true",
        help="Skip output metric evaluation and only run input metric.",
    )
    parser.add_argument(
        "--explanation-batch-size",
        type=int,
        default=8,
        help="Batch size for checkpoint explanation materialization in dpo_checkpoint mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_input and args.skip_output:
        raise SystemExit("Cannot set both --skip-input and --skip-output at the same time.")

    from src.feature_description_eval import (
        DPOCheckpointExplanationProvider,
        DatasetFieldExplanationProvider,
        FeatureDescriptionEvaluator,
        JsonlExplanationProvider,
    )
    from src.data_collection.neuronpedia_autointerp import load_jsonl
    from src.dpo.eval import _build_openai_client

    input_jsonl = args.input_jsonl if args.input_jsonl.is_absolute() else (PROJECT_ROOT / args.input_jsonl).resolve()
    sft_config = args.sft_config if args.sft_config.is_absolute() else (PROJECT_ROOT / args.sft_config).resolve()
    output_jsonl = args.output_jsonl if args.output_jsonl.is_absolute() else (PROJECT_ROOT / args.output_jsonl).resolve()
    output_summary = args.output_summary if args.output_summary.is_absolute() else (PROJECT_ROOT / args.output_summary).resolve()
    progress_json = None
    if args.progress_json is not None:
        progress_json = args.progress_json if args.progress_json.is_absolute() else (PROJECT_ROOT / args.progress_json).resolve()

    device = torch.device(args.device)
    metric_client = _build_openai_client(
        api_key_env=args.openai_api_key_env,
        base_url=args.openai_base_url,
    )

    if args.explanation_source == "dataset_field":
        provider = DatasetFieldExplanationProvider(args.explanation_field)
    elif args.explanation_source == "jsonl":
        if args.explanation_jsonl is None:
            raise SystemExit("--explanation-jsonl is required when --explanation-source=jsonl.")
        explanation_jsonl = (
            args.explanation_jsonl
            if args.explanation_jsonl.is_absolute()
            else (PROJECT_ROOT / args.explanation_jsonl).resolve()
        )
        provider = JsonlExplanationProvider(explanation_jsonl, field_name=args.explanation_field)
    else:
        if args.checkpoint_dir is None:
            raise SystemExit("--checkpoint-dir is required when --explanation-source=dpo_checkpoint.")
        checkpoint_dir = (
            args.checkpoint_dir
            if args.checkpoint_dir.is_absolute()
            else (PROJECT_ROOT / args.checkpoint_dir).resolve()
        )
        provider = DPOCheckpointExplanationProvider(
            sft_config_path=sft_config,
            checkpoint_dir=checkpoint_dir,
            input_jsonl=input_jsonl,
            project_root=PROJECT_ROOT,
            device=device,
            explanation_batch_size=args.explanation_batch_size,
        )
        # Materialize explanations first, then release the checkpoint explainer
        # before constructing the activation scorer / steerer, so we do not keep
        # two 27B-class models resident on the same GPU at once.
        records = load_jsonl(input_jsonl)
        active_indices = list(range(len(records)))
        if args.max_features is not None:
            active_indices = active_indices[: args.max_features]
        if args.sample_features is not None:
            rng = random.Random(args.sample_seed)
            active_indices = rng.sample(active_indices, k=min(args.sample_features, len(active_indices)))
        provider.materialize_explanations(active_indices)

    evaluator = FeatureDescriptionEvaluator(
        sft_config_path=sft_config,
        input_jsonl=input_jsonl,
        project_root=PROJECT_ROOT,
        provider=provider,
        metric_client=metric_client,
        input_metric_llm_model=args.metric_llm_model,
        output_metric_judge_model=args.metric_llm_model,
        device=device,
        max_features=args.max_features,
        sample_features=args.sample_features,
        sample_seed=args.sample_seed,
        steering_max_new_tokens=args.steering_max_new_tokens,
        steering_temperature=args.steering_temperature,
        run_input_metric=not args.skip_input,
        run_output_metric=not args.skip_output,
    )
    try:
        summary = evaluator.evaluate(
            output_jsonl=output_jsonl,
            output_summary=output_summary,
            progress_json=progress_json,
            overwrite=args.overwrite,
        )
    finally:
        evaluator.close()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved per-feature results to {output_jsonl}")
    print(f"Saved summary to {output_summary}")


if __name__ == "__main__":
    main()
