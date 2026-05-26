#!/usr/bin/env python3
"""Run local generation-only evaluation for a DPO checkpoint or explanation file."""

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
        description=(
            "Evaluate a trained DPO checkpoint on a JSONL test set using "
            "local SAE activation scoring for generation-only metrics."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=False,
        default=None,
        help="Path to the DPO checkpoint directory to evaluate. Optional when --explanations-jsonl is set.",
    )
    parser.add_argument(
        "--sft-config",
        type=Path,
        required=True,
        help="Path to the SFT config used to build the explainer model.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "test.jsonl",
        help="JSONL file containing feature metadata and activation contexts.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file for per-feature detailed evaluation results.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Optional JSON file for aggregate summary statistics.",
    )
    parser.add_argument(
        "--generated-explanations-jsonl",
        type=Path,
        default=None,
        help="Optional shared cache file for generated checkpoint explanations.",
    )
    parser.add_argument(
        "--explanations-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL of pre-generated explanations to evaluate directly.",
    )
    parser.add_argument(
        "--explanation-field",
        type=str,
        default="explanation",
        help="Field name in --explanations-jsonl that contains the explanation text.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Optional progress snapshot JSON path.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Limit evaluation to the first N features in the input JSONL.",
    )
    parser.add_argument(
        "--sample-features",
        type=int,
        default=None,
        help="Randomly sample N features from the candidate set before evaluation.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when sampling features.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model loading and local activation scoring.",
    )
    parser.add_argument(
        "--explainer-max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens used when the DPO model generates explanations.",
    )
    parser.add_argument(
        "--explanation-batch-size",
        type=int,
        default=8,
        help="Batch size for checkpoint explanation generation before evaluation.",
    )
    parser.add_argument(
        "--generation-llm-model",
        type=str,
        default="gpt-5.4-mini",
        help="OpenAI model used to generate evaluation texts from explanations.",
    )
    parser.add_argument(
        "--num-generation-examples",
        type=int,
        default=10,
        help="Number of texts to generate per explanation for generation evaluation.",
    )
    parser.add_argument(
        "--fallback-activation-threshold",
        type=float,
        default=8.0,
        help="Fallback threshold when test.jsonl does not provide top activation contexts.",
    )
    parser.add_argument(
        "--openai-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI API key.",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="Optional base URL for compatible OpenAI-style endpoints.",
    )
    parser.add_argument(
        "--generation-api-max-concurrency",
        type=int,
        default=4,
        help="Maximum concurrent API requests for generation evaluation.",
    )
    parser.add_argument(
        "--skip-generation-eval",
        action="store_true",
        help="Only generate checkpoint explanations and skip generation evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.dpo.eval import evaluate_dpo_on_jsonl, evaluate_explanations_on_jsonl

    checkpoint_dir = None
    if args.checkpoint_dir is not None:
        checkpoint_dir = (
            args.checkpoint_dir
            if args.checkpoint_dir.is_absolute()
            else (PROJECT_ROOT / args.checkpoint_dir)
        )
    sft_config = args.sft_config if args.sft_config.is_absolute() else (PROJECT_ROOT / args.sft_config)
    input_jsonl = args.input_jsonl if args.input_jsonl.is_absolute() else (PROJECT_ROOT / args.input_jsonl)
    explanations_jsonl = args.explanations_jsonl
    if explanations_jsonl is not None and not explanations_jsonl.is_absolute():
        explanations_jsonl = PROJECT_ROOT / explanations_jsonl

    output_jsonl = args.output_jsonl
    if output_jsonl is None:
        if explanations_jsonl is not None:
            output_jsonl = PROJECT_ROOT / "outputs" / "explanations_eval.jsonl"
        else:
            if checkpoint_dir is None:
                raise SystemExit("--checkpoint-dir is required unless --explanations-jsonl is set.")
            output_jsonl = checkpoint_dir / "dpo_eval.jsonl"
    elif not output_jsonl.is_absolute():
        output_jsonl = PROJECT_ROOT / output_jsonl

    output_summary = args.output_summary
    if output_summary is None:
        if explanations_jsonl is not None:
            output_summary = PROJECT_ROOT / "outputs" / "explanations_eval.summary.json"
        else:
            if checkpoint_dir is None:
                raise SystemExit("--checkpoint-dir is required unless --explanations-jsonl is set.")
            output_summary = checkpoint_dir / "dpo_eval.summary.json"
    elif not output_summary.is_absolute():
        output_summary = PROJECT_ROOT / output_summary

    generated_explanations_jsonl = args.generated_explanations_jsonl
    if generated_explanations_jsonl is not None and not generated_explanations_jsonl.is_absolute():
        generated_explanations_jsonl = PROJECT_ROOT / generated_explanations_jsonl

    progress_json = args.progress_json
    if progress_json is not None and not progress_json.is_absolute():
        progress_json = PROJECT_ROOT / progress_json

    if explanations_jsonl is not None:
        summary = evaluate_explanations_on_jsonl(
            sft_config_path=sft_config.resolve(),
            input_jsonl=input_jsonl.resolve(),
            explanations_jsonl=explanations_jsonl.resolve(),
            explanation_field=args.explanation_field,
            project_root=PROJECT_ROOT,
            output_jsonl=output_jsonl.resolve(),
            max_features=args.max_features,
            sample_features=args.sample_features,
            sample_seed=args.sample_seed,
            device=args.device,
            generation_llm_model=args.generation_llm_model,
            num_generation_examples=args.num_generation_examples,
            fallback_activation_threshold=args.fallback_activation_threshold,
            openai_api_key_env=args.openai_api_key_env,
            openai_base_url=args.openai_base_url,
            run_generation_eval=not args.skip_generation_eval,
            generation_api_max_concurrency=args.generation_api_max_concurrency,
            progress_json=progress_json.resolve() if progress_json is not None else None,
        )
    else:
        if checkpoint_dir is None:
            raise SystemExit("--checkpoint-dir is required unless --explanations-jsonl is set.")
        summary = evaluate_dpo_on_jsonl(
            sft_config_path=sft_config.resolve(),
            checkpoint_dir=checkpoint_dir.resolve(),
            input_jsonl=input_jsonl.resolve(),
            project_root=PROJECT_ROOT,
            output_jsonl=output_jsonl.resolve(),
            max_features=args.max_features,
            sample_features=args.sample_features,
            sample_seed=args.sample_seed,
            device=args.device,
            explainer_max_new_tokens=args.explainer_max_new_tokens,
            generation_llm_model=args.generation_llm_model,
            num_generation_examples=args.num_generation_examples,
            fallback_activation_threshold=args.fallback_activation_threshold,
            openai_api_key_env=args.openai_api_key_env,
            openai_base_url=args.openai_base_url,
            run_generation_eval=not args.skip_generation_eval,
            generation_api_max_concurrency=args.generation_api_max_concurrency,
            generated_explanations_jsonl=(
                generated_explanations_jsonl.resolve() if generated_explanations_jsonl is not None else None
            ),
            progress_json=progress_json.resolve() if progress_json is not None else None,
            explanation_batch_size=args.explanation_batch_size,
        )

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {output_jsonl}")
    print(f"Saved summary to {output_summary}")


if __name__ == "__main__":
    main()
