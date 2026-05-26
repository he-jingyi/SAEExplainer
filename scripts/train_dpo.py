#!/usr/bin/env python3
"""Train a vec-only DPO model using injected feature vectors."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vec-only DPO with injected feature vectors.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "explamma_15" / "config_dpo_explamma_15.yaml",
        help="Path to the DPO YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets and models, then exit before training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional Trainer checkpoint directory to resume full training state from "
            "(model/optimizer/scheduler/global_step)."
        ),
    )
    return parser.parse_args()


def _load_policy_or_reference_model(
    *,
    checkpoint_dir: Path,
    sft_config_path: Path,
    is_trainable: bool,
):
    from peft import PeftModel

    from src.sft.config import load_sft_config
    from src.sft.model import HybridExplainerModel
    from src.sft.train_utils import load_base_model, load_tokenizer

    sft_cfg = load_sft_config(sft_config_path, PROJECT_ROOT)
    tokenizer = load_tokenizer(sft_cfg.model)

    base_model = load_base_model(sft_cfg.model)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(base_model, "generation_config", None) is not None:
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    base_model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_dir),
        is_trainable=is_trainable,
    )
    model = HybridExplainerModel(
        base_model=base_model,
        inject_layer=sft_cfg.injection.inject_layer,
        steering_coefficient=sft_cfg.injection.steering_coefficient,
        use_projector=sft_cfg.injection.use_projector,
        vector_input_dim=sft_cfg.injection.vector_input_dim,
    )
    model.load_projector_from_checkpoint(checkpoint_dir)
    return sft_cfg, tokenizer, model


def _try_get_wandb_run() -> Any | None:
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    return getattr(wandb, "run", None)


def _iter_summary_scalars(prefix: str, payload: Any):
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from _iter_summary_scalars(next_prefix, value)
        return
    if isinstance(payload, (int, float, bool, str)) or payload is None:
        yield prefix, payload


def _save_eval_outputs_to_wandb(*paths: Path) -> None:
    run = _try_get_wandb_run()
    if run is None:
        return
    try:
        import wandb  # type: ignore
    except Exception:
        return

    for path in paths:
        if not path.exists():
            continue
        try:
            wandb.save(str(path), policy="now")
        except Exception:
            continue


def _write_summary_to_wandb(prefix: str, payload: dict[str, Any]) -> None:
    run = _try_get_wandb_run()
    if run is None:
        return

    for key, value in _iter_summary_scalars(prefix, payload):
        try:
            run.summary[key] = value
        except Exception:
            continue


def _looks_like_dpo_eval_jsonl(path: Path) -> bool:
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return False
            return (
                isinstance(record, dict)
                and "feature_id" in record
                and (
                    "top_activating_contexts" in record
                    or "feature_url" in record
                )
            )
    return False


def main() -> None:
    args = parse_args()

    from src.dpo.collator import VecOnlyDPOCollator
    from src.dpo.config import load_dpo_config
    from src.dpo.dataset import build_dpo_train_val_datasets
    from src.dpo.eval import evaluate_dpo_on_jsonl
    from src.dpo.reference_cache import build_reference_cache_path, ensure_reference_logp_cache
    from src.dpo.trainer import HybridDPOTrainer, TieBreakEarlyStoppingCallback
    from src.sft.eval import evaluate_checkpoint_preview, evaluate_checkpoint_preview_on_jsonl
    from src.sft.train_utils import build_training_arguments, set_seed

    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    cfg = load_dpo_config(config_path, PROJECT_ROOT)
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is not None and not resume_from_checkpoint.is_absolute():
        resume_from_checkpoint = (PROJECT_ROOT / resume_from_checkpoint).resolve()

    set_seed(cfg.trainer.seed)

    cache_batch_size = max(2, cfg.trainer.per_device_eval_batch_size)
    reference_cache_dir = cfg.trainer.output_dir
    train_reference_cache_path = build_reference_cache_path(reference_cache_dir, "train")
    val_reference_cache_path = build_reference_cache_path(reference_cache_dir, "val")
    reference_cache_device = cfg.model.reference_device or "cuda"

    train_cache_summary = ensure_reference_logp_cache(
        input_jsonl=cfg.data.preference_jsonl,
        cache_path=train_reference_cache_path,
        dpo_cfg=cfg,
        project_root=PROJECT_ROOT,
        batch_size=cache_batch_size,
        device=reference_cache_device,
    )
    val_cache_summary = None
    if cfg.data.val_preference_jsonl is not None:
        val_cache_summary = ensure_reference_logp_cache(
            input_jsonl=cfg.data.val_preference_jsonl,
            cache_path=val_reference_cache_path,
            dpo_cfg=cfg,
            project_root=PROJECT_ROOT,
            batch_size=cache_batch_size,
            device=reference_cache_device,
        )

    train_dataset, val_dataset = build_dpo_train_val_datasets(
        preference_jsonl=cfg.data.preference_jsonl,
        val_preference_jsonl=cfg.data.val_preference_jsonl,
        sft_config_path=cfg.model.sft_config_path,
        project_root=PROJECT_ROOT,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
        train_reference_logp_cache_path=train_reference_cache_path,
        val_reference_logp_cache_path=(
            val_reference_cache_path
            if cfg.data.val_preference_jsonl is not None
            else train_reference_cache_path
        ),
    )
    sft_cfg, tokenizer, policy_model = _load_policy_or_reference_model(
        checkpoint_dir=cfg.model.init_checkpoint_dir,
        sft_config_path=cfg.model.sft_config_path,
        is_trainable=True,
    )
    reference_dir = cfg.model.reference_checkpoint_dir or cfg.model.init_checkpoint_dir
    need_reference_model = not (
        train_dataset.all_have_cached_reference_logps
        and (len(val_dataset) == 0 or val_dataset.all_have_cached_reference_logps)
    )
    reference_model = None
    if need_reference_model:
        _, _, reference_model = _load_policy_or_reference_model(
            checkpoint_dir=reference_dir,
            sft_config_path=cfg.model.sft_config_path,
            is_trainable=False,
        )

    summary = {
        "config_path": str(config_path),
        "sft_config_path": str(cfg.model.sft_config_path),
        "policy_checkpoint": str(cfg.model.init_checkpoint_dir),
        "reference_checkpoint": str(reference_dir),
        "reference_device": cfg.model.reference_device,
        "reference_model_loaded": need_reference_model,
        "train_reference_cache_path": str(train_reference_cache_path),
        "val_reference_cache_path": (
            str(val_reference_cache_path) if cfg.data.val_preference_jsonl is not None else None
        ),
        "train_reference_cache_generated": bool(train_cache_summary["generated"]),
        "val_reference_cache_generated": (
            bool(val_cache_summary["generated"]) if val_cache_summary is not None else None
        ),
        "train_preference_jsonl": str(cfg.data.preference_jsonl),
        "val_preference_jsonl": str(cfg.data.val_preference_jsonl) if cfg.data.val_preference_jsonl else None,
        "test_jsonl": str(cfg.eval.test_jsonl) if cfg.eval.test_jsonl else None,
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "train_cached_reference_pairs": train_dataset.num_cached_reference_pairs,
        "val_cached_reference_pairs": val_dataset.num_cached_reference_pairs,
        "beta": cfg.loss.beta,
        "best_model_metric": cfg.trainer.metric_for_best_model,
        "best_model_metric_greater_is_better": cfg.trainer.greater_is_better,
        "best_model_secondary_metric": cfg.trainer.secondary_metric_for_best_model,
        "best_model_secondary_greater_is_better": cfg.trainer.secondary_greater_is_better,
        "best_model_tie_epsilon": cfg.trainer.best_model_tie_epsilon,
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    training_args = build_training_arguments(cfg.trainer)
    collator = VecOnlyDPOCollator(tokenizer)
    callbacks = []
    if cfg.trainer.early_stopping_patience is not None:
        callbacks.append(
            TieBreakEarlyStoppingCallback(
                early_stopping_patience=cfg.trainer.early_stopping_patience,
                metric_for_best_model=cfg.trainer.metric_for_best_model or "eval_loss",
                greater_is_better=bool(cfg.trainer.greater_is_better),
                secondary_metric_for_best_model=cfg.trainer.secondary_metric_for_best_model,
                secondary_greater_is_better=cfg.trainer.secondary_greater_is_better,
                tie_epsilon=cfg.trainer.best_model_tie_epsilon,
            )
        )

    trainer = HybridDPOTrainer(
        model=policy_model,
        reference_model=reference_model,
        reference_device=cfg.model.reference_device,
        beta=cfg.loss.beta,
        average_log_prob=cfg.loss.average_log_prob,
        secondary_metric_for_best_model=cfg.trainer.secondary_metric_for_best_model,
        secondary_greater_is_better=cfg.trainer.secondary_greater_is_better,
        best_model_tie_epsilon=cfg.trainer.best_model_tie_epsilon,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
        keep_only_best_checkpoint=cfg.trainer.keep_only_best_checkpoint,
    )

    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
    )

    best_checkpoint = trainer.state.best_model_checkpoint
    if cfg.trainer.load_best_model_at_end and best_checkpoint:
        print(
            json.dumps(
                {
                    "best_checkpoint": best_checkpoint,
                    "best_metric": trainer.state.best_metric,
                    "best_secondary_metric": getattr(trainer.state, "best_secondary_metric", None),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    source_dir = Path(best_checkpoint) if best_checkpoint else cfg.trainer.output_dir
    final_dir = cfg.trainer.output_dir / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(source_dir, final_dir)
    print(f"Saved final model and tokenizer to {final_dir}")

    if cfg.eval.enabled:
        checkpoint_for_eval = final_dir
        try:
            if cfg.eval.test_jsonl is not None and _looks_like_dpo_eval_jsonl(cfg.eval.test_jsonl):
                dpo_output_jsonl = cfg.eval.output_jsonl or (cfg.trainer.output_dir / "dpo_eval.jsonl")
                dpo_summary_json = dpo_output_jsonl.with_suffix(".summary.json")
                eval_summary = evaluate_dpo_on_jsonl(
                    sft_config_path=cfg.model.sft_config_path,
                    checkpoint_dir=checkpoint_for_eval,
                    input_jsonl=cfg.eval.test_jsonl,
                    project_root=PROJECT_ROOT,
                    output_jsonl=dpo_output_jsonl,
                )
                dpo_summary_json.parent.mkdir(parents=True, exist_ok=True)
                dpo_summary_json.write_text(
                    json.dumps(eval_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _write_summary_to_wandb("post_train_eval", eval_summary)
                _save_eval_outputs_to_wandb(dpo_output_jsonl, dpo_summary_json)
                print(json.dumps({"dpo_generation_eval": eval_summary}, ensure_ascii=False, indent=2))
                print(f"Saved DPO evaluation summary to {dpo_summary_json}")
            elif cfg.eval.test_jsonl is not None:
                eval_summary = evaluate_checkpoint_preview_on_jsonl(
                    config_path=cfg.model.sft_config_path,
                    checkpoint_dir=checkpoint_for_eval,
                    project_root=PROJECT_ROOT,
                    input_jsonl=cfg.eval.test_jsonl,
                    num_examples=cfg.data.eval_num_examples,
                    output_jsonl=cfg.eval.output_jsonl,
                    use_full_validation_set=cfg.data.use_full_validation_set,
                    sample_randomly=cfg.data.eval_sample_randomly,
                    sample_seed=cfg.data.eval_sample_seed,
                )
                if cfg.eval.output_jsonl is not None:
                    _save_eval_outputs_to_wandb(cfg.eval.output_jsonl)
                _write_summary_to_wandb("post_train_eval", eval_summary)
                print(json.dumps({"dpo_eval": eval_summary}, ensure_ascii=False, indent=2))
            else:
                eval_summary = evaluate_checkpoint_preview(
                    config_path=cfg.model.sft_config_path,
                    checkpoint_dir=checkpoint_for_eval,
                    project_root=PROJECT_ROOT,
                    num_examples=cfg.data.eval_num_examples,
                    output_jsonl=cfg.eval.output_jsonl,
                    use_full_validation_set=cfg.data.use_full_validation_set,
                    sample_randomly=cfg.data.eval_sample_randomly,
                    sample_seed=cfg.data.eval_sample_seed,
                )
                if cfg.eval.output_jsonl is not None:
                    _save_eval_outputs_to_wandb(cfg.eval.output_jsonl)
                _write_summary_to_wandb("post_train_eval", eval_summary)
                print(json.dumps({"dpo_eval": eval_summary}, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "post_train_eval_error": str(exc),
                        "checkpoint_for_eval": str(checkpoint_for_eval),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
