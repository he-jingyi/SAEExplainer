#!/usr/bin/env python3
"""Train the vector-conditioned SAE explainer."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

from transformers import EarlyStoppingCallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a vector-conditioned SAE explainer with transformers.Trainer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "explamma_15" / "config_sft_explamma_15.yaml",
        help="Path to the SFT YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets and model config, then exit before training.",
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


def main() -> None:
    args = parse_args()
    import torch

    from src.sft.collator import HybridSFTCollator
    from src.sft.config import load_sft_config
    from src.sft.dataset import build_train_val_datasets
    from src.sft.eval import evaluate_checkpoint_preview
    from src.sft.model import HybridExplainerModel, resolve_injection_submodule_with_path
    from src.sft.trainer import HybridSFTTrainer
    from src.sft.train_utils import (
        build_training_arguments,
        load_base_model,
        load_tokenizer,
        maybe_apply_lora,
        set_seed,
    )

    config_path = args.config if args.config.is_absolute() else (PROJECT_ROOT / args.config).resolve()
    cfg = load_sft_config(config_path=config_path, project_root=PROJECT_ROOT)
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is not None and not resume_from_checkpoint.is_absolute():
        resume_from_checkpoint = (PROJECT_ROOT / resume_from_checkpoint).resolve()

    set_seed(cfg.trainer.seed)
    tokenizer = load_tokenizer(cfg.model)
    train_dataset, val_dataset = build_train_val_datasets(
        tokenizer=tokenizer,
        data_config=cfg.data,
        injection_config=cfg.injection,
        project_root=PROJECT_ROOT,
    )

    base_model = load_base_model(cfg.model)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(base_model, "generation_config", None) is not None:
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    base_model = maybe_apply_lora(base_model, cfg.lora)
    injection_module_path, injection_module = resolve_injection_submodule_with_path(
        base_model,
        cfg.injection.inject_layer,
    )
    model = HybridExplainerModel(
        base_model=base_model,
        inject_layer=cfg.injection.inject_layer,
        steering_coefficient=cfg.injection.steering_coefficient,
        use_projector=cfg.injection.use_projector,
        vector_input_dim=cfg.injection.vector_input_dim,
    )

    summary = {
        "config_path": str(config_path),
        "model_name": cfg.model.model_name,
        "train_jsonl": str(cfg.data.train_jsonl),
        "val_jsonl": str(cfg.data.val_jsonl) if cfg.data.val_jsonl is not None else None,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "placeholder_text": cfg.injection.placeholder_text,
        "placeholder_token": train_dataset.placeholder_token,
        "placeholder_token_id": train_dataset.placeholder_token_id,
        "inject_layer": cfg.injection.inject_layer,
        "injection_module_path": injection_module_path,
        "injection_module_class": type(injection_module).__name__,
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    training_args = build_training_arguments(cfg.trainer)
    collator = HybridSFTCollator(tokenizer)
    callbacks = []
    if cfg.trainer.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.trainer.early_stopping_patience,
            )
        )

    trainer = HybridSFTTrainer(
        model=model,
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
        print(json.dumps({"best_checkpoint": best_checkpoint}, ensure_ascii=False, indent=2))

    checkpoint_for_eval = Path(best_checkpoint) if best_checkpoint else cfg.trainer.output_dir
    source_dir = checkpoint_for_eval

    # Eval preview loads a fresh checkpoint model. Free the training-time model
    # first so large checkpoints like Gemma-2-27B do not OOM due to double
    # residency on the same GPU.
    del trainer
    del model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    if cfg.eval.enabled:
        eval_summary = evaluate_checkpoint_preview(
            config_path=config_path,
            checkpoint_dir=checkpoint_for_eval,
            project_root=PROJECT_ROOT,
            num_examples=cfg.eval.num_examples,
            output_jsonl=cfg.eval.output_jsonl,
        )
        print(json.dumps({"basic_eval": eval_summary}, ensure_ascii=False, indent=2))

    final_dir = cfg.trainer.output_dir / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(source_dir, final_dir)
    print(f"Saved final model and tokenizer to {final_dir}")


if __name__ == "__main__":
    main()
