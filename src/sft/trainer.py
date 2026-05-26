"""Custom Trainer hooks for hybrid SAE explainer SFT."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any
import logging
import shutil

import torch
from transformers import Trainer

logger = logging.getLogger(__name__)

FULL_MODEL_FILENAMES = (
    "pytorch_model.bin",
    "model.safetensors",
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
)
ADAPTER_MODEL_FILENAMES = (
    "adapter_model.bin",
    "adapter_model.safetensors",
)


class HybridSFTTrainer(Trainer):
    """Trainer that saves HybridExplainerModel checkpoints via save_pretrained."""

    def __init__(self, *args, keep_only_best_checkpoint: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_only_best_checkpoint = keep_only_best_checkpoint

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model = self.model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(output_path))
        else:
            super().save_model(output_dir=output_dir, _internal_call=_internal_call)
            return

        processing_class = getattr(self, "processing_class", None)
        if processing_class is not None and hasattr(processing_class, "save_pretrained"):
            processing_class.save_pretrained(str(output_path))

    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        self._prune_checkpoints_keep_best()

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        checkpoint_dir = Path(resume_from_checkpoint)
        if self._should_use_hybrid_adapter_resume(checkpoint_dir, model):
            self._load_hybrid_adapter_checkpoint(checkpoint_dir, model)
            return

        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _load_best_model(self):
        # We handle best-checkpoint reloading manually after trainer.train() so we can
        # avoid the torch<2.6 safety restriction inside transformers' default loader.
        return

    def _should_use_hybrid_adapter_resume(self, checkpoint_dir: Path, model: Any) -> bool:
        if not checkpoint_dir.exists():
            return False

        if any((checkpoint_dir / filename).is_file() for filename in FULL_MODEL_FILENAMES):
            return False

        has_adapter_weights = any((checkpoint_dir / filename).is_file() for filename in ADAPTER_MODEL_FILENAMES)
        has_projector_or_metadata = any(
            (checkpoint_dir / filename).is_file()
            for filename in ("feature_projector.pt", "hybrid_sft_metadata.json")
        )
        if not (has_adapter_weights or has_projector_or_metadata):
            return False

        base_model = getattr(model, "base_model", None)
        return base_model is not None and hasattr(base_model, "load_adapter")

    def _get_active_adapter_name(self, base_model: Any) -> str:
        if hasattr(base_model, "active_adapters"):
            active_adapters = list(getattr(base_model, "active_adapters"))
            if active_adapters:
                return str(active_adapters[0])
        if hasattr(base_model, "active_adapter"):
            active_adapter = getattr(base_model, "active_adapter")
            if isinstance(active_adapter, str) and active_adapter:
                return active_adapter
        return "default"

    def _load_hybrid_adapter_checkpoint(self, checkpoint_dir: Path, model: Any) -> None:
        base_model = model.base_model
        adapter_name = self._get_active_adapter_name(base_model)
        is_trainable = any(param.requires_grad for param in base_model.parameters())

        logger.info(
            "Loading hybrid adapter checkpoint from %s using adapter '%s'.",
            checkpoint_dir,
            adapter_name,
        )
        base_model.load_adapter(
            str(checkpoint_dir),
            adapter_name=adapter_name,
            is_trainable=is_trainable,
        )
        if hasattr(base_model, "set_adapter"):
            base_model.set_adapter(adapter_name)

        if hasattr(model, "load_projector_from_checkpoint"):
            model.load_projector_from_checkpoint(checkpoint_dir)

    def _prune_checkpoints_keep_best(self) -> None:
        if not getattr(self.args, "load_best_model_at_end", False):
            return
        if not self.keep_only_best_checkpoint:
            return

        best_checkpoint = self.state.best_model_checkpoint
        if not best_checkpoint:
            return

        best_path = Path(best_checkpoint).resolve()
        output_dir = Path(self.args.output_dir).resolve()
        checkpoint_dirs = sorted(output_dir.glob("checkpoint-*"))

        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.resolve() == best_path:
                continue
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
