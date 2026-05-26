"""Wrapped causal LM model for hybrid vector-injected SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.sft.injection import add_hook, make_norm_matched_injection_hook


def _try_get_attr_chain(root: Any, chain: tuple[str, ...]) -> Any | None:
    current = root
    for attr in chain:
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def _resolve_layers_container(model: nn.Module, model_name: str, layer_idx: int) -> Any:
    candidate_chains = []

    if "gemma-2" in model_name or "mistral" in model_name or "llama" in model_name or "qwen" in model_name:
        candidate_chains = [
            ("model", "layers"),
            ("model", "model", "layers"),
            ("base_model", "model", "layers"),
            ("base_model", "model", "model", "layers"),
        ]
    elif "gemma-3" in model_name:
        candidate_chains = [
            ("model", "model", "language_model", "layers"),
            ("base_model", "model", "model", "language_model", "layers"),
            ("model", "language_model", "layers"),
            ("base_model", "model", "language_model", "layers"),
            ("language_model", "layers"),
            ("base_model", "language_model", "layers"),
        ]
    elif "gpt2" in model_name:
        candidate_chains = [
            ("transformer", "h"),
            ("base_model", "model", "transformer", "h"),
            ("base_model", "transformer", "h"),
        ]

    for chain in candidate_chains:
        layers = _try_get_attr_chain(model, chain)
        if layers is not None:
            return layers

    for module_name, module in model.named_modules():
        if module_name.split(".")[-1] not in {"layers", "h"}:
            continue
        if not hasattr(module, "__len__"):
            continue
        try:
            module_len = len(module)
        except Exception:
            continue
        if module_len <= 0 or layer_idx >= module_len:
            continue
        try:
            first_layer = module[0]
        except Exception:
            continue
        if hasattr(first_layer, "self_attn") or hasattr(first_layer, "mlp") or hasattr(first_layer, "attn"):
            return module

    raise ValueError(
        f"Unsupported model architecture for injection: {getattr(model.config, '_name_or_path', type(model).__name__)}"
    )


def resolve_injection_submodule(model: nn.Module, layer_idx: int) -> nn.Module:
    model_name = getattr(model.config, "_name_or_path", "").lower()
    layers = _resolve_layers_container(model, model_name, layer_idx)
    return layers[layer_idx]


def resolve_injection_submodule_with_path(model: nn.Module, layer_idx: int) -> tuple[str, nn.Module]:
    model_name = getattr(model.config, "_name_or_path", "").lower()
    candidate_chains: list[tuple[str, ...]] = []

    if "gemma-2" in model_name or "mistral" in model_name or "llama" in model_name or "qwen" in model_name:
        candidate_chains = [
            ("model", "layers"),
            ("model", "model", "layers"),
            ("base_model", "model", "layers"),
            ("base_model", "model", "model", "layers"),
        ]
    elif "gemma-3" in model_name:
        candidate_chains = [
            ("model", "model", "language_model", "layers"),
            ("base_model", "model", "model", "language_model", "layers"),
            ("model", "language_model", "layers"),
            ("base_model", "model", "language_model", "layers"),
            ("language_model", "layers"),
            ("base_model", "language_model", "layers"),
        ]
    elif "gpt2" in model_name:
        candidate_chains = [
            ("transformer", "h"),
            ("base_model", "model", "transformer", "h"),
            ("base_model", "transformer", "h"),
        ]

    for chain in candidate_chains:
        layers = _try_get_attr_chain(model, chain)
        if layers is None:
            continue
        return (".".join(chain + (str(layer_idx),)), layers[layer_idx])

    resolved_module = resolve_injection_submodule(model, layer_idx)
    resolved_id = id(resolved_module)
    for module_name, module in model.named_modules():
        if id(module) == resolved_id:
            resolved_path = module_name or "<root>"
            lowered = resolved_path.lower()
            if "gemma-3" in model_name and (
                "vision_tower" in lowered or "vision" in lowered or "siglip" in lowered
            ):
                raise ValueError(
                    "Resolved Gemma-3 injection module points to the vision tower instead of the text tower: "
                    f"{resolved_path}"
                )
            return (resolved_path, resolved_module)

    return (f"<unresolved-path>.{layer_idx}", resolved_module)


def _resolve_hidden_size(config: Any) -> int:
    for attr_chain in (
        ("hidden_size",),
        ("text_config", "hidden_size"),
        ("language_config", "hidden_size"),
    ):
        current = config
        found = True
        for attr in attr_chain:
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found and current is not None:
            return int(current)

    raise AttributeError(
        "Could not resolve hidden_size from model config. "
        "Tried config.hidden_size, config.text_config.hidden_size, and config.language_config.hidden_size."
    )


class HybridExplainerModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        inject_layer: int,
        steering_coefficient: float,
        use_projector: bool = False,
        vector_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.inject_layer = inject_layer
        self.steering_coefficient = steering_coefficient
        object.__setattr__(
            self,
            "_injection_submodule_ref",
            resolve_injection_submodule(base_model, inject_layer),
        )
        hidden_size = _resolve_hidden_size(base_model.config)
        self.feature_projector: nn.Module | None = None
        if use_projector:
            if vector_input_dim is None:
                raise ValueError("vector_input_dim must be provided when use_projector=True")
            self.feature_projector = nn.Linear(vector_input_dim, hidden_size, bias=False)

    @property
    def config(self) -> Any:
        return self.base_model.config

    @property
    def injection_submodule(self) -> nn.Module:
        return self._injection_submodule_ref

    def save_pretrained(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(str(output_path))
        if self.feature_projector is not None:
            torch.save(self.feature_projector.state_dict(), output_path / "feature_projector.pt")
        metadata = {
            "inject_layer": self.inject_layer,
            "steering_coefficient": self.steering_coefficient,
            "has_feature_projector": self.feature_projector is not None,
        }
        (output_path / "hybrid_sft_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_projector_from_checkpoint(self, checkpoint_dir: str | Path) -> None:
        if self.feature_projector is None:
            return

        projector_path = Path(checkpoint_dir) / "feature_projector.pt"
        if not projector_path.exists():
            return

        try:
            state_dict = torch.load(projector_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(projector_path, map_location="cpu")
        self.feature_projector.load_state_dict(state_dict)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        feature_vectors: torch.Tensor | None = None,
        act_positions: torch.Tensor | None = None,
        inject_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        should_inject = (
            feature_vectors is not None
            and act_positions is not None
            and inject_mask is not None
            and bool(inject_mask.any().item())
        )

        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            **kwargs,
        }

        if not should_inject:
            return self.base_model(**model_kwargs)

        if self.feature_projector is not None:
            feature_vectors = self.feature_projector(feature_vectors.to(self.feature_projector.weight.dtype))

        param = next(self.base_model.parameters())
        hook = make_norm_matched_injection_hook(
            feature_vectors_BD=feature_vectors,
            act_positions_B=act_positions,
            inject_mask_B=inject_mask,
            steering_coefficient=self.steering_coefficient,
            device=param.device,
            dtype=param.dtype,
        )
        with add_hook(self.injection_submodule, hook):
            return self.base_model(**model_kwargs)

    @torch.no_grad()
    def generate_with_injection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        feature_vectors: torch.Tensor | None = None,
        act_positions: torch.Tensor | None = None,
        inject_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        should_inject = (
            feature_vectors is not None
            and act_positions is not None
            and inject_mask is not None
            and bool(inject_mask.any().item())
        )

        if not should_inject:
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        if self.feature_projector is not None:
            feature_vectors = self.feature_projector(feature_vectors.to(self.feature_projector.weight.dtype))

        param = next(self.base_model.parameters())
        hook = make_norm_matched_injection_hook(
            feature_vectors_BD=feature_vectors,
            act_positions_B=act_positions,
            inject_mask_B=inject_mask,
            steering_coefficient=self.steering_coefficient,
            device=param.device,
            dtype=param.dtype,
        )
        with add_hook(self.injection_submodule, hook):
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )