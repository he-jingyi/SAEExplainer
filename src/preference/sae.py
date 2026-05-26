"""Minimal JumpReLU SAE scorer for feature activation ranking."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch

from src.data_collection.sae_vectors import SAEFeatureLocator, SAEVectorStore


@dataclass(frozen=True)
class SAEFeatureActivationScore:
    max_activation: float
    mean_activation: float
    sum_activation: float
    argmax_token_index: int
    per_token_activations: list[float]


@dataclass(frozen=True)
class SAEFeatureActivationComparison:
    optimized: SAEFeatureActivationScore
    reference: SAEFeatureActivationScore
    max_activation_diff: float
    mean_activation_diff: float
    sum_activation_diff: float
    max_abs_per_token_diff: float
    mean_abs_per_token_diff: float
    same_argmax_token_index: bool
    per_token_allclose: bool


class JumpReluSAEStore:
    def __init__(
        self,
        local_dir: str | Path = "data/cache/hf_saes",
        *,
        max_feature_cache_size: int | None = 4096,
        max_full_sae_cache_size: int | None = 4,
        max_torch_feature_cache_size: int | None = 8192,
        backend: str | None = None,
    ):
        self.vector_store = SAEVectorStore(local_dir=local_dir)
        self.max_feature_cache_size = max_feature_cache_size
        self.max_full_sae_cache_size = max_full_sae_cache_size
        self.max_torch_feature_cache_size = max_torch_feature_cache_size
        self.backend = (
            backend or os.environ.get("SAE_EXPLAINER_SAE_SCORER_BACKEND", "saelens")
        ).strip().lower()
        if self.backend not in {"auto", "manual", "saelens"}:
            raise ValueError(
                f"Unsupported SAE scorer backend {self.backend!r}. "
                "Expected one of: auto, manual, saelens."
            )
        self._feature_cache: OrderedDict[tuple[str, str, int], tuple[np.ndarray, float, float]] = OrderedDict()
        self._torch_feature_cache: OrderedDict[
            tuple[str, str, int, str, str],
            tuple[torch.Tensor, float, float],
        ] = OrderedDict()
        self._full_sae_cache: OrderedDict[
            tuple[str, str],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = OrderedDict()
        self._saelens_cache: OrderedDict[
            tuple[str, str, str],
            Any,
        ] = OrderedDict()
        self._warned_messages: set[str] = set()

    @staticmethod
    def _get_cached(
        cache: OrderedDict[Any, Any],
        key: Any,
    ) -> Any | None:
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value

    @staticmethod
    def _set_cached(
        cache: OrderedDict[Any, Any],
        key: Any,
        value: Any,
        *,
        max_size: int | None,
    ) -> Any:
        cache[key] = value
        cache.move_to_end(key)
        if max_size is not None and max_size > 0 and len(cache) > max_size:
            cache.popitem(last=False)
        return value

    def _warn_once(self, message: str) -> None:
        if message in self._warned_messages:
            return
        self._warned_messages.add(message)
        warnings.warn(message, stacklevel=2)

    def _resolve_saelens_target(self, locator: SAEFeatureLocator) -> tuple[str, str] | None:
        release = locator.saelens_release or locator.hf_repo_id
        sae_id = locator.saelens_sae_id or locator.hook_name
        if not release or not sae_id:
            return None
        return release, sae_id

    def _should_try_saelens(self, locator: SAEFeatureLocator) -> bool:
        if self.backend == "manual":
            return False

        target = self._resolve_saelens_target(locator)
        if target is not None:
            return True

        if self.backend == "saelens":
            raise ValueError(
                "SAE Lens backend was requested, but the record is missing both "
                "metadata.saelens_sae_id and metadata.hook_name."
            )
        return False

    def _load_saelens_sae(
        self,
        locator: SAEFeatureLocator,
        *,
        device: torch.device,
    ) -> Any:
        target = self._resolve_saelens_target(locator)
        if target is None:
            raise ValueError("No SAE Lens release/sae_id target is available for this feature.")
        release, sae_id = target
        cache_key = (release, sae_id, str(device))
        cached = self._get_cached(self._saelens_cache, cache_key)
        if cached is not None:
            return cached

        try:
            from sae_lens import SAE
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: sae-lens. Install it first, for example:\n"
                "  pip install sae-lens"
            ) from exc

        try:
            loaded = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=str(device),
            )
        except TypeError:
            try:
                loaded = SAE.from_pretrained(release, sae_id, device=str(device))
            except TypeError:
                loaded = SAE.from_pretrained(release=release, sae_id=sae_id)

        sae = loaded[0] if isinstance(loaded, tuple) else loaded
        if hasattr(sae, "to"):
            sae = sae.to(device=device)
        if hasattr(sae, "eval"):
            sae.eval()
        return self._set_cached(
            self._saelens_cache,
            cache_key,
            sae,
            max_size=self.max_full_sae_cache_size,
        )

    def _load_single_feature_params(self, locator: SAEFeatureLocator) -> tuple[np.ndarray, float, float]:
        cache_key = (locator.hf_repo_id, locator.hf_folder_id, locator.feature_index)
        cached = self._get_cached(self._feature_cache, cache_key)
        if cached is not None:
            return cached

        # Reuse the per-SAE cache so we do not repeatedly decompress the full
        # W_enc matrix once per feature during large scoring runs.
        w_enc, b_enc, b_dec, threshold = self._load_full_sae_params(locator)

        feature_index = locator.feature_index
        if feature_index < 0 or feature_index >= w_enc.shape[1]:
            raise IndexError(
                f"Feature index {feature_index} is out of bounds for W_enc with shape {w_enc.shape}"
            )
        feature_w_enc = np.asarray(w_enc[:, feature_index], dtype=np.float32)
        feature_b_enc = float(np.asarray(b_enc[feature_index], dtype=np.float32))
        decoder_bias = np.asarray(b_dec, dtype=np.float32)
        if decoder_bias.ndim != 1:
            raise ValueError(f"Expected b_dec to be 1D, got shape {decoder_bias.shape}")
        if decoder_bias.shape[0] != feature_w_enc.shape[0]:
            raise ValueError(
                f"Shape mismatch between b_dec {decoder_bias.shape} and feature encoder column {feature_w_enc.shape}"
            )

        # Fold decoder bias into the encoder bias so scoring matches the SAE encode() definition:
        # pre = (x - b_dec) @ W_enc[:, f] + b_enc[f]
        effective_feature_b_enc = feature_b_enc - float(np.dot(decoder_bias, feature_w_enc))
        feature_threshold = float(np.asarray(threshold[feature_index], dtype=np.float32))

        return self._set_cached(
            self._feature_cache,
            cache_key,
            (feature_w_enc, effective_feature_b_enc, feature_threshold),
            max_size=self.max_feature_cache_size,
        )

    def _load_single_feature_torch_params(
        self,
        locator: SAEFeatureLocator,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, float, float]:
        device_key = str(device)
        dtype_key = str(dtype)
        cache_key = (
            locator.hf_repo_id,
            locator.hf_folder_id,
            locator.feature_index,
            device_key,
            dtype_key,
        )
        cached = self._get_cached(self._torch_feature_cache, cache_key)
        if cached is not None:
            return cached

        feature_w_enc, feature_b_enc, feature_threshold = self._load_single_feature_params(locator)
        weight_D = torch.from_numpy(feature_w_enc).to(device=device, dtype=dtype)
        return self._set_cached(
            self._torch_feature_cache,
            cache_key,
            (weight_D, feature_b_enc, feature_threshold),
            max_size=self.max_torch_feature_cache_size,
        )

    def _load_full_sae_params(
        self,
        locator: SAEFeatureLocator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = (locator.hf_repo_id, locator.hf_folder_id)
        cached = self._get_cached(self._full_sae_cache, cache_key)
        if cached is not None:
            return cached

        params_path = self.vector_store._download_params_path(locator)  # noqa: SLF001
        with np.load(params_path) as params:
            w_enc = np.asarray(params["W_enc"], dtype=np.float32)
            b_enc = np.asarray(params["b_enc"], dtype=np.float32)
            b_dec = np.asarray(params["b_dec"], dtype=np.float32)
            threshold = np.asarray(params["threshold"], dtype=np.float32)

        if w_enc.ndim != 2:
            raise ValueError(f"Expected W_enc to be 2D, got shape {w_enc.shape}")
        if b_enc.ndim != 1 or b_dec.ndim != 1 or threshold.ndim != 1:
            raise ValueError(
                f"Expected b_enc, b_dec, threshold to be 1D, got shapes "
                f"{b_enc.shape}, {b_dec.shape}, {threshold.shape}"
            )

        return self._set_cached(
            self._full_sae_cache,
            cache_key,
            (w_enc, b_enc, b_dec, threshold),
            max_size=self.max_full_sae_cache_size,
        )

    def _apply_valid_mask(
        self,
        acts_L: torch.Tensor,
        valid_token_mask_L: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if acts_L.ndim != 1:
            raise ValueError(f"Expected 1D token activations, got shape {tuple(acts_L.shape)}")

        if valid_token_mask_L is None:
            valid_mask_L = torch.ones_like(acts_L, dtype=torch.bool)
        else:
            if valid_token_mask_L.ndim != 1:
                raise ValueError(
                    f"valid_token_mask_L must be 1D with shape [seq_len], got {tuple(valid_token_mask_L.shape)}"
                )
            valid_mask_L = valid_token_mask_L.to(device=acts_L.device, dtype=torch.bool)
            if valid_mask_L.shape[0] != acts_L.shape[0]:
                raise ValueError(
                    f"Mask length {valid_token_mask_L.shape[0]} does not match sequence length {acts_L.shape[0]}"
                )

        masked_acts_L = acts_L.masked_fill(~valid_mask_L, 0.0)
        valid_count = int(valid_mask_L.sum().item())
        return masked_acts_L, valid_mask_L, valid_count

    def _aggregate_token_scores(
        self,
        acts_L: torch.Tensor,
        valid_token_mask_L: torch.Tensor | None = None,
    ) -> SAEFeatureActivationScore:
        masked_acts_L, valid_mask_L, valid_count = self._apply_valid_mask(acts_L, valid_token_mask_L)

        if valid_count == 0:
            return SAEFeatureActivationScore(
                max_activation=0.0,
                mean_activation=0.0,
                sum_activation=0.0,
                argmax_token_index=0,
                per_token_activations=[float(value) for value in masked_acts_L.detach().cpu().tolist()],
            )

        masked_for_max_L = acts_L.masked_fill(~valid_mask_L, float("-inf"))
        max_value, max_index = torch.max(masked_for_max_L, dim=0)
        sum_value = torch.sum(masked_acts_L, dim=0)
        mean_value = sum_value / valid_count
        return SAEFeatureActivationScore(
            max_activation=float(max_value.item()),
            mean_activation=float(mean_value.item()),
            sum_activation=float(sum_value.item()),
            argmax_token_index=int(max_index.item()),
            per_token_activations=[float(value) for value in masked_acts_L.detach().cpu().tolist()],
        )

    def _score_feature_manual(
        self,
        hidden_states_LD: torch.Tensor,
        locator: SAEFeatureLocator,
        valid_token_mask_L: torch.Tensor | None = None,
    ) -> SAEFeatureActivationScore:
        weight_D, feature_b_enc, feature_threshold = self._load_single_feature_torch_params(
            locator,
            device=hidden_states_LD.device,
            dtype=hidden_states_LD.dtype,
        )
        pre_acts_L = hidden_states_LD @ weight_D + feature_b_enc
        acts_L = torch.relu(pre_acts_L) * (pre_acts_L > feature_threshold)
        return self._aggregate_token_scores(acts_L, valid_token_mask_L)

    def _score_feature_saelens(
        self,
        hidden_states_LD: torch.Tensor,
        locator: SAEFeatureLocator,
        valid_token_mask_L: torch.Tensor | None = None,
    ) -> SAEFeatureActivationScore:
        sae = self._load_saelens_sae(locator, device=hidden_states_LD.device)

        with torch.no_grad():
            acts_LF = sae.encode(hidden_states_LD)

        if acts_LF.ndim != 2:
            raise ValueError(
                f"Expected SAE Lens encode(...) to return [seq_len, d_sae], got shape {tuple(acts_LF.shape)}"
            )
        if locator.feature_index < 0 or locator.feature_index >= acts_LF.shape[1]:
            raise IndexError(
                f"Feature index {locator.feature_index} is out of bounds for SAE activations with shape "
                f"{tuple(acts_LF.shape)}"
            )

        acts_L = acts_LF[:, locator.feature_index]
        return self._aggregate_token_scores(acts_L, valid_token_mask_L)

    def score_feature(
        self,
        hidden_states_LD: torch.Tensor,
        locator: SAEFeatureLocator,
        valid_token_mask_L: torch.Tensor | None = None,
    ) -> SAEFeatureActivationScore:
        """Optimized single-feature scorer.

        Computes exactly one feature using the folded-bias formulation, which is
        equivalent to the full SAE encode path for that feature but much cheaper.
        """
        if hidden_states_LD.ndim != 2:
            raise ValueError(
                f"score_feature expects hidden states with shape [seq_len, hidden_dim], got {tuple(hidden_states_LD.shape)}"
            )
        if self._should_try_saelens(locator):
            try:
                return self._score_feature_saelens(
                    hidden_states_LD,
                    locator,
                    valid_token_mask_L=valid_token_mask_L,
                )
            except Exception as exc:
                if self.backend == "saelens":
                    raise
                target = self._resolve_saelens_target(locator)
                target_name = f"{target[0]}/{target[1]}" if target is not None else locator.feature_id or "unknown"
                self._warn_once(
                    "SAE Lens scoring was unavailable for "
                    f"{target_name}; falling back to the manual scorer. Reason: {exc}"
                )

        return self._score_feature_manual(
            hidden_states_LD,
            locator,
            valid_token_mask_L=valid_token_mask_L,
        )

    def score_feature_reference(
        self,
        hidden_states_LD: torch.Tensor,
        locator: SAEFeatureLocator,
        valid_token_mask_L: torch.Tensor | None = None,
    ) -> SAEFeatureActivationScore:
        """Reference scorer matching the official full-SAE encode path.

        This computes all features first, then slices the target feature. It is
        slower and more memory-hungry, but useful for sanity-checking the
        optimized single-feature scorer.
        """
        if hidden_states_LD.ndim != 2:
            raise ValueError(
                f"score_feature_reference expects hidden states with shape [seq_len, hidden_dim], got {tuple(hidden_states_LD.shape)}"
            )

        w_enc, b_enc, b_dec, threshold = self._load_full_sae_params(locator)
        _, hidden_dim = hidden_states_LD.shape
        if b_dec.shape[0] != hidden_dim:
            raise ValueError(
                f"Expected b_dec to match hidden dim {hidden_dim}, got shape {b_dec.shape}"
            )
        if locator.feature_index < 0 or locator.feature_index >= w_enc.shape[1]:
            raise IndexError(
                f"Feature index {locator.feature_index} is out of bounds for W_enc with shape {w_enc.shape}"
            )

        w_enc_DF = torch.from_numpy(w_enc).to(hidden_states_LD.device, hidden_states_LD.dtype)
        b_enc_F = torch.from_numpy(b_enc).to(hidden_states_LD.device, hidden_states_LD.dtype)
        b_dec_D = torch.from_numpy(b_dec).to(hidden_states_LD.device, hidden_states_LD.dtype)
        threshold_F = torch.from_numpy(threshold).to(hidden_states_LD.device, hidden_states_LD.dtype)

        centered_LD = hidden_states_LD - b_dec_D
        pre_acts_LF = centered_LD @ w_enc_DF + b_enc_F
        acts_LF = torch.relu(pre_acts_LF) * (pre_acts_LF > threshold_F)
        acts_L = acts_LF[:, locator.feature_index]
        return self._aggregate_token_scores(acts_L, valid_token_mask_L)

    def compare_score_feature(
        self,
        hidden_states_LD: torch.Tensor,
        locator: SAEFeatureLocator,
        valid_token_mask_L: torch.Tensor | None = None,
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> SAEFeatureActivationComparison:
        optimized = self.score_feature(hidden_states_LD, locator, valid_token_mask_L)
        reference = self.score_feature_reference(hidden_states_LD, locator, valid_token_mask_L)

        optimized_tensor = torch.tensor(optimized.per_token_activations, dtype=torch.float32)
        reference_tensor = torch.tensor(reference.per_token_activations, dtype=torch.float32)
        abs_diff = torch.abs(optimized_tensor - reference_tensor)

        return SAEFeatureActivationComparison(
            optimized=optimized,
            reference=reference,
            max_activation_diff=abs(optimized.max_activation - reference.max_activation),
            mean_activation_diff=abs(optimized.mean_activation - reference.mean_activation),
            sum_activation_diff=abs(optimized.sum_activation - reference.sum_activation),
            max_abs_per_token_diff=float(abs_diff.max().item()) if abs_diff.numel() > 0 else 0.0,
            mean_abs_per_token_diff=float(abs_diff.mean().item()) if abs_diff.numel() > 0 else 0.0,
            same_argmax_token_index=optimized.argmax_token_index == reference.argmax_token_index,
            per_token_allclose=bool(torch.allclose(optimized_tensor, reference_tensor, atol=atol, rtol=rtol)),
        )
