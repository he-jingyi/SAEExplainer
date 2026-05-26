"""Utilities for resolving SAE decoder vectors from Hugging Face assets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PARAMS_FILENAME = "params.npz"
PARAMS_SAFETENSORS_FILENAME = "params.safetensors"
SAFETENSORS_FILENAME = "checkpoints/final.safetensors"
R1_DISTILL_SAFETENSORS_FILENAME = "sae_weights.safetensors"
SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
LLAMA_OPENMOSS_REPO_IDS = {
    "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
    "fnlp/Llama3_1-8B-Base-LXR-8x",
}
LLAMA_SCOPE_R1_DISTILL_REPO_IDS = {
    "OpenMOSS-Team/Llama-Scope-R1-Distill",
    "fnlp/Llama-Scope-R1-Distill",
}
TARGET_MODEL_ALIAS_TO_HF_BASE_MODEL = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-27b": "google/gemma-2-27b",
    "deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}


@dataclass(frozen=True)
class SAEFeatureLocator:
    hf_repo_id: str
    hf_folder_id: str
    feature_index: int
    d_in: int | None = None
    d_sae: int | None = None
    feature_id: str | None = None
    saelens_release: str | None = None
    saelens_sae_id: str | None = None
    hook_name: str | None = None


def _sanitize_component(value: str) -> str:
    return SAFE_COMPONENT_RE.sub("_", value).strip("_")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _fallback_to_base_model_name(model_name: str) -> str:
    stripped = model_name.strip()
    if not stripped:
        return stripped
    if stripped.endswith("-Instruct"):
        return stripped[: -len("-Instruct")]
    if stripped.endswith("-it"):
        return stripped[: -len("-it")]
    return stripped


def resolve_record_target_base_model_name(
    record: dict[str, Any],
    *,
    fallback_model_name: str | None = None,
) -> str:
    metadata = record.get("metadata") or {}
    target_model = str(metadata.get("target_model") or "").strip()
    if target_model:
        if "/" in target_model:
            return target_model
        mapped = TARGET_MODEL_ALIAS_TO_HF_BASE_MODEL.get(target_model.lower())
        if mapped:
            return mapped

    if fallback_model_name:
        return _fallback_to_base_model_name(fallback_model_name)

    if target_model:
        return target_model

    raise ValueError(
        "Could not resolve a target base model name from record metadata.target_model "
        "and no fallback model name was provided."
    )


def locator_from_record(record: dict[str, Any]) -> SAEFeatureLocator:
    metadata = record.get("metadata") or {}
    hf_repo_id = metadata.get("hf_repo_id")
    hf_folder_id = metadata.get("hf_folder_id")
    feature_index = metadata.get("feature_index")

    missing = []
    if not hf_repo_id:
        missing.append("metadata.hf_repo_id")
    if not hf_folder_id:
        missing.append("metadata.hf_folder_id")
    if feature_index is None:
        missing.append("metadata.feature_index")

    if missing:
        raise ValueError(
            "Record is missing required fields for SAE vector lookup: "
            + ", ".join(missing)
        )

    return SAEFeatureLocator(
        hf_repo_id=str(hf_repo_id),
        hf_folder_id=str(hf_folder_id),
        feature_index=int(feature_index),
        d_in=int(metadata["d_in"]) if metadata.get("d_in") is not None else None,
        d_sae=int(metadata["d_sae"]) if metadata.get("d_sae") is not None else None,
        feature_id=record.get("feature_id"),
        saelens_release=(
            str(metadata["saelens_release"])
            if metadata.get("saelens_release") is not None
            else None
        ),
        saelens_sae_id=(
            str(metadata["saelens_sae_id"])
            if metadata.get("saelens_sae_id") is not None
            else None
        ),
        hook_name=str(metadata["hook_name"]) if metadata.get("hook_name") is not None else None,
    )


class SAEVectorStore:
    """Small cache over downloaded SAE params files."""

    def __init__(self, local_dir: str | Path = "data/cache/hf_saes"):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self._decoder_cache: dict[tuple[str, str], np.ndarray] = {}

    def _looks_like_llama_openmoss_sae(self, locator: SAEFeatureLocator) -> bool:
        return (
            locator.hf_repo_id in LLAMA_OPENMOSS_REPO_IDS
            or "Llama3_1-8B-Base-L" in locator.hf_folder_id
        )

    def _looks_like_llama_scope_r1_distill_sae(self, locator: SAEFeatureLocator) -> bool:
        return (
            locator.hf_repo_id in LLAMA_SCOPE_R1_DISTILL_REPO_IDS
            or "OpenR1-Math-220k" in locator.hf_folder_id
        )

    def _candidate_r1_distill_folder_ids(self, locator: SAEFeatureLocator) -> list[str]:
        raw_folder_id = locator.hf_folder_id.strip().strip("/")
        candidates: list[str] = []
        if raw_folder_id:
            candidates.append(raw_folder_id)

        # Some metadata variants use `0-Slimpajama-800M-OpenR1-Math-220k/L15R`,
        # while the current HF repo layout uses `800M-Slimpajama-0-OpenR1-Math-220k/L15R`.
        match = re.match(r"^0-Slimpajama-(\d+M)-OpenR1-Math-220k/(L\d+R)$", raw_folder_id)
        if match:
            model_size, layer_id = match.groups()
            candidates.append(f"{model_size}-Slimpajama-0-OpenR1-Math-220k/{layer_id}")

        # Keep order stable while deduplicating.
        deduped: list[str] = []
        seen: set[str] = set()
        for folder_id in candidates:
            if folder_id in seen:
                continue
            seen.add(folder_id)
            deduped.append(folder_id)
        return deduped

    def _looks_like_gemma_scope2_sae(self, locator: SAEFeatureLocator) -> bool:
        return locator.hf_repo_id.startswith("google/gemma-scope-2-")

    def _has_saelens_target(self, locator: SAEFeatureLocator) -> bool:
        return bool(locator.saelens_release or locator.saelens_sae_id)

    def _load_decoder_matrix_via_saelens(self, locator: SAEFeatureLocator) -> np.ndarray:
        try:
            from sae_lens import SAE
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: sae-lens. Install it first, for example:\n"
                "  pip install sae-lens"
            ) from exc

        release = locator.saelens_release or locator.hf_repo_id
        sae_id = locator.saelens_sae_id or locator.hook_name
        if not release or not sae_id:
            raise ValueError(
                "Cannot load gemma-scope-2 SAE via sae-lens because the record is missing "
                "metadata.saelens_release / metadata.saelens_sae_id / metadata.hook_name."
            )

        try:
            loaded = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device="cpu",
            )
        except TypeError:
            try:
                loaded = SAE.from_pretrained(release, sae_id, device="cpu")
            except TypeError:
                loaded = SAE.from_pretrained(release=release, sae_id=sae_id)

        sae = loaded[0] if isinstance(loaded, tuple) else loaded
        if not hasattr(sae, "W_dec"):
            raise AttributeError(f"Loaded SAE object for {release}/{sae_id} does not expose W_dec")

        decoder = sae.W_dec.detach().to(dtype=torch.float32).cpu().numpy()
        return np.asarray(decoder, dtype=np.float32)

    def _download_params_path(self, locator: SAEFeatureLocator) -> Path:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: huggingface_hub. Install it first, for example:\n"
                "  pip install huggingface_hub numpy"
            ) from exc

        candidate_filenames = [f"{locator.hf_folder_id}/{PARAMS_FILENAME}"]
        if self._looks_like_gemma_scope2_sae(locator):
            candidate_filenames.insert(0, f"{locator.hf_folder_id}/{PARAMS_SAFETENSORS_FILENAME}")
        if self._looks_like_llama_openmoss_sae(locator):
            candidate_filenames.insert(0, f"{locator.hf_folder_id}/{SAFETENSORS_FILENAME}")
        if self._looks_like_llama_scope_r1_distill_sae(locator):
            candidate_filenames = [
                f"{folder_id}/{R1_DISTILL_SAFETENSORS_FILENAME}"
                for folder_id in self._candidate_r1_distill_folder_ids(locator)
            ] + candidate_filenames

        last_error: Exception | None = None
        for filename in candidate_filenames:
            try:
                downloaded = hf_hub_download(
                    repo_id=locator.hf_repo_id,
                    filename=filename,
                    local_dir=str(self.local_dir),
                    force_download=False,
                )
                return Path(downloaded)
            except (EntryNotFoundError, RemoteEntryNotFoundError) as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(
            f"Could not download SAE parameters for repo={locator.hf_repo_id} folder={locator.hf_folder_id}"
        )

    def _load_decoder_matrix_from_safetensors(self, params_path: Path) -> np.ndarray:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: safetensors. Install it first, for example:\n"
                "  pip install safetensors"
            ) from exc

        state_dict = load_file(str(params_path), device="cpu")
        decoder_key = None
        for key in ("W_dec", "decoder.weight", "decoder"):
            if key in state_dict:
                decoder_key = key
                break
        if decoder_key is None:
            for key in state_dict:
                lowered = key.lower()
                if "decoder" in lowered and "weight" in lowered:
                    decoder_key = key
                    break
        if decoder_key is None:
            raise KeyError(f"{params_path} does not contain a recognizable decoder weight key")

        decoder = state_dict[decoder_key].detach().to(dtype=torch.float32).cpu().numpy()
        if decoder_key != "W_dec":
            decoder = decoder.T
        return np.asarray(decoder, dtype=np.float32)

    def _load_decoder_matrix(self, locator: SAEFeatureLocator) -> np.ndarray:
        cache_key = (locator.hf_repo_id, locator.hf_folder_id)
        if cache_key in self._decoder_cache:
            return self._decoder_cache[cache_key]

        if self._looks_like_gemma_scope2_sae(locator) or self._has_saelens_target(locator):
            decoder = self._load_decoder_matrix_via_saelens(locator)
        else:
            params_path = self._download_params_path(locator)
            if params_path.suffix == ".safetensors":
                decoder = self._load_decoder_matrix_from_safetensors(params_path)
            else:
                with np.load(params_path) as params:
                    if "W_dec" not in params:
                        raise KeyError(f"{params_path} does not contain W_dec")
                    decoder = params["W_dec"]

        if decoder.ndim != 2:
            raise ValueError(f"Expected 2D W_dec, got shape {decoder.shape}")

        if locator.d_in is not None and locator.d_sae is not None:
            expected = (locator.d_sae, locator.d_in)
            transposed = (locator.d_in, locator.d_sae)
            if decoder.shape == transposed:
                decoder = decoder.T
            elif decoder.shape != expected:
                raise ValueError(
                    f"Unexpected W_dec shape {decoder.shape}; expected {expected} "
                    f"or {transposed} based on record metadata."
                )

        decoder = np.asarray(decoder, dtype=np.float32)
        self._decoder_cache[cache_key] = decoder
        return decoder

    def get_decoder_vector(self, locator: SAEFeatureLocator) -> np.ndarray:
        decoder = self._load_decoder_matrix(locator)
        if locator.feature_index < 0 or locator.feature_index >= decoder.shape[0]:
            raise IndexError(
                f"Feature index {locator.feature_index} is out of bounds for W_dec with "
                f"shape {decoder.shape}"
            )
        return decoder[locator.feature_index].copy()

    def save_decoder_vector(
        self,
        locator: SAEFeatureLocator,
        output_dir: str | Path = "data/cache/sae_decoder_vectors",
    ) -> Path:
        output_dir = Path(output_dir)
        repo_part = _sanitize_component(locator.hf_repo_id)
        folder_part = _sanitize_component(locator.hf_folder_id)
        feature_part = locator.feature_id or f"feature_{locator.feature_index:05d}"
        feature_part = _sanitize_component(feature_part)

        target_dir = output_dir / repo_part / folder_part
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{feature_part}.npy"

        vector = self.get_decoder_vector(locator)
        np.save(output_path, vector)
        return output_path
