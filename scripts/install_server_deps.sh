#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/install_server_deps.sh
#   bash scripts/install_server_deps.sh --with-flash-attn
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash scripts/install_server_deps.sh --with-flash-attn

WITH_FLASH_ATTN=0
for arg in "$@"; do
  case "$arg" in
    --with-flash-attn)
      WITH_FLASH_ATTN=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
MAX_JOBS="${MAX_JOBS:-8}"

echo "[1/4] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "[2/4] Installing PyTorch..."
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "$TORCH_INDEX_URL"

echo "[3/4] Installing project requirements..."
python -m pip install -r requirements.txt
python -m pip install openai wandb

if [[ "$WITH_FLASH_ATTN" -eq 1 ]]; then
  echo "[4/4] Installing flash-attn..."
  python -m pip install -U packaging psutil ninja
  MAX_JOBS="$MAX_JOBS" python -m pip install flash-attn --no-build-isolation
else
  echo "[4/4] Skipping flash-attn install. Re-run with --with-flash-attn to enable it."
fi

echo "Dependency installation complete."
