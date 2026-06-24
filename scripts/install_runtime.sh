#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${BACKEND_DIR:-/opt/tritonserver/backends/vllm}"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends git
  rm -rf /var/lib/apt/lists/*
fi

mkdir -p "$BACKEND_DIR"
cp -r "$REPO_ROOT/src/"* "$BACKEND_DIR/"

pip install --no-cache-dir --force-reinstall \
  "numpy<2" \
  "pydantic==2.10.6" \
  huggingface_hub

pip install --no-cache-dir --force-reinstall \
  "git+https://github.com/lkm2835/vllm.git@add-exaone4_5"

pip install --no-cache-dir --force-reinstall \
  "git+https://github.com/nuxlear/transformers.git@add-exaone4_5-v5.3.0.dev0"

python -c "import vllm, transformers, huggingface_hub, numpy, pydantic; print('deps ok', transformers.__version__, numpy.__version__)"
