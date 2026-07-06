#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/samples/model_repository/gemma4_31b_it/1}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required." >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"
MODEL_DIR="$MODEL_DIR" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-4-31B-it",
    revision="main",
    local_dir=os.environ["MODEL_DIR"],
    local_dir_use_symlinks=False,
    token=os.environ["HF_TOKEN"],
    allow_patterns=[
        "*.py",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "processor_config.json",
    ],
)
PY

test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/tokenizer.json" || test -f "$MODEL_DIR/tokenizer.model"
echo "model downloaded to $MODEL_DIR"
