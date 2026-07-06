#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-exaone45awq-triton:latest}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/tritonserver:26.05-trtllm-python-py3}"
INCLUDE_MODEL="${INCLUDE_MODEL:-0}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/.build/docker}"
CONTEXT_DIR="$BUILD_DIR/context"

mkdir -p "$BUILD_DIR"
rm -rf "$CONTEXT_DIR"
mkdir -p "$CONTEXT_DIR"
cp -R "$REPO_ROOT/src" "$CONTEXT_DIR/src"
cp -R "$REPO_ROOT/scripts" "$CONTEXT_DIR/scripts"
cp -R "$REPO_ROOT/samples" "$CONTEXT_DIR/samples"

shopt -s nullglob

for MODEL_JSON in "$CONTEXT_DIR"/samples/model_repository/*/1/model.json; do
  MODEL_NAME="$(basename "$(dirname "$(dirname "$MODEL_JSON")")")"
  MODEL_PATH="/workspace/vllm_backend/samples/model_repository/$MODEL_NAME/1"
  MODEL_JSON="$MODEL_JSON" MODEL_PATH="$MODEL_PATH" python - <<'PY'
import json
import os

path = os.environ["MODEL_JSON"]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
data["model"] = os.environ["MODEL_PATH"]
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY
done

cat > "$BUILD_DIR/Dockerfile.generated" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /workspace/vllm_backend

COPY src/ /opt/tritonserver/backends/vllm/
COPY samples/ /workspace/vllm_backend/samples/
COPY scripts/ /workspace/vllm_backend/scripts/

RUN chmod +x /workspace/vllm_backend/scripts/*.sh \
    && /workspace/vllm_backend/scripts/install_runtime.sh

CMD ["/workspace/vllm_backend/scripts/run_triton.sh"]
DOCKERFILE

if [[ "$INCLUDE_MODEL" == "1" ]]; then
  echo "building image with local model files included"
else
  for MODEL_DIR in "$CONTEXT_DIR"/samples/model_repository/*/1; do
    rm -f "$MODEL_DIR"/*.py
    rm -f "$MODEL_DIR"/config.json
    rm -f "$MODEL_DIR"/generation_config.json
    rm -f "$MODEL_DIR"/tokenizer.json
    rm -f "$MODEL_DIR"/tokenizer.model
    rm -f "$MODEL_DIR"/tokenizer_config.json
    rm -f "$MODEL_DIR"/special_tokens_map.json
    rm -f "$MODEL_DIR"/chat_template.jinja
    rm -f "$MODEL_DIR"/model.safetensors
    rm -f "$MODEL_DIR"/model-*.safetensors
    rm -f "$MODEL_DIR"/model.safetensors.index.json
    rm -f "$MODEL_DIR"/preprocessor_config.json
    rm -f "$MODEL_DIR"/processor_config.json
    rm -rf "$MODEL_DIR"/.cache
    rm -rf "$MODEL_DIR"/.locks
  done
  echo "building runtime image without model weights"
fi

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f "$BUILD_DIR/Dockerfile.generated" \
  -t "$IMAGE_TAG" \
  "$CONTEXT_DIR"

echo "built $IMAGE_TAG"
