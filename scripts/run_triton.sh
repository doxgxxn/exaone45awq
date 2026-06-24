#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-$REPO_ROOT/samples/model_repository}"

exec tritonserver --model-repository "$MODEL_REPOSITORY" "$@"
