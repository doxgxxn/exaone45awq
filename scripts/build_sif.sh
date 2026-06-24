#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-exaone45awq-triton:latest}"
OUTPUT="${OUTPUT:-exaone45awq-triton.sif}"
SOURCE="${SOURCE:-docker-daemon://$IMAGE_TAG}"

if command -v apptainer >/dev/null 2>&1; then
  apptainer build "$OUTPUT" "$SOURCE"
elif command -v singularity >/dev/null 2>&1; then
  singularity build "$OUTPUT" "$SOURCE"
else
  echo "apptainer or singularity is required." >&2
  exit 1
fi

echo "built $OUTPUT from $SOURCE"
