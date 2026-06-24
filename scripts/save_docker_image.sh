#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-exaone45awq-triton:latest}"
OUTPUT="${OUTPUT:-exaone45awq-triton.tar}"

docker image inspect "$IMAGE_TAG" >/dev/null
docker save "$IMAGE_TAG" -o "$OUTPUT"
echo "saved $IMAGE_TAG to $OUTPUT"
