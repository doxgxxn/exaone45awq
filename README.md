# EXAONE 4.5 AWQ Triton vLLM Backend

This repository packages a Triton vLLM backend setup for
`LGAI-EXAONE/EXAONE-4.5-33B-AWQ`.

The repository intentionally keeps only source, scripts, and model repository
configuration. Do not commit Hugging Face tokens, model weights, Docker image
archives, or SIF files.

## Scripts

- `scripts/install_runtime.sh`: installs runtime Python dependencies and copies `src/` into `/opt/tritonserver/backends/vllm`.
- `scripts/download_model.sh`: downloads EXAONE model files into the Triton model repository. Requires `HF_TOKEN`.
- `scripts/run_triton.sh`: starts Triton with the sample model repository.
- `scripts/build_docker_image.sh`: generates a temporary Dockerfile/context and builds a Docker image.
- `scripts/save_docker_image.sh`: saves a built Docker image to a tar archive for offline transfer.
- `scripts/build_sif.sh`: builds a Singularity/Apptainer SIF from a Docker image or Docker archive.

## VastAI Online Bring-Up

Start from `nvcr.io/nvidia/tritonserver:26.05-trtllm-python-py3` or another
compatible Triton Server image on VastAI.

Set your Hugging Face token first:

```bash
export HF_TOKEN="hf_xxx"
```

Then install and run:

```bash
set -euo pipefail

test -n "$HF_TOKEN" || { echo "HF_TOKEN is required"; exit 1; }

apt-get update
apt-get install -y --no-install-recommends git

rm -rf /tmp/vllm_backend
git clone https://github.com/doxgxxn/exaone45awq.git /tmp/vllm_backend
cd /tmp/vllm_backend

./scripts/install_runtime.sh
./scripts/download_model.sh
./scripts/run_triton.sh
```

The sample model name is `exaone4_5_awq`.

```bash
python /tmp/vllm_backend/samples/client.py \
  -m exaone4_5_awq \
  --return-reasoning \
  --exclude-inputs-in-outputs
```

## Docker Image Build

After the VastAI flow succeeds, build a reusable Docker image from the repository.
By default, the image excludes model weights.

```bash
IMAGE_TAG=exaone45awq-triton:latest ./scripts/build_docker_image.sh
```

To include locally downloaded model files in the image, first run
`scripts/download_model.sh`, then build with `INCLUDE_MODEL=1`.

Set your Hugging Face token first:

```bash
export HF_TOKEN="hf_xxx"
```

Then download and build:

```bash
./scripts/download_model.sh
INCLUDE_MODEL=1 IMAGE_TAG=exaone45awq-triton:offline ./scripts/build_docker_image.sh
```

Run the image:

```bash
docker run --gpus all --rm --net=host \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  exaone45awq-triton:latest
```

Save it for offline transfer:

```bash
IMAGE_TAG=exaone45awq-triton:offline \
OUTPUT=exaone45awq-triton-offline.tar \
./scripts/save_docker_image.sh
```

Load it on another machine:

```bash
docker load -i exaone45awq-triton-offline.tar
```

## Singularity / Apptainer SIF

Build a SIF from a local Docker image:

```bash
IMAGE_TAG=exaone45awq-triton:offline \
OUTPUT=exaone45awq-triton-offline.sif \
./scripts/build_sif.sh
```

Build a SIF from a Docker archive:

```bash
SOURCE=docker-archive://exaone45awq-triton-offline.tar \
OUTPUT=exaone45awq-triton-offline.sif \
./scripts/build_sif.sh
```

Run the SIF:

```bash
apptainer run --nv --network host exaone45awq-triton-offline.sif
```

If `apptainer run` does not use the image `CMD` in your environment, run Triton
explicitly:

```bash
apptainer exec --nv --network host exaone45awq-triton-offline.sif \
  /workspace/vllm_backend/scripts/run_triton.sh
```

## Offline Guidance

For an internet-free environment, prepare artifacts on an internet-connected
machine first.

1. Build the runtime Docker image with `scripts/build_docker_image.sh`.
2. Download the model with `scripts/download_model.sh`.
3. If a single self-contained artifact is required, rebuild with `INCLUDE_MODEL=1`.
4. Save the Docker image with `scripts/save_docker_image.sh`.
5. Optionally convert the image/archive to SIF with `scripts/build_sif.sh`.
6. Transfer only the tar/SIF artifact to the offline machine.

Do not bake `HF_TOKEN` into the image. Use it only during the online download or
build step. Public redistribution of model weights or NGC-derived images may be
restricted by their respective licenses and terms.

## Troubleshooting

- `HF_TOKEN` must be set and must have access to `LGAI-EXAONE/EXAONE-4.5-33B-AWQ`.
- Model files must exist under `samples/model_repository/exaone4_5_awq/1` before running offline.
- `samples/model_repository/exaone4_5_awq/1/model.json` must point to `/tmp/vllm_backend/samples/model_repository/exaone4_5_awq/1` for the VastAI clone flow.
- `scripts/build_docker_image.sh` patches the generated Docker build context to use `/workspace/vllm_backend/samples/model_repository/exaone4_5_awq/1` inside the image.
- If model loading runs out of memory, lower `max_model_len` or `gpu_memory_utilization` in `model.json`.
