# EXAONE 4.5 / Gemma Triton vLLM Backend

This repository packages a Triton vLLM backend setup with sample model
repositories for the following Hugging Face models:

- `LGAI-EXAONE/EXAONE-4.5-33B-AWQ` as `exaone4_5_awq`
- `LGAI-EXAONE/EXAONE-4.5-33B` as `exaone4_5_33b`
- `google/gemma-4-31B-it` as `gemma4_31b_it`

The repository intentionally keeps only source, scripts, and model repository
configuration. Do not commit Hugging Face tokens, model weights, Docker image
archives, or SIF files.

## Repository Layout

- `samples/model_repository/exaone4_5_awq`: Triton model config for EXAONE 4.5 33B AWQ.
- `samples/model_repository/exaone4_5_33b`: Triton model config for EXAONE 4.5 33B BF16/non-AWQ.
- `samples/model_repository/gemma4_31b_it`: Triton model config for Gemma 4 31B IT.
- `scripts/download_model.sh`: downloads the original EXAONE AWQ sample model.
- `scripts/download_exaone4_5_33b.sh`: downloads `LGAI-EXAONE/EXAONE-4.5-33B`.
- `scripts/download_gemma4_31b_it.sh`: downloads `google/gemma-4-31B-it`.
- `scripts/install_runtime.sh`: installs runtime Python dependencies and copies `src/` into `/opt/tritonserver/backends/vllm`.
- `scripts/run_triton.sh`: starts Triton with the sample model repository.
- `scripts/build_docker_image.sh`: generates a temporary Dockerfile/context and builds a Docker image.
- `scripts/save_docker_image.sh`: saves a built Docker image to a tar archive for offline transfer.
- `scripts/build_sif.sh`: builds a Singularity/Apptainer SIF from a Docker image or Docker archive.

## Compatibility Notes

Each model has its own Triton model directory and `model.json`. Do not reuse the
AWQ `model.json` for non-AWQ models.

- `exaone4_5_awq` keeps `quantization: compressed-tensors` and MTP speculative decoding.
- `exaone4_5_33b` removes AWQ quantization and keeps the EXAONE reasoning parser.
- `gemma4_31b_it` removes EXAONE-specific quantization, speculative decoding, and reasoning parser options.

`scripts/build_docker_image.sh` patches every
`samples/model_repository/*/1/model.json` so the `model` path points to the
container path under `/workspace/vllm_backend/samples/model_repository/...`.

The runtime currently installs custom vLLM and Transformers branches from
`scripts/install_runtime.sh`. If Gemma 4 support is not available in those
branches, update the vLLM/Transformers revisions there before serving
`google/gemma-4-31B-it`.

## VastAI Online Bring-Up

Start from `nvcr.io/nvidia/tritonserver:26.05-trtllm-python-py3` or another
compatible Triton Server image on VastAI.

Set your Hugging Face token first:

```bash
export HF_TOKEN="hf_xxx"
```

Install the backend:

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1
set -euo pipefail

test -n "$HF_TOKEN" || { echo "HF_TOKEN is required"; exit 1; }

apt-get update
apt-get install -y --no-install-recommends git

rm -rf /tmp/vllm_backend
git clone https://github.com/doxgxxn/exaone45awq.git /tmp/vllm_backend
cd /tmp/vllm_backend

./scripts/install_runtime.sh
```

Download the model you want to serve:

```bash
./scripts/download_model.sh
./scripts/download_exaone4_5_33b.sh
./scripts/download_gemma4_31b_it.sh
```

You can download all three, but loading all three at the same time requires
enough GPU memory. For most test environments, load one model explicitly.

## Run With Triton

Run only EXAONE 4.5 33B AWQ:

```bash
TORCH_NVSHMEM_DISABLE=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
./scripts/run_triton.sh \
  --model-control-mode=explicit \
  --load-model=exaone4_5_awq
```

Run only EXAONE 4.5 33B:

```bash
TORCH_NVSHMEM_DISABLE=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
./scripts/run_triton.sh \
  --model-control-mode=explicit \
  --load-model=exaone4_5_33b
```

Run only Gemma 4 31B IT:

```bash
TORCH_NVSHMEM_DISABLE=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
./scripts/run_triton.sh \
  --model-control-mode=explicit \
  --load-model=gemma4_31b_it
```

Run every model in `samples/model_repository`:

```bash
TORCH_NVSHMEM_DISABLE=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
./scripts/run_triton.sh
```

Test the Triton gRPC client:

```bash
python samples/client.py \
  -m exaone4_5_awq \
  --return-reasoning \
  --exclude-inputs-in-outputs
```

```bash
python samples/client.py \
  -m exaone4_5_33b \
  --return-reasoning \
  --exclude-inputs-in-outputs
```

```bash
python samples/client.py \
  -m gemma4_31b_it \
  --exclude-inputs-in-outputs
```

Test Triton HTTP generation:

```bash
curl -X POST http://localhost:8000/v2/models/exaone4_5_33b/generate \
  -H "Content-Type: application/json" \
  -d '{
        "text_input": "싸이는 어떤 가수야?",
        "parameters": {
          "stream": false,
          "return_reasoning": true,
          "enable_thinking": true,
          "temperature": 0.8,
          "top_p": 0.95,
          "max_tokens": 1024
        }
      }'
```

```bash
curl -X POST http://localhost:8000/v2/models/gemma4_31b_it/generate \
  -H "Content-Type: application/json" \
  -d '{
        "text_input": "Explain what Triton Inference Server does.",
        "parameters": {
          "stream": false,
          "temperature": 0.7,
          "top_p": 0.95,
          "max_tokens": 1024
        }
      }'
```

## Run With vLLM Directly

Use these commands when you want to bypass Triton and validate the vLLM runtime
directly.

EXAONE 4.5 33B AWQ:

```bash
HF_HOME=/tmp/hf_cache \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
TORCH_NVSHMEM_DISABLE=1 \
vllm serve \
  /tmp/vllm_backend/samples/model_repository/exaone4_5_awq/1 \
  --served-model-name exaone4_5_awq \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 16000 \
  --host 0.0.0.0 \
  --port 9000 \
  --quantization compressed-tensors \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 64}'
```

EXAONE 4.5 33B:

```bash
HF_HOME=/tmp/hf_cache \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
TORCH_NVSHMEM_DISABLE=1 \
vllm serve \
  /tmp/vllm_backend/samples/model_repository/exaone4_5_33b/1 \
  --served-model-name exaone4_5_33b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 16000 \
  --host 0.0.0.0 \
  --port 9001 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 64}'
```

Gemma 4 31B IT:

```bash
HF_HOME=/tmp/hf_cache \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
TORCH_NVSHMEM_DISABLE=1 \
vllm serve \
  /tmp/vllm_backend/samples/model_repository/gemma4_31b_it/1 \
  --served-model-name gemma4_31b_it \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 16000 \
  --host 0.0.0.0 \
  --port 9002 \
  --limit-mm-per-prompt '{"image":64}' \
  --max-num-batched-tokens 4096 \
  --trust-remote-code
```

Test the vLLM OpenAI-compatible endpoints:

```bash
curl -X POST http://localhost:9001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "exaone4_5_33b",
        "messages": [
          {"role": "user", "content": "싸이는 어떤 가수야?"}
        ],
        "max_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
        "chat_template_kwargs": {
          "enable_thinking": true
        }
      }'
```

```bash
curl -X POST http://localhost:9002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gemma4_31b_it",
        "messages": [
          {"role": "user", "content": "Explain what Triton Inference Server does."}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95
      }'
```

## Docker Image Build

After the VastAI flow succeeds, build a reusable Docker image from the repository.
By default, the image excludes model weights.

```bash
IMAGE_TAG=exaone45awq-triton:latest ./scripts/build_docker_image.sh
```

To include locally downloaded model files in the image, first run the relevant
download script, then build with `INCLUDE_MODEL=1`.

```bash
export HF_TOKEN="hf_xxx"
./scripts/download_exaone4_5_33b.sh
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
2. Download the target model with its download script.
3. If a single self-contained artifact is required, rebuild with `INCLUDE_MODEL=1`.
4. Save the Docker image with `scripts/save_docker_image.sh`.
5. Optionally convert the image/archive to SIF with `scripts/build_sif.sh`.
6. Transfer only the tar/SIF artifact to the offline machine.

Do not bake `HF_TOKEN` into the image. Use it only during the online download or
build step. Public redistribution of model weights or NGC-derived images may be
restricted by their respective licenses and terms.
