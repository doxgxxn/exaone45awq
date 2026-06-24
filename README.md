# EXAONE 4.5 AWQ Triton vLLM Backend

This repository packages a Triton vLLM backend setup for
`LGAI-EXAONE/EXAONE-4.5-33B-AWQ`.

## VastAI Quickstart

Run the commands below inside a Triton Server container on VastAI. Set
`HF_TOKEN` to a Hugging Face token that can access the EXAONE model repository.

```bash
set -euo pipefail

export HF_TOKEN="hf_xxx"
test -n "$HF_TOKEN"

apt-get update
apt-get install -y --no-install-recommends git

rm -rf /tmp/vllm_backend
mkdir -p /opt/tritonserver/backends/vllm
git clone https://github.com/doxgxxn/exaone45awq.git /tmp/vllm_backend
cp -r /tmp/vllm_backend/src/* /opt/tritonserver/backends/vllm

pip install --no-cache-dir --force-reinstall \
  "numpy<2" \
  "pydantic==2.10.6" \
  huggingface_hub

pip install --no-cache-dir --force-reinstall \
  "git+https://github.com/lkm2835/vllm.git@add-exaone4_5"

pip install --no-cache-dir --force-reinstall \
  "git+https://github.com/nuxlear/transformers.git@add-exaone4_5-v5.3.0.dev0"

python /tmp/vllm_backend/samples/model_down.py

python -c "import vllm, transformers, huggingface_hub, numpy, pydantic; print('deps ok', transformers.__version__, numpy.__version__)"
test -f /tmp/vllm_backend/samples/model_repository/exaone4_5_awq/1/config.json
test -f /tmp/vllm_backend/samples/model_repository/exaone4_5_awq/1/tokenizer.json

tritonserver --model-repository /tmp/vllm_backend/samples/model_repository
```

## Test Request

The sample model name is `exaone4_5_awq`.

```bash
python /tmp/vllm_backend/samples/client.py \
  -m exaone4_5_awq \
  --return-reasoning \
  --exclude-inputs-in-outputs
```

## Troubleshooting

- `HF_TOKEN` must be set and must have access to `LGAI-EXAONE/EXAONE-4.5-33B-AWQ`.
- Model files must exist under `/tmp/vllm_backend/samples/model_repository/exaone4_5_awq/1`.
- `samples/model_repository/exaone4_5_awq/1/model.json` must point to `/tmp/vllm_backend/samples/model_repository/exaone4_5_awq/1`.
- If model loading runs out of memory, lower `max_model_len` or `gpu_memory_utilization` in `model.json`.
