```
mkdir -p /opt/tritonserver/backends/vllm
git clone  https://github.com/doxgxxn/exaone45awq.git /tmp/vllm_backend
cp -r /tmp/vllm_backend/src/* /opt/tritonserver/backends/vllm

export HF_TOKEN=""
python /tmp/vllm_backend/samples/model_down.py

pip install --no-cache-dir --force-reinstall \
  "git+https://github.com/nuxlear/transformers.git@add-exaone4_5-v5.3.0.dev0" \
  "numpy<2" \
  "pydantic==2.10.6"

pip install --no-cache-dir --force-reinstall \
  "git+https://github.com/lkm2835/vllm.git@add-exaone4_5"

tritonserver --model-repository /tmp/vllm_backend/samples/model_repository

```
