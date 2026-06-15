

from huggingface_hub import snapshot_download
import os

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("환경변수 HF_TOKEN이 설정되어 있지 않습니다.")

snapshot_download(
    repo_id="LGAI-EXAONE/EXAONE-4.5-33B-AWQ",
    revision="main",
    local_dir="/tmp/vllm_backend/samples/model_repository/exaon4_5_awq", # 작성 ..
    local_dir_use_symlinks=False,
    token=os.getenv("HF_TOKEN"),
    allow_patterns=[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "processor_config.json",
        "assets/*",
    ],
)

# tritonserver --model-repository /tmp/vllm_backend/samples/model_repository
