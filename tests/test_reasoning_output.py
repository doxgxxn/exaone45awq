"""Lightweight unit tests for reasoning_output plumbing.

These tests run without a Triton or vLLM installation by stubbing the minimal
APIs that src/utils/request.py imports.

Run:
  python3 tests/test_reasoning_output.py
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass



# -------------------------
# Stub external dependencies
# -------------------------


def _install_stub_modules() -> None:
    # numpy (minimal stub)
    np = types.ModuleType("numpy")

    class _FakeNDArray(list):
        def tolist(self):
            return list(self)

    def _asarray(obj, dtype=None):
        if isinstance(obj, _FakeNDArray):
            return obj
        if isinstance(obj, list):
            return _FakeNDArray(obj)
        return _FakeNDArray([obj])

    np.asarray = _asarray
    np.array = _asarray
    np.dtype = object  # used for type annotations in the backend
    np.object_ = object()
    np.float32 = object()
    np.uint32 = object()

    sys.modules["numpy"] = np

    # triton_python_backend_utils
    pb_utils = types.ModuleType("triton_python_backend_utils")

    class _Logger:
        @staticmethod
        def log_warn(msg: str) -> None:
            pass

        @staticmethod
        def log_error(msg: str) -> None:
            raise AssertionError(msg)

    class _Tensor:
        def __init__(self, name: str, array):
            self.name = name
            self._array = array

        def as_numpy(self):
            return self._array

    class _InferenceResponse:
        def __init__(self, output_tensors=None, error=None):
            self.output_tensors = list(output_tensors or [])
            self.error = error

        def tensor_dict(self):
            return {t.name: t.as_numpy() for t in self.output_tensors}

    def _get_input_tensor_by_name(req, name: str):
        return req.inputs.get(name)

    pb_utils.Logger = _Logger
    pb_utils.Tensor = _Tensor
    pb_utils.InferenceResponse = _InferenceResponse
    pb_utils.get_input_tensor_by_name = _get_input_tensor_by_name

    sys.modules["triton_python_backend_utils"] = pb_utils

    # PIL
    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")

    class _Image:
        pass

    pil_image.Image = _Image
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = pil_image

    # vllm (minimal submodules referenced by request.py)
    vllm = types.ModuleType("vllm")
    vllm_inputs = types.ModuleType("vllm.inputs")

    @dataclass
    class TokensPrompt:
        prompt_token_ids: list[int]

    vllm_inputs.TokensPrompt = TokensPrompt

    vllm_lora = types.ModuleType("vllm.lora")
    vllm_lora_request = types.ModuleType("vllm.lora.request")

    class LoRARequest:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    vllm_lora_request.LoRARequest = LoRARequest

    vllm_outputs = types.ModuleType("vllm.outputs")

    class RequestOutput:  # placeholder for typing
        pass

    class PoolingRequestOutput:
        def __class_getitem__(cls, item):
            return cls

    class EmbeddingOutput:
        pass

    class EmbeddingRequestOutput:
        @staticmethod
        def from_base(base):
            return base

    vllm_outputs.RequestOutput = RequestOutput
    vllm_outputs.PoolingRequestOutput = PoolingRequestOutput
    vllm_outputs.EmbeddingOutput = EmbeddingOutput
    vllm_outputs.EmbeddingRequestOutput = EmbeddingRequestOutput

    vllm_pooling_params = types.ModuleType("vllm.pooling_params")

    class PoolingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm_pooling_params.PoolingParams = PoolingParams

    vllm_utils = types.ModuleType("vllm.utils")

    def random_uuid() -> str:
        return "uuid"

    vllm_utils.random_uuid = random_uuid

    sys.modules["vllm"] = vllm
    sys.modules["vllm.inputs"] = vllm_inputs
    sys.modules["vllm.lora"] = vllm_lora
    sys.modules["vllm.lora.request"] = vllm_lora_request
    sys.modules["vllm.outputs"] = vllm_outputs
    sys.modules["vllm.pooling_params"] = vllm_pooling_params
    sys.modules["vllm.utils"] = vllm_utils

    # utils.vllm_backend_utils
    backend_utils = types.ModuleType("utils.vllm_backend_utils")

    def coerce_parameters_payload(payload, logger=None):
        if payload is None:
            return {}
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return {}
        if isinstance(payload, dict):
            return dict(payload)
        if hasattr(payload, "items"):
            return dict(payload.items())
        return {}

    class TritonSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.lora_name = kwargs.get("lora_name")
            self.return_reasoning = kwargs.get("return_reasoning")

        @staticmethod
        def from_dict(params_dict_str, logger):
            d = coerce_parameters_payload(params_dict_str, logger)
            # mimic real behavior (strip out custom keys from init)
            init_kwargs = dict(d)
            init_kwargs.pop("return_reasoning", None)
            inst = TritonSamplingParams(**init_kwargs)
            if "return_reasoning" in d:
                inst.return_reasoning = bool(d["return_reasoning"])
            return inst

    backend_utils.coerce_parameters_payload = coerce_parameters_payload
    backend_utils.TritonSamplingParams = TritonSamplingParams

    sys.modules["utils.vllm_backend_utils"] = backend_utils


# -------------------------
# Dummy request/output types
# -------------------------


class DummyInputTensor:
    def __init__(self, array):
        self._array = array

    def as_numpy(self):
        return self._array


class DummyTritonRequest:
    def __init__(self, inputs: dict[str, DummyInputTensor], parameters=None):
        self.inputs = inputs
        self._parameters = parameters or {}

    def parameters(self):
        return self._parameters


@dataclass
class DummySequenceOutput:
    text: str
    reasoning_output: str | None = None


@dataclass
class DummyRequestOutput:
    prompt: str
    outputs: list[DummySequenceOutput]
    finished: bool = False
    prompt_token_ids: list[int] | None = None


# -------------------------
# Tests
# -------------------------


def test_return_reasoning_tensor_is_merged_into_parameters():
    _install_stub_modules()

    # Import after stubs are installed.
    sys.path.insert(0, "src")
    request_mod = importlib.import_module("utils.request")

    np = sys.modules["numpy"]

    sampling_params = {"temperature": 0.0}
    req = DummyTritonRequest(
        inputs={
            "text_input": DummyInputTensor([b"hi"]),
            "stream": DummyInputTensor([False]),
            "exclude_input_in_output": DummyInputTensor([True]),
            "sampling_parameters": DummyInputTensor(
                [json.dumps(sampling_params).encode("utf-8")]
            ),
            "return_reasoning": DummyInputTensor([True]),
        }
    )

    gr = request_mod.GenerateRequest(
        req,
        executor_callback=lambda *args, **kwargs: None,
        output_dtype=np.object_,
        logger=sys.modules["triton_python_backend_utils"].Logger,
        tokenizer=None,
    )

    (
        _prompt,
        _stream,
        _prepend_input,
        parameters,
        _additional_outputs,
        return_reasoning,
        _enable_thinking,
    ) = gr._get_input_tensors()

    assert return_reasoning is True
    assert parameters["return_reasoning"] is True
    assert parameters["reasoning"]["enable"] is True


def test_enable_thinking_renders_template_from_text_input():
    _install_stub_modules()

    sys.path.insert(0, "src")
    request_mod = importlib.import_module("utils.request")

    np = sys.modules["numpy"]

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        ):
            assert messages[0]["role"] == "user"
            assert kwargs.get("enable_thinking") is True
            return (
                "<|user|>\n"
                + messages[0]["content"]
                + "<|endofturn|>\n<|assistant|>\n<think>\n"
            )

    sampling_params = {"enable_thinking": True}
    req = DummyTritonRequest(
        inputs={
            "text_input": DummyInputTensor([b"hi"]),
            "stream": DummyInputTensor([False]),
            "exclude_input_in_output": DummyInputTensor([True]),
            "sampling_parameters": DummyInputTensor(
                [json.dumps(sampling_params).encode("utf-8")]
            ),
            "return_reasoning": DummyInputTensor([True]),
        }
    )

    gr = request_mod.GenerateRequest(
        req,
        executor_callback=lambda *args, **kwargs: None,
        output_dtype=np.object_,
        logger=sys.modules["triton_python_backend_utils"].Logger,
        tokenizer=FakeTokenizer(),
    )

    prompt, *_rest = gr._get_input_tensors()
    assert isinstance(prompt, str)
    assert prompt.startswith("<|user|>")


def test_reasoning_output_is_incremental_and_encoded():
    _install_stub_modules()
    sys.path.insert(0, "src")
    request_mod = importlib.import_module("utils.request")

    np = sys.modules["numpy"]

    req = DummyTritonRequest(
        inputs={
            "text_input": DummyInputTensor([b"prompt"]),
        }
    )

    gr = request_mod.GenerateRequest(
        req,
        executor_callback=lambda *args, **kwargs: None,
        output_dtype=np.object_,
        logger=sys.modules["triton_python_backend_utils"].Logger,
        tokenizer=None,
    )
    gr.return_reasoning = True
    gr.stream = True
    gr.enable_thinking = True
    gr.additional_outputs = {
        "return_finish_reason": False,
        "return_cumulative_logprob": False,
        "return_logprobs": False,
        "return_num_input_tokens": False,
        "return_num_output_tokens": False,
    }

    state = {}

    out1 = DummyRequestOutput(
        prompt="prompt",
        outputs=[DummySequenceOutput(text="Hello", reasoning_output="THINK1")],
        finished=False,
        prompt_token_ids=[1, 2],
    )
    resp1 = gr.create_response(out1, state, prepend_input=False)
    tensors1 = resp1.tensor_dict()
    assert tensors1["text_output"].tolist() == [b"Hello"]
    assert tensors1["reasoning_output"].tolist() == [b"THINK1"]

    out2 = DummyRequestOutput(
        prompt="prompt",
        outputs=[
            DummySequenceOutput(text="Hello world", reasoning_output="THINK1 THINK2")
        ],
        finished=True,
        prompt_token_ids=[1, 2],
    )
    resp2 = gr.create_response(out2, state, prepend_input=False)
    tensors2 = resp2.tensor_dict()
    assert tensors2["text_output"].tolist() == [b" world"]
    assert tensors2["reasoning_output"].tolist() == [b" THINK2"]


def test_fallback_extracts_think_from_prompt_when_reasoning_fields_missing():
    _install_stub_modules()
    sys.path.insert(0, "src")
    request_mod = importlib.import_module("utils.request")

    np = sys.modules["numpy"]

    req = DummyTritonRequest(inputs={"text_input": DummyInputTensor([b"prompt"])})

    gr = request_mod.GenerateRequest(
        req,
        executor_callback=lambda *args, **kwargs: None,
        output_dtype=np.object_,
        logger=sys.modules["triton_python_backend_utils"].Logger,
        tokenizer=None,
    )
    gr.return_reasoning = True
    gr.enable_thinking = True
    gr.additional_outputs = {
        "return_finish_reason": False,
        "return_cumulative_logprob": False,
        "return_logprobs": False,
        "return_num_input_tokens": False,
        "return_num_output_tokens": False,
    }

    state = {}
    out = DummyRequestOutput(
        # Simulate a chat-template prompt where <think> is in the prompt.
        prompt="<|assistant|>\n<think>\n",
        outputs=[DummySequenceOutput(text="ABC", reasoning_output=None)],
        finished=True,
        prompt_token_ids=[1],
    )
    resp = gr.create_response(out, state, prepend_input=False)
    tensors = resp.tensor_dict()
    assert tensors["reasoning_output"].tolist() == [b"ABC"]


def test_reasoning_parser_splits_content_and_reasoning():
    _install_stub_modules()
    sys.path.insert(0, "src")
    request_mod = importlib.import_module("utils.request")

    np = sys.modules["numpy"]

    class FakeReasoningParser:
        def extract_reasoning(self, model_output, request):
            reasoning, _, content = model_output.partition("</think>")
            return reasoning, content

    req = DummyTritonRequest(inputs={"text_input": DummyInputTensor([b"prompt"])})

    gr = request_mod.GenerateRequest(
        req,
        executor_callback=lambda *args, **kwargs: None,
        output_dtype=np.object_,
        logger=sys.modules["triton_python_backend_utils"].Logger,
        tokenizer=None,
    )
    gr.return_reasoning = True
    gr.stream = False
    gr.enable_thinking = True
    gr.reasoning_parser = FakeReasoningParser()
    gr.additional_outputs = {
        "return_finish_reason": False,
        "return_cumulative_logprob": False,
        "return_logprobs": False,
        "return_num_input_tokens": False,
        "return_num_output_tokens": False,
    }

    out = DummyRequestOutput(
        prompt="prompt",
        outputs=[DummySequenceOutput(text="reasoning</think>final")],
        finished=True,
        prompt_token_ids=[1],
    )
    resp = gr.create_response(out, {}, prepend_input=False)
    tensors = resp.tensor_dict()
    assert tensors["reasoning_output"].tolist() == [b"reasoning"]
    assert tensors["text_output"].tolist() == [b"final"]


def main() -> None:
    test_return_reasoning_tensor_is_merged_into_parameters()
    test_enable_thinking_renders_template_from_text_input()
    test_reasoning_output_is_incremental_and_encoded()
    test_fallback_extracts_think_from_prompt_when_reasoning_fields_missing()
    test_reasoning_parser_splits_content_and_reasoning()
    print("OK")


if __name__ == "__main__":
    main()
