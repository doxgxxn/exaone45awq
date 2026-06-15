# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Union

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
try:
    from vllm.sampling_params import (
        ReasoningParams,
        SamplingParams,
        StructuredOutputsParams,
    )
except ImportError:  # pragma: no cover - older vLLM versions
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams  # type: ignore
    ReasoningParams = None  # type: ignore
from vllm.usage.usage_lib import UsageContext
from vllm.v1.metrics.loggers import StatLoggerFactory


def coerce_parameters_payload(
    payload: Any, logger: "pb_utils.Logger | None" = None
) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            if logger is not None:
                logger.log_warn(
                    "[vllm] Received sampling parameters that could not be parsed as JSON."
                )
            return {}
    if isinstance(payload, dict):
        return dict(payload)
    if hasattr(payload, "items"):
        return dict(payload.items())
    if logger is not None:
        logger.log_warn(
            f"[vllm] Unsupported sampling parameter payload type: {type(payload)}. "
            "Using default sampling parameters."
        )
    return {}


class TritonSamplingParams(SamplingParams):
    """
    Extended sampling parameters for text generation via
    Triton Inference Server and vLLM backend.

    Attributes:
        lora_name (Optional[str]): The name of the LoRA (Low-Rank Adaptation)
        to use for inference.
        return_reasoning (Optional[bool]): Whether reasoning tokens should be returned.
    """

    lora_name: Optional[str] = None
    return_reasoning: Optional[bool] = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the `TritonSamplingParams` object.

        This method overrides the `__repr__` method of the parent class
        to include additional attributes in the string representation.

        Returns:
            A string representation of the object.
        """
        base = super().__repr__()
        return (
            f"{base}, lora_name={self.lora_name}, return_reasoning={self.return_reasoning}"
        )

    @staticmethod
    def from_dict(
        params_dict_str: Union[str, bytes, dict, None],
        logger: "pb_utils.Logger",
    ) -> "TritonSamplingParams":
        """
        Creates a `TritonSamplingParams` object from a dictionary-like payload.

        The payload can be provided as:
          * A JSON string (or bytes) containing sampling parameters
          * A Python ``dict`` with the sampling parameters
          * ``None`` to fall back to the default ``SamplingParams``

        Returns:
            TritonSamplingParams: An instance of TritonSamplingParams.
        """

        def _normalize_bool(value: Any) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        try:
            params_dict = coerce_parameters_payload(params_dict_str, logger)

            # Some clients wrap the sampling configuration inside a `sampling_parameters`
            # key. If present, unwrap it before continuing so we only pass sampling
            # arguments to vLLM.
            if "sampling_parameters" in params_dict:
                params_dict = coerce_parameters_payload(
                    params_dict["sampling_parameters"], logger
                )

            structured_outputs_cfg = params_dict.get("structured_outputs")
            if structured_outputs_cfg is not None:
                structured_outputs_cfg = coerce_parameters_payload(
                    structured_outputs_cfg, logger
                )
                if structured_outputs_cfg:
                    params_dict["structured_outputs"] = StructuredOutputsParams(
                        **structured_outputs_cfg
                    )

            reasoning_cfg = params_dict.get("reasoning")
            if reasoning_cfg is not None:
                reasoning_cfg = coerce_parameters_payload(reasoning_cfg, logger)
                if reasoning_cfg and "enable" in reasoning_cfg and not reasoning_cfg.get(
                    "enable", True
                ):
                    # When reasoning is explicitly disabled we simply remove the block to
                    # avoid passing unnecessary structures to older vLLM versions.
                    params_dict.pop("reasoning", None)
                elif reasoning_cfg and ReasoningParams is not None:
                    params_dict["reasoning"] = ReasoningParams(**reasoning_cfg)
                elif reasoning_cfg:
                    logger.log_warn(
                        "[vllm] Reasoning parameters provided, but this vLLM version does "
                        "not expose `ReasoningParams`. Ignoring reasoning configuration."
                    )
                    params_dict.pop("reasoning", None)

            init_kwargs = dict(params_dict)
            lora_name = init_kwargs.pop("lora_name", None)
            return_reasoning = init_kwargs.pop("return_reasoning", None)

            try:
                sampling_params = TritonSamplingParams(**init_kwargs)
            except TypeError as exc:
                known_fields = set(SamplingParams.__annotations__.keys())
                filtered_kwargs = {
                    key: value for key, value in init_kwargs.items() if key in known_fields
                }
                logger.log_warn(
                    "[vllm] Dropped unsupported sampling parameters while creating "
                    f"`TritonSamplingParams`: {exc}"
                )
                sampling_params = TritonSamplingParams(**filtered_kwargs)

            if lora_name is not None:
                sampling_params.lora_name = lora_name

            if return_reasoning is not None:
                sampling_params.return_reasoning = _normalize_bool(return_reasoning)

            return sampling_params
        except Exception as e:
            logger.log_error(
                f"[vllm] Was trying to create `TritonSamplingParams`, but got exception: {e}"
            )

        return TritonSamplingParams()


# Copy from vllm/vllm/entrypoints/openai/api_server.py with custom stat_loggers
@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    logger: "pb_utils.Logger",
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: bool = False,
    stat_loggers: Optional[list[StatLoggerFactory]] = None,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Create the EngineConfig
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if disable_frontend_multiprocessing:
        logger.log_warn("V1 is enabled, but got --disable-frontend-multiprocessing.")

    from vllm.v1.engine.async_llm import AsyncLLM

    async_llm: AsyncLLM | None = None

    try:
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
        )

        # Don't keep the dummy data in memory
        assert async_llm is not None
        await async_llm.reset_mm_cache()

        yield async_llm
    finally:
        if async_llm:
            async_llm.shutdown()
