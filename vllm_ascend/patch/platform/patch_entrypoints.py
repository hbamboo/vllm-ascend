from __future__ import annotations

import asyncio
import io
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator
from http import HTTPStatus
from typing import Any, cast

import numpy as np
import pybase64 as base64
from fastapi import Request
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    get_history_tool_calls_cnt,
    make_tool_call_id,
)
from vllm.entrypoints.openai import api_server as _api_server_module
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    FunctionCall,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    clamp_prompt_logprobs,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    parse_chat_output,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)
from vllm.inputs import EngineInput, SingletonPrompt, TokensPrompt
from vllm.logger import logger
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.abstract_parser import Parser
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.renderers import merge_kwargs
from vllm.renderers.inputs.preprocess import prompt_to_seq
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParser
from vllm.utils.async_utils import merge_async_iterators
from vllm.utils.collection_utils import as_list
from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser

from vllm_ascend import envs


def _p_then_d_token_reuse(request: object) -> bool:
    """该请求是否走"提前派发"的首 token 补投路径(见 recompute_scheduler).

    为真时 entrypoint 不做复用/合成首输出: 请求下发时 P 还没出首 token, 这里本来也
    取不到; 复用由 scheduler 在 KV 到达时从 spool 取, 首 token 文本由 proxy 下发.
    """
    params = getattr(request, "kv_transfer_params", None)
    return bool(params and params.get("p_then_d_token_reuse"))


def _package_prefilled_text(tokenizer: TokenizerLike, kv_transfer_params: dict) -> None:
    """给已打包的 prefilled_token 补上解码后的文本(供 p_then_d 的提前派发用).

    提前派发时 D 的请求是在 P 出首 token 之前下发的, D 侧 entrypoint 读不到
    prefilled_token, 也就不会用 _wrap_with_prefilled 合成首输出 —— 首 token 改由
    proxy 直接下发给客户端。proxy 没有 tokenizer, 所以文本在这里(有 tokenizer 的
    P 引擎)算好带走。

    与 D 侧"放弃复用"的判据保持一致(见 _create_completion 里的同名判断): 文本以
    替换字符结尾(不完整 UTF-8)或就是 EOS 时, 标记为不可复用, 让 proxy 整条回退到
    "D 自己生成首 token"。
    """
    tokens = kv_transfer_params.get("prefilled_token")
    if not tokens:
        return
    try:
        # convert_ids_to_tokens 收单个 id; convert_tokens_to_string 收 token 列表.
        new_token = tokenizer.convert_ids_to_tokens(tokens[0])
        delta_text = tokenizer.convert_tokens_to_string([new_token])
        eos_id = getattr(tokenizer, "eos_token_id", None)
    except Exception as e:  # 分词器异常不该影响响应本身
        logger.warning("p_then_d: failed to decode prefilled token %s: %s", tokens, e)
        return
    kv_transfer_params["prefilled_text"] = delta_text
    if delta_text.endswith("�") or (eos_id is not None and tokens[0] == eos_id):
        kv_transfer_params["prefilled_text_reusable"] = False


async def _wrap_with_prefilled(
    generator: AsyncGenerator[RequestOutput, None],
    request_id: str,
    prompt_token_ids: list[int] | None,
    prefilled_text: str,
    prefilled_tokens: list[int],
) -> AsyncGenerator[RequestOutput, None]:
    """Yield a synthetic first output for prefilled token, then continue."""
    output = RequestOutput(
        request_id=request_id,
        prompt=None,
        finished=False,
        prompt_logprobs=None,
        prompt_token_ids=prompt_token_ids,
        outputs=[
            CompletionOutput(
                index=0,
                cumulative_logprob=None,
                logprobs=None,
                text=prefilled_text,
                token_ids=prefilled_tokens,
            )
        ],
    )
    yield output
    async for res in generator:
        yield res


async def _create_completion(
    self,
    request: CompletionRequest,
    raw_request: Request | None = None,
) -> AsyncGenerator[str, None] | CompletionResponse | ErrorResponse:
    if request.stream and request.use_beam_search:
        return self.create_error_response("Streaming is not currently supported with beam search")
    result = await self.render_completion_request(request)
    if isinstance(result, ErrorResponse):
        return result
    engine_inputs = result
    request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"
    created_time = int(time.time())
    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata
    lora_request = self._maybe_get_adapters(request)
    # Extract data_parallel_rank from header (router can inject it)
    data_parallel_rank = self._get_data_parallel_rank(raw_request)
    # Schedule the request and get the result generator.
    max_model_len = self.model_config.max_model_len
    generators: list[AsyncGenerator[RequestOutput, None]] = []
    for i, engine_input in enumerate(engine_inputs):
        prompt_token_ids = self._extract_prompt_components(engine_input).token_ids
        max_tokens = get_max_tokens(
            max_model_len,
            request.max_tokens,
            self._extract_prompt_len(engine_input),
            self.default_sampling_params,
            self.override_max_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )
        sampling_params: SamplingParams | BeamSearchParams
        if request.use_beam_search:
            sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
        else:
            sampling_params = request.to_sampling_params(
                max_tokens,
                self.default_sampling_params,
            )
        # ascend vllm adapt start, reuse prefill token (streaming only)
        # p_then_d 提前派发: 请求下发时 P 还没出首 token, 这里拿不到 prefilled_token,
        # 复用交给 scheduler 在 KV 到达时从 spool 取(recompute_scheduler 里注入) ——
        # 且因为此处没有合成首输出的 wrapper, 首 token 由 proxy 直接下发给客户端.
        if envs.REUSE_PREFILLED_TOKENS and request.stream and not _p_then_d_token_reuse(request):
            tokenizer = self.renderer.tokenizer
            if request.kv_transfer_params and "prefilled_token" in request.kv_transfer_params:
                new_tokens = tokenizer.convert_ids_to_tokens(request.kv_transfer_params["prefilled_token"][0])
                delta_text = tokenizer.convert_tokens_to_string([new_tokens])
                # If the decoded text ends with the replacement char, it
                # indicates an incomplete UTF-8 sequence — abandon reuse.
                # If the prefilled side already triggered a stop reason, fall
                # back to normal generation.
                if (
                    min(max_tokens, max_model_len - len(prompt_token_ids)) == 1
                    or delta_text.endswith("�")
                    or any(s is not None for s in request.kv_transfer_params["stop_reasons"])
                    or request.kv_transfer_params["prefilled_token"][0] == tokenizer.eos_token_id
                ):
                    request.kv_transfer_params.pop("prefilled_token", None)
                else:
                    request.kv_transfer_params["prefilled_texts"] = delta_text
        # ascend vllm adapt end
        request_id_item = f"{request_id}-{i}"
        self._log_inputs(
            request_id_item,
            engine_input,
            params=sampling_params,
            lora_request=lora_request,
        )
        trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)
        if isinstance(sampling_params, BeamSearchParams):
            generator = self.beam_search(
                prompt=engine_input,
                request_id=request_id,
                params=sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
            )
        else:
            generator = self.engine_client.generate(
                engine_input,
                sampling_params,
                request_id_item,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
                data_parallel_rank=data_parallel_rank,
            )
            # ascend vllm adapt start, reuse prefill token (streaming only)
            if envs.REUSE_PREFILLED_TOKENS and request.stream:
                if (
                    request.kv_transfer_params
                    and "prefilled_token" in request.kv_transfer_params
                    and "prefilled_texts" in request.kv_transfer_params
                ):
                    prefilled_text = request.kv_transfer_params["prefilled_texts"]
                    prefilled_tokens = request.kv_transfer_params["prefilled_token"]
                    generator = _wrap_with_prefilled(
                        generator,
                        request_id_item,
                        prompt_token_ids,
                        prefilled_text,
                        prefilled_tokens,
                    )
            # ascend vllm adapt end
        generators.append(generator)
    result_generator = merge_async_iterators(*generators)
    model_name = self.models.model_name(lora_request)
    num_prompts = len(engine_inputs)
    # Streaming response
    tokenizer = self.renderer.tokenizer
    if request.stream:
        return self.completion_stream_generator(
            request,
            engine_inputs,
            result_generator,
            request_id,
            created_time,
            model_name,
            num_prompts=num_prompts,
            tokenizer=tokenizer,
            request_metadata=request_metadata,
        )
    # Non-streaming response
    final_res_batch: list[RequestOutput | None] = [None] * num_prompts
    try:
        async for i, res in result_generator:
            final_res_batch[i] = res
        for i, final_res in enumerate(final_res_batch):
            assert final_res is not None
            # The output should contain the input text
            # We did not pass it into vLLM engine to avoid being redundant
            # with the inputs token IDs
            if final_res.prompt is None:
                final_res.prompt = self._extract_prompt_text(engine_inputs[i])
        # ascend vllm adapt start
        if envs.REUSE_PREFILLED_TOKENS:
            # Prefill side: package first generated token for downstream Decode.
            for final_res in final_res_batch:
                if final_res and final_res.kv_transfer_params:
                    final_res.kv_transfer_params["prefilled_token"] = [final_res.outputs[0].token_ids[0]]
                    final_res.kv_transfer_params["stop_reasons"] = [output.stop_reason for output in final_res.outputs]
        if envs.SKIP_DECODE_TOKENIZE:
            for final_res in final_res_batch:
                if final_res and final_res.kv_transfer_params:
                    final_res.kv_transfer_params["prompt_token_ids"] = final_res.prompt_token_ids
        # ascend vllm adapt end
        final_res_batch_checked = cast(list[RequestOutput], final_res_batch)
        response = self.request_output_to_completion_response(
            final_res_batch_checked,
            request,
            request_id,
            created_time,
            model_name,
            tokenizer,
            request_metadata,
        )
    except asyncio.CancelledError:
        return self.create_error_response("Client disconnected")
    # When user requests streaming but we don't stream, we still need to
    # return a streaming response with a single event.
    if request.stream:
        response_json = response.model_dump_json()

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return fake_stream_generator()
    return response


async def _create_completion(
    self,
    request: CompletionRequest,
    raw_request: Request | None = None,
) -> AsyncGenerator[str, None] | CompletionResponse | ErrorResponse:
    if request.stream and request.use_beam_search:
        return self.create_error_response("Streaming is not currently supported with beam search")

    result = await self.render_completion_request(request)
    if isinstance(result, ErrorResponse):
        return result

    engine_inputs = result

    request_id = f"cmpl-{self._base_request_id(raw_request, request.request_id)}"
    created_time = int(time.time())

    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata

    lora_request = self._maybe_get_adapters(request)

    # Extract data_parallel_rank from header (router can inject it)
    data_parallel_rank = self._get_data_parallel_rank(raw_request)

    # Schedule the request and get the result generator.
    max_model_len = self.model_config.max_model_len
    generators: list[AsyncGenerator[RequestOutput, None]] = []
    for i, engine_input in enumerate(engine_inputs):
        prompt_token_ids = self._extract_prompt_components(engine_input).token_ids

        max_tokens = get_max_tokens(
            max_model_len,
            request.max_tokens,
            self._extract_prompt_len(engine_input),
            self.default_sampling_params,
            self.override_max_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )

        sampling_params: SamplingParams | BeamSearchParams
        if request.use_beam_search:
            sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
        else:
            sampling_params = request.to_sampling_params(
                max_tokens,
                self.default_sampling_params,
            )

        # ascend vllm adapt start, reuse prefill token
        if (
            envs.REUSE_PREFILLED_TOKENS
            and request.kv_transfer_params
            and "prefilled_token" in request.kv_transfer_params
        ):
            tokenizer = self.renderer.tokenizer
            new_tokens = tokenizer.convert_ids_to_tokens(request.kv_transfer_params["prefilled_token"][0])
            delta_text = tokenizer.convert_tokens_to_string([new_tokens])
            # If the decoded text ends with the replacement char, it
            # indicates an incomplete UTF-8 sequence — abandon reuse.
            # If the prefilled side already triggered a stop reason, fall
            # back to normal generation.
            if (
                min(max_tokens, max_model_len - len(prompt_token_ids)) == 1
                or delta_text.endswith("�")
                or any(s is not None for s in request.kv_transfer_params["stop_reasons"])
                or request.kv_transfer_params["prefilled_token"][0] == tokenizer.eos_token_id
            ):
                request.kv_transfer_params.pop("prefilled_token", None)
            else:
                request.kv_transfer_params["prefilled_texts"] = delta_text
        # ascend vllm adapt end

        request_id_item = f"{request_id}-{i}"

        self._log_inputs(
            request_id_item,
            engine_input,
            params=sampling_params,
            lora_request=lora_request,
        )

        trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)

        if isinstance(sampling_params, BeamSearchParams):
            generator = self.beam_search(
                prompt=engine_input,
                request_id=request_id,
                params=sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
            )
        else:
            generator = self.engine_client.generate(
                engine_input,
                sampling_params,
                request_id_item,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
                data_parallel_rank=data_parallel_rank,
            )

            # ascend vllm adapt start, reuse prefill token (streaming only)
            if envs.REUSE_PREFILLED_TOKENS and request.stream:
                if (
                    request.kv_transfer_params
                    and "prefilled_token" in request.kv_transfer_params
                    and "prefilled_texts" in request.kv_transfer_params
                ):
                    prefilled_text = request.kv_transfer_params["prefilled_texts"]
                    prefilled_tokens = request.kv_transfer_params["prefilled_token"]
                    generator = _wrap_with_prefilled(
                        generator, request_id_item, prompt_token_ids, prefilled_text, prefilled_tokens
                    )
            # ascend vllm adapt end

        generators.append(generator)

    result_generator = merge_async_iterators(*generators)

    model_name = self.models.model_name(lora_request)
    num_prompts = len(engine_inputs)

    # Streaming response
    tokenizer = self.renderer.tokenizer

    if request.stream:
        return self.completion_stream_generator(
            request,
            engine_inputs,
            result_generator,
            request_id,
            created_time,
            model_name,
            num_prompts=num_prompts,
            tokenizer=tokenizer,
            request_metadata=request_metadata,
        )

    # Non-streaming response
    final_res_batch: list[RequestOutput | None] = [None] * num_prompts
    try:
        async for i, res in result_generator:
            final_res_batch[i] = res

        for i, final_res in enumerate(final_res_batch):
            assert final_res is not None

            # The output should contain the input text
            # We did not pass it into vLLM engine to avoid being redundant
            # with the inputs token IDs
            if final_res.prompt is None:
                final_res.prompt = self._extract_prompt_text(engine_inputs[i])

        # ascend vllm adapt start
        if envs.REUSE_PREFILLED_TOKENS:
            # Prefill side: package first generated token for downstream Decode.
            for final_res in final_res_batch:
                if final_res and final_res.kv_transfer_params:
                    # Prefill side: package first generated token for D.
                    final_res.kv_transfer_params["prefilled_token"] = [final_res.outputs[0].token_ids[0]]
                    final_res.kv_transfer_params["stop_reasons"] = [output.stop_reason for output in final_res.outputs]
                    # p_then_d 提前派发: 首 token 的文本由 proxy 直接下发(见 _package_prefilled_text).
                    _package_prefilled_text(self.renderer.tokenizer, final_res.kv_transfer_params)
                elif request.kv_transfer_params and request.kv_transfer_params.get("prefilled_token"):
                    # Decode side (non-streaming): prepend the prefilled token
                    # reused from Prefill side into the output token_ids/text.
                    prefilled = request.kv_transfer_params.get("prefilled_token")
                    for output in final_res.outputs:
                        output.token_ids = list(prefilled) + list(output.token_ids)
                        output.text = request.kv_transfer_params.get("prefilled_texts", "") + output.text
        if envs.SKIP_DECODE_TOKENIZE:
            for final_res in final_res_batch:
                if final_res and final_res.kv_transfer_params:
                    final_res.kv_transfer_params["prompt_token_ids"] = final_res.prompt_token_ids
        # ascend vllm adapt end

        final_res_batch_checked = cast(list[RequestOutput], final_res_batch)

        response = self.request_output_to_completion_response(
            final_res_batch_checked,
            request,
            request_id,
            created_time,
            model_name,
            tokenizer,
            request_metadata,
        )
    except asyncio.CancelledError:
        return self.create_error_response("Client disconnected")

    # When user requests streaming but we don't stream, we still need to
    # return a streaming response with a single event.
    if request.stream:
        response_json = response.model_dump_json()

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return fake_stream_generator()

    return response


async def preprocess_completion(
    self,
    request: Any,
    prompt_input: str | list[str] | list[int] | list[list[int]] | None,
    prompt_embeds: bytes | list[bytes] | None,
    *,
    skip_mm_cache: bool = False,
) -> list[EngineInput]:
    """Copied from OpenAIServing._preprocess_completion."""
    # ascend vllm adapt start, decode skip tokenize
    kv_transfer_params = request.kv_transfer_params
    if kv_transfer_params and kv_transfer_params.get("prompt_token_ids"):
        return [TokensPrompt(prompt_token_ids=kv_transfer_params.get("prompt_token_ids"))]
    # ascend vllm adapt end
    prompts = list[SingletonPrompt | bytes]()
    if prompt_embeds is not None:  # embeds take higher priority
        prompts.extend(prompt_to_seq(prompt_embeds))
    if prompt_input is not None:
        prompts.extend(prompt_to_seq(prompt_input))
    return await self.preprocess_cmpl(request, prompts, skip_mm_cache=skip_mm_cache)


async def chat_completion_full_generator(
    self,
    request: ChatCompletionRequest,
    result_generator: AsyncIterator[RequestOutput],
    request_id: str,
    model_name: str,
    conversation: list[ConversationMessage],
    tokenizer: TokenizerLike,
    request_metadata: RequestResponseMetadata,
    parser: Parser | None = None,
) -> ErrorResponse | ChatCompletionResponse:
    created_time = int(time.time())
    final_res: RequestOutput | None = None

    try:
        async for res in result_generator:
            final_res = res
    except asyncio.CancelledError:
        return self.create_error_response("Client disconnected")

    if final_res is None:
        return self.create_error_response(
            "No output received from the engine.",
            err_type="InternalServerError",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    # ascend vllm adapt start, reuse prefill token and decode skip tokenizer
    if envs.REUSE_PREFILLED_TOKENS:
        if final_res.kv_transfer_params:
            # Prefill side: package first generated token for D.
            final_res.kv_transfer_params["prefilled_token"] = [final_res.outputs[0].token_ids[0]]
            # p_then_d 提前派发: 首 token 的文本由 proxy 直接下发(见 _package_prefilled_text).
            _package_prefilled_text(self.renderer.tokenizer, final_res.kv_transfer_params)
        elif request.kv_transfer_params and request.kv_transfer_params.get("prefilled_token"):
            prefilled = request.kv_transfer_params.get("prefilled_token")
            for output in final_res.outputs:
                output.token_ids = list(prefilled) + list(output.token_ids)
                output.text = request.kv_transfer_params.get("prefilled_texts", "") + output.text
    ## In prefill, the response will carry prompt_token_ids with kv_transfer_params
    if envs.SKIP_DECODE_TOKENIZE and final_res.kv_transfer_params:
        final_res.kv_transfer_params["prompt_token_ids"] = final_res.prompt_token_ids
    # ascend vllm adapt end

    choices: list[ChatCompletionResponseChoice] = []
    if self.tool_call_id_type == "kimi_k2":
        history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
    else:
        history_tool_call_cnt = 0

    role = self.get_chat_request_role(request)
    for output in final_res.outputs:
        # check for error finish reason and raise GenerationError
        # finish_reason='error' indicates a retryable request-level internal error
        self._raise_if_error(output.finish_reason, request_id)
        token_ids = output.token_ids
        out_logprobs = output.logprobs
        tool_call_info = None

        if request.logprobs and request.top_logprobs is not None:
            assert out_logprobs is not None, "Did not output logprobs"
            logprobs = self._create_chat_logprobs(
                token_ids=token_ids,
                top_logprobs=out_logprobs,
                num_output_top_logprobs=request.top_logprobs,
                tokenizer=tokenizer,
                return_as_token_id=request.return_tokens_as_token_ids,
            )
        else:
            logprobs = None

        if self.use_harmony:
            reasoning, content, _ = parse_chat_output(token_ids)
            if not request.include_reasoning:
                reasoning = None

            if self.tool_parser is not None:
                if tokenizer is None:
                    raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

                tool_parser = self.tool_parser(tokenizer, request.tools)
                # NOTE: We use token_ids for openai tool parser
                tool_call_info = tool_parser.extract_tool_calls(
                    "",
                    request=request,
                    token_ids=token_ids,  # type: ignore
                )
                content = tool_call_info.content
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_call_info.tool_calls,
                )
            else:
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content=content,
                )

            # Encode routed_experts for transport. JSON can't carry raw
            # bytes, so we write the ndarray as a ``.npy`` byte stream
            # and base64-encode it. ``pybase64`` is ~3x faster than the
            # stdlib ``base64`` on large payloads thanks to SIMD.
            routed_experts_b64 = None
            if output.routed_experts is not None:
                buf = io.BytesIO()
                np.save(buf, output.routed_experts)
                routed_experts_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason=(
                    "tool_calls"
                    if (tool_call_info is not None and tool_call_info.tools_called)
                    else output.finish_reason
                    if output.finish_reason
                    else "stop"
                ),
                stop_reason=output.stop_reason,
                token_ids=(as_list(output.token_ids) if request.return_token_ids else None),
                routed_experts=routed_experts_b64,
            )
            choices.append(choice_data)
            continue

        if parser is not None:
            reasoning, content, tool_calls = parser.parse(
                output.text,
                request,
                enable_auto_tools=self.enable_auto_tools,
            )
            if not request.include_reasoning:
                reasoning = None
        else:
            reasoning = None
            content = output.text
            tool_calls = []

        auto_tools_called = False
        if is_mistral_tokenizer(tokenizer):
            from vllm.tool_parsers.mistral_tool_parser import MistralToolCall

            tool_call_class: type[ToolCall] = MistralToolCall
        else:
            tool_call_class = ToolCall

        use_mistral_tool_parser = request._grammar_from_tool_parser
        if use_mistral_tool_parser:
            from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

            tool_call_items = MistralToolParser.build_non_streaming_tool_calls(tool_calls)
            if tool_call_items:
                auto_tools_called = request.tool_choice is None or request.tool_choice == "auto"
            message = ChatMessage(
                role=role,
                reasoning=reasoning,
                content=content,
                tool_calls=tool_call_items,
            )

        elif (not self.enable_auto_tools or not self.tool_parser) and (
            not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
            and request.tool_choice != "required"
        ):
            message = ChatMessage(role=role, reasoning=reasoning, content=content)

        elif request.tool_choice and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam:
            tool_call_class_items = []
            tool_calls = tool_calls or []
            for idx, tc in enumerate(tool_calls):
                # Use native ID if available (e.g., Kimi K2),
                # otherwise generate ID with correct id_type
                if tc.id:
                    tool_call_class_items.append(tool_call_class(id=tc.id, function=tc))
                else:
                    # Generate ID using the correct format (kimi_k2 or random),
                    # but leave it to the class if it's Mistral to preserve
                    # 9-char IDs
                    if is_mistral_tokenizer(tokenizer):
                        tool_call_class_items.append(tool_call_class(function=tc))
                    else:
                        generated_id = make_tool_call_id(
                            id_type=self.tool_call_id_type,
                            func_name=tc.name,
                            idx=history_tool_call_cnt,
                        )
                        tool_call_class_items.append(tool_call_class(id=generated_id, function=tc))
                history_tool_call_cnt += 1
            message = ChatMessage(
                role=role,
                reasoning=reasoning,
                content="",
                tool_calls=tool_call_class_items,
            )

        elif request.tool_choice and request.tool_choice == "required":
            tool_call_class_items = []
            tool_calls = tool_calls or []
            for idx, tool_call in enumerate(tool_calls):
                # Use native ID if available,
                # otherwise generate ID with correct id_type
                if tool_call.id:
                    tool_call_class_items.append(tool_call_class(id=tool_call.id, function=tool_call))
                else:
                    # Generate ID using the correct format (kimi_k2 or random),
                    # but leave it to the class if it's Mistral to preserve
                    # 9-char IDs
                    if is_mistral_tokenizer(tokenizer):
                        tool_call_class_items.append(tool_call_class(function=tool_call))
                    else:
                        generated_id = make_tool_call_id(
                            id_type=self.tool_call_id_type,
                            func_name=tool_call.name,
                            idx=history_tool_call_cnt,
                        )
                        tool_call_class_items.append(tool_call_class(id=generated_id, function=tool_call))
                history_tool_call_cnt += 1
            message = ChatMessage(
                role=role,
                content="",
                tool_calls=tool_call_class_items,
                reasoning=reasoning,
            )

        # if the request doesn't use tool choice
        # OR specifies to not use a tool
        elif not request.tool_choice or request.tool_choice == "none":
            message = ChatMessage(role=role, reasoning=reasoning, content=content)

        # handle when there are tools and tool choice is auto
        elif (
            request.tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
            and self.enable_auto_tools
            and self.tool_parser
        ):
            # In the OpenAI API the finish_reason is "tools_called"
            # if the tool choice is auto and the model produced a tool
            # call. The same is not true for named function calls
            auto_tools_called = tool_calls is not None and len(tool_calls) > 0
            if tool_calls:
                tool_call_items = []
                for idx, tc in enumerate(tool_calls):
                    # Use native ID if available (e.g., Kimi K2),
                    # otherwise generate ID with correct id_type
                    if tc.id:
                        tool_call_items.append(tool_call_class(id=tc.id, function=tc))
                    else:
                        # Generate ID using the correct format (kimi_k2 or random),
                        # but leave it to the class if it's Mistral to preserve
                        # 9-char IDs
                        if is_mistral_tokenizer(tokenizer):
                            tool_call_items.append(tool_call_class(function=tc))
                        else:
                            generated_id = make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tc.name,
                                idx=history_tool_call_cnt,
                            )
                            tool_call_items.append(tool_call_class(id=generated_id, function=tc))
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_call_items,
                )

            else:
                # FOR NOW make it a chat message; we will have to detect
                # the type to make it later.
                ret_content = content

                # try to use content return from tool parser first,
                # tool parser may do some modify for the content.
                if content and len(content) > 0:
                    ret_content = content
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content=ret_content,
                )

        # undetermined case that is still important to handle
        else:
            logger.error(
                "Error in chat_completion_full_generator - cannot determine"
                " if tools should be extracted. Returning a standard chat "
                "completion."
            )
            message = ChatMessage(role=role, reasoning=reasoning, content=content)
        # In OpenAI's API, when a tool is called, the finish_reason is:
        # "tool_calls" for "auto" or "required" tool calls,
        # and "stop" for named tool calls.
        is_finish_reason_tool_calls = auto_tools_called or (
            request.tool_choice and request.tool_choice == "required" and output.finish_reason == "stop"
        )

        # Encode routed_experts for transport. JSON can't carry raw
        # bytes, so we write the ndarray as a ``.npy`` byte stream
        # and base64-encode it. ``pybase64`` is ~3x faster than the
        # stdlib ``base64`` on large payloads thanks to SIMD.
        routed_experts_b64 = None
        if output.routed_experts is not None:
            buf = io.BytesIO()
            np.save(buf, output.routed_experts)
            routed_experts_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=message,
            logprobs=logprobs,
            finish_reason="tool_calls"
            if is_finish_reason_tool_calls
            else output.finish_reason
            if output.finish_reason
            else "stop",
            stop_reason=output.stop_reason,
            token_ids=(as_list(output.token_ids) if request.return_token_ids else None),
            routed_experts=routed_experts_b64,
        )
        choice_data = maybe_filter_parallel_tool_calls(choice_data, request)

        choices.append(choice_data)

    if request.echo:
        last_msg_content: str | list[dict[str, str]] = ""
        if conversation and "content" in conversation[-1] and conversation[-1].get("role") == role:
            last_msg_content = conversation[-1]["content"] or ""
        if isinstance(last_msg_content, list):
            last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

        for choice in choices:
            full_message = last_msg_content + (choice.message.content or "")
            choice.message.content = full_message

    assert final_res.prompt_token_ids is not None
    num_prompt_tokens = len(final_res.prompt_token_ids)
    if final_res.encoder_prompt_token_ids is not None:
        num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)

    # ascend vllm adapt start, reuse prefill token
    if envs.REUSE_PREFILLED_TOKENS and final_res.kv_transfer_params:
        final_res.kv_transfer_params["stop_reasons"] = [output.stop_reason for output in final_res.outputs]
    # ascend vllm adapt end

    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
        usage.prompt_tokens_details = PromptTokenUsageInfo(cached_tokens=final_res.num_cached_tokens)

    request_metadata.final_usage_info = usage

    # ``final_res.prompt`` is the rendered chat-templated prompt text
    prompt_text = final_res.prompt if request.return_prompt_text else None

    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
        system_fingerprint=self.system_fingerprint,
        prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
        prompt_token_ids=(final_res.prompt_token_ids if request.return_token_ids else None),
        prompt_text=prompt_text,
        kv_transfer_params=final_res.kv_transfer_params,
    )

    # Log complete response if output logging is enabled
    if self.enable_log_outputs and self.request_logger:
        for choice in choices:
            output_text = ""
            if choice.message.content:
                output_text = choice.message.content
            elif choice.message.tool_calls:
                # For tool calls, log the function name and arguments
                tool_call_descriptions = []
                for tc in choice.message.tool_calls:  # type: ignore
                    function_call: FunctionCall = tc.function  # type: ignore
                    tool_call_descriptions.append(f"{function_call.name}({function_call.arguments})")
                tool_calls_str = ", ".join(tool_call_descriptions)
                output_text = f"[tool_calls: {tool_calls_str}]"

            if output_text:
                # Get the corresponding output token IDs
                output_token_ids = None
                if choice.index < len(final_res.outputs):
                    output_token_ids = final_res.outputs[choice.index].token_ids

                self.request_logger.log_outputs(
                    request_id=request_id,
                    outputs=output_text,
                    output_token_ids=output_token_ids,
                    finish_reason=choice.finish_reason,
                    is_streaming=False,
                    delta=False,
                )

    return response


async def preprocess_chat(
    self,
    request: Any,
    messages: list[Any],
    default_template: str | None,
    default_template_content_format: ChatTemplateContentFormatOption,
    default_template_kwargs: dict[str, Any] | None,
    tool_dicts: list[dict[str, Any]] | None = None,
    tool_parser: type[ToolParser] | None = None,
    reasoning_parser: type[ReasoningParser] | None = None,
    *,
    skip_mm_cache: bool = False,
) -> tuple[list[ConversationMessage], list[EngineInput]]:
    """Copied from OpenAIServing._preprocess_chat."""
    renderer = self.renderer
    mm_config = self.model_config.multimodal_config
    default_template_kwargs = merge_kwargs(
        default_template_kwargs,
        dict(
            tools=tool_dicts,
            tokenize=(is_mistral_tokenizer(renderer.tokenizer) or self.model_config.enable_prompt_embeds),
        ),
    )
    tok_params = request.build_tok_params(self.model_config)
    chat_params = request.build_chat_params(default_template, default_template_content_format).with_defaults(
        default_template_kwargs,
        default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
        default_mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
    )
    # ascend vllm adapt start, decode skip tokenize
    kv_transfer_params = request.kv_transfer_params
    if kv_transfer_params and kv_transfer_params.get("prompt_token_ids"):
        conversation = []
        engine_input = TokensPrompt(prompt_token_ids=kv_transfer_params.get("prompt_token_ids"))
    # ascend vllm adapt end
    else:
        (conversation,), (engine_input,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v for k in ("mm_processor_kwargs", "cache_salt") if (v := getattr(request, k, None)) is not None
            },
            skip_mm_cache=skip_mm_cache,
        )
    if reasoning_parser is not None:
        tokenizer = renderer.get_tokenizer()
        request = reasoning_parser(
            tokenizer,
            model_config=self.model_config,
            chat_template_kwargs=chat_params.chat_template_kwargs,
        ).adjust_request(request=request)
    # tool parsing is done only if a tool_parser has been set and if
    # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
    # is set, we want to prevent parsing a tool_call hallucinated by the LLM
    #
    # Exception: Mistral grammar-capable tokenizers always call
    # adjust_request — even for tool_choice="none" — so that the grammar
    # factory can prevent special-token leakage.
    if tool_parser is not None:
        tool_choice = getattr(request, "tool_choice", "none")
        tokenizer = renderer.get_tokenizer()
        is_mistral_grammar_eligible = (
            is_mistral_tool_parser(tool_parser) and is_mistral_tokenizer(tokenizer) and tokenizer.supports_grammar
        )
        if tool_choice != "none" or is_mistral_grammar_eligible:
            if not isinstance(request, ChatCompletionRequest | ResponsesRequest):
                msg = (
                    "Tool usage is only supported "
                    "for Chat Completions API or Responses API requests, "
                    f"but got {type(request).__name__}"
                )
                raise NotImplementedError(msg)
            request = tool_parser(tokenizer, request.tools).adjust_request(request=request)
    return conversation, [engine_input]


async def _create_chat_completion(
    self,
    request: ChatCompletionRequest,
    raw_request: Request | None = None,
) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
    # Streaming response
    tokenizer = self.renderer.tokenizer
    assert tokenizer is not None
    chat_template_kwargs = self._effective_chat_template_kwargs(request)
    reasoning_parser: ReasoningParser | None = None
    if self.reasoning_parser_cls:
        reasoning_parser = self.reasoning_parser_cls(
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
        )
    result = await self.render_chat_request(request)
    if isinstance(result, ErrorResponse):
        return result

    conversation, engine_inputs = result

    request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"

    request_metadata = RequestResponseMetadata(request_id=request_id)
    if raw_request:
        raw_request.state.request_metadata = request_metadata

    lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

    model_name = self.models.model_name(lora_request)

    # Extract data_parallel_rank from header (router can inject it)
    data_parallel_rank = self._get_data_parallel_rank(raw_request)

    # Schedule the request and get the result generator.
    max_model_len = self.model_config.max_model_len
    generators: list[AsyncGenerator[RequestOutput, None]] = []
    for i, engine_input in enumerate(engine_inputs):
        prompt_token_ids = self._extract_prompt_components(engine_input).token_ids

        # If we are creating sub requests for multiple prompts, ensure that they
        # have unique request ids.
        sub_request_id = request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"

        max_tokens = get_max_tokens(
            max_model_len,
            request.max_completion_tokens if request.max_completion_tokens is not None else request.max_tokens,
            self._extract_prompt_len(engine_input),
            self.default_sampling_params,
            self.override_max_tokens,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )

        sampling_params: SamplingParams | BeamSearchParams
        if request.use_beam_search:
            sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
        else:
            sampling_params = request.to_sampling_params(
                max_tokens,
                self.default_sampling_params,
            )

        # ascend vllm adapt start, reuse prefill token
        # 提前派发(p_then_d)时这里取不到 token, 见 _p_then_d_token_reuse.
        if (
            envs.REUSE_PREFILLED_TOKENS
            and not _p_then_d_token_reuse(request)
            and request.kv_transfer_params
            and "prefilled_token" in request.kv_transfer_params
        ):
            new_tokens = tokenizer.convert_ids_to_tokens(request.kv_transfer_params["prefilled_token"][0])
            delta_text = tokenizer.convert_tokens_to_string([new_tokens])
            # If the decoded text ends with '�', it indicates an incomplete UTF-8 — abandon reuse.
            # If the prefilled side has already triggered a stop reason, fall back to normal generation.
            if (
                min(max_tokens, max_model_len - len(prompt_token_ids)) == 1
                or delta_text.endswith("�")
                or any(s is not None for s in request.kv_transfer_params["stop_reasons"])
                or request.kv_transfer_params["prefilled_token"][0] == tokenizer.eos_token_id
            ):
                request.kv_transfer_params.pop("prefilled_token", None)
            else:
                request.kv_transfer_params["prefilled_texts"] = delta_text
        # ascend vllm adapt end

        self._log_inputs(
            sub_request_id,
            engine_input,
            params=sampling_params,
            lora_request=lora_request,
        )

        trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)

        if isinstance(sampling_params, BeamSearchParams):
            generator = self.beam_search(
                prompt=engine_input,
                request_id=sub_request_id,
                params=sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
            )
        else:
            if not request.include_reasoning:
                reasoning_ended = True
            elif request._grammar_from_tool_parser:
                # The Mistral grammar already includes an optional
                # `think?` rule that handles both reasoning and
                # non-reasoning outputs.
                reasoning_ended = True
            elif reasoning_parser:
                reasoning_ended = reasoning_parser.is_reasoning_end(prompt_token_ids or [])
            else:
                reasoning_ended = None

            generator = self.engine_client.generate(
                engine_input,
                sampling_params,
                sub_request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
                data_parallel_rank=data_parallel_rank,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs={
                    "chat_template_kwargs": chat_template_kwargs,
                }
                if reasoning_parser
                else None,
            )

            # ascend vllm adapt start, reuse prefill token (streaming only;
            if envs.REUSE_PREFILLED_TOKENS and request.stream:
                if (
                    request.kv_transfer_params
                    and "prefilled_token" in request.kv_transfer_params
                    and "prefilled_texts" in request.kv_transfer_params
                ):
                    prefilled_text = request.kv_transfer_params["prefilled_texts"]
                    prefilled_tokens = request.kv_transfer_params["prefilled_token"]
                    generator = _wrap_with_prefilled(
                        generator, sub_request_id, prompt_token_ids, prefilled_text, prefilled_tokens
                    )
            # ascend vllm adapt end

        generators.append(generator)

    assert len(generators) == 1
    (result_generator,) = generators

    parser: Parser | None = None
    if self.parser_cls is not None:
        parser = self.parser_cls(
            tokenizer,
            request.tools,
            chat_template_kwargs=chat_template_kwargs,
        )

    if request.stream:
        return self.chat_completion_stream_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            reasoning_parser,
            chat_template_kwargs=chat_template_kwargs,
        )

    return await self.chat_completion_full_generator(
        request,
        result_generator,
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        parser,
    )


def _install_p_then_d_token_route(app) -> None:
    """p_then_d: 注册接收"首 token 补投"的端点(见 PrefilledTokenSpool).

    proxy 提前派发 D 时, 请求体里没有 prefilled_token; 拿到 P 的首 token 后它 POST
    到这里, 由本进程把记录原子写进 spool, 供同机 EngineCore 在 KV 接收完成的转型点
    认领(recompute_scheduler._update_waiting_for_remote_kv).
    """
    from fastapi import Body

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        PTD_TOKEN_SPOOL_ROOT,
        PrefilledTokenSpool,
    )

    @app.post("/v1/p_then_d_token")
    async def _p_then_d_token(raw_request: Request, payload: dict = Body(...)) -> dict:  # noqa: B008
        # X-Request-Id 与派发 D 时用的一致(见 proxy 的 deliver_prefiller_token).
        request_id = raw_request.headers.get("X-Request-Id") or ""
        kv_port = payload.get("kv_port")
        if not request_id or kv_port is None:
            return {"ok": False, "error": "X-Request-Id header and kv_port are required"}
        spool_dir = os.path.join(PTD_TOKEN_SPOOL_ROOT, str(kv_port))
        try:
            PrefilledTokenSpool.write_entry(
                spool_dir,
                request_id,
                {
                    "prefilled_token": payload.get("prefilled_token"),
                    "stop_reasons": payload.get("stop_reasons"),
                    "reuse": bool(payload.get("reuse", True)),
                },
            )
        except OSError as e:
            logger.error("p_then_d: failed to store token for request %s: %s", request_id, e)
            return {"ok": False, "error": str(e)}
        return {"ok": True}


def _build_app_with_p_then_d_token_route(*args, **kwargs):
    # api_server 内部是 `app = build_app(...)`(模块全局名解析), 所以替换模块属性即可生效.
    app = _original_build_app(*args, **kwargs)
    if envs.REUSE_PREFILLED_TOKENS:
        # 只有解码侧(复用 P 首 token)才需要这条路由.
        _install_p_then_d_token_route(app)
    return app


_original_build_app = _api_server_module.build_app  # noqa: F811
_api_server_module.build_app = _build_app_with_p_then_d_token_route

OpenAIServingChat._create_chat_completion = _create_chat_completion
OpenAIServingChat.chat_completion_full_generator = chat_completion_full_generator
OpenAIServingCompletion._create_completion = _create_completion
OpenAIServingRender.preprocess_completion = preprocess_completion
OpenAIServingRender.preprocess_chat = preprocess_chat
