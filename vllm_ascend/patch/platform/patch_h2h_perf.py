#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# H2H 时间线: 引擎前端(api_server)侧的请求生命周期打点。
#
# 只补两个时刻, 全部由 MC_TCP_PERF_LOG 门控(h2h_perf.PERF_ON, 关掉时零输出):
#   1) api_arrive      —— 请求到达 api_server(此刻 external request id 已铸造);
#   2) first_token_out —— 首 token 已写出前端(流式: 第一个 chunk; 非流式: 整包返回)。
# 这两个时刻 + worker 侧的 p_fwd_first/last(p_fwd_*)、connector 的传输段, 合起来
# 就是 TTFT 的端到端闭环; proxy 的同名打点(in/pf/meta/tok)作为跨跳对照。
#
# P/D 两个引擎跑同一份前端代码, 用 kv_transfer_config.is_kv_producer/is_kv_consumer
# 区分角色; 请求 id 形如 chatcmpl-<uuid>, 与 proxy 生成的 uuid 同源(proxy 在两次
# 派发上带同一个 X-Request-Id), 所以生成器可以直接按 id join。
#
# 说明: 这里**不**做任何耗时统计 —— 只记录绝对时刻(h2h_perf 自带 ts/wall), 差值是
# 时间线工具的事。

import functools
import json
import time
from typing import Any

from vllm.logger import init_logger

from vllm_ascend.distributed.kv_transfer.utils import h2h_perf

logger = init_logger(__name__)

# 包装标记: 模块被重复加载 / _apply() 被调用两次时避免重复包一层(会把每条记录
# 打两遍、首 chunk 判定也失效)。
_WRAP_FLAG = "_h2h_wrapped"


def _pick(args: tuple, kwargs: dict, name: str, index: int) -> Any:
    """按名取参, 取不到再按位置.

    index 是 **args 里的下标**(包装器签名是 (self, *args), 所以 index 不含 self):
    chat 的 request_id 在 args[2], request_metadata 在 args[6];
    completion 的 request_id 在 args[3], request_metadata 在 args[8]。
    """
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


def _already(original: Any) -> bool:
    return getattr(original, _WRAP_FLAG, False)


def _mark(fn: Any) -> Any:
    setattr(fn, _WRAP_FLAG, True)
    return fn


def _chunk_has_token(chunk: Any) -> bool:
    """SSE chunk 里是否带**实际输出内容**(role chunk 不算首 token).

    流式第一个 chunk 是 role chunk, 与首个 RequestOutput 同轮产出; 真正带文本的
    chunk 会紧随其后 —— 判据与 proxy 的 _first_token_event_to_forward 一致。
    """
    if not isinstance(chunk, str):
        return False
    body = chunk[6:].strip() if chunk.startswith("data: ") else chunk.strip()
    if not body or body == "[DONE]":
        return False
    try:
        obj = json.loads(body)
    except Exception:
        return False
    for choice in obj.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content"):
            return True
        if choice.get("text"):
            return True
    return False


def _role(serving: Any) -> str:
    """producer(P) / consumer(D) / ""(未启用 PD)。"""
    cfg = getattr(getattr(serving, "engine_client", None), "vllm_config", None)
    kt = getattr(cfg, "kv_transfer_config", None)
    if kt is None:
        return ""
    if kt.is_kv_producer:
        return "producer"
    if kt.is_kv_consumer:
        return "consumer"
    return ""


def _arrival_id(serving: Any, raw_request: Any) -> str | None:
    """到达时刻的 id: _create_* 内部会把它挂在 raw_request.state 上。"""
    state = getattr(raw_request, "state", None)
    md = getattr(state, "request_metadata", None)
    rid = getattr(md, "request_id", None)
    return rid if isinstance(rid, str) and rid else None


def _wrap_create(cls: Any, method_name: str) -> None:
    """包 *_create_*(chat/completion): 记"请求到达 api_server"。

    顺带覆盖**非流式**: `_create_*` 对流式返回 async generator, 非流式则已经把
    整包生成完再返回 —— 后者在这里补一条 first_token_out(stream=False), 因为
    vLLM 0.23 已无独立的 completion_full_generator 可包。
    """
    original = getattr(cls, method_name, None)
    if original is None or _already(original):
        return

    @functools.wraps(original)
    async def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = await original(self, *args, **kwargs)
        if h2h_perf.PERF_ON:
            raw_request = _pick(args, kwargs, "raw_request", 1)
            rid = _arrival_id(self, raw_request)
            if rid:
                # ts 用进入本方法前的时刻: await 期间包含了 chat template 渲染、
                # 分词与 mm 预处理, 那正是"到达 → 交给引擎"的前处理段。
                h2h_perf.emit("api_arrive", ts=t0, role=_role(self), req=rid)
                if not hasattr(result, "__anext__"):
                    h2h_perf.emit("first_token_out", role=_role(self), req=rid, stream=False)
            else:
                logger.debug("h2h perf: no external request id at %s; skip api_arrive", method_name)
        return result

    setattr(cls, method_name, _mark(wrapper))


def _wrap_stream(cls: Any, method_name: str, id_index: int, md_index: int) -> None:
    """包流式生成器: 记"首 chunk 写出"与"首个带内容的 token 写出"。

    包装器自己也是 async generator —— 必须原样 yield 每一个 chunk, 否则客户端
    会缺 chunk(这是本文件唯一有踩坑风险的地方, 见 P_THEN_D.md 的排障记录)。
    """
    original = getattr(cls, method_name, None)
    if original is None or _already(original):
        return

    @functools.wraps(original)
    async def wrapper(self, *args, **kwargs):
        rid = _pick(args, kwargs, "request_id", id_index)
        if not isinstance(rid, str) or not rid:
            md = _pick(args, kwargs, "request_metadata", md_index)
            rid = getattr(md, "request_id", None)
        first_chunk = True
        got_token = False
        async for chunk in original(self, *args, **kwargs):
            if h2h_perf.PERF_ON and isinstance(rid, str) and rid:
                if first_chunk:
                    first_chunk = False
                    # 首个出站 chunk 是 role chunk(与首个 RequestOutput 同轮产出).
                    h2h_perf.emit("first_chunk_out", role=_role(self), req=rid, stream=True)
                elif not got_token and _chunk_has_token(chunk):
                    got_token = True
                    # 真正带输出内容的第一个 chunk = "首 token 已写出前端".
                    h2h_perf.emit("first_token_out", role=_role(self), req=rid, stream=True)
            yield chunk

    setattr(cls, method_name, _mark(wrapper))


def _apply() -> None:
    if not h2h_perf.PERF_ON:
        # 关掉时连包装都不装: 前端热路径保持零改动.
        return
    wrapped: list[str] = []
    skipped: list[str] = []

    try:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    except Exception as e:  # pragma: no cover
        OpenAIServingChat = None  # type: ignore[assignment]
        logger.warning("h2h perf: OpenAIServingChat import failed: %s", e)

    if OpenAIServingChat is not None:
        # chat: (self, request, result_generator, request_id, model_name,
        #        conversation, tokenizer, request_metadata, ...)
        for fn, args_ in (
            (lambda: _wrap_create(OpenAIServingChat, "_create_chat_completion"), "chat.arrive"),
            (lambda: _wrap_stream(OpenAIServingChat, "chat_completion_stream_generator", 2, 6), "chat.stream"),
        ):
            try:
                fn()
                wrapped.append(args_)
            except Exception as e:  # pragma: no cover
                skipped.append(args_)
                logger.warning("h2h perf: wrap %s failed: %s", args_, e)

    try:
        from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    except Exception as e:  # pragma: no cover
        OpenAIServingCompletion = None  # type: ignore[assignment]
        logger.warning("h2h perf: OpenAIServingCompletion import failed: %s", e)

    if OpenAIServingCompletion is not None:
        # completion: (self, request, engine_inputs, result_generator, request_id,
        #              created_time, model_name, num_prompts, tokenizer, request_metadata)
        for fn, args_ in (
            (lambda: _wrap_create(OpenAIServingCompletion, "_create_completion"), "cmpl.arrive"),
            (lambda: _wrap_stream(OpenAIServingCompletion, "completion_stream_generator", 3, 8), "cmpl.stream"),
        ):
            try:
                fn()
                wrapped.append(args_)
            except Exception as e:  # pragma: no cover
                skipped.append(args_)
                logger.warning("h2h perf: wrap %s failed: %s", args_, e)

    # 包成功/失败都记一条: 版本漂移导致某些时刻缺失时, 日志里能直接看出来,
    # 而不是时间线上静默少两条线.
    h2h_perf.emit("perf_cfg", role="frontend", source="patch_h2h_perf", wrapped=wrapped, skipped=skipped)


_apply()
