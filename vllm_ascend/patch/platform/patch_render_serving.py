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
# 先 P 后 D: 把 reuse_prefilled_tokens 透传进请求的 kv_transfer_params。
#
# P 侧采样出的首 token 会随 KV 一起投给 D, D 把它当作 prompt 的最后一格重算
# KV 并继续 decode, 因此 D 不会再输出这个 token —— 客户端的第 1 个 token 来自
# P, 其余来自 D, 由 proxy 把两段响应拼成一个流。只有流式请求能这么拼, 所以
# 非流式一律置 False, 让 D 自己重算首 token。
#
# 上游 OpenAI 前端会在 preprocess_* 里按需重写 request.kv_transfer_params,
# 这里在原实现外面包一层, 在它写完之后再盖上 reuse_prefilled_tokens。
#

import functools
from typing import Any

from vllm.entrypoints.serve.render.serving import OpenAIServingRender

from vllm_ascend import envs


def _mark_reuse_prefilled_tokens(request: Any) -> None:
    if request is None or not request.kv_transfer_params:
        return
    request.kv_transfer_params["reuse_prefilled_tokens"] = envs.REUSE_PREFILLED_TOKENS
    if hasattr(request, "stream") and not request.stream:
        request.kv_transfer_params["reuse_prefilled_tokens"] = False


def _wrap_preprocess(method_name: str) -> None:
    original = getattr(OpenAIServingRender, method_name)

    @functools.wraps(original)
    async def wrapper(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        # preprocess_* 的第一个位置参数就是请求对象。
        _mark_reuse_prefilled_tokens(args[0] if args else None)
        return result

    setattr(OpenAIServingRender, method_name, wrapper)


for _method_name in ("preprocess_completion", "preprocess_cmpl", "preprocess_chat"):
    if hasattr(OpenAIServingRender, _method_name):
        _wrap_preprocess(_method_name)
