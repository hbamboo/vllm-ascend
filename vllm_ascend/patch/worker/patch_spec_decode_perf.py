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
# H2H 时间线: MTP / draft 提案(草案模型的投机提案)耗时打点, kind="mtp_fwd".
#
# 打点位置演进(别退回第一版):
#   1) 最初包 MTP 层自己的 forward -> 它是 torch.compile(fullgraph=True) 编译的,
#      编译区域内的 time.perf_counter 让 dynamo 报 "Attempted to call function
#      marked as skipped", **D 侧编图直接失败**;
#   2) 改包 vLLM 的 SpecDecodeBaseProposer.propose -> 一条记录都没有: ascend 的
#      model runner 调的是 **self.drafter._propose(...)**(带下划线, ascend 自己的
#      实现在 spec_decode/llm_base_proposer.py, 基类没有公开 propose), 公开 propose
#      根本不在这条路径上;
#   3) 现在包 AscendSpecDecodeBaseProposer._propose -> 每次 draft 一步一条,
#      且不会与公开 propose 双重计数(ascend 基类无公开 propose).
#
# 为什么打在这里而不是 MTP 层自己的 forward 里:
#   MTP 模型是 torch.compile(fullgraph=True) 编译的("Attempted to call function
#   marked as skipped"), 编译区域内任何宿主侧调用 —— time.perf_counter、日志、
#   json —— 都会让 dynamo 直接报错并导致 D 侧编图失败(2026-09-23 实测踩过)。
#   spec decode 的 propose() 是编译区之外的纯 Python: 每个 decode 步调用一次,
#   内部跑完 num_speculative_tokens 次 MTP 层前向 + 采样, 所以这里测到的
#   "提案耗时 ≈ 该步 MTP 前向总和"(记录里以 scope="propose" 标注口径)。
#
# 门控与全仓既有约定一致(见 vllm_ascend/distributed/kv_transfer/utils/h2h_perf.py):
#   - 记录与打印由 MC_TCP_PERF_LOG 管;
#   - 设备同步只在 ENABLE_PERF_DEBUG 下做, 否则耗时是异步提交时间(记录里 sync=false);
#   - 请求归属靠 h2h_perf.CTX(model runner 在 fwd 打点处 set_step_ctx)。

import functools
import time

import torch

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.kv_transfer.utils import h2h_perf

_WRAP_FLAG = "_h2h_wrapped"


def _wrap_propose(cls, method_name: str = "_propose") -> None:
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, _WRAP_FLAG, False):
        return

    @functools.wraps(original)
    def propose(self, *args, **kwargs):
        ctx = h2h_perf.CTX
        # ctx 不新鲜 => 本 rank 不是打点 rank(model runner 只在那里 set_step_ctx),
        # 或本步还没走前向: 不记录, 免得用空/陈旧请求集刷记录.
        # is_compiling() 是保险: 万一将来 propose 被卷进 torch.compile 区域, 本包装器
        # 自动跳过, 不会像"在 MTP forward 里打点"那样把编图搞崩.
        if not h2h_perf.PERF_ON or torch.compiler.is_compiling() or not ctx.fresh():
            return original(self, *args, **kwargs)
        t0 = time.perf_counter()
        out = original(self, *args, **kwargs)
        sync = bool(getattr(ascend_envs, "ENABLE_PERF_DEBUG", False))
        if sync:
            torch.npu.synchronize()
        h2h_perf.emit(
            "mtp_fwd",
            ts=t0,
            role=ctx.role,
            reqs=list(ctx.reqs),
            ms=h2h_perf.perf_ms(t0),
            sync=sync,
            scope="propose",
            proposer=type(self).__name__,
        )
        return out

    setattr(propose, _WRAP_FLAG, True)
    setattr(cls, method_name, propose)


def _apply() -> None:
    try:
        # ascend 的 _propose: eagle/mtp/draft_model 等所有跑 draft 模型前向的提案器
        # 都继承它(ngram/suffix 是纯 CPU 提案, 不跑 draft 前向, 也没必要打点).
        from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
    except Exception:  # pragma: no cover - 版本不匹配时静默跳过
        return
    _wrap_propose(AscendSpecDecodeBaseProposer, "_propose")


_apply()
