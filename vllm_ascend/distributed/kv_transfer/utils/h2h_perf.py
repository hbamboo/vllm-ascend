# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H2H / PD 逐层传输的统一性能打点(记录 + 打印都由 MC_TCP_PERF_LOG 管).

约定
----
- **唯一开关读取点**: 其它模块不要各自 `os.getenv("MC_TCP_PERF_LOG")`, 一律
  `from ... import h2h_perf` 后用 `h2h_perf.PERF_ON` / `h2h_perf.emit(...)`.
  关掉时零输出、零字符串拼接(emit 第一步就 return).
- **每条记录自带两个绝对量**: `ts` = `time.perf_counter()`(CLOCK_MONOTONIC, 同机
  跨进程可比, 与 P/D 引擎的现有打点同一坐标系) 与 `wall` = 本地墙钟 ISO8601 毫秒.
  时间线工具直接用这两个字段画图/悬停, 不再需要从 vLLM 日志前缀反推 epoch
  (该前缀只有秒精度且不含年份).
- **行格式**: `[h2h][perf] {json}` 一行一条; 解析器先按 JSON 解, 失败再退回历史
  文本格式(见 test_script/layerwise/analysis/parse_h2h_chain.py)。
- 引擎进程内走 vllm logger, 因此日志行仍带 `(Worker_TP0_EP0 pid=...)` 前缀,
  时间线靠它区分 rank; 没有 logger 的进程(如 proxy)用 `set_writer()` 换成
  `print(flush=True)` 直出(nohup 下 logger 可能缓冲)。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime

# 唯一开关读取点(import 时冻结, 与旧 _PERF_LOG 行为一致)。
PERF_ON: bool = os.getenv("MC_TCP_PERF_LOG", "0") == "1"

_TAG = "[h2h][perf]"

# 单调钟与墙钟的换算偏移, import 时采一次(误差 ~ 调用开销, 足够时间线展示)。
_MONO_AT_IMPORT = time.perf_counter()
_WALL_AT_IMPORT = time.time()


def perf_ms(t0: float) -> float:
    """perf_counter 差值的毫秒数(与旧 connector 的 `_perf_ms` 同语义)."""
    return (time.perf_counter() - t0) * 1e3


def win_str(a: float | None, b: float | None) -> str:
    """绝对窗口的可读形式: "-" 或 "起点,终点"(秒, CLOCK_MONOTONIC).

    只用于文本兜底与排障; 结构化记录里窗口直接是 `[a, b]` 两元组。
    """
    return "-" if a is None else f"{a:.6f},{b:.6f}"


def external_id(request_id: str) -> str:
    """引擎内部 req_id → external req id(与 proxy 的 uuid / 前端 id 同源).

    引擎内部 id = f"{external}-{random_uuid():.8}"(见 vllm/v1/engine/
    input_processor.py), 后缀固定 9 个字符("-" + 8 位 hex)。worker/scheduler 侧
    打点必须过这一层, 否则时间线上与前端/proxy 的记录 join 不上。
    """
    return request_id[:-9] if len(request_id) > 9 and request_id[-9] == "-" else request_id


def wall_str(ts: float | None = None) -> str:
    """CLOCK_MONOTONIC 秒 → 本地墙钟 ISO8601(毫秒); 缺省取当前时刻."""
    if ts is None:
        return datetime.now().isoformat(timespec="milliseconds")
    return datetime.fromtimestamp(_WALL_AT_IMPORT + (ts - _MONO_AT_IMPORT)).isoformat(timespec="milliseconds")


def _write(line: str) -> None:
    """默认输出通道: 引擎侧走 vllm logger(保留 worker/pid 前缀)."""
    try:
        from vllm.logger import logger

        logger.info("%s %s", _TAG, line)
    except Exception:  # pragma: no cover - 打点失败绝不影响主流程
        print(f"{_TAG} {line}", flush=True)


def set_writer(fn: Callable[[str], None]) -> None:
    """替换输出通道(非引擎进程用, 例如 proxy 的 `print(flush=True)`)."""
    global _write
    _write = fn


def emit(kind: str, ts: float | None = None, **fields) -> None:
    """输出一条性能记录.

    kind 决定时间线的泳道/事件类型; ts 缺省为当前 perf_counter。多余字段原样
    进 JSON, 因此新增打点不需要改解析器的正则。
    """
    if not PERF_ON:
        return
    rec: dict = {"kind": kind, "ts": time.perf_counter() if ts is None else ts}
    rec["wall"] = wall_str(rec["ts"])
    rec.update(fields)
    _write(json.dumps(rec, separators=(",", ":"), default=str))


class StepCtx:
    """最近一次 step 的上下文, 供"模块内 forward"读请求归属.

    典型用户是 MTP(draft) 层的 forward: 它由 drafter 在 model runner 的 step 内部
    调用, 拿不到 scheduler_output, 但打点需要知道"这次前向属于哪些请求"。model
    runner 在发 fwd 记录时顺手 set_step_ctx(), MTP 的包装器就按这份上下文归属。

    只在同一个进程/同一个 step 内使用(调用是同步的), 不做跨线程保护。
    """

    __slots__ = ("reqs", "role", "step_id", "ts")

    def __init__(self) -> None:
        self.reqs: list[str] = []
        self.role: str = ""
        self.step_id: int = 0
        self.ts: float = 0.0

    def set(self, reqs, role: str) -> None:
        self.reqs = list(reqs)
        self.role = role
        self.step_id += 1
        self.ts = time.perf_counter()

    def fresh(self, max_age_s: float = 5.0) -> bool:
        """上下文是否还算"当前 step"(防止跨 step 误挂到旧批次)."""
        return bool(self.reqs) and (time.perf_counter() - self.ts) <= max_age_s


CTX = StepCtx()


def set_step_ctx(reqs, role: str) -> None:
    """在 step 的 fwd 打点处调用(见 NPUModelRunner.execute_model)."""
    CTX.set(reqs, role)


class FirstLastTracker:
    """请求级"首次/末次被调度"跟踪: P 的前向两点与 D 的首次前向共用.

    判定方式与调度语义解耦, 只依赖"请求是否出现在本 step":
      - 请求首次出现在本 step       → 发 first(ts = 本 step 的 forward start)
      - 请求从本 step 起不再出现     → 发 last(ts = 上一次出现的 step 的 end)

    该判定成立的前提(已核实):
      - P 侧: proxy 对 P 的派发带 max_tokens=1, 请求 prefill + 采 1 个 token 后即
        完成, 故"最后一次出现"就是 prefill 结束(含首 token)那一步;
      - D 侧: 等 KV 的请求处于 WAITING_FOR_REMOTE_KVS, 此前不出现在任何
        SchedulerOutput, 故"首次出现"就是它的第一次 decode 前向。
    分块 prefill 天然正确(chunk0 是 first, 最后一个 chunk 那次是 last); 请求被
    preempt/abort 后重新调度会多出一对 last/first —— 在时间线上可见, 属预期。
    """

    def __init__(self, role: str) -> None:
        self.role = role
        # req_id -> {"last_end": 上一次出现的 step 结束时刻}
        self._live: dict[str, dict[str, float]] = {}

    def step(
        self,
        req_ids,
        ts_start: float,
        ts_end: float,
        first_kind: str,
        last_kind: str,
        **fields,
    ) -> None:
        """在每个 step 的前向边界调用一次(需 PERF_ON, 调用方自行判空)."""
        cur = set(req_ids)
        for rid in list(self._live):
            if rid in cur:
                continue
            st = self._live.pop(rid)
            emit(last_kind, ts=st["last_end"], role=self.role, req=rid, steps=int(st["n"]), **fields)
        if not cur:
            # 空批(引擎已空闲): 上一批里还挂着的请求就是"最后一次被调度", 否则
            # 本次运行最后一个请求的 last 永远发不出来(P 引擎跑完即空闲).
            for rid in list(self._live):
                st = self._live.pop(rid)
                emit(last_kind, ts=st["last_end"], role=self.role, req=rid, steps=int(st["n"]), **fields)
            return
        for rid in cur:
            st = self._live.get(rid)
            if st is None:
                self._live[rid] = {"last_end": ts_end, "n": 1.0}
                emit(first_kind, ts=ts_start, role=self.role, req=rid, **fields)
            else:
                st["last_end"] = ts_end
                st["n"] += 1
