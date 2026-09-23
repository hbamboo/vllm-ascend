#!/usr/bin/env python3
"""Parse P/D perf logs (layerwise H2H) into a transfer-chain timeline HTML.

Reads ../logs/p0.log (P / kv_producer) and ../logs/d0.log (D / kv_consumer).

Instrumentation. 新格式(推荐)= 结构化 JSON 行 `[h2h][perf] {"kind":...,"ts":...,"wall":...}`,
由 h2h_perf.emit 输出、统一受 MC_TCP_PERF_LOG 门控; 旧文本格式仍支持(见下)。两种
格式可以混读, 旧日志可原样重渲染。

  新格式 kinds: fwd(前向, 请求级 first/last 由 p_fwd_first/last、d_fwd_first 给出) /
  p_batch, d_batch, p_session, p_req, d_req(P/D 批次与请求级汇总) /
  p_wait_params(等 D 投递 block table), p_save_blocked(发送队列背压) /
  te_d2h, te_h2d(传输侧独立测量) / api_arrive, first_chunk_out, first_token_out(前端) /
  p_first_token_send, d_first_token_recv, d_recv_done, p_prefill_done / proxy.

  旧文本格式(gated, 见 run_p.sh / run_d.sh):
  1) Model execution (ENABLE_PERF_DEBUG=1, NPUModelRunner, TP0):
       reqs=<id>[,<id>...]Recv at <t>, forward start at <t>, ... Model forward
       time: <ms> ms            (P prefill forward / D decode step per batch)
  2) Transfer stages (MC_TCP_PERF_LOG=1, MooncakeLayerwiseConnector, TP0):
     P per layer-batch:  [mooncake][perf] P batch=N layers=[..] reqs=[..]
       wait= event= flush= write= layerdone= misc= total= ms t0=<abs>
       flush_win=<a,b> write_win=<a,b> layerdone_win=<a,b>
       flush = P D2H (NPU->CPU staging), write = H2H TCP 本体(不含 D 侧 H2D),
       layerdone = LAYER_DONE REQ-REP 往返(内含 D 侧 H2D, 仅用于配对/参照)
     D per LAYER_DONE:  [mooncake][perf] D batch layerdone reqs=[..] ranges=N
       h2d=<ms> total=<ms> ms t0=<abs>   H2D 窗口 = [t0, t0+h2d/1000]

Concurrency 支持: 并发请求会合批 — 每个 P/D perf 行携带批内全部 reqs; 同一批
次对其中每个请求共享同一份墙钟窗口. P 批 ↔ D h2d 通过 P 侧 layerdone_win
(REQ-REP) 包含 D h2d 的 t0 配对, 对并发/合批/PIPE 模式的重叠时序稳健.

同一请求可被多次推送(分块 prefill / 多 round): 按 layers 从 0 重新开始切分
round, 时间线按 round+层组展示. PIPE 模式(MC_TCP_PIPE_WRITER=1)下 write 由
独立写线程执行, 相邻批的 flush/write/h2d 有意重叠 — 图中如实显示, 不做串行
假设. 两引擎同机, perf_counter (CLOCK_MONOTONIC) 跨进程可比.
"""

import json
import re
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ---- 新格式: 结构化 JSON 行 (h2h_perf.emit) ----
# `[h2h][perf] {"kind":...,"ts":<CLOCK_MONOTONIC>,"wall":"<本地 ISO8601 毫秒>",...}`
# 解析优先走这里; 不匹配的行退回下面的历史文本正则, 因此旧日志仍可重渲染。
H2H_JSON = re.compile(r"\[h2h\]\[perf\] (\{.*\})\s*$")


def _json_line(ln: str) -> dict | None:
    m = H2H_JSON.search(ln)
    if not m:
        return None
    try:
        rec = json.loads(m.group(1))
    except Exception:
        return None
    return rec if isinstance(rec, dict) else None


def _norm_u(u: str) -> str:
    """事件里的请求 id 归一成 chatcmpl-<uuid>, 与 proxy uuid 对齐.

    要处理两种形态: proxy 记的是**裸 uuid**; 而少数事件(如 d_recv_done 来自
    connector 的 get_finished, 用的是 request_map)带的是**引擎内部 id** =
    external + "-<8hex>"(9 字符)。后缀判据与 h2h_perf.external_id 一致(-9 位是
    "-"), uuid4 的最后一组是 12 位 hex, 不会误伤。
    """
    u = (u or "").strip()
    if not u:
        return u
    if len(u) > 9 and u[-9] == "-":
        u = u[:-9]
    return u if u.startswith("chatcmpl-") else "chatcmpl-" + u


def _count_perf_lines(fn: Path) -> dict:
    """统计打点行数与"既非 JSON 也不能被任何旧正则消费"的行数.

    这是防止**解析静默漏行**的哨兵: 连接器新增/改名一个字段时, 正则匹配不上
    不会报错, 只会让那一批从时间线上消失 —— 计数对不上能立刻看出来。
    """
    st = {"total": 0, "json": 0, "legacy": 0, "unparsed": 0}
    if not fn.exists():
        return st
    for ln in fn.open(errors="ignore"):
        if "[h2h][perf]" not in ln and "[mooncake][perf]" not in ln:
            continue
        st["total"] += 1
        if _json_line(ln) is not None:
            st["json"] += 1
        elif PXY.search(ln) or P_BATCH.search(ln) or D_H2D.search(ln) or LEGACY_KNOWN.search(ln):
            st["legacy"] += 1
        else:
            st["unparsed"] += 1
    return st


# worker 来源的事件 kind: 只有这些才按 --p-worker/--d-worker 过滤.
# 前端打点(api_arrive/first_chunk_out/first_token_out)来自 api_server 进程,
# 日志前缀是 (APIServer pid=...), 拿 worker 名过滤会**整类丢掉**(实测踩过).
_WORKER_KINDS = {
    "fwd",
    "p_batch",
    "d_batch",
    "p_session",
    "p_req",
    "d_req",
    "te_d2h",
    "te_h2d",
    "p_wait_params",
    "p_save_blocked",
    "p_first_token_send",
    "d_first_token_recv",
    "d_recv_done",
    "p_prefill_done",
    "p_fwd_first",
    "p_fwd_last",
    "d_fwd_first",
    "d_fwd_last",
    "mtp_fwd",
}


def _parse_events(fn: Path, worker: str = "") -> tuple[dict[str, list[dict]], dict[str, dict], dict | None]:
    """收集结构化记录, 返回 (按请求分组的事件, proxy 记录, 绝对时间锚点)。

    - 事件按 external req id 分组, 时间线上再按 kind 落到泳道/标记;
    - clk = 任一 (ts, wall) 都有的记录 —— TMPL 用它把任意单调时刻换算成墙钟,
      不必再从日志前缀(秒级、无年份)反推 epoch。
    """
    by_req: dict[str, list[dict]] = {}
    proxy: dict[str, dict] = {}
    clk: dict | None = None
    if not fn.exists():
        return by_req, proxy, clk
    for ln in fn.open(errors="ignore"):
        rec = _json_line(ln)
        if rec is None:
            continue
        if worker and rec.get("kind") in _WORKER_KINDS and worker not in ln:
            continue
        ts = rec.get("ts")
        if clk is None and ts is not None and rec.get("wall"):
            clk = {"ts": float(ts), "wall": rec["wall"]}
        if rec.get("kind") == "proxy":
            proxy[_norm_u(rec.get("req", ""))] = {
                "in": rec.get("in", rec.get("ts")),
                "meta": rec.get("meta"),
                "pf": rec.get("pf"),
                "tok": rec.get("tok"),
                "done": rec.get("done"),
                "ts": ts,
                "wall": rec.get("wall"),
            }
            continue
        rids = rec.get("reqs")
        if rids is None:
            one = rec.get("req")
            rids = [one] if one else []
        for u in rids:
            by_req.setdefault(_norm_u(u), []).append(rec)
    return by_req, proxy, clk


# 结构化事件 → 请求级字段的归类(见 TMPL 的 drawB):
#   api_arrive(前端受理, role 区分 P/D) / first_token_out(首 token 写出前端)
#   p_fwd_first,p_fwd_last(P 前向两点) / d_fwd_first(D 开始计算)
#   p_wait_params,p_save_blocked(关键停顿) / te_d2h,te_h2d(传输段独立测量)
_REQ_EVENT_KEYS = {
    "api_arrive": "api_in",
    "first_chunk_out": "first_chunk",
    "first_token_out": "first_out",
    "p_fwd_first": "p_fwd_first",
    "p_fwd_last": "p_fwd_last",
    "d_fwd_first": "d_fwd_first",
    "d_fwd_last": "d_fwd_last",
    "p_prefill_done": "prefill_done",
    "p_first_token_send": "token_sent",
    "d_first_token_recv": "token_recv",
    "d_recv_done": "recv_done",
}


def _attach_events(req: dict, events: list[dict], rel) -> None:
    """把结构化事件挂到请求记录上(就地修改).

    t = 相对锚点 ms(画图/对齐用), wall = 该时刻的本地墙钟(悬停显示绝对时刻用)。
    """

    def stamp(rec: dict) -> dict:
        ts = rec.get("ts")
        return {"t": rel(float(ts)) if ts is not None else None, "wall": rec.get("wall"), "role": rec.get("role")}

    probes: list[dict] = []
    te: list[dict] = []
    mtp: list[dict] = []
    # 注意: 事件**按上报条数原样挂上**, 不做按请求的合并 —— 有些 kind 天然一个
    # 请求多条(如 d_recv_done: 同一 DP 组的每个 TP rank 各报一次自己的 KV 分片
    # 收齐; first_token_out: P/D 两个引擎各一条), 时间线上就该画成多条, 由使用者
    # 自己看两份上报的先后.
    for rec in events:
        kind = rec.get("kind")
        key = _REQ_EVENT_KEYS.get(kind)
        if key is not None:
            st = stamp(rec)
            if kind == "api_arrive":
                # 两侧各一条(P/D), 按 role 分开存
                req.setdefault("api_in", {})[rec.get("role", "")] = st
            else:
                req.setdefault(key, []).append(st)
        elif kind in ("p_wait_params", "p_save_blocked"):
            st = stamp(rec)
            st.update(kind=kind, ms=rec.get("ms"), mode=rec.get("mode") or rec.get("layer"))
            probes.append(st)
        elif kind == "mtp_fwd":
            # MTP(draft) 层前向: 每个投机步一条(num_speculative_tokens=3 → 每步 3 条),
            # 量大, 由 build() 按 decode 窗口裁剪后再挂到请求上.
            st = stamp(rec)
            st.update(
                ms=rec.get("ms"),
                step=rec.get("step"),
                sync=rec.get("sync"),
                scope=rec.get("scope"),
                proposer=rec.get("proposer"),
            )
            mtp.append(st)
        elif kind in ("te_d2h", "te_h2d"):
            st = stamp(rec)
            st.update(kind=kind, ms=rec.get("ms"), src=rec.get("src"))
            te.append(st)
    if mtp:
        # 全量留给 build() 裁剪(它才知道 decode 窗口); 这里先原样挂上.
        req["_mtp_all"] = mtp
    if probes:
        req["probes"] = probes
    if te:
        req["te"] = te


UUID = r"chatcmpl-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
REQS = re.compile(UUID)

# 模型执行行: reqs=id[,id...]Recv at t, forward start at t, ... Model forward time: X ms
# 引擎内部 id = uuid + 变长后缀(如 -a53f0c8e), 外部 ext id 只取 uuid 段.
_FULL = UUID + r"(?:[0-9a-f-]*[0-9a-f])?"
MODEL = re.compile(
    rf"reqs=((?:{_FULL},)*{_FULL})Recv at ([\d.]+), forward start at ([\d.]+),"
    rf" .*?Model forward time: ([\d.]+) ms"
)
# P 批行 (bytes= 为可选的传输 payload 字节数, 旧日志无该字段)
P_BATCH = re.compile(
    r"\[mooncake\]\[perf\] P batch=(\d+) layers=\[([^\]]*)\] reqs=\[(.*?)\]"
    r" wait=([\d.]+) event=([\d.]+) flush=([\d.]+) write=([\d.]+) layerdone=([\d.]+)"
    r" misc=([\d.]+) total=([\d.]+) ms t0=([\d.]+)"
    r" flush_win=(\S+) write_win=(\S+) layerdone_win=(\S+)"
    r"(?: bytes=(\d+))?"
)
# D h2d 行
D_H2D = re.compile(
    r"\[mooncake\]\[perf\] D batch layerdone reqs=\[(.*?)\] ranges=(\d+)"
    r" h2d=([\d.]+) total=([\d.]+) ms t0=([\d.]+)"
)
WALL = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d)")
# 旧文本格式里"可识别但不进时间线"的行(请求级汇总): 计数时算已消费, 避免哨兵
# 误报. 新格式下这两类都有结构化 kind(p_req / d_req).
LEGACY_KNOWN = re.compile(r"\[mooncake\]\[perf\] [PD] req=")


def _ids(s: str) -> list[str]:
    return REQS.findall(s)


def _win(s: str) -> tuple[float, float] | None:
    m = re.match(r"^([\d.]+),([\d.]+)$", s)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _parse_model_rows(fn: Path, worker: str = "Worker_TP0") -> tuple[list[dict], str | None]:
    """模型执行行 (P 前向或 D decode), reqs 支持逗号分隔的多请求.

    worker: 进程名前缀过滤. 旧配置(P/D 都是 TP4, dp=1)下 worker 名是 `Worker_TP0`;
    decode 侧开了 dp 时名字变成 `Worker_DP1_TP0`, 由调用方传入.
    """
    rows, wall, seen = [], None, set()
    for ln in fn.open(errors="ignore"):
        if worker and worker not in ln:
            continue
        rec = _json_line(ln)
        if rec is not None:
            # 新格式: kind=fwd 一条即该批的前向边界(P/D 同一条结构).
            if rec.get("kind") == "fwd":
                fs, fe = rec.get("fs"), rec.get("fe")
                if fs is None or fe is None:
                    continue
                recv = rec.get("recv")
                row = {
                    "u": [_norm_u(u) for u in rec.get("reqs", [])],
                    "recv": float(recv) if recv is not None else float(fs),
                    "fs": float(fs),
                    "fe": float(fe),
                    "fwd": float(rec.get("fwd_ms") or (float(fe) - float(fs)) * 1e3),
                }
                # TP 内各 rank 记的是同一个 step(时间戳几乎一致): 宽匹配(--d-worker 传
                # Worker_DP 这类前缀)时按 (reqs, fs) 去重, 否则 decode 步数会被算成两倍、
                # 步周期中位数被压成 ~0.
                key = (",".join(row["u"]), round(row["fs"], 4))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
            continue
        w = WALL.search(ln)
        wall = wall or (w.group(1) if w else None)
        m = MODEL.search(ln)
        if m:
            rows.append(
                {
                    "u": _ids(m.group(1)),
                    "recv": float(m.group(2)),
                    "fs": float(m.group(3)),
                    "fwd": float(m.group(4)),
                }
            )
    return rows, wall


def _parse_connector_rows(fn: Path, worker: str = "Worker_TP0") -> tuple[list[dict], list[dict], str | None]:
    """connector perf 行: P batch 行 + D h2d 行 (同一文件里只会出现一种).

    worker 同 _parse_model_rows: decode 侧开 dp 时传 `Worker_DP1_TP0`(取其中一个
    dp 组即可, 各组时间接近; 不传的话 dp>1 的两组会同时命中, 批次被算成双份).
    """
    pb, dh, wall, _seen_d = [], [], None, set()
    for ln in fn.open(errors="ignore"):
        if worker not in ln:
            continue
        rec = _json_line(ln)
        if rec is not None:
            k = rec.get("kind")
            _wm = re.search(r"\((Worker_[A-Za-z0-9_]+)", ln)  # 前缀里的 worker 名(D 侧每 TP rank 一条)
            _worker = _wm.group(1) if _wm else ""
            if k == "p_batch":
                pb.append(
                    {
                        "u": [_norm_u(u) for u in rec.get("reqs", [])],
                        "layers": [int(x) for x in rec.get("layers", [])],
                        "wait": rec.get("wait"),
                        "event": rec.get("event"),
                        "flush": rec.get("flush"),
                        "write": rec.get("write"),
                        "layerdone": rec.get("layerdone"),
                        "misc": rec.get("misc"),
                        "total": rec.get("total"),
                        "t0": rec.get("ts"),
                        "fw": rec.get("flush_win"),
                        "ww": rec.get("write_win"),
                        "lw": rec.get("layerdone_win"),
                        "bytes": rec.get("bytes", rec.get("nbytes")),
                        "wall": rec.get("wall"),
                        "sender": rec.get("sender"),
                    }
                )
            elif k == "d_batch":
                # 同一批会被对端的多个 TP rank 各收一次(各收自己的分片, t0 几乎一致):
                # 按 (reqs, t0) 去重, 让"一个 P 批 ↔ 一条 D 行"的配对在 P/D 不等分时
                # 依然成立; 配对阶段再按窗口做多对一兜底.
                _k = (",".join(sorted(_norm_u(x) for x in rec.get("reqs", []))), round(float(rec.get("ts", 0.0)), 4))
                if _k in _seen_d:
                    continue
                _seen_d.add(_k)
                # 新格式把 pull(H2H read) 与 h2d_only 拆开了: 这里 h2d 取纯 H2D,
                # 与旧文本行的 h2d 语义(曾经含 pull)对齐; pull/H2H 由独立字段带出.
                dh.append(
                    {
                        "u": [_norm_u(u) for u in rec.get("reqs", [])],
                        "ranges": rec.get("ranges"),
                        "h2d": rec.get("h2d_only"),
                        "h2d_all": rec.get("h2d"),
                        "pull": rec.get("pull"),
                        "pull_t0": rec.get("pull_t0"),
                        "h2d_t0": rec.get("h2d_t0"),
                        "ack_ts": rec.get("ack_ts"),
                        "worker": _worker,
                        "sender": rec.get("sender"),
                        "total": rec.get("total"),
                        "t0": rec.get("ts"),
                        "wall": rec.get("wall"),
                    }
                )
            continue
        w = WALL.search(ln)
        wall = wall or (w.group(1) if w else None)
        m = P_BATCH.search(ln)
        if m:
            pb.append(
                {
                    "u": _ids(m.group(3)),
                    "layers": [int(x) for x in re.findall(r"\d+", m.group(2))],
                    "wait": float(m.group(4)),
                    "event": float(m.group(5)),
                    "flush": float(m.group(6)),
                    "write": float(m.group(7)),
                    "layerdone": float(m.group(8)),
                    "misc": float(m.group(9)),
                    "total": float(m.group(10)),
                    "t0": float(m.group(11)),
                    "fw": _win(m.group(12)),
                    "ww": _win(m.group(13)),
                    "lw": _win(m.group(14)),
                    "bytes": int(m.group(15)) if m.group(15) else None,
                }
            )
            continue
        m = D_H2D.search(ln)
        if m:
            dh.append(
                {
                    "u": _ids(m.group(1)),
                    "ranges": int(m.group(2)),
                    "h2d": float(m.group(3)),
                    "total": float(m.group(4)),
                    "t0": float(m.group(5)),
                }
            )
    return pb, dh, wall


# proxy 端到端打点行 (load_balance_proxy_layerwise_server_example.py):
# [h2h][perf] proxy req=<裸 uuid> in=.. meta=.. pf=.. tok=.. done=..
PXY = re.compile(
    r"\[h2h\]\[perf\] proxy req=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r" in=(\S+) meta=(\S+) pf=(\S+) tok=(\S+) done=(\S+)"
)


def _parse_proxy(fn: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not fn.exists():
        return out
    for ln in fn.open(errors="ignore"):
        rec = _json_line(ln)
        if rec is not None:
            if rec.get("kind") != "proxy":
                continue
            vals = {}
            if rec.get("in") is None and rec.get("ts") is not None:
                rec["in"] = rec["ts"]  # 兼容未带 in 字段的历史记录
            for k in ("in", "meta", "pf", "tok", "done"):
                v = rec.get(k)
                try:
                    vals[k] = float(v) if v is not None else None
                except (TypeError, ValueError):
                    vals[k] = None
            out[_norm_u(rec.get("req", ""))] = vals
            continue
        m = PXY.search(ln)
        if m:
            u = "chatcmpl-" + m.group(1)  # proxy 的 uuid 无前缀, 与引擎 ext id 对齐
            vals = {}
            for k, v in zip(("in", "meta", "pf", "tok", "done"), m.groups()[1:]):
                try:
                    vals[k] = float(v)
                except ValueError:
                    vals[k] = None
            out[u] = vals
    return out


def build(logs: Path, p_worker: str = "Worker_TP0", d_worker: str = "Worker_TP0") -> dict:
    """p_worker/d_worker: 两侧取哪条 worker 的日志行(见 _parse_model_rows 的说明).

    decode 侧 dp>1 时必须指定 d_worker, 否则同一批次会被两个 dp 组各算一次.
    """
    pfwds, pwall = _parse_model_rows(logs / "p0.log", p_worker)
    ppb, _, _ = _parse_connector_rows(logs / "p0.log", p_worker)
    ddecs, dwell = _parse_model_rows(logs / "d0.log", d_worker)
    _, ddh, _ = _parse_connector_rows(logs / "d0.log", d_worker)
    pxy_all = _parse_proxy(logs / "proxy.log")
    # 结构化事件(新格式): 前端时刻 / P 前向两点 / D 开始计算 / 停顿探针 / 传输侧测量
    p_events, _, p_clk = _parse_events(logs / "p0.log", p_worker)
    d_events, _, d_clk = _parse_events(logs / "d0.log", d_worker)
    clk = p_clk or d_clk
    ts_all: list[float] = [b["t0"] for b in ppb if b.get("t0")] + [d["t0"] for d in ddh if d.get("t0")]
    for _src in (p_events, d_events):
        for _evs in _src.values():
            for _rec in _evs:
                if _rec.get("ts") is not None:
                    ts_all.append(float(_rec["ts"]))
    span = (min(ts_all), max(ts_all)) if ts_all else None
    parse_stat = {
        "p": _count_perf_lines(logs / "p0.log"),
        "d": _count_perf_lines(logs / "d0.log"),
        "proxy": _count_perf_lines(logs / "proxy.log"),
    }

    # ---- P 批 ↔ D h2d 配对 ----
    # 同机: P 的 layerdone REQ-REP 窗口(lw)直接包含 D 的 t0(同一 CLOCK_MONOTONIC).
    # 跨机: 时钟原点不同, 先用逐批 LAYER_DONE 交换估计时钟偏移 —— 每批给一个
    # (lw 中点, D t0) 样本, 取中位数即偏移; 校准后 D 时间折算到 P 时钟再配对.
    cross = {"mode": "same-host", "offset_ms": 0.0}
    pb_sorted = sorted(ppb, key=lambda b: b["t0"])
    d_sorted = sorted(ddh, key=lambda d: d["t0"])
    paired = 0
    for b in pb_sorted:
        b["dh"] = None
    if len(pb_sorted) == len(d_sorted) and pb_sorted:
        # 同机快速路径: 窗口包含计数
        di, nd = 0, len(d_sorted)
        ok_same = 0
        for b in pb_sorted:
            lw = b["lw"]
            if lw is None:
                continue
            while di < nd and d_sorted[di]["t0"] < lw[0] - 0.005:
                di += 1
            if di < nd and d_sorted[di]["t0"] <= lw[1] + 0.005:
                ok_same += 1
                di += 1
        if ok_same < max(4, len(pb_sorted) * 0.8):
            # 跨机: 顺序 zip 求偏移样本(lw 中点 vs D t0), 中位数 = 时钟偏移
            offs = [d["t0"] - (b["lw"][0] + b["lw"][1]) / 2 for b, d in zip(pb_sorted, d_sorted) if b["lw"]]
            if offs:
                off = statistics.median(offs)
                cross = {"mode": "cross-node", "offset_ms": round(off * 1000, 3)}
                # D 侧所有绝对时间折算到 P 时钟
                for d in d_sorted:
                    d["t0"] -= off
                for r in ddecs:
                    r["recv"] -= off
                    r["fs"] -= off
    # 窗口包含配对(校准后跨机同样适用).
    # 一个 P 批可能对上**多条** D 行: P/D 不等分(P tp4 → D tp2×dp2)时, 同一批数据
    # 会被对端每个 TP rank 各收一次(去重后仍可能剩多条 —— 收的是不同分片). 这里把
    # 窗口内的行聚合: 窗口取并集, 耗时取各行的最大值(各 rank 并行, 取最大即该批的
    # 墙钟下界), 并记录条数与 range 总数.
    # 逐批取窗口(用 bisect, 不用前向游标 —— pipe 写线程让相邻批的 layerdone 窗口
    # 重叠, 游标会跳过后面批需要的行, 表现为莫名其妙的"缺失").
    import bisect

    dt0 = [d["t0"] for d in d_sorted]
    for b in pb_sorted:
        lw = b["lw"]
        if lw is None:
            continue
        i = bisect.bisect_left(dt0, lw[0] - 0.005)
        j = bisect.bisect_right(dt0, lw[1] + 0.005)
        matches = d_sorted[i:j]
        if not matches:
            continue
        # 同窗口内若有多个请求的批(合批/流水重叠), 只留与本批 reqs 相交的行.
        # 注意: **不要**再按 sender(P 侧档位) 过滤 —— 实测每个 D rank 的数据来自
        # **不同的 P rank**(同一批 layers 由多个 P rank 各发一份给各自负责的 D rank),
        # 按 sender 过滤会把"另一个 TP rank 的行"全滤掉, DP 泳道里就只剩一个 rank 了.
        # 精度靠下面 rows/ref 用的 ±0.5ms 窗口保证(同机单调钟).
        reqs_b = set(b["u"])
        hit = [d for d in matches if reqs_b & set(d["u"])]
        matches = hit or matches

        # 新格式的字段语义(见 d_batch 解析): d["h2d"] = 纯 H2D(实测),
        # d["h2d_all"] = recv0→H2D 完成的旧口径(含 pull read); 旧文本日志只有后者,
        # 那时"纯 H2D"无从区分, 只给 dur.
        def _incl(d):
            return d["h2d_all"] if d.get("h2d_all") is not None else (d.get("h2d") or 0.0)

        def _pure(d):
            return d.get("h2d") if d.get("h2d_all") is not None else None

        s = min(d["t0"] for d in matches)
        e = max(d["t0"] + _incl(d) / 1000 for d in matches)
        dh = {
            "s": s,
            "e": e,
            "dur": max(_incl(d) for d in matches),
            "ranges": sum(d["ranges"] or 0 for d in matches),
            "n": len(matches),
        }
        pures = [x for x in (_pure(d) for d in matches) if x is not None]
        if pures:
            dh["h2d_only"] = max(pures)
        if len(matches) > 1:
            dh["multi"] = True  # 图上标注"该批被对端 N 个 rank 各收一次"
        pulls = [d.get("pull") for d in matches if d.get("pull") is not None]
        pull_t0s = [d.get("pull_t0") for d in matches if d.get("pull_t0") is not None]
        h2d_t0s = [d.get("h2d_t0") for d in matches if d.get("h2d_t0") is not None]
        # 每 rank 明细(按 DP 分泳道 / 泳道内按 TP 分行用): 落在本批窗口内且带回 ACK 的行.
        # 要求**整段往返都在本批窗口内**(t0 与 ACK 都落在 [lw0,lw1]): 只按 t0 判会把
        # 下一批的往返也收进来(t0 落在窗口重叠区、ACK 在窗口外), 画出来是一条冲出窗口
        # 的长斜线, 且和下一批自己的折线重叠(实测: 每个 TP 泳道右侧那条长线).
        dh["rows"] = sorted(
            (
                d
                for d in matches
                if d.get("ack_ts") is not None
                and lw[0] - 0.0005 <= d["t0"] <= lw[1] + 0.0005
                and lw[0] - 0.0005 <= d["ack_ts"] <= lw[1] + 0.0005
            ),
            key=lambda x: x["t0"],
        )
        # LAYER_DONE 折线的参考行: 必须是**同一个 rank 的收/回时刻**。P 的
        # layerdone_win 是多 peer 并集、D 行又是多 rank 各一条, 若取 min(t0) 配
        # max(ack) 会把 A rank 的收与 B rank 的回拼在一起, 折线上出现"D 回 ACK 早于
        # 收到"之类物理不可能的段(实测 -19ms 级)。这里只挑一行(取最早收到的那条,
        # 它对应 P 窗口起点那次发送)。
        # 容差收紧到 0.5ms(同机单调钟, D 的收到必然晚于 P 的发, 只有网络/排队差):
        # 若沿用配对窗口的 ±5ms, 会把**上一批**的 D 行(pipe 重叠时 t0 落在本窗口内)
        # 当成参考行, 折线第一阶段被夹成 0(实测 13/143 条).
        ref = None
        for d in matches:
            if d.get("ack_ts") is None:
                continue
            if not (lw[0] - 0.0005 <= d["t0"] <= lw[1] + 0.0005):
                continue
            if d["ack_ts"] > lw[1] + 0.0005:
                continue
            if ref is None or d["t0"] < ref["t0"]:
                ref = d
        if ref is not None:
            # 参考行的完整一套(单 rank): 时间线画图/tooltip 以它为准, 聚合值(max/min)
            # 只作对照 —— 混用会出现"②=37.83ms 而 H2H 8.09 + H2D 36.02 对不上"。
            dh["flow_s"] = ref["t0"]
            dh["flow_ack"] = ref["ack_ts"]
            dh["flow_rank_rows"] = len(matches)
            dh["ref"] = {
                "pull": ref.get("pull"),
                "pull_t0": ref.get("pull_t0"),
                "h2d_only": ref.get("h2d"),
                "h2d_t0": ref.get("h2d_t0"),
                "h2d_all": ref.get("h2d_all"),
                "t0": ref["t0"],
            }
        acks = [ref["ack_ts"]] if ref is not None else []
        if pulls:
            dh["pull"] = max(pulls)
        if pull_t0s:
            dh["pull_t0"] = min(pull_t0s)
        if h2d_t0s:
            dh["h2d_t0"] = min(h2d_t0s)
        if acks:
            dh["ack_ts"] = max(acks)
        if matches[0].get("h2d_all") is not None:
            # 保留含 pull 的旧口径(与历史日志可比), 供 tooltip/汇总参考
            dh["h2d_all"] = max(_incl(d) for d in matches)
        b["dh"] = dh
        paired += 1

    # ---- 请求聚合 ----
    by_fwd: dict[str, list] = {}
    for r in pfwds:
        for u in r["u"]:
            by_fwd.setdefault(u, []).append(r)
    by_dec: dict[str, list] = {}
    for r in ddecs:
        for u in r["u"]:
            by_dec.setdefault(u, []).append(r)
    by_b: dict[str, list] = {}
    for b in ppb:
        for u in b["u"]:
            by_b.setdefault(u, []).append(b)

    all_u = set(by_b) | set(by_fwd)
    reqs = []
    for u in sorted(all_u, key=lambda k: min((x["t0"] for x in by_b.get(k, [])), default=1e18)):
        bs = sorted(by_b.get(u, []), key=lambda b: b["t0"])
        fs = sorted(by_fwd.get(u, []), key=lambda r: r["recv"])
        ds = sorted(by_dec.get(u, []), key=lambda r: r["recv"])
        if not bs and not fs:
            continue
        # round 切分: layers 重新从 0 开始即为新的一轮推送
        for b in bs:
            b["rnd"] = 0
        if bs:
            bs[0]["rnd"] = 0
            for i in range(1, len(bs)):
                bs[i]["rnd"] = bs[i - 1]["rnd"] + (1 if bs[i]["layers"][0] <= bs[i - 1]["layers"][0] else 0)
        # 锚点: P 首次前向 recv, 否则首批 flush 起点; 旧版日志无 flush/模型
        # 打点(flush_win=-)时回退首批 write 窗口起点, 再无则取批 t0; 单位秒
        if fs:
            anchor = fs[0]["recv"]
        else:
            fws = [b["fw"][0] for b in bs if b["fw"]]
            wws = [b["ww"][0] for b in bs if b["ww"]]
            anchor = min(fws) if fws else (min(wws) if wws else bs[0]["t0"])
        # 全部转相对锚点 ms
        rel = lambda t, _a=anchor: (t - _a) * 1000
        d1 = None
        if ds:
            d1 = {"recv": rel(ds[0]["recv"]), "fs": rel(ds[0]["fs"]), "fe": rel(ds[0]["fs"]) + ds[0]["fwd"]}
        reqs.append(
            {
                "u": u,
                "pf": [
                    {"recv": rel(r["recv"]), "fs": rel(r["fs"]), "fe": rel(r["fs"]) + r["fwd"], "fwd": r["fwd"]}
                    for r in fs
                ],
                "groups": [
                    {
                        "lay": b["layers"],
                        "rnd": b["rnd"],
                        "fw": [rel(b["fw"][0]), rel(b["fw"][1])] if b["fw"] else None,
                        "ww": [rel(b["ww"][0]), rel(b["ww"][1])] if b["ww"] else None,
                        "lw": [rel(b["lw"][0]), rel(b["lw"][1])] if b["lw"] else None,
                        "dh": None
                        if not b["dh"]
                        else {
                            "s": rel(b["dh"]["s"]),
                            "e": rel(b["dh"]["e"]),
                            "dur": b["dh"]["dur"],
                            "rng": b["dh"]["ranges"],
                            # P/D 不等分时同一批被对端 N 个 rank 各收一次(n>1), 图上标注
                            "n": b["dh"].get("n"),
                            "multi": b["dh"].get("multi"),
                            # 新格式专有: pull=H2H read 耗时(属 H2H 而非 H2D), pt=其绝对
                            # 起点(相对 ms); h2d_only=纯 H2D 耗时; hd_t=H2D 绝对起点.
                            "pull": b["dh"].get("pull"),
                            "pt": rel(b["dh"]["pull_t0"]) if b["dh"].get("pull_t0") is not None else None,
                            "h2d_only": b["dh"].get("h2d_only"),
                            "hd_t": rel(b["dh"]["h2d_t0"]) if b["dh"].get("h2d_t0") is not None else None,
                            # LAYER_DONE 往返折线的第 3 个点: D 回 ACK 的时刻(第 1/2/4
                            # 个点分别是 lw[0](P 发) / dh.s(D 收) / lw[1](P 收到 ACK)).
                            "ack": rel(b["dh"]["ack_ts"]) if b["dh"].get("ack_ts") is not None else None,
                            # 折线专用: 单个 rank 的收/回时刻(见配对处), 以及该批命中的
                            # D 行数(>1 说明该批被多个 rank 接收, 折线只画最早那条).
                            "fs": rel(b["dh"]["flow_s"]) if b["dh"].get("flow_s") is not None else None,
                            "fa": rel(b["dh"]["flow_ack"]) if b["dh"].get("flow_ack") is not None else None,
                            "frows": b["dh"].get("flow_rank_rows"),
                            # 每 rank 明细: w=接收方 worker, snd=发送方端口尾号, t0/ack 相对 ms
                            "rows": [
                                {
                                    "w": _r.get("worker", ""),
                                    "snd": str(_r.get("sender", ""))[-5:],
                                    "t0": rel(_r["t0"]),
                                    "ack": rel(_r["ack_ts"]),
                                    "pull": _r.get("pull"),
                                    "h2d_only": _r.get("h2d"),
                                }
                                for _r in (b["dh"].get("rows") or [])
                            ],
                            # 单 rank 参考行(与折线同一行): H2H/H2D 的条与 tooltip 都用它,
                            # 保证与 LAYER_DONE 折线的 ② 段对得上.
                            "rf": None
                            if not b["dh"].get("ref")
                            else {
                                "pull": b["dh"]["ref"].get("pull"),
                                "pt": rel(b["dh"]["ref"]["pull_t0"])
                                if b["dh"]["ref"].get("pull_t0") is not None
                                else None,
                                "h2d_only": b["dh"]["ref"].get("h2d_only"),
                                "hd_t": rel(b["dh"]["ref"]["h2d_t0"])
                                if b["dh"]["ref"].get("h2d_t0") is not None
                                else None,
                                "t0": rel(b["dh"]["ref"]["t0"]),
                            },
                        },
                        "event": b["event"],
                        "flush": b["flush"],
                        "write": b["write"],
                        "ld": b["layerdone"],
                        "total": b["total"],
                        "by": b["bytes"],
                    }
                    for b in bs
                ],
                "d1": d1,
            }
        )
        # decode 行窗口裁剪: 只保留首个 decode 后 ~350ms(画图足够), 中位数在
        # python 侧算好, 避免 67 请求 × 720 步的 JSON 膨胀.
        dcut = (d1["fs"] + 350) if d1 else 1e18
        reqs[-1]["n"] = len(ds)
        reqs[-1]["med"] = (
            round(statistics.median((ds[k + 1]["recv"] - ds[k]["recv"]) * 1000 for k in range(len(ds) - 1)), 2)
            if len(ds) > 1
            else None
        )
        reqs[-1]["dec"] = [{"fs": rel(r["fs"]), "fe": rel(r["fs"]) + r["fwd"]} for r in ds if rel(r["fs"]) <= dcut]
        reqs[-1]["d1fwd"] = round(d1["fe"] - d1["fs"], 2) if d1 else None
        # proxy 端到端锚点(绝对秒 → 相对 P 受理锚点 ms; in/meta/pf 在锚点前为负).
        # 跨机时 proxy 所在机时钟未知, 绝对位置不能上图; 保留同进程内差值
        # (tok-in, 与时钟无关) 供 TTFT 表使用.
        p = pxy_all.get(u)
        reqs[-1]["pxy"] = None
        reqs[-1]["pttft"] = None
        if p:
            if cross["mode"] == "same-host":
                reqs[-1]["pxy"] = {k: (round((t - anchor) * 1000, 2) if t is not None else None) for k, t in p.items()}
            if p.get("tok") is not None and p.get("in") is not None:
                reqs[-1]["pttft"] = round((p["tok"] - p["in"]) * 1000, 2)
        # 结构化事件: 前端 api_arrive / first_token_out、P 前向两点、D 首次前向、
        # 等参数与背压探针、TE 侧 H2D/D2H 独立测量(缺则这些字段不存在 → TMPL 跳过).
        ev = p_events.get(u, []) + d_events.get(u, [])
        if ev:
            _attach_events(reqs[-1], ev, rel)
        # MTP(draft) 层前向(需在 _attach_events 之后取): 与 dec 同样裁到首个 decode
        # 后 ~350ms, 避免上千投机步把 JSON/SVG 撑爆; 总次数与中位耗时按全量口径给.
        mtp_all = reqs[-1].pop("_mtp_all", [])
        if mtp_all:
            reqs[-1]["mtp"] = [e for e in mtp_all if d1 is not None and e["t"] <= dcut]
            reqs[-1]["mtp_n"] = len(mtp_all)
            msv = [e["ms"] for e in mtp_all if e.get("ms") is not None]
            reqs[-1]["mtp_ms"] = round(statistics.median(msv), 3) if msv else None
            reqs[-1]["mtp_sync"] = bool(mtp_all[0].get("sync"))
        if clk:
            reqs[-1]["clk"] = clk
            # "相对锚点 ms" 的绝对起点(锚点的墙钟 ISO):
            #   wallAt(r, rel) = new Date(r.aw) + rel
            # 注意不能用 clk.ts 去减 rel —— clk.ts 是**绝对**单调钟(秒级 ~4.1e6), rel 是
            # 相对 anchor 的毫秒, 相减会差出几十天(实测 -48 天, 传输段与事件标记的日期
            # 因此对不上). aw 由 clk(clk.ts↔clk.wall 一对) 与 anchor 换算得到.
            try:
                _base = datetime.fromisoformat(clk["wall"])
                reqs[-1]["aw"] = (_base + timedelta(seconds=float(anchor) - float(clk["ts"]))).isoformat(
                    timespec="milliseconds"
                )
            except Exception:
                pass

    # 汇总(每请求全部层组累计, ms; 合批时共享墙钟窗口如实计入)
    def _s(k: str):
        return [round(sum(g[k] for g in r["groups"]), 2) for r in reqs]

    sums = {"d2h": _s("flush"), "h2h": _s("write"), "ld": _s("ld"), "event": _s("event")}
    sums["h2d"] = [round(sum(g["dh"]["dur"] for g in r["groups"] if g["dh"]), 2) for r in reqs]
    # pull 模式: P 几乎不写对端(write≈0), 真正的 H2H 搬运是 D 侧的 pull read —— 单列,
    # 免得看汇总表时把"h2h=0"误读成"没传".
    sums["pull"] = [round(sum((g["dh"] or {}).get("pull") or 0.0 for g in r["groups"]), 2) for r in reqs]
    sums["h2d_only"] = [round(sum((g["dh"] or {}).get("h2d_only") or 0.0 for g in r["groups"]), 2) for r in reqs]
    return {
        "reqs": reqs,
        "wall": f"{pwall} – {dwell}",
        "sums": sums,
        "meta": {
            "p_batches": len(ppb),
            "d_h2d": len(ddh),
            "paired": paired,
            "pfwd_rows": len(pfwds),
            "ddec_rows": len(ddecs),
            "n_pairs_missing": len(ppb) - paired,
            "cross": cross,
            # 防静默漏行: 每个文件里 [h2h][perf] 行的总数/JSON 数/旧正则数/
            # 两者都没吃下的行数(unparsed 必须为 0, 否则说明格式漂移).
            "parse": parse_stat,
        },
        "clk": clk,
        # 全体记录的绝对时间跨度(单调钟秒): proxy 打点缺席时, 页面墙钟窗口用它 + clk
        # 换算, 而不是退回"从日志前缀反推"(新格式日志里没有 t0= 可推).
        "span": span,
        "tcp": _tcp_stats(ppb),
    }


def _tcp_stats(pb: list[dict]) -> dict:
    """批级 TCP 传输统计: 总字节/耗时/平均吞吐 + 固定开销与渐近带宽回归.

    (bytes_i, write_ms_i) 线性回归 write = fixed + bytes/GB * ms_per_gb:
    fixed 即与字节无关的每批固定开销(控制/协议头等), 1/斜率 为数据面渐近带宽.
    样本字节恒定时无法分离固定开销, fixed 置 None 并给出 min/median 参考.
    """
    pts = [(b["bytes"], b["write"]) for b in pb if b["bytes"] is not None and b["write"] > 0]
    st = {
        "n": len(pts),
        "bytes": 0,
        "write_ms": 0.0,
        "fixed_ms": None,
        "slope_gbs": None,
        "distinct": 0,
        "min_write_ms": None,
        "med_write_ms": None,
    }
    if not pts:
        return st
    tot_b = sum(x for x, _ in pts)
    tot_w = sum(w for _, w in pts)
    st.update(bytes=tot_b, write_ms=round(tot_w, 2), distinct=len({x for x, _ in pts}))
    st["min_write_ms"] = round(min(w for _, w in pts), 2)
    st["med_write_ms"] = round(statistics.median(w for _, w in pts), 2)
    if st["distinct"] >= 2:
        n = len(pts)
        x = [b / 1e9 for b, _ in pts]
        y = list(w for _, w in pts)
        mx, my = sum(x) / n, sum(y) / n
        sxx = sum((xi - mx) ** 2 for xi in x)
        sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        if sxx > 0:
            b = sxy / sxx  # ms per GB
            a = my - b * mx
            st["fixed_ms"] = round(max(0.0, a), 3)
            st["slope_gbs"] = round(1000.0 / b, 2) if b > 0 else None
    return st


def _short(u: str) -> str:
    return u[len("chatcmpl-") :][:8]


def fmt(x) -> str:
    return "–" if x is None else f"{x:.1f}"


def main() -> int:
    logs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "logs"
    data = build(logs)
    # 输出路径可选: argv[2] 为文件或目录; 缺省 analysis/h2h_chain_timeline.html.
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    if out_arg is not None:
        op = Path(out_arg)
        out = op if op.suffix == ".html" else op / "h2h_chain_timeline.html"
    else:
        out = Path(__file__).resolve().parent / "h2h_chain_timeline.html"
    out.write_text(TMPL.replace("__DATA__", json.dumps(data)))
    s, reqs, meta = data["sums"], data["reqs"], data["meta"]
    print(f"wrote {out}: {len(reqs)} reqs, window {data['wall']}")
    print(
        f"P batch={meta['p_batches']} D h2d={meta['d_h2d']} 配对成功={meta['paired']} "
        f"缺失={meta['n_pairs_missing']} | P前向行={meta['pfwd_rows']} D decode行={meta['ddec_rows']}"
    )
    hdr = ["req", "P前向ms", "批组", "D2H", "H2H", "H2D", "LD往返", "等数据", "decode步", "TPOT中位"]
    print(f"{'':4s} " + "".join(f"{h:>9s}" for h in hdr))
    for i, r in enumerate(reqs):
        vals = [
            f"{sum(p['fwd'] for p in r['pf']):.0f}" if r["pf"] else "-",
            str(len(r["groups"])),
            f"{s['d2h'][i]:.0f}",
            f"{s['h2h'][i]:.0f}",
            f"{s['h2d'][i]:.0f}",
            f"{s['ld'][i]:.0f}",
            f"{s['event'][i]:.0f}",
            str(r["n"]),
            fmt(r["med"]),
        ]
        print(f"{_short(r['u'])} " + "".join(f"{v:>9s}" for v in vals))
    print("各列=该请求全部 layer-batch/round 累计(ms); H2H 列=TCP write, 不含 D 侧 H2D。")
    return 0


TMPL = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>H2H layerwise 传输链路时间线</title>
<style>
  :root { --surface-1:#fcfcfb; --surface-2:#f2f1ee; --text-1:#0b0b0b; --text-2:#52514e;
          --line:#d9d7d1;
          --p-fwd:#eb6834; --d-fwd:#2a78d6; --d2h:#0e9f6e; --net:#7454d6; --h2d:#c9841c; --mtp:#c026d3;
          --prep-p:#f6b48f; --prep-d:#9fc3ec; --idle:#e5e3de; --ext:#7d8490; }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) { color-scheme: dark; --surface-1:#1a1a19;
      --surface-2:#232322; --text-1:#ffffff; --text-2:#c3c2b7; --line:#3a3a37;
      --p-fwd:#d95926; --d-fwd:#3987e5; --d2h:#21b383; --net:#8a6ae8; --h2d:#dba13a; --mtp:#e879f9;
      --prep-p:#7a4a2e; --prep-d:#2c4a6e; --idle:#2c2c2a; --ext:#8b93a1; }
  }
  body { margin:0; background:var(--surface-1); color:var(--text-1);
         font:14px/1.45 system-ui,"Segoe UI",sans-serif; }
  .wrap { max-width:1500px; margin:0 auto; padding:18px 24px 60px; }
  h1 { font-size:18px; margin:0 0 2px; } h2 { font-size:13px; font-weight:600; margin:24px 0 6px; }
  .sub { color:var(--text-2); font-size:12px; margin-bottom:10px; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color:var(--text-2); margin:4px 0 10px; }
  .legend .sw { display:inline-block; width:12px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
  #themeBtn { float:right; font-size:12px; color:var(--text-2); background:var(--surface-2);
              border:1px solid var(--line); border-radius:6px; padding:3px 10px; cursor:pointer; }
  svg { display:block; background:var(--surface-1); }
  .axis-t { fill:var(--text-2); font-size:10.5px; } .axis-line { stroke:var(--line); stroke-width:1; }
  .lane-lbl { fill:var(--text-2); font:11px ui-monospace,Consolas,monospace; }
  .lbl-main { fill:var(--text-1); font:600 12px ui-monospace,Consolas,monospace; }
  .dim { fill:var(--text-2); font-size:10.5px; } .note { fill:var(--text-2); font-size:11px; }
  .chips { display:flex; gap:6px; flex-wrap:wrap; margin:4px 0 8px; }
  .chip { font:11px ui-monospace,Consolas,monospace; color:var(--text-2);
          background:var(--surface-2); border:1px solid var(--line); border-radius:14px;
          padding:2px 10px; cursor:pointer; }
  .chip.on { color:var(--text-1); border-color:var(--d-fwd); }
  table { border-collapse:collapse; font-size:12px; margin-top:8px; }
  th,td { padding:3px 10px 3px 0; text-align:right; border-bottom:1px solid var(--line);
          white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--text-2); font-weight:500; } td.num { font-variant-numeric:tabular-nums; }
  #tip { position:fixed; pointer-events:none; background:var(--surface-2); color:var(--text-1);
         border:1px solid var(--line); border-radius:6px; padding:6px 9px; font-size:11.5px;
         display:none; z-index:10; max-width:460px; }
  .bar { stroke:var(--surface-1); stroke-width:1px; }
  .bar.prep-p { fill:var(--prep-p); }
  .bar.p-fwd { fill:var(--p-fwd); }
  .bar.d2h { fill:var(--d2h); }
  .bar.net { fill:var(--net); }
  .bar.h2d { fill:var(--h2d); }
  .bar.mtp{fill:var(--mtp)}
  /* LAYER_DONE 往返折线: 上= P 侧, 下= D 侧; 三段各自一段斜/平线 */
  .ldflow { fill:none; stroke:var(--net); stroke-width:1.3; }
  .ldpt   { fill:var(--net); }
.bar.d-fwd { fill:var(--d-fwd); }
  .bar.ext { fill:var(--ext); }
  .bchk { stroke:var(--text-2); stroke-width:1; stroke-dasharray:2,2; }
  #err { position:fixed; left:12px; bottom:12px; max-width:70%; background:#7f1d1d; color:#fff;
         font:12px ui-monospace,monospace; padding:8px 12px; border-radius:6px; display:none;
         white-space:pre-wrap; z-index:99; }
</style>
</head>
<body><div class="wrap">
<button id="themeBtn">dark</button>
<h1>H2H layerwise · 传输链路时间线（P D2H / H2H TCP / D H2D / 模型执行）</h1>
<div class="sub">Qwen3-4B · logs window {WALL} · 并发合批运行(部分 decode/推送批次含多请求, 共享同一墙钟窗口; 同一请求多 round 推送按层组区分)。
时间轴 CLOCK_MONOTONIC(同机跨进程可比), 原点 = P 首次前向(无则首批 flush)。H2H 条/列 = TCP write 本体, <b>不含 D 侧 H2D</b>。PIPE 模式下相邻批 flush/write/H2D 有意重叠, 图中如实显示。</div>
<div class="legend">
  <span><i class="sw" style="background:var(--p-fwd)"></i>P 模型执行(prefill)</span>
  <span><i class="sw" style="background:var(--d2h)"></i>P D2H: NPU→CPU staging</span>
  <span><i class="sw" style="background:var(--net)"></i>H2H 传输: TCP write P→D</span>
  <span><i class="sw" style="background:var(--h2d)"></i>D H2D: staging→NPU</span>
  <span><i class="sw" style="background:var(--d-fwd)"></i>D 模型执行(decode)</span>
  <span><i class="sw" style="background:var(--mtp)"></i>MTP/draft 提案(每 decode 步一次, 含 MTP 层前向)</span>
  <span><i class="sw" style="background:var(--net);height:3px"></i>LAYER_DONE 往返折线: 上沿=P 发/收 ACK, 下沿=D 收/回 ACK(斜率段=网络, 平段=D 侧处理)</span>
  <span><i class="sw" style="background:var(--ext)"></i>图外: 客户端受理→P 受理 / 首token 回传(proxy 打点)</span>
  <span><i class="sw" style="background:var(--prep-p);opacity:.5"></i>forward 前准备(收批/调度)</span>
</div>

<h2>链路细览(点击泳道/标签切换请求)</h2>
<div class="chips" id="chips"></div>
<div id="panelB"></div>
<div class="sub">每批 4 层(MC_TCP_LAYER_BATCH=4), 批次标注 r{round}·{首层}-{末层}; 悬停任意条查看明细(含合批请求数与 H2D 配对)。
D2H 条 = P 侧 flush 窗口, H2H 条 = write 窗口(独立写线程时与相邻批重叠), H2D 条 = D 侧打点窗口(配对自 P 的 LAYER_DONE 往返)。</div>

<div class="sub" id="crossnote"></div>
<h2>TTFT 端到端闭环分解(ms, proxy 打点)</h2>
<div style="overflow-x:auto"><table id="ttft"></table></div>
<div class="sub">0 点 = P 引擎受理(worker 收批)。列: ①受理→P 受理(proxy 收请求→D 转发/触发 remote prefill→派发 P→P 收批) ②P 受理→D 首个 decode 前向起(前向+链路推送+调度) ③首个 decode 前向 ④其结束→proxy 转发首 token(采样/流式回传)。①②间详情(proxy 受理→D 触发、派发 P)悬停首 token/受理段可见。proxy 观测 TTFT 应≈vllm bench 实测(客户端侧, 差一次网络往返)。</div>

<h2>汇总表(每请求全部 layer-batch 累计, ms)</h2>
<div class="sub" id="netstat"></div>
<div style="overflow-x:auto"><table id="sum"></table></div>
<div class="sub">并发合批请求共享同一批次的墙钟, 各请求如实计入; decode 步为该请求参与解码的调度步数(合批步共享墙钟)。</div>
</div><div id="tip"></div>
<div id="err"></div>

<script>
const DATA = __DATA__;
const R = DATA.reqs;
const FMT = (ms,d=1) => (ms==null?"–":(ms<10000?ms.toFixed(d):ms.toFixed(0)))+" ms";
const U8 = u => u.slice("chatcmpl-".length, "chatcmpl-".length+8);
const tip = (html,x,y) => { const t=document.getElementById("tip");
  t.innerHTML=html; t.style.display="block"; t.style.left=(x+14)+"px"; t.style.top=(y+12)+"px"; };
const untip = () => document.getElementById("tip").style.display="none";
/* 单调时刻 → 绝对墙钟(本地时区). r.clk 是任一结构化记录自带的 (ts, wall) 对,
   由生成器从日志里原样带出 —— 不再从 vLLM 日志前缀反推 epoch(那只有秒精度且
   没有年份). 无 clk(旧日志)时返回 "–", 旧页面行为不变. */
function wallAt(r, rel){
  // rel 是"相对锚点"的毫秒: 直接加到锚点的墙钟上即可(见 build 里 aw 的推导)。
  if(!r.aw || rel==null) return "–";
  const t = new Date(r.aw).getTime() + rel;          // 浮点毫秒(epoch)
  let base = Math.floor(t), us = Math.round((t - base)*1000);
  if (us >= 1000) { base += 1; us -= 1000; }
  const d = new Date(base), p = n => String(n).padStart(2,"0");
  // 带 3 位微秒: 亚毫秒段(网络单程 0.2~0.9ms)也能看出起止不同.
  return `${p(d.getMonth()+1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:`
       + `${p(d.getSeconds())}.${String(d.getMilliseconds()).padStart(3,"0")}${String(us).padStart(3,"0")}`;
}
/* 一个时刻 → 一条竖标记(带 tooltip). y 为 lane 顶部, 标记画在 lane 内 4px 起,
   高 14px; 命中区比线宽, 便于悬停. */
function mark(svg, r, px, t, y, html){
  if (t==null) return;
  const x = px(t);
  svg.append(S("line",{x1:x,x2:x,y1:y+3,y2:y+17,stroke:"var(--text-2)","stroke-width":1.5,
    "stroke-dasharray":"2,2",opacity:.9}));
  const hit=S("rect",{x:x-3,y:y+3,width:6,height:14,fill:"transparent"});
  hit.addEventListener("mousemove",e=>tip(html,e.clientX,e.clientY));
  hit.addEventListener("mouseleave",untip);
  svg.append(hit);
}
/* 时刻列表 → 取首个/末个(缺则 null). */
const t1 = (a) => (a && a.length) ? a[0].t : null;
const tN = (a) => (a && a.length) ? a[a.length-1].t : null;
const f1 = (a) => (a && a.length) ? a[0] : null;
const fN = (a) => (a && a.length) ? a[a.length-1] : null;
const SVGNS = "http://www.w3.org/2000/svg";
function S(tag, attrs={}) { const el=document.createElementNS(SVGNS, tag);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,v); return el; }
function axis(svg, x, y, w, x0, x1) {
  const px = t => x + (t-x0)/(x1-x0) * w;
  const span = x1-x0,
        step = span>20000?2000:(span>800?100:(span>200?50:25));
  const g = S("g");
  for (let t=Math.ceil(x0/step)*step; t<x1; t+=step) {
    const xx = px(t);
    g.append(S("line", {x1:xx,x2:xx,y1:y-3,y2:y+3,"class":"axis-line"}));
    const txt = S("text", {x:xx,y:y+13,"class":"axis-t","text-anchor":"middle"});
    txt.textContent = Math.round(t); g.append(txt);
  }
  svg.append(g);
}
const bar = (t0,t1,px,y,h,cls) => {
  if (t1<=t0) t1=t0+0.5;
  // 最小可见宽度 1.8px: 亚像素宽条(如 1-2ms 的 D2H/H2D)在低缩放时仍可辨,
  // 实际起止由 tooltip 给出.
  return S("rect", {x:px(t0), y, width:Math.max(1.8,px(t1)-px(t0)), height:h, rx:Math.min(2,h/2), "class":"bar "+cls});
};
/* 该请求时间线范围 (含 proxy 图外锚点).
   tok(首 token)只在存在 decode 段(d1)时才拉长窗口 — 否则链路批会被压成
   亚像素; 无 decode 时 tok 以细标记线形式叠加, 不进窗口. */
function win(r) {
  let a = Infinity, b = -Infinity;
  if (r.pf.length) { a = Math.min(a, r.pf[0].recv); b = Math.max(b, r.pf.at(-1).fe); }
  for (const g of r.groups) {
    if (g.fw) { a = Math.min(a, g.fw[0]); b = Math.max(b, g.fw[1]); }
    if (g.ww) b = Math.max(b, g.ww[1]);
    if (g.lw) b = Math.max(b, g.lw[1]);
    if (g.dh) b = Math.max(b, g.dh.e);
  }
  // in/tok 是图外锚点, 只用于 tooltip/标注, 不进窗口(否则传输批被压成亚像素);
  // 无 pf/groups 的请求退回 in..tok 全景.
  if (r.pxy && r.pxy.in != null && !r.pf.length && !r.groups.length) {
    a = Math.min(a, r.pxy.in);
  }
  if (r.d1 && r.pxy && r.pxy.tok != null) b = Math.max(b, r.pxy.tok);
  // 6 个请求级时刻(新格式日志才有): 前端受理/首 token 写出、P 前向两点、
  // D 首次前向、等参数与背压段 —— 必须纳入窗口, 否则标记被裁掉.
  const marks = [];
  for (const k of ["producer","consumer"]) {
    if (r.api_in && r.api_in[k]) marks.push(r.api_in[k].t);
  }
  for (const k of ["p_fwd_first","p_fwd_last","d_fwd_first","first_out","first_chunk","prefill_done"]) {
    if (r[k]) for (const x of r[k]) marks.push(x.t);
  }
  if (r.mtp) for (const e of r.mtp) { marks.push(e.t); if (e.ms!=null) marks.push(e.t + e.ms); }
  if (r.probes) for (const p of r.probes) if (p.t!=null) { marks.push(p.t); marks.push(p.t + (p.ms||0)); }
  for (const m of marks) { if (m==null) continue; a = Math.min(a, m); b = Math.max(b, m); }
  const x0 = Math.min(a-15, 0);
  const x1 = Math.max(b+60, (r.d1 ? r.d1.fs+300 : b+60), 60);
  return [x0, x1];
}

/* ---------- Panel: 单请求 7 lane 链路细览 ---------- */
let sel=0;
function select(i){ sel=i; document.querySelectorAll(".chip").forEach((c,j)=>c.classList.toggle("on",j===i)); drawB(); }
const LANES = [
  ["受理→P(图外)", "ext"],
  ["P 模型", "p-fwd"],
  ["P D2H", "d2h"],
  ["H2H TCP", "net"],
  ["D H2D", "h2d"],
  ["D 模型", "d-fwd"],
  ["首token 回传(图外)", "ext"],
];
// 短标签: Worker_DP0_TP1_EP1 -> DP0·TP1
const rankTag = w => String(w).replace("Worker_","").split("_").slice(0,2).join("·");
function drawB() {
  const host=document.getElementById("panelB"); host.innerHTML="";
  const r=R[sel], W=1450, LABEL=128, AX=24;
  const LD_Y0 = 7;
  // LAYER_DONE: **该请求的每个 D rank 一条泳道**(DP 域内的 TP 各占一条; 只画本请求的
  // DP 域, 不出现空泳道) —— 泳道之间的横向错位就是"各 rank 收齐的时差".
  const reqRanks = [...new Set(r.groups.flatMap(g=>((g.dh&&g.dh.rows)||[]).map(x=>x.w)))]
    .filter(Boolean).sort();
  const LD_LANES = (reqRanks.length ? reqRanks : [""]).map(w=>["LAYER_DONE · " + (w? rankTag(w) : "?"), "ldflow"]);
  const ALL_LANES = LANES.concat(LD_LANES);
  const yOf=[30,58,86,114,142,170,198].concat(LD_LANES.map((_,i)=>226+i*28));
  const H=274 + Math.max(0, LD_LANES.length-1)*28;
  const [x0,x1] = win(r);
  const span=x1-x0, px=t=>LABEL+(t-x0)/span*(W-LABEL-8);
  const svg=S("svg",{width:W,height:H});
  const lb=S("text",{x:6,y:10,"class":"lbl-main"});
  lb.textContent=`req${sel+1} ${U8(r.u)} · 原点=P 首次前向`; svg.append(lb);
  ALL_LANES.forEach(([name,c],k)=>{
    const lbl=S("text",{x:6,y:yOf[k]+3,"class":"lane-lbl"}); lbl.textContent=name; svg.append(lbl);
    svg.append(S("line",{x1:0,x2:W,y1:yOf[k]+24,y2:yOf[k]+24,"class":"axis-line",opacity:.4}));
  });
  const grpLbl = (txt,x,y) => { const t=S("text",{x,y,"class":"dim"}); t.textContent=txt; svg.append(t); };
  // 图外段: 客户端受理 → P 受理 (lane 0). 起点可能在窗口外(不拉宽窗口),
  // 此时画截断条: 从窗口左缘到 P 受理(0).
  if (r.pxy && r.pxy.in != null && r.pxy.in < 0) {
    const t0 = Math.max(r.pxy.in, x0);
    const eb=bar(t0,0,px,yOf[0]+7,10,"ext");
    svg.append(eb);
    eb.addEventListener("mousemove",e=>tip(
      `<b>${r.u}</b> · 受理→P 受理 ${FMT(-r.pxy.in,1)}`+
      (r.pxy.in < x0?` (起点 −${FMT(x0-r.pxy.in,0)} 在窗口外, 条为截断显示)`:"")+
      `<br>`+
      (r.pxy.meta!=null?`proxy受理→D 触发remote prefill ${FMT(r.pxy.meta-r.pxy.in,1)}<br>`:"")+
      (r.pxy.pf!=null&&r.pxy.meta!=null?`proxy 选择P并派发 ${FMT(r.pxy.pf-r.pxy.meta,1)}<br>`:"")+
      (r.pxy.pf!=null?`派发P→P引擎收批(网络+API+调度) ${FMT(-r.pxy.pf,1)}`:"")+
      `<br><span style="color:var(--text-2)">proxy 打点, 与引擎同机单调钟</span>`,e.clientX,e.clientY));
    eb.addEventListener("mouseleave",untip);
    const ttl=S("text",{x:px(0)+4,y:yOf[0]+10,"class":"dim"});
    ttl.textContent=`受理→P ${FMT(-r.pxy.in,0)}`; svg.append(ttl);
  }
  // P 前向 (lane 1)
  r.pf.forEach(p=>{
    svg.append(bar(p.recv,p.fs,px,yOf[1]+7,4,"prep-p"));
    svg.append(bar(p.fs,p.fe,px,yOf[1]+7,10,"p-fwd"));
  });
  if (r.mtp_n) {
    const mt=S("text",{x:LABEL+2,y:yOf[5]+30,"class":"dim"});
    mt.textContent=`MTP/draft 提案 ×${r.mtp_n} 中位 ${(r.mtp_ms??0).toFixed(2)} ms`; svg.append(mt);
  }
  if (r.pf.length) {
    const pt=S("text",{x:px(r.pf.at(-1).fe)+4,y:yOf[1]+20,"class":"dim"});
    pt.textContent=`P 前向 ${FMT(r.pf.map(p=>p.fwd).reduce((a,b)=>a+b,0),0)}`;
    svg.append(pt);
  }
  // 层组批次: D2H(2) H2H(3) H2D(4)
  r.groups.forEach(g=>{
    const yF=yOf[2]+7, yW=yOf[3]+7, yH=yOf[4]+7;
    const fl=g.fw? bar(g.fw[0],g.fw[1],px,yF,10,"d2h") : null;
    // H2H 段: pull 模式下 P 不写对端(ww≈0 宽), 真正的搬运在 D 侧 = dh.pull(起点 pt);
    // push 模式才用 P 的 write 窗口(ww).
    // H2H read / H2D 都用**单 rank 参考行**(与 LAYER_DONE 折线同一行), 这样图上三
    // 条与折线的 ② 段能互相对上; 多 rank 的聚合值(max 耗时 / min 起点)只在 tooltip
    // 里作对照 —— 混用会出现 "②=37.8ms 而 H2H 8.1 + H2D 36.0 对不上".
    const rf = g.dh && g.dh.rf;
    const pullT = (rf && rf.pull!=null && rf.pt!=null) ? [rf.pt, rf.pt+rf.pull]
                : ((g.dh && g.dh.pull!=null && g.dh.pt!=null) ? [g.dh.pt, g.dh.pt+g.dh.pull] : null);
    const hw = pullT || g.ww;
    const wr = hw? bar(hw[0],hw[1],px,yW,10,"net") : null;
    const hdWin = (rf && rf.hd_t!=null && rf.h2d_only!=null) ? [rf.hd_t, rf.hd_t + rf.h2d_only]
                : ((g.dh && g.dh.hd_t!=null && g.dh.h2d_only!=null) ? [g.dh.hd_t, g.dh.hd_t + g.dh.h2d_only]
                : (g.dh? [g.dh.s, g.dh.e] : null));
    const hd = hdWin? bar(hdWin[0],hdWin[1],px,yH,10,"h2d") : null;
    for (const el of [fl,wr,hd]) {
      if (!el) continue;
      svg.append(el);
      el.addEventListener("mousemove",e=>tip(
        `<b>${r.u}</b> · round ${g.rnd+1} 批 layers=[${g.lay.join(",")}]`+
        (g.event>0.01?`<br>等 NPU 数据就绪 ${FMT(g.event)}`:"")+
        `<br>P D2H ${FMT(g.flush)}`+
          (g.fw?` <span style="color:var(--text-2)">· 绝对 ${wallAt(r,g.fw[0])} → ${wallAt(r,g.fw[1])}</span>`:"")+
        `<br>H2H TCP(write) ${FMT(g.write)}`+
          (g.ww?` <span style="color:var(--text-2)">· 绝对 ${wallAt(r,g.ww[0])} → ${wallAt(r,g.ww[1])}</span>`:"")+
        (rf&&rf.pull!=null?`<br>H2H read(pull, 单 rank) ${FMT(rf.pull)}`+
          (rf.pt!=null?` <span style="color:var(--text-2)">· 绝对 ${wallAt(r,rf.pt)} → ${wallAt(r,rf.pt+rf.pull)}</span>`:"")+
          ((g.dh.frows||1)>1&&g.dh.pull!=null?`<br><span style="color:var(--text-2)">多 rank 聚合(max) pull=${FMT(g.dh.pull)} / 起点 min=${FMT(g.dh.pt)}</span>`:"")
          :"")+
        (g.by!=null?`<br>H2H payload ${(g.by/1048576).toFixed(1)} MiB`+
          (g.write>0?` · 吞吐 ${(g.by/1e9/(g.write/1000)).toFixed(2)} GB/s (${(g.by*8/1e9/(g.write/1000)).toFixed(1)} Gbps)`:"")
          :"")+
        (rf&&rf.h2d_only!=null?`<br>D H2D(纯, 单 rank) ${FMT(rf.h2d_only)}`+
          (rf.hd_t!=null?` <span style="color:var(--text-2)">· 绝对 ${wallAt(r,rf.hd_t)} → ${wallAt(r,rf.hd_t+rf.h2d_only)}</span>`:"")+
          ((g.dh.frows||1)>1?`<br><span style="color:var(--text-2)">多 rank 聚合(max) h2d_only=${FMT(g.dh.h2d_only)}; 含 pull 的旧口径 dur=${FMT(g.dh.dur)}</span>`:"")
          :(g.dh?`<br>D H2D ${FMT(g.dh.h2d_only!=null? g.dh.h2d_only : g.dh.dur)}`+
            ` <span style="color:var(--text-2)">· 绝对 ${wallAt(r,g.dh.s)} → ${wallAt(r,g.dh.e)}</span>`:""))+
        (g.lw && g.dh && g.ack!=null?`<br>LAYER_DONE 三段: P→D ${FMT(g.dh.s-g.lw[0],2)} · D 处理 ${FMT(g.ack-g.dh.s,2)} · D→P ${FMT(g.lw[1]-g.ack,2)}`:"")+
        (g.lw?`<br>LAYER_DONE 往返 ${FMT(g.lw[1]-g.lw[0])}`+
          ` <span style="color:var(--text-2)">· 绝对 ${wallAt(r,g.lw[0])} → ${wallAt(r,g.lw[1])}</span>`+
          ` (内含 H2D, 已单列不计入 H2H)`:"")+
        `<br>批墙钟 ${FMT(g.total)}`,e.clientX,e.clientY));
      el.addEventListener("mouseleave",untip);
    }
    if (g.lw) {  // LAYER_DONE 往返细线(含 D H2D 的参照窗口)
      const l=S("line",{x1:px(g.lw[0]),x2:px(g.lw[1]),y1:yW+15,y2:yW+15,"class":"bchk"});
      svg.append(l);
    }
    // LAYER_DONE 折线: **每个 lane(=一个 D rank) 每批只画一条** —— 取该 rank 在本批窗口内
    // 最早的"整段往返"(t0/ACK 都在窗口内, 由解析器保证); 该 rank 若在本批另有落在窗口内
    // 的往返(多个 P rank 各发一次), 不在图上重复画, 只写进悬停(避免同一 TP 出现两条线).
    const rows = (g.dh && g.dh.rows) || [];
    if (g.lw && rows.length) {
      reqRanks.forEach((w, ri) => {
        const rs = rows.filter(x => x.w === w);
        if (!rs.length) return;
        const yBase = yOf[7 + ri], yP = yBase + 6, yD = yBase + 20;
        const rr = rs[0];                       // rows 已按 t0 排序
        const t0 = rr.t0, ack = Math.max(rr.ack, t0);
        const x0 = g.lw[0], x1 = Math.max(t0, x0), x2 = Math.max(ack, x1), x3 = Math.max(g.lw[1], x2);
        // 绘制截断: P 侧窗口是**多 peer 并集**, 个别批会很长(实测 157ms, 因为 P 在等其它
        // rank 的 ACK) —— 原样画会把折线拉成长斜线横穿后面几批. 这里把 ①/③ 两条腿的
        // 绘制长度限制在 3×(该 rank 自己的往返) 或 15ms 内; **数值仍按真实值给**,
        // 悬停里标注"绘制已截断".
        const own = Math.max(x2 - x1, 1), cap = Math.max(own * 3, 15);
        const x0d = Math.max(x0, x1 - cap), x3d = Math.min(x3, x2 + cap);
        const clipped = (x0d > x0 + 1e-3) || (x3d < x3 - 1e-3);
        const xs = [x0d, x1, x2, x3d], ys = [yP, yD, yD, yP];
        svg.append(S("polyline", {points: xs.map((t,i)=>(px(t)+","+ys[i])).join(" "), "class":"ldflow"}));
        xs.forEach((t,i)=>svg.append(S("circle", {cx:px(t), cy:ys[i], r:1.5, "class":"ldpt"})));
        const hit=S("rect",{x:px(x3)-3, y:yBase+1, width:6, height:23, fill:"transparent"});
        hit.addEventListener("mousemove",e=>tip(
          `<b>${r.u}</b> · LAYER_DONE 往返 round ${g.rnd+1} 批 layers=[${g.lay.join(",")}]<br>`+
          `接收方 <b>${rankTag(w)}</b>${rr.snd?` · 发送方 ⋯${rr.snd}`:""}<br>`+
          `① P 发 → D 收(网络) <b>${FMT(x1-x0,2)}</b><br>`+
          `② D 收 → D 回 ACK(该 rank: pull ${FMT(rr.pull,2)} + H2D(纯) ${FMT(rr.h2d_only,2)}) <b>${FMT(x2-x1,2)}</b><br>`+
          `③ D 回 ACK → P 收齐 ACK <b>${FMT(x3-x2,2)}</b><br>`+
          `绝对 ${wallAt(r,x0)} → ${wallAt(r,x3)}`+
          (clipped?`<br><span style="color:var(--text-2)">(① / ③ 实际 ${FMT(x1-x0,2)} / ${FMT(x3-x2,2)} ms, 绘制已截断到 ${FMT(cap,1)} ms ×2)</span>`:"")+
          (rs.length>1?`<br><span style="color:var(--text-2)">本批该 rank 另有 ${rs.length-1} 次落窗往返(未重复画): `+
            rs.slice(1).map(x=>`收 ${wallAt(r,x.t0)}`).join(" / ")+`</span>`:"")+
          ((g.dh.frows||1)>1?`<br><span style="color:var(--text-2)">该批共 ${g.dh.frows} 个 D rank 各收一次(P 侧窗口为并集, 仅供参照)</span>`:""),
          e.clientX,e.clientY));
        hit.addEventListener("mouseleave",untip);
        svg.append(hit);
      });
    } else if (g.lw && g.dh && g.dh.fs!=null && g.dh.fa!=null) {
      // 老格式日志(无 per-rank 明细): 退回单条四点折线 —— P发 → D收 → D回ACK → P收齐.
      const yP = yOf[7]+6, yD = yOf[7]+20;
      const x0 = g.lw[0];
      const x1 = Math.max(g.dh.fs, x0);
      const x2 = Math.max(g.dh.fa, x1);
      const x3 = Math.max(g.lw[1], x2);
      const xs = [x0, x1, x2, x3], ys = [yP, yD, yD, yP];
      svg.append(S("polyline", {points: xs.map((t,i)=>(px(t)+","+ys[i])).join(" "), "class":"ldflow"}));
      xs.forEach((t,i)=>svg.append(S("circle", {cx:px(t), cy:ys[i], r:1.6, "class":"ldpt"})));
      const hit=S("rect",{x:px(x0)-2, y:yOf[7]+2, width:Math.max(3,px(x3)-px(x0)+4), height:22, fill:"transparent"});
      hit.addEventListener("mousemove",e=>tip(
        `<b>${r.u}</b> · LAYER_DONE 往返 round ${g.rnd+1} 批 layers=[${g.lay.join(",")}]<br>`+
        `① P 发 → D 收 <b>${FMT(x1-x0,2)}</b> · 绝对 ${wallAt(r,x0)} → ${wallAt(r,x1)}<br>`+
        `② D 收 → D 回 ACK <b>${FMT(x2-x1,2)}</b> · 绝对 ${wallAt(r,x1)} → ${wallAt(r,x2)}<br>`+
        `③ D 回 ACK → P 收齐 <b>${FMT(x3-x2,2)}</b> · 绝对 ${wallAt(r,x2)} → ${wallAt(r,x3)}`,
        e.clientX,e.clientY));
      hit.addEventListener("mouseleave",untip);
      svg.append(hit);
    }
    if (fl) grpLbl(`r${g.rnd+1}·${g.lay[0]}-${g.lay.at(-1)}`, px(g.fw[0])+1, yF-6);
  });
  // decode 步 (lane 5)
  let drawn=0;
  for (const s of r.dec) {
    if (s.fs > x1) break;
    if (s.fe < x0) continue;
    svg.append(bar(s.fs,s.fe,px,yOf[5]+7,10,"d-fwd"));
    drawn++;
  }
  if (r.d1) {
    const m=S("line",{x1:px(r.d1.recv),x2:px(r.d1.recv),y1:yOf[5]+19,y2:yOf[5]+23,
      stroke:"var(--d-fwd)","stroke-width":2});
    svg.append(m);
    grpLbl(`decode×${r.n}`, px(r.d1.recv)+4, yOf[5]+20);
    const hit=S("rect",{x:px(x0),y:yOf[5]+7,width:px(x1)-px(x0),height:10,fill:"transparent"});
    hit.addEventListener("mousemove",e=>tip(
      `<b>${r.u}</b> · decode 参与 ${r.n} 步, 本窗口画出 ${drawn} 步<br>`+
      `步周期中位 ${FMT(DATA.summary[r.si].step_med,2)} · 首步 fwd ${FMT(r.d1? DATA.summary[r.si].d1fwd:0,1)}`,
      e.clientX,e.clientY));
    hit.addEventListener("mouseleave",untip);
    svg.append(hit);
  }
  // 图外段: D 首 decode 前向结束 → proxy 转发首 token (lane 6)
  if (r.pxy && r.pxy.tok != null && r.d1 && r.pxy.tok > r.d1.fe) {
    const eb=bar(r.d1.fe,r.pxy.tok,px,yOf[6]+7,10,"ext");
    svg.append(eb);
    eb.addEventListener("mousemove",e=>tip(
      `<b>${r.u}</b> · 首 token 回传 ${FMT(r.pxy.tok-r.d1.fe,1)}<br>`+
      `(首个 decode 前向结束 → proxy 转发含文本的首 chunk, 含采样/解码与流式封装)<br>`+
      `proxy 观测 TTFT = 受理→首token ${FMT(r.pxy.tok-r.pxy.in,1)}`,e.clientX,e.clientY));
    eb.addEventListener("mouseleave",untip);
  } else if (r.pxy && r.pxy.tok != null && !r.d1) {
    // 无 decode 段(日志缺模型执行打点): tok 不进窗口, 图右上角给 TTFT 标注.
    const nt=S("text",{x:W-8,y:yOf[0]-8,"class":"dim","text-anchor":"end"});
    nt.textContent=`无 decode 段日志 — proxy 观测 TTFT=${FMT(r.pxy.tok-r.pxy.in,1)} (首token 于链尾 ${FMT(r.pxy.tok-0,0)}ms)`;
    svg.append(nt);
  }
  // MTP(draft) 层前向: 画在对应模型泳道的下半格(细杠), 悬停给耗时与绝对起止.
  // 每投机步一条, 已在上游裁到 decode 起始后 ~350ms 内(见 build 的 dcut).
  (r.mtp||[]).forEach(e=>{
    const yy = (e.role==="producer"? yOf[1] : yOf[5]) + 18;
    const b = bar(e.t, e.t + (e.ms||0), px, yy, 6, "mtp");
    svg.append(b);
    b.addEventListener("mousemove",ev=>tip(
      `<b>${r.u}</b> · MTP/draft 提案${e.step!=null?` #${e.step}`:""}`+
      (e.proposer?` <span style="color:var(--text-2)">(${e.proposer})</span>`:"")+
      (e.scope==="propose"?'<br><span style="color:var(--text-2)">口径=该 decode 步的提案(含 num_speculative_tokens 次 MTP 层前向)</span>':"")+
      `<br>耗时 ${FMT(e.ms,2)}${e.sync===false?` <span style="color:var(--text-2)">(未同步, 偏小)</span>`:""}`+
      `<br>绝对 ${wallAt(r,e.t)} → ${wallAt(r,e.t+(e.ms||0))}`,
      ev.clientX,ev.clientY));
    b.addEventListener("mouseleave",untip);
  });
  // ---- 6 个请求级时刻 + 停顿探针(新格式日志才有; 旧日志整段跳过, 页面不变) ----
  // 前端: 到达 P/D 的 api_server; worker: P 前向两点、D 开始计算; 前端: 首 token
  // 写出. 悬停显示**绝对时刻**(本地墙钟, 来自记录自带的 (ts, wall) 对).
  const mAbs = (rec) => (rec && rec.wall) ? rec.wall : wallAt(r, rec && rec.t);
  const put = (rec, y, label) =>
    mark(svg, r, px, rec && rec.t, y, `<b>${r.u}</b> · ${label}<br>绝对 ${mAbs(rec)}`);
  const roleCN = v => v==="producer" ? "P" : (v==="consumer" ? "D" : "?");
  if (r.api_in) {
    put(r.api_in.producer, yOf[1], "请求到达 P 的 api_server");
    put(r.api_in.consumer, yOf[5], "请求到达 D 的 api_server");
  }
  put(f1(r.p_fwd_first), yOf[1], "P 开始前向计算");
  put(fN(r.p_fwd_last), yOf[1], "P 结束前向计算(prefill 完成, 含首 token 采样)");
  put(f1(r.d_fwd_first), yOf[5], "D 开始计算(首次 decode 前向起)");
  (r.recv_done||[]).forEach(rec=>put(rec, yOf[5],
    `D: KV 收齐并交给 scheduler (${roleCN(rec.role)})` +
    ((r.recv_done.length>1)?` · 该请求共 ${r.recv_done.length} 个 rank 各报一次`:"")
  ));
  (r.first_chunk||[]).forEach(rec=>put(rec, yOf[6], `首 chunk 写出前端(${roleCN(rec.role)})`));
  (r.first_out||[]).forEach(rec=>put(rec, yOf[6],
    `首 token 写出前端(${roleCN(rec.role)}, 流式首个带内容 chunk)`));
  if (r.prefill_done) put(f1(r.prefill_done), yOf[1], "P: prefill 完成 + 首 token 已产出(校验点)");
  (r.token_sent||[]).forEach((rec,i)=>put(rec, yOf[6], `P→D 投递首 token (peer ${i+1})`));
  // P 侧过程量: 等 D 投递 block table(常为 P 最长停顿) 与发送队列背压段.
  if (r.probes) for (const b of r.probes) {
    mark(svg, r, px, b.t, yOf[1],
      `<b>${r.u}</b> · ${b.kind==="p_wait_params" ? "等 D 投递 block table("+(b.mode||"")+")"
        : "前向被发送队列背压钳制(layer="+(b.mode||"")+")"}`+
      `<br>持续 ${FMT(b.ms)}<br>绝对 ${wallAt(r,b.t)} → ${wallAt(r,b.t+(b.ms||0))}`);
  }
  axis(svg,LABEL,H-AX+6,W-LABEL-8,x0,x1);
  const axl=S("text",{x:LABEL,y:H-AX+22,"class":"dim"}); axl.textContent=`距请求原点 ms`;
  svg.append(axl);
  host.append(svg);
}

/* ---------- TTFT 闭环表 ---------- */
function drawTTFT(){
  const heads=["req","受理→P受理","P受理→D1前向起","D1前向","D1前向末→首token","proxy TTFT(受理→首token)"];
  const tr=document.createElement("tr");
  heads.forEach(h=>{const th=document.createElement("th"); th.textContent=h; tr.append(th);});
  document.getElementById("ttft").append(tr);
  R.forEach((r,i)=>{
    const p=r.pxy, tr=document.createElement("tr");
    const d1f = r.d1 ? r.d1.fe - r.d1.fs : null;
    const cells = [["req"+(i+1)+" "+U8(r.u)]];
    if (p && p.in != null) {
      const d1fe = r.d1 ? r.d1.fe : null;
      cells.push(-p.in, r.d1 ? r.d1.fs : null,
                  d1f, (p.tok!=null && d1fe!=null) ? p.tok-d1fe : null,
                  p.tok!=null ? p.tok-p.in : null);
    } else if (r.pttft != null) {
      // 跨机: 绝对位置无法与 P 时钟对齐, 只给 proxy 进程内 TTFT 差值.
      for (let k=1;k<5;k++) cells.push("–");
      cells.push(r.pttft.toFixed(1)+"*");
    } else {
      for (let k=1;k<5;k++) cells.push("–");  // 无 proxy 打点
    }
    cells.forEach((v,j)=>{ const td=document.createElement("td");
      td.textContent = typeof v==="number" ? (v<0?"−"+(-v).toFixed(1):v.toFixed(1)) : v;
      if (j!==0) td.className="num"; tr.append(td); });
    document.getElementById("ttft").append(tr);
  });
}

/* ---------- 汇总表 ---------- */
function drawTable(){
  const heads=["req","P前向(ms/次)","批组数","P D2H","H2H write","H2H read(pull)","H2H 总MiB","H2H 吞吐GB/s","D H2D","D H2D(纯)","LD往返","等数据","decode步","TPOT中位","D1 fwd","已decode","MTP提案(中位)","MTP×次数"];
  const tr=document.createElement("tr");
  heads.forEach(h=>{const th=document.createElement("th"); th.textContent=h; tr.append(th);});
  document.getElementById("sum").append(tr);
  R.forEach((r,i)=>{
    const c=DATA.summary[i], tr=document.createElement("tr");
    const cells=[["req"+(i+1)+" "+U8(r.u)],
      r.pf.length? (r.pf.reduce((a,p)=>a+p.fwd,0).toFixed(0)+" / "+r.pf.length) : "–",
      String(r.groups.length), c.d2h, c.h2h, c.pull||0,
      c.mib==null?"–":c.mib.toFixed(1),
      c.gbs_eff==null?"–":c.gbs_eff.toFixed(2),
      c.h2d, c.h2d_only||0, c.ld, c.event,
      String(r.n), c.step_med==null?"–":c.step_med.toFixed(1),
      c.d1fwd==null?"–":c.d1fwd.toFixed(1), c.decoded?"✓":"✗",
      r.mtp_ms==null?"–":r.mtp_ms.toFixed(2), r.mtp_n==null?"–":String(r.mtp_n)];
    cells.forEach((v,j)=>{ const td=document.createElement("td"); td.textContent=v;
      if (j!==0) td.className="num"; tr.append(td); });
    document.getElementById("sum").append(tr);
  });
}

/* ---------- 预计算 per-request 汇总(索引对齐) ---------- */
const SUMMARY = R.map((r,i)=>{
  r.si=i;
  const totBytes=sum(r.groups,g=>g.by||0), totW=sum(r.groups,g=>g.write);
  const totPull=sum(r.groups,g=>(g.dh&&g.dh.pull)||0);
  // h2h_eff: 该 run 的实际 H2H 搬运量 —— pull 模式下 P 不写对端, 搬运在 D 侧的 pull
  // read; push 模式才是 P 的 write. 两个都列, 避免把 pull 模式的 h2h≈0 读成"没传".
  const totH2hEff=totPull>0? totPull : totW;
  return {d2h:sum(r.groups,g=>g.flush), h2h:totW, pull:totPull, h2h_eff:totH2hEff,
          h2d:sum(r.groups,g=>g.dh?g.dh.dur:0),
          h2d_only:sum(r.groups,g=>(g.dh&&g.dh.h2d_only!=null)?g.dh.h2d_only:0),
          ld:sum(r.groups,g=>g.lw?g.lw[1]-g.lw[0]:0),
          event:sum(r.groups,g=>g.event), step_med:r.med, d1fwd:r.d1fwd, decoded:r.n>0,
          mib: totBytes>0? totBytes/1048576 : null,
          // 吞吐按"实际 H2H 段"算: pull 用 pull read, push 用 write
          gbs: (totBytes>0&&(totH2hEff>0))? totBytes/1e9/(totH2hEff/1000) : null,
          gbs_eff: (totBytes>0&&(totH2hEff>0))? totBytes/1e9/(totH2hEff/1000) : null};
});
function sum(a,f){ let s=0; for (const x of a) s+=f(x); return Math.round(s*10)/10; }
DATA.summary = SUMMARY;

function drawChips(){
  const c=document.getElementById("chips");
  R.forEach((r,i)=>{ const b=document.createElement("button"); b.className="chip"+(i===sel?" on":"");
    b.textContent=`req${i+1} ${U8(r.u)}`; b.onclick=()=>select(i); c.append(b); });
}
function fillNet(){
  const el=document.getElementById("netstat");
  if(!el) return;
  const t=DATA.tcp||{};
  let txt=`TCP 传输全局: ${t.n} 批 · Σ ${(t.bytes/1048576).toFixed(0)} MiB · Σ write ${FMT(t.write_ms,1)}`;
  // pull 模式: P 侧不写对端(write≈0), 真正的搬运是 D 侧 pull read —— 这时按 write 算
  // 吞吐会得到天文数字(实测 1.8e5 GB/s), 直接改成给出 D 侧口径的提示.
  const pullMs = (DATA.sums && DATA.sums.pull) ? DATA.sums.pull.reduce((a,b)=>a+(b||0),0) : 0;
  if (t.n>0 && t.write_ms > t.n*0.05) {
    txt+=` → 平均吞吐(write) ${(t.bytes/1e9/(t.write_ms/1000)).toFixed(2)} GB/s (${(t.bytes*8/1e9/(t.write_ms/1000)).toFixed(1)} Gbps)`;
  } else if (t.n>0) {
    txt+=` → <b>pull 模式</b>: P 不写对端(write≈0), 实际搬运见 D 侧 pull read`;
    if (pullMs>0) txt+=` (Σ pull ${FMT(pullMs,0)} → 平均 ${(t.bytes/1e9/(pullMs/1000)).toFixed(2)} GB/s)`;
  }
  if (t.fixed_ms!=null && t.slope_gbs!=null && t.slope_gbs<1e4) {
    txt+=`<br>每批固定开销(线性回归截距)= ${t.fixed_ms.toFixed(2)} ms · 数据面渐近带宽(1/斜率)= ${t.slope_gbs.toFixed(1)} GB/s (样本 ${t.distinct} 种字节量)`;
  } else if (t.n>0) {
    txt+=`<br>全部批字节量相同(${(t.bytes/t.n/1048576).toFixed(1)} MiB/批): 固定开销无法由回归分离; 参考 write min=${FMT(t.min_write_ms,2)} med=${FMT(t.med_write_ms,2)}`;
  }
  el.innerHTML=txt;
}
function fillCross(){
  const el=document.getElementById("crossnote");
  if(!el) return;
  const c=(DATA.meta&&DATA.meta.cross)||{};
  if (c.mode==="cross-node") {
    el.innerHTML=`<b>跨节点模式</b>: P/D 时钟不同原点, 已用逐批 LAYER_DONE 往返样本估计时钟偏移 `+
      `${c.offset_ms.toFixed(1)} ms(P10-P90 约 ±0.2ms), D 侧时间已折算到 P 时钟; `+
      `TTFT 闭环表中 * 列为 proxy 进程内差值(无时钟依赖), 图外受理/回传段不上图.`;
    el.style.color="var(--h2d)";
  } else { el.innerHTML=""; }
}
function themeInit(){
  const btn=document.getElementById("themeBtn");
  const apply=()=>btn.textContent=document.documentElement.dataset.theme==="dark"?"light":"dark";
  btn.onclick=()=>{document.documentElement.dataset.theme=document.documentElement.dataset.theme==="dark"?"light":"dark";apply();};
  apply();
}
window.addEventListener("error", ev => {
  const e=document.getElementById("err");
  if (e) { e.style.display="block";
    e.textContent="页面脚本错误: "+ev.message+"\\n@"+(ev.filename||"")+":"+(ev.lineno||"");
  }
});
try { drawChips(); drawB(); drawTTFT(); drawTable(); fillNet(); fillCross(); themeInit(); }
catch (err) {
  const e=document.getElementById("err");
  if (e) { e.style.display="block"; e.textContent="页面脚本错误:\\n"+(err && err.stack ? err.stack : err); }
}
</script>
</body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
