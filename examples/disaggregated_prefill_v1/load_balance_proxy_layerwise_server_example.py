# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.8+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 9000 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# Optional: p_then_d flow (--p-then-d)
# ------------------------------------
# By default the proxy dispatches the decoder first; the decoder triggers the
# remote prefill by POSTing its kv_transfer_params to this proxy's
# /v1/metaserver endpoint, and this proxy then dispatches the prefiller.
#
# With --p-then-d the order is reversed: the prefiller goes first (non-streaming,
# max_tokens=1) and generates the first token. The decoder is dispatched right
# after the prefiller — in parallel, using the prefiller's transport identity
# declared on the command line (--prefiller-kv-ports/--prefiller-tp-sizes/
# --prefiller-pcp-sizes) — so that the decoder pushes its block table back to
# the prefiller's direct params channel while the prefiller is still running its
# prefill. That is what lets the layerwise KV transfer overlap with the prefill
# instead of starting only after the prefiller's response comes back.
#
# The first token is then injected into that already-dispatched decoder: the
# prefiller returns its text (decoded with its own tokenizer) plus the token id
# in kv_transfer_params, the proxy emits it as the first stream chunk, and posts
# the token id to the decoder's /v1/p_then_d_token endpoint so the decoder can
# account for it (see vllm_ascend/core/recompute_scheduler.py). No metaserver is
# involved in this mode.
#
# Requires on the engines:
#   prefiller: P_THEN_D=1, REUSE_PREFILLED_TOKENS=1, (recommended) SKIP_DECODE_TOKENIZE=1
#   decoder:   REUSE_PREFILLED_TOKENS=1, (recommended) SKIP_DECODE_TOKENIZE=1
# See vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import copy
import functools
import heapq
import ipaddress
import json
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.logger import init_logger

logger = init_logger(__name__)

# Add uvloop for faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class ServerState:
    def __init__(self, host, port, kv_meta=None):
        self.host = host
        self.port = port
        # p_then_d: P 的 KV 传输身份({kv_port, tp_size, pcp_size}), 由命令行声明;
        # 未声明时为 None(退回"等 P 响应才知道 P 身份"的老时序).
        self.kv_meta = kv_meta
        self.url = f"http://{host}:{port}/v1"
        # Auto-completion for ipv6
        try:
            ip = ipaddress.ip_address(self.host)
            if isinstance(ip, ipaddress.IPv6Address):
                self.url = f"http://[{host}]:{port}/v1"
        except Exception:
            pass
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0  # Number of active requests
        self.aborted_requests = set()  # Track aborted requests
        # Removed individual server lock - will use global locks instead


class ProxyState:
    def __init__(self, prefiller_instances, decoder_instances, prefiller_meta=None):
        meta = prefiller_meta or [None] * len(prefiller_instances)
        self.prefillers: list[ServerState] = [
            ServerState(h, p, kv_meta=m) for (h, p), m in zip(prefiller_instances, meta)
        ]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        # Removed selection locks - no longer needed for synchronous methods

        # Initialize priority queues for efficient server selection
        # Each entry is (priority_score, server_index, server_reference)
        # Lower priority score = higher priority (less loaded)
        self.prefiller_heap = [(0, i, server) for i, server in enumerate(self.prefillers)]
        self.decoder_heap = [(0, i, server) for i, server in enumerate(self.decoders)]
        heapq.heapify(self.prefiller_heap)
        heapq.heapify(self.decoder_heap)
        self.req_id_future = {}
        self.req_data_dict = {}
        # perf: 请求级绝对时刻打点 (uuid -> dict), 供 TTFT 端到端分解:
        # in=proxy 受理 / meta=D 触发 remote prefill / pf=派发 P / tok=首 token 转发.
        # perf_counter 为 CLOCK_MONOTONIC, 与 P/D 引擎同机可比.
        self.req_perf: dict[str, dict] = {}

    def _update_prefiller_priority(self, server_idx: int):
        """Update the priority of a prefiller server in the heap."""
        server = self.prefillers[server_idx]
        # Priority based on active_tokens and active_kv_cache
        priority = server.active_tokens + server.active_kv_cache * 0.3
        # Remove old entry and add new one
        self.prefiller_heap = [(p, i, s) for p, i, s in self.prefiller_heap if i != server_idx]
        heapq.heappush(self.prefiller_heap, (priority, server_idx, server))  # type: ignore

    def _update_decoder_priority(self, server_idx: int):
        """Update the priority of a decoder server in the heap."""
        server = self.decoders[server_idx]
        priority = server.active_tokens
        # Remove old entry and add new one
        self.decoder_heap = [(p, i, s) for p, i, s in self.decoder_heap if i != server_idx]
        heapq.heappush(self.decoder_heap, (priority, server_idx, server))  # type: ignore

    def abort_prefiller_request(self, server_idx: int, request_id):  # Changed to synchronous
        """
        Mark a request as aborted. This will helps to release kv cache in
        prefiller node.
        """
        # No lock needed - atomic operation
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int):  # Changed to synchronous
        """
        Get the set of aborted requests and clear it.
        This is used to release kv cache in prefiller node.
        """
        # No lock needed - atomic operation
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_prefiller(self, token_count):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.prefiller_heap:
            raise RuntimeError("No prefiller servers available")

        priority, chosen, server = heapq.heappop(self.prefiller_heap)

        # Update the chosen server atomically
        self.prefillers[chosen].active_tokens += token_count
        self.prefillers[chosen].active_kv_cache += token_count

        # Update priority and re-add to heap
        self._update_prefiller_priority(chosen)

        return chosen

    def release_prefiller(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        self.prefillers[idx].active_tokens -= token_count
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if self.prefillers[idx].active_kv_cache > 0:
            self.prefillers[idx].active_kv_cache -= token_count
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def select_decoder(self, token_count):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.decoder_heap:
            raise RuntimeError("No decoder servers available")

        priority, chosen, server = heapq.heappop(self.decoder_heap)

        # Update the chosen server atomically
        self.decoders[chosen].active_tokens += token_count

        # Update priority and re-add to heap
        self._update_decoder_priority(chosen)

        return chosen

    def release_decoder(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        self.decoders[idx].active_tokens -= token_count
        # Update priority queue after releasing
        self._update_decoder_priority(idx)

    # Omni_infer's calculate_input_scores function
    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        input_score = length_score * 0.0345 + 120.0745
        return input_score

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length


proxy_state = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    # p_then_d: P 的 KV 传输身份. proxy 需要它在**发给 P 之前**就构造出"D → P
    # 回推参数"所需的字段, 从而能把 D 的派发提前到 P 响应之前(不等 P 的首 token),
    # 让 P 的逐层 KV 传输与前向重叠. 三个参数一一对应 prefiller 实例, 必须同时给:
    #   kv_port  = P 引擎 --kv-transfer-config 里的 kv_port
    #   tp_size  = P 的 tensor-parallel-size
    #   pcp_size = P 的 prefill_context_parallel_size(未开则为 1)
    # 端口推导与 P 侧 kv_transfer_params_zmq_port_base() 一致:
    #   base = kv_port + tp_size*pcp_size, 每个 rank 绑 base + pcp_rank*tp + tp_rank.
    parser.add_argument("--prefiller-kv-ports", type=int, nargs="+", default=None)
    parser.add_argument("--prefiller-tp-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--prefiller-pcp-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument(
        "--p-then-d",
        action="store_true",
        help="Enable the p_then_d flow: dispatch the request to the prefiller first (non-streaming, "
        "max_tokens=1), take the prefilled first token + connection info from its kv_transfer_params, "
        "then dispatch the decoder with those params and stream from it. Requires P_THEN_D=1 and "
        "REUSE_PREFILLED_TOKENS=1 on the prefiller (SKIP_DECODE_TOKENIZE=1 recommended on both) and "
        "REUSE_PREFILLED_TOKENS=1 on the decoder; the /v1/metaserver endpoint is unused in this mode.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    args = parser.parse_args()
    logger.info(
        "Decoder hosts will access Proxy host:port/metaserver, ensure that %s can access %s:%s/metaserver",
        set(args.decoder_hosts),
        args.host,
        args.port,
    )
    # Wildcard address is not allowed for layerwise connector
    if args.host in ["0.0.0.0", "::", "0:0:0:0:0:0:0:0"]:
        raise ValueError(
            f"Decoder hosts will access Proxy host:port/metaserver, to avoid configuration errors, "
            f"the Wildcard Address {args.host} is not allowed for proxy"
        )
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    # p_then_d 的 P 身份声明: 三个列表要么都不给(退回"等 P 响应"的老时序),
    # 要么给全且与 prefiller 实例数一致.
    ptd_lists = {
        "kv_ports": args.prefiller_kv_ports,
        "tp_sizes": args.prefiller_tp_sizes,
        "pcp_sizes": args.prefiller_pcp_sizes,
    }
    given = {name: values for name, values in ptd_lists.items() if values is not None}
    if given:
        if len(given) != len(ptd_lists):
            raise ValueError(
                f"--prefiller-kv-ports / --prefiller-tp-sizes / --prefiller-pcp-sizes must be given together; "
                f"missing: {sorted(set(ptd_lists) - set(given))}"
            )
        for name, values in given.items():
            if len(values) != len(args.prefiller_instances):
                raise ValueError(f"--prefiller-{name.replace('_', '-')} must have one value per prefiller instance")
    args.prefiller_meta = [
        (
            {
                "kv_port": args.prefiller_kv_ports[i],
                "tp_size": args.prefiller_tp_sizes[i],
                "pcp_size": args.prefiller_pcp_sizes[i],
            }
            if given
            else None
        )
        for i in range(len(args.prefiller_instances))
    ]
    if args.p_then_d and not given:
        logger.warning(
            "--p-then-d without --prefiller-kv-ports/--prefiller-tp-sizes/--prefiller-pcp-sizes: "
            "the decoder will only be dispatched after the prefiller's response (no prefill/transfer overlap)."
        )
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.prefiller_instances, global_args.decoder_instances, global_args.prefiller_meta)
    print(f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients.")
    yield
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()


async def listen_for_disconnect(request: Request) -> None:
    """Return if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_id: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    """给 P(prefill) 发一次非流式请求(max_tokens=1), 返回响应里的 kv_transfer_params.

    D 先派发的流程里只关心"P 已被触发"(返回的 params 为 None); p_then_d 流程里
    proxy 正好需要这份 params(prefilled_token/prompt_token_ids + P 的身份信息)
    原样转给 D。
    """
    proxy_state.acquire_aborted_prefiller_requests(prefiller_id)
    req_data = req_data.copy()
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    req_data["min_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            kv_transfer_params = response.json().get("kv_transfer_params")
            if request_id in proxy_state.req_id_future:
                result_future = proxy_state.req_id_future[request_id]
                result_future.set_result(kv_transfer_params)
            return kv_transfer_params
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, e)
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for %s.", max_retries, endpoint)
                raise last_exc


async def dispatch_prefiller_p_then_d(
    api: str, req_data: dict, request_id: str, request_length: int, prefiller_idx: int
) -> dict | None:
    """p_then_d: 选一个 P 并把请求发给它, 返回响应里的 kv_transfer_params.

    req_data 的 kv_transfer_params 必须是 {"do_remote_decode": True}(不带任何 D 参数)
    —— P 侧 connector 靠"没有 D 参数"判定这是 P 先派发的请求: 打 _p_then_d 标记、
    等 D 直连推参数后才逐层推 KV(块不再延迟释放, 见 connector 的 request_finished)。

    返回的 params 含 P 的身份(remote_host / remote_tp_size /
    kv_transfer_params_zmq_port)与首 token(prefilled_token / prompt_token_ids,
    entrypoint patch 打的)。proxy 不再用它派发 D —— D 的派发提前到 P 响应的**前面**
    (见 build_prefiller_fwd_params / deliver_prefiller_token), 这里只取首 token。

    注: 这里按现有 metaserver 路径的粒度释放 P 的负载计数(active_kv_cache 也一起
    释放), 单 P 部署无影响。
    """
    prefiller = proxy_state.prefillers[prefiller_idx]
    prefiller_score = proxy_state.calculate_prefill_scores(request_length)
    logger.debug("Prefiller score: %f", prefiller_score)
    # perf: 派发 P 的时刻(P 受理 ≈ 此后 + 网络/API/调度).
    st = proxy_state.req_perf.get(request_id)
    if st is not None:
        st["pf"] = time.perf_counter()
    try:
        return await send_request_to_service(
            prefiller.client,
            prefiller_idx,
            api,
            req_data,
            request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
        )
    finally:
        proxy_state.release_prefiller(prefiller_idx, prefiller_score)
        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)


def prefiller_fwd_params(prefiller: ServerState) -> dict | None:
    """p_then_d: 由命令行声明的 P 身份推出"D → P 回推参数"所需的字段.

    与 P 侧 `_p_then_d_send_params` 产出的那组字段同源、同端口公式(见 argparse 里
    --prefiller-kv-ports 的注释), 但不需要等 P 的 HTTP 响应 —— 这正是能把 D 的派发
    提前到 P 前向期间的关键。
    """
    meta = prefiller.kv_meta
    if meta is None:
        return None
    tp_size, pcp_size = meta["tp_size"], meta["pcp_size"]
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        # D 的推参数路径只读 host/port/tp/pcp 四项(缺一个就找不到 P 的参数通道);
        # engine_id 只用于 P 侧日志与兼容字段, 从命令行推导不出, 留空即可。
        "remote_host": prefiller.host,
        "remote_tp_size": tp_size,
        "remote_pcp_size": pcp_size,
        # base = kv_port + dp*pcp*tp + dp_rank*pcp*tp; D 侧按 base + rank_idx 逐 rank 推。
        "kv_transfer_params_zmq_port": meta["kv_port"] + tp_size * pcp_size,
        # D 侧据此把首 token 的记账推迟到 KV 到达时(见 recompute_scheduler);
        # kv_port 同时用来定位首 token 的 spool 目录。
        "p_then_d_token_reuse": True,
        "kv_port": meta["kv_port"],
    }


def build_first_token_chunk(api: str, request_id: str, model: str, text: str) -> bytes:
    """p_then_d 提前派发: 构造首 token 的 SSE chunk(与 vLLM 的流式格式一致).

    提前派发时 D 的请求是在 P 出首 token 之前下发的, D 侧 entrypoint 读不到
    prefilled_token, 也就不会用 _wrap_with_prefilled 合成首输出 —— 首 token 由
    proxy 直接下发。D 侧仍会把该 token 记进 output_token_ids(scheduler 在 KV 到达
    时补投), 所以它的流从第 2 个 token 开始, 不会重复。
    """
    chunk_id = f"chatcmpl-{request_id}" if api == "/chat/completions" else f"cmpl-{request_id}"
    choice = {"index": 0, "finish_reason": None, "stop_reason": None}
    if api == "/chat/completions":
        choice["delta"] = {"role": "assistant", "content": text}
    else:
        choice["text"] = text
        choice["token_ids"] = None
        choice["prompt_logprobs"] = None
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk" if api == "/chat/completions" else "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def deliver_prefiller_token(decoder: ServerState, request_id: str, params: dict) -> None:
    """p_then_d: 把 P 的首 token 补给已提前派发的 D.

    D 侧 entrypoint 在构造请求时读不到 prefilled_token(= 触发它走普通生成路径,
    也会丢掉"复用 P 首 token"的语义), 所以 token 改由这条通道异步送达, 由 D 在
    KV 接收完成的转型点注入(见 vllm_ascend/core/recompute_scheduler.py)。

    失败不阻断请求: D 会在自己的等待点超时后落回普通生成路径(首 token 会与 P 的
    重复), 打 error 便于定位。
    """
    payload = {
        "prefilled_token": params.get("prefilled_token"),
        "stop_reasons": params.get("stop_reasons"),
        # P 侧判断"这个 token 的文本能复用吗"(不完整 UTF-8 / EOS); 不可复用时 D
        # 也不该记账 —— 否则 D 会跳过这个位置, 而客户端并没有收到对应的文本。
        "reuse": bool(params.get("prefilled_text_reusable", True)),
        "kv_port": params.get("kv_port"),
    }
    url = f"{decoder.url}/p_then_d_token"
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    try:
        response = await decoder.client.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.error("Failed to deliver prefilled token for request %s to %s: %s", request_id, url, e)


def _log_task_failure(task: "asyncio.Task") -> None:
    """提前派发的 D 请求若在客户端开始收流前就失败, 至少留下一条日志."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("p_then_d: decoder stream task failed: %r", exc)


async def _pump_decoder_stream(agent: "AsyncIterator[bytes]", buffered: list[bytes], done: asyncio.Event) -> None:
    """把 D 的流搬进 buffer —— 由 create_task 立刻驱动, 从而让请求真正提前发出.

    见 `stream_service_response_with_retry` 是异步生成器: 不驱动它, HTTP 请求就不发。
    这里逐个 chunk 搬运并置 done 事件, 消费方(generate_stream)按缓冲区水位跟随。
    """
    try:
        async for chunk in agent:
            buffered.append(chunk)
    finally:
        done.set()


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return  # Success, exit after streaming
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, e)
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for streaming %s.", max_retries, endpoint)
                raise e
        except Exception as e:
            # If any chunk has been sent, do not retry, just log and drop
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error("Streaming to client interrupted after response started: %s", e)
                return
            else:
                if attempt < max_retries:
                    logger.warning("Attempt %s failed for streaming %s: %s", attempt, endpoint, e)
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                else:
                    logger.error("All %s attempts failed for streaming %s.", max_retries, endpoint)
                    raise e


def get_api_request_id(api, req_id):
    if api == "/completions":
        return "cmpl-" + req_id + "-0"
    elif api == "/chat/completions":
        return "chatcmpl-" + req_id


def get_origin_request_id(api, req_id):
    if api == "/completions":
        return req_id.replace("cmpl-", "")[:-2]
    elif api == "/chat/completions":
        return req_id.replace("chatcmpl-", "")


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        request_id = await proxy_state.next_req_id()
        request_id_api = get_api_request_id(api, request_id)
        proxy_state.req_perf[request_id] = {"in": time.perf_counter()}
        proxy_state.req_data_dict[request_id_api] = (copy.deepcopy(req_data), request_length, api)
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data
        prefiller_idx = None
        prefiller_params = None
        # p_then_d 提前派发: D 的流搬进这个 buffer, 由 create_task 立刻驱动(见下).
        decoder_chunks: list[bytes] = []
        decoder_done = asyncio.Event()
        decoder_task: asyncio.Task | None = None
        if global_args.p_then_d:
            # p_then_d: P 是第一跳(非流式, max_tokens=1), 首 token 与 P 的身份从它的
            # 响应里拿; D 先流程里"D 触发 remote prefill"那一步在这里变成了"proxy
            # 直接派发 P", 因此不再有 metaserver。
            prefiller_score = proxy_state.calculate_prefill_scores(request_length)
            prefiller_idx = proxy_state.select_prefiller(prefiller_score)
            # 提前派发只对流式有意义: 首 token 由 proxy 作为第一个 chunk 下发(见
            # build_first_token_chunk), 非流式没有 chunk 可发, 走老时序由 P 的响应
            # 带着 token 派发 D。
            fwd_params = prefiller_fwd_params(proxy_state.prefillers[prefiller_idx]) if stream_flag else None
            if fwd_params is not None:
                # 提前派发: D 一被派发就会把块表直连推给 P(见 connector 的
                # update_state_after_alloc), 于是 P 的发送线程在**前向期间**就能解析出
                # peer 映射并逐层推送 —— 传输与前向重叠, 不再等 P 的响应往返。
                # 首 token 此刻还不存在, 由 deliver_prefiller_token 在拿到后补给 D。
                #
                # 关键: task 必须在这里创建 —— 一旦下面 await 上 P 的响应, 事件循环就
                # 被占住, 直到 P 响应回来才会轮到它; 实测那样 D 的请求要晚 ~270 ms
                # (=P 的前向时长)才发出去, 正好把要重叠的那段又串行化了。
                # 请求体用快照: req_data 马上要装 P 的那份 kv_transfer_params.
                d_req_data = {**req_data, "kv_transfer_params": dict(fwd_params)}
                # perf: meta 在提前派发下的语义 = "派发 D"的时刻(旧时序里它是
                # "D 触发 remote prefill"). 图上把它当 D 的起点, 所以打在 task 创建处.
                st = proxy_state.req_perf.get(request_id)
                if st is not None:
                    st["meta"] = time.perf_counter()
                decoder_client = proxy_state.decoders[
                    proxy_state.select_decoder(proxy_state.calculate_decode_scores(request_length))
                ].client
                decoder_task = asyncio.create_task(
                    _pump_decoder_stream(
                        stream_service_response_with_retry(
                            decoder_client,
                            api,
                            d_req_data,
                            request_id=request_id,
                            max_retries=global_args.max_retries,
                            base_delay=global_args.retry_delay,
                        ),
                        decoder_chunks,
                        decoder_done,
                    )
                )
                decoder_task.add_done_callback(_log_task_failure)
                logger.debug(
                    "p_then_d: dispatched decoder %s (client->D %.1f ms)",
                    request_id,
                    1000 * (time.perf_counter() - proxy_state.req_perf[request_id]["in"]),
                )
            # 发给 P 的请求必须只带 do_remote_decode(P 靠"没有 D 参数"判定先派发),
            # 提前派发的那份参数要换回来 —— send_request_to_service 是浅拷贝, 嵌套的
            # kv_transfer_params 会被两跳共享, 不换回就会把 D 的语义带给 P。
            req_data["kv_transfer_params"] = {"do_remote_decode": True, "do_remote_prefill": False}
            prefiller_params = await dispatch_prefiller_p_then_d(
                api, req_data, request_id, request_length, prefiller_idx
            )
            if not prefiller_params:
                logger.error(
                    "Prefiller returned no kv_transfer_params for request %s; check P_THEN_D=1 / "
                    "REUSE_PREFILLED_TOKENS=1 on the prefiller engine.",
                    request_id,
                )
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": "prefiller returned no kv_transfer_params; check P_THEN_D=1 / "
                        "REUSE_PREFILLED_TOKENS=1 on the prefiller engine"
                    },
                )
            if fwd_params is None:
                # 未声明 P 身份: 回到老时序 —— 用 P 响应里的身份派发 D(不重叠).
                # perf: 对齐 D 先流程里 meta=D 触发 remote prefill 的位置 —— P 已出首 token.
                st = proxy_state.req_perf.get(request_id)
                if st is not None:
                    st["meta"] = time.perf_counter()
                req_data["kv_transfer_params"] = {
                    **prefiller_params,
                    "do_remote_prefill": True,
                    "do_remote_decode": False,
                }
            else:
                # 提前派发: 换回 D 的那份参数(上面为了派发 P 临时改过).
                req_data["kv_transfer_params"] = dict(fwd_params)
        else:
            req_data["kv_transfer_params"] = {
                "do_remote_decode": False,
                "do_remote_prefill": True,
                "metaserver": f"http://{global_args.host}:{global_args.port}/v1/metaserver",
            }
        # Select decoder
        decoder_score = proxy_state.calculate_decode_scores(request_length)
        logger.debug("Decoder score: %f", decoder_score)
        # Use the prefiller's kv_transfer_params to select decoder
        decoder_idx = proxy_state.select_decoder(decoder_score)
        decoder = proxy_state.decoders[decoder_idx]
        if fwd_params is not None and prefiller_params:
            # 提前派发场景: 首 token 一到位就补给 D(req_data 里发的是 fwd_params,
            # 没有 token)。放在 D 的 task 建好之后, 因为 D 要先起来监听
            # /v1/p_then_d_token; 也放在这里而不是 generate_stream 里, 是因为生成器
            # 要等客户端开始收流才被驱动, 那时 D 可能已经开始生成。
            # 老时序(fwd_params is None)不需要注入 —— token 本来就在 D 的请求里。
            await deliver_prefiller_token(decoder, request_id, prefiller_params)
        # logger.debug("Using %s %s", prefiller.url, decoder.url)
        # Stream response from decoder
        released_kv = False

        # Record request info for recompute
        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
            if isinstance(origin_prompt, list):
                origin_prompt = origin_prompt[0].get("text", "")
        else:
            origin_prompt = ""
        # refer to vLLM sampling_params: max_token default value
        origin_max_tokens = req_data.get("max_tokens", 16)
        # p_then_d 提前派发: P 算好的首 token 文本(见 _package_prefilled_text), 由
        # proxy 作为第一个 chunk 下发; P 判定不可复用(不完整 UTF-8/EOS)时为空, 此时
        # 不通知 D 记账, 客户端看到的首 token 由 D 自己的流给出。
        # 首轮由 P 的响应填充, recompute 重试时用新一轮的值追加下发。
        prefilled_text = ""
        if fwd_params is not None and prefiller_params.get("prefilled_text_reusable", True):
            prefilled_text = prefiller_params.get("prefilled_text") or ""

        async def iter_decoder_chunks(first_attempt: bool):
            """按"轮次"产出 D 的 chunk: 首轮来自提前派发的 task, 重试轮现场重发.

            首轮必须消费那个已启动的 task(而不是重新发请求), 否则每次迭代都会多一次
            派发; 重试轮在 `generate_stream` 里已经重跑过 P, 这里照旧现发。
            """
            if first_attempt and decoder_task is not None:
                pos = 0
                while True:
                    while pos < len(decoder_chunks):
                        chunk = decoder_chunks[pos]
                        pos += 1
                        yield chunk
                    if decoder_done.is_set():
                        break
                    await asyncio.sleep(0.001)
                # 把异常从 task 里取出来(如连接失败), 交给上面的重试逻辑.
                await decoder_task
            else:
                async for chunk in stream_service_response_with_retry(
                    decoder.client,
                    api,
                    req_data,
                    request_id=request_id,
                    max_retries=global_args.max_retries,
                    base_delay=global_args.retry_delay,
                ):
                    yield chunk

        async def generate_stream():
            nonlocal released_kv
            generated_token = ""
            released_kv = False
            nonlocal prefilled_text
            if global_args.p_then_d and prefiller_idx is not None:
                # perf: 提前派发时 meta = 真正发出 D 请求的时刻(老时序里它是
                # "D 触发 remote prefill"的时刻). 放在生成器体内是因为 D 的请求在
                # 迭代开始时才发出, 上一行 await deliver_prefiller_token 已先完成。
                st = proxy_state.req_perf.get(request_id)
                if st is not None:
                    st["meta"] = time.perf_counter()
            if prefilled_text:
                # 提前派发: 首 token 的 chunk 由 proxy 下发(原因见 build_first_token_chunk).
                # 放在这里产出、且在 D 的流之前, 保证顺序正确。
                # perf: 首 token 转发时刻 = 客户端可见的首 chunk 时刻.
                st = proxy_state.req_perf.get(request_id)
                if st is not None and "tok" not in st:
                    st["tok"] = time.perf_counter()
                yield build_first_token_chunk(api, request_id, req_data.get("model", "unknown"), prefilled_text)
                prefilled_text = ""
            retry_count = 0
            retry = True
            completion_tokens = 0
            # Only one await per chunk, minimal logic in loop
            try:
                while retry:
                    retry = False
                    if global_args.p_then_d and retry_count > 0:
                        # recompute 重试: prompt 里已追加了已生成内容, 必须整条重走
                        # "P 先派发" —— P 用新 prompt 重新 prefill 并把新 KV 推给同一个
                        # D, 再用新的首 token 重发 D 请求.
                        # 重试沿用与首轮相同的时序: 声明了 P 身份就提前派发 D(新配置的
                        # KV 与前向重叠), 首 token 到位后再注入; D 在 KV 到达的转型点
                        # 等新 token, 上一轮残留的那份会被覆盖.
                        retry_fwd_params = (
                            prefiller_fwd_params(proxy_state.prefillers[prefiller_idx])
                            if prefiller_idx is not None
                            else None
                        )
                        # 同首轮: 给 P 的请求只带 do_remote_decode, D 的那份参数换回来
                        # 再发(浅拷贝下嵌套 dict 共享).
                        req_data["kv_transfer_params"] = {"do_remote_decode": True, "do_remote_prefill": False}
                        retry_prefiller_params = await dispatch_prefiller_p_then_d(
                            api, req_data, request_id, request_length, prefiller_idx
                        )
                        if not retry_prefiller_params:
                            logger.error(
                                "Recompute retry for request %s: prefiller returned no kv_transfer_params, "
                                "dropping the retry.",
                                request_id,
                            )
                            break
                        if retry_fwd_params is not None:
                            # 换回 D 的那份参数(上面为了派发 P 临时改过).
                            req_data["kv_transfer_params"] = dict(retry_fwd_params)
                        await deliver_prefiller_token(decoder, request_id, retry_prefiller_params)
                        if retry_fwd_params is None:
                            req_data["kv_transfer_params"] = {
                                **retry_prefiller_params,
                                "do_remote_prefill": True,
                                "do_remote_decode": False,
                            }
                    async for chunk in iter_decoder_chunks(retry_count == 0):
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            logger.debug("Skipping chunk: %s", chunk)
                            yield chunk
                            continue
                        if not chunk_str:
                            continue
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: ") :]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            # if chunk is [done], skip it.
                            logger.debug("Skipping chunk: %s", chunk_str)
                            yield chunk
                            continue
                        choices = chunk_json.get("choices", [])
                        if not choices:
                            yield chunk
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        content = delta.get("content") or message.get("content") or choice.get("text") or ""
                        generated_token += content
                        # perf: 首个含文本的 chunk 时刻 = 首 token 可用(转发给客户端前)
                        if content and "tok" not in proxy_state.req_perf.get(request_id, {}):
                            proxy_state.req_perf[request_id]["tok"] = time.perf_counter()

                        stop_reason = choice.get("stop_reason")
                        usage = chunk_json.get("usage", {})
                        completion_tokens = (
                            (completion_tokens + 1)
                            if stream_flag
                            else (completion_tokens + usage.get("completion_tokens"))
                        )
                        if stop_reason == "recomputed":
                            retry = True
                            retry_count += 1
                            if chat_flag:
                                messages[0]["content"] = origin_prompt + generated_token
                            else:
                                req_data["prompt"] = origin_prompt + generated_token
                            req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                            break
                        if retry_count > 0 and not stream_flag:
                            if chat_flag:
                                choice["message"]["content"] = generated_token
                            else:
                                choice["text"] = generated_token
                            chunk = json.dumps(chunk_json).encode("utf-8")
                        yield chunk
            except Exception as e:
                logger.error(
                    "Error during streaming from decoder %s: %s the aborted request %s "
                    "will be routing to the target prefiller when new request is ready to dispatch to it",
                    decoder.url,
                    e,
                    request_id,
                )
            finally:
                # After streaming done, release tokens
                proxy_state.release_decoder(decoder_idx, decoder_score)
                # perf: 请求收尾时打印端到端分解时刻 (与引擎日志 uuid 同源).
                st = proxy_state.req_perf.pop(request_id, None)
                if st:

                    def _f(k: str) -> str:
                        return f"{st[k]:.6f}" if k in st else "-"

                    # nohup 下 vllm logger 行会缓冲, 改用 print(flush=True) 直出.
                    print(
                        f"[h2h][perf] proxy req={request_id} in={_f('in')} meta={_f('meta')} "
                        f"pf={_f('pf')} tok={_f('tok')} done={time.perf_counter():.6f}",
                        flush=True,
                    )

        if stream_flag:
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            return StreamingResponse(generate_stream(), media_type="application/json")
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders),
    }


@app.post("/reset_prefix_cache")
async def reset_prefix_cache(request: Request):
    params = dict(request.query_params)
    failures = []
    for client, base_url in [
        (s.client, f"http://{s.host}:{s.port}") for s in proxy_state.prefillers + proxy_state.decoders
    ]:
        try:
            resp = await client.post(f"{base_url}/reset_prefix_cache", params=params)
            resp.raise_for_status()
        except Exception as e:
            logger.error("reset_prefix_cache failed for %s: %s", base_url, e)
            failures.append(base_url)
    if failures:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"failed": failures})
    from fastapi.responses import Response as FastAPIResponse

    return FastAPIResponse(status_code=200)


@app.post("/v1/metaserver")
async def metaserver(request: Request):
    try:
        kv_transfer_params = await request.json()

        request_id = kv_transfer_params["request_id"]
        assert request_id in proxy_state.req_data_dict
        req_data, request_length, api = proxy_state.req_data_dict[request_id]
        request_id = get_origin_request_id(api, request_id)
        req_data["kv_transfer_params"] = kv_transfer_params
        # perf: D 触发 remote prefill 到达 proxy 的时刻.
        st = proxy_state.req_perf.get(request_id)
        if st is not None:
            st["meta"] = time.perf_counter()
        prefiller_score = proxy_state.calculate_prefill_scores(request_length)
        logger.debug("Request length: %s, Prefiller score: %s", request_length, prefiller_score)

        # Select prefiller
        prefiller_idx = proxy_state.select_prefiller(prefiller_score)
        prefiller = proxy_state.prefillers[prefiller_idx]
        logger.debug("Using prefill prefiller.url=%r req_data=%r", prefiller.url, req_data)
        # perf: 派发 P 的时刻 (P 受理 ≈ 此后 +网络/API/调度).
        st = proxy_state.req_perf.get(request_id)
        if st is not None:
            st["pf"] = time.perf_counter()
        # Send request to prefiller
        await send_request_to_service(
            prefiller.client,
            prefiller_idx,
            api,
            req_data,
            request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
        )

    except Exception as e:
        logger.error("Post metaserver failed with: %s", e)
    finally:
        proxy_state.release_prefiller(prefiller_idx, prefiller_score)
        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
