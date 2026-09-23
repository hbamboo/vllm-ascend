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
# Step 5: 先 P 后 D (p_then_d, 可选)
# ----------------------------------
# 默认流程是 D 先受理, 再由 D 通过 proxy 的 /v1/metaserver 触发 P 做 prefill。
# 加上 --p-then-d 则改成 P 先派发:
#
#   P 只做 prefill 并采样出第 1 个 token, 其 KV 逐层直接写进 D 的显存; D 收到
#   KV 后把首 token 当作 prompt 的最后一格重算 KV 并继续 decode, 因此不会重复
#   输出它。客户端的第 1 个 token 来自 P, 其余来自 D, proxy 把两段流拼接起来。
#
# 需要:
#   - P/D 引擎都设置 REUSE_PREFILLED_TOKENS=1 (流式请求才生效, 非流式由 D 自己
#     重算首 token), 并使用 MooncakeLayerwisePrefillThenDecodeConnector;
#   - 所有引擎设置相同的 kv_port 基准, 并给出每个 prefiller 的 params 通道端口
#     (P 引擎启动日志里的 kv_transfer_params_zmq_port) 与 tp_size:
#
#   python load_balance_proxy_layerwise_server_example.py \
#     --host 127.0.0.1 --port 9000 --p-then-d \
#     --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
#     --prefiller-kv-params-ports 14580 --prefiller-tp-sizes 8 \
#     --decoder-hosts 127.0.0.1 --decoder-ports 8200
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
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
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from vllm.logger import init_logger

logger = init_logger(__name__)

# perf: 与引擎侧同一套打点开关 + 同一套 JSON 行(权威实现在
# vllm_ascend.distributed.kv_transfer.utils.h2h_perf)。
# 刻意不复用那个模块: 它所在的包 import 时会注册 vllm_ascend 的 platform patch
# (连带 torch_npu), 而 proxy 是纯前端进程, 不该背这份启动开销与副作用。
# 需要 proxy.sh 也 export MC_TCP_PERF_LOG=1 才有输出。
_H2H_PERF_ON = os.getenv("MC_TCP_PERF_LOG", "0") == "1"


def _emit_proxy_perf(request_id: str, st: dict) -> None:
    """请求收尾时输出端到端分解时刻(与引擎日志 uuid 同源)。

    in=proxy 受理 / meta=D 触发 remote prefill / pf=派发 P / tok=首 token 转发,
    都是 CLOCK_MONOTONIC 绝对值, 与 P/D 引擎、前端、connector 的打点同一坐标系,
    因此时间线可以直接按 req(uuid) join 出端到端闭环。
    """
    if not _H2H_PERF_ON or not st:
        return
    rec = {"kind": "proxy", "role": "proxy", "req": request_id, "ts": st.get("in", time.perf_counter())}
    rec["wall"] = datetime.fromtimestamp(time.time() - time.perf_counter() + rec["ts"]).isoformat(
        timespec="milliseconds"
    )
    for key, value in st.items():
        # in 同时保留成字段(与 ts 同值): 下游解析器/时间线按 "in" 取受理时刻, 删掉
        # 会让 lane0 的"受理→P"段整段消失(实测踩过)。
        rec[key] = value
    rec["done"] = time.perf_counter()
    # nohup 下 vllm logger 行会缓冲, 用 print(flush=True) 直出.
    print(f"[h2h][perf] {json.dumps(rec, separators=(',', ':'))}", flush=True)

# Add uvloop for faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class ServerState:
    def __init__(self, host, port, kv_params_port=None, tp_size=1):
        self.host = host
        self.port = port
        # 先 P 后 D: P 引擎接收 D 侧 kv_transfer_params 的 ZMQ 基址
        # (引擎启动日志里的 kv_transfer_params_zmq_port, tp_rank=0 那个),
        # 以及 P 的 tp_size (D 需要按 rank 逐个投递)。仅 prefiller 需要。
        self.kv_params_port = kv_params_port
        self.tp_size = tp_size
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
    def __init__(self, prefiller_instances, decoder_instances):
        # parse_args 给出的是扁平四元组 (host, port, kv_params_port, tp_size)
        self.prefillers: list[ServerState] = [
            ServerState(host, port, kv_params_port=kv_params_port, tp_size=tp_size)
            for host, port, kv_params_port, tp_size in prefiller_instances
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
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    # ---- 先 P 后 D (p_then_d) ----
    parser.add_argument(
        "--p-then-d",
        action="store_true",
        help="先派发 P 再派发 D: P 的流式输出(首 token)先转发给客户端, D 复用该 token "
        "继续 decode。需要 P/D 引擎都实现 first token 投递, 且各引擎设置 "
        "REUSE_PREFILLED_TOKENS=1 (流式请求才生效)。",
    )
    parser.add_argument(
        "--prefiller-kv-params-ports",
        type=int,
        nargs="+",
        default=None,
        help="每个 prefiller 接收 D 侧 kv_transfer_params 的 ZMQ 基址 (即 P 引擎启动日志里的 "
        "kv_transfer_params_zmq_port)。开启 --p-then-d 时必须与 --prefiller-hosts 一一对应。",
    )
    parser.add_argument(
        "--prefiller-tp-sizes",
        type=int,
        nargs="+",
        default=None,
        help="每个 prefiller 的 tensor_parallel_size。开启 --p-then-d 时必须与 --prefiller-hosts 一一对应。",
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
    if args.p_then_d:
        if args.prefiller_kv_params_ports is None or args.prefiller_tp_sizes is None:
            raise ValueError("--p-then-d 需要同时给出 --prefiller-kv-params-ports 与 --prefiller-tp-sizes")
        if len(args.prefiller_kv_params_ports) != len(args.prefiller_hosts):
            raise ValueError("Number of prefiller kv params ports must match number of prefiller hosts")
        if len(args.prefiller_tp_sizes) != len(args.prefiller_hosts):
            raise ValueError("Number of prefiller tp sizes must match number of prefiller hosts")
    else:
        args.prefiller_kv_params_ports = [None] * len(args.prefiller_hosts)
        args.prefiller_tp_sizes = [1] * len(args.prefiller_hosts)
    args.prefiller_instances = [
        (host, port, kv_ports, tp)
        for host, port, kv_ports, tp in zip(
            args.prefiller_hosts, args.prefiller_ports, args.prefiller_kv_params_ports, args.prefiller_tp_sizes
        )
    ]
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.prefiller_instances, global_args.decoder_instances)
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
            if request_id in proxy_state.req_id_future:
                result_future = proxy_state.req_id_future[request_id]
                result_future.set_result(response.json()["kv_transfer_params"])
            return
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning("Attempt %s failed for %s: %s", attempt, endpoint, e)
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error("All %s attempts failed for %s.", max_retries, endpoint)
                raise last_exc


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


def _first_token_event_to_forward(event: bytes) -> bytes | None:
    """P 侧的输出只保留"带内容"的 chunk(role + 首 token), 并把收尾字段清掉。

    P 的 max_tokens=1, 首 token 与 finish_reason 常常在**同一个** chunk 里(末 chunk
    就是带 finish_reason 的那个)——所以不能整块丢: 那会把首 token 一起丢掉, 客户端
    就再也看不到 P 出的首 token 了。这里保留 chunk、只把 finish_reason/usage 清空,
    客户端才不会以为流已结束、看不到 D 续写的部分。usage/[DONE] 直接丢。

    重新序列化时用 ensure_ascii=False: json.dumps 默认会把非 ASCII 转成 `\\uXXXX`
    (中文首 token 会变成 `"\\u55ef"`), 与 vLLM 自己发的 chunk(D 的 chunk 是原样透传)
    风格不一致 —— 语义等价, 但对着 curl 看很容易被当成乱码。
    """
    text = event.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return None
    payload = text[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        chunk_json = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if chunk_json.get("usage"):
        return None
    keep = False
    for choice in chunk_json.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("role") or choice.get("text"):
            keep = True
        choice["finish_reason"] = None
        if "stop_reason" in choice:
            choice["stop_reason"] = None
    if not keep:
        return None
    return b"data: " + json.dumps(chunk_json, ensure_ascii=False).encode("utf-8")


async def _handle_p_then_d(
    api: str,
    req_data: dict,
    request_id: str,
    request_length: int,
    stream_flag: bool,
    chat_flag: bool,
    origin_prompt: str,
    origin_max_tokens: int,
):
    """先 P 后 D: 先派发 P, 再把 P 的首 token 与 D 的后续输出拼成一个流返回。

    P 只算 prefill 并采样出第 1 个 token, 它的 KV 逐层直接写进 D 的显存; D 侧
    收到 KV 后把首 token 当作 prompt 的最后一格重算 KV, 因此不会重复输出它 ——
    客户端看到的第 1 个 token 来自 P, 其余来自 D。非流式请求无法拼接两段响应,
    此时 D 会自己重算首 token (reuse_prefilled_tokens=False), P 的输出丢弃。
    """
    prefiller_score = proxy_state.calculate_prefill_scores(request_length)
    prefiller_idx = proxy_state.select_prefiller(prefiller_score)
    prefiller = proxy_state.prefillers[prefiller_idx]
    decoder_score = proxy_state.calculate_decode_scores(request_length)
    decoder_idx = proxy_state.select_decoder(decoder_score)
    decoder = proxy_state.decoders[decoder_idx]

    # P 侧: do_remote_decode=True 让 P 把本请求登记为"待发送", 并等待 D 的
    # block table 经 kv_transfer_params channel 直投过来 (不走 metaserver)。
    req_data_p = {k: v for k, v in req_data.items() if k != "kv_transfer_params"}
    req_data_p["kv_transfer_params"] = {"do_remote_decode": True}
    req_data_p["stream"] = stream_flag
    req_data_p["max_tokens"] = 1
    req_data_p["min_tokens"] = 1
    if "max_completion_tokens" in req_data_p:
        req_data_p["max_completion_tokens"] = 1
    req_data_p.pop("stream_options", None)

    # D 侧: do_remote_prefill=True 触发拉取; kv_transfer_params_zmq_port /
    # remote_tp_size 告诉 D 把 block table 投给哪个 P (逐个 tp rank)。
    req_data_d = copy.deepcopy(req_data)
    req_data_d["kv_transfer_params"] = {
        "do_remote_prefill": True,
        # D 要经专线把 block table 投给 P 的每个 tp rank: 基址 + 目标主机
        "kv_transfer_params_zmq_port": prefiller.kv_params_port,
        "remote_tp_size": prefiller.tp_size,
        "remote_host": prefiller.host,
    }

    async def _pump(gen, out_queue):
        try:
            async for chunk in gen:
                await out_queue.put(chunk)
        except Exception as e:
            # 交给消费端统一处理 (与 D-first 路径的报错方式保持一致)
            await out_queue.put(e)
        finally:
            await out_queue.put(None)

    async def _decoder_chunks():
        """消费 D 的流, 保留 recompute 重试语义 (与 D-first 路径一致)。"""
        generated_token = ""
        retry_count = 0
        retry = True
        completion_tokens = 0
        while retry:
            retry = False
            async for chunk in stream_service_response_with_retry(
                decoder.client,
                api,
                req_data_d,
                request_id=request_id,
                max_retries=global_args.max_retries,
                base_delay=global_args.retry_delay,
            ):
                try:
                    chunk_str = chunk.decode("utf-8").strip()
                except UnicodeDecodeError:
                    yield chunk
                    continue
                if not chunk_str:
                    continue
                if chunk_str.startswith("data: "):
                    chunk_str = chunk_str[len("data: ") :]
                try:
                    chunk_json = json.loads(chunk_str)
                except json.JSONDecodeError:
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
                usage = chunk_json.get("usage", {})
                completion_tokens = (
                    (completion_tokens + 1)
                    if stream_flag
                    else (completion_tokens + (usage.get("completion_tokens") or 0))
                )
                if choice.get("stop_reason") == "recomputed":
                    retry = True
                    retry_count += 1
                    if chat_flag:
                        messages = req_data_d["messages"]
                        messages[0]["content"] = origin_prompt + generated_token
                    else:
                        req_data_d["prompt"] = origin_prompt + generated_token
                    req_data_d["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                    break
                if retry_count > 0 and not stream_flag:
                    if chat_flag:
                        choice["message"]["content"] = generated_token
                    else:
                        choice["text"] = generated_token
                    chunk = json.dumps(chunk_json).encode("utf-8")
                yield chunk

    async def generate_stream():
        decoder_queue: asyncio.Queue = asyncio.Queue()
        decoder_task = asyncio.create_task(_pump(_decoder_chunks(), decoder_queue))
        prefiller_task = None
        try:
            # 先把 P 放出去 (它一受理就开始 prefill), 再转发 D 的输出。
            # 流式: P 的首 token 直接转发; 非流式: P 的输出丢弃, D 会自己算。
            if stream_flag:
                proxy_state.req_perf.setdefault(request_id, {})["pf"] = time.perf_counter()
                prefiller_queue: asyncio.Queue = asyncio.Queue()
                prefiller_task = asyncio.create_task(
                    _pump(
                        stream_service_response_with_retry(
                            prefiller.client,
                            api,
                            req_data_p,
                            request_id=request_id,
                            max_retries=global_args.max_retries,
                            base_delay=global_args.retry_delay,
                        ),
                        prefiller_queue,
                    )
                )
                # 按 SSE 事件边界切整齐再挑: 只转发 P 的首 token (含 role),
                # 清掉 finish_reason/usage/[DONE], 否则客户端会提前收流。
                pending = b""
                while True:
                    item = await prefiller_queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        logger.warning("Prefiller stream failed for %s: %s", request_id, item)
                        break
                    pending += item
                    while b"\n\n" in pending:
                        event, pending = pending.split(b"\n\n", 1)
                        forward = _first_token_event_to_forward(event)
                        if forward is None:
                            continue
                        if "tok" not in proxy_state.req_perf.get(request_id, {}):
                            proxy_state.req_perf[request_id]["tok"] = time.perf_counter()
                        yield forward + b"\n\n"
                if pending.strip():
                    forward = _first_token_event_to_forward(pending)
                    if forward is not None:
                        yield forward
            else:
                # 非流式: 不需要转发 P 的输出, 但**必须**和 D 并发派发 —— P 在
                # 自己的前向里等 D 投来 block table, 若先 await P 再派发 D 就成
                # 了互相等待。这里只把 P 发出去, 完成与否不阻塞 D。
                proxy_state.req_perf.setdefault(request_id, {})["pf"] = time.perf_counter()
                prefiller_task = asyncio.create_task(
                    send_request_to_service(
                        prefiller.client,
                        prefiller_idx,
                        api,
                        req_data_p,
                        request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    )
                )

                def _log_prefiller_failure(task):
                    if not task.cancelled() and task.exception() is not None:
                        logger.error("Prefiller request failed for %s: %s", request_id, task.exception())

                prefiller_task.add_done_callback(_log_prefiller_failure)

            while True:
                item = await decoder_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            for task in (prefiller_task, decoder_task):
                if task is not None and not task.done():
                    task.cancel()
            proxy_state.release_prefiller(prefiller_idx, prefiller_score)
            proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
            proxy_state.release_decoder(decoder_idx, decoder_score)
            st = proxy_state.req_perf.pop(request_id, None)
            _emit_proxy_perf(request_id, st)

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream" if stream_flag else "application/json",
    )


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        request_id = await proxy_state.next_req_id()
        request_id_api = get_api_request_id(api, request_id)
        proxy_state.req_perf[request_id] = {"in": time.perf_counter()}
        proxy_state.req_data_dict[request_id_api] = (copy.deepcopy(req_data), request_length, api)
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
        # logger.debug("Using %s %s", prefiller.url, decoder.url)
        # Stream response from decoder
        released_kv = False

        # Record request info for recompute
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data
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

        if global_args.p_then_d:
            # 先 P 后 D: P 先受理并把首 token 流回来, D 复用该 token 续跑。
            return await _handle_p_then_d(
                api=api,
                req_data=req_data,
                request_id=request_id,
                request_length=request_length,
                stream_flag=stream_flag,
                chat_flag=chat_flag,
                origin_prompt=origin_prompt,
                origin_max_tokens=origin_max_tokens,
            )

        async def generate_stream():
            nonlocal released_kv
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0
            # Only one await per chunk, minimal logic in loop
            try:
                while retry:
                    retry = False
                    async for chunk in stream_service_response_with_retry(
                        decoder.client,
                        api,
                        req_data,
                        request_id=request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    ):
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
                _emit_proxy_perf(request_id, st)

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
