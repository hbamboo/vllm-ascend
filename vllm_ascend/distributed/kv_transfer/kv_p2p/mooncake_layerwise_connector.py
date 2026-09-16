# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import hashlib
import json
import logging
import math
import os
import queue
import re
import struct
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import numpy.typing as npt
import torch
import torch_npu
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    get_world_group,
)
from vllm.logger import logger
from vllm.utils.math_utils import round_down
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import extract_layer_index

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import GET_META_MSG
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    bind_current_thread_to_idle_cpu,
    global_te,
)
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    RegisterRegions,
    align_memory,
    collect_storage_merged_register_regions,
    context_parallel_parameters_check,
    get_cp_group,
    get_local_remote_block_port_mappings,
    get_transfer_mappings,
    get_transfer_timeout_value,
    kv_alltoall_and_rearrange,
    parallel_info,
    validate_register_region_count,
)
from vllm_ascend.distributed.utils import get_decode_context_model_parallel_rank
from vllm_ascend.utils import npu_stream_switch, trans_nd_to_nz

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

DONE_SENDING_MSG = b"done_sending_msg"
FAILED_SENDING_MSG = b"failed_sending_msg"
LAYER_DONE_SENDING_MSG = b"layer_done_sending_msg"
# D→P 参数直连通道(p_then_d): 载荷 (KV_TRANSFER_PARAMS_MSG, external_req_id, params).
KV_TRANSFER_PARAMS_MSG = b"kv_transfer_params_msg"

# 逐层 LAYER_DONE 握手使用独立连接, 需覆盖 D 侧 H2D 时间.
LAYER_DONE_TIMEOUT_S = 30.0

# p_then_d (envs.P_THEN_D): P 侧等 D 参数的超时与参数缓存过期时间(秒).
# 与参考实现(预填充后解码版 connector)的 prefill extra config 同名, 但等待超时
# 取更小的默认值: 发送线程是串行的, 等一个请求的参数会阻塞其他请求的传输, 正常
# 情况下参数在 P 响应回 proxy → D 派发后毫秒级就到.
DEFAULT_WAIT_TRANSFER_PARAMS_TIMEOUT_SEC = 30
DEFAULT_PREFILL_TRANSFER_PARAMS_EXPIRE_SEC = 300
# 发送线程轮询等待 D 参数的检查间隔: 比一次长等待更容易打点/观测, 也便于在
# 超时前先处理已完成 write 的收尾(pipe 路径).
WAIT_TRANSFER_PARAMS_POLL_S = 0.005

# p_then_d: 首 token 注入通道(spool 文件). proxy 提前派发 D 后, P 的首 token 由它
# 经由 D 的 API server 写入本机 spool, D 的 EngineCore 在 KV 接收完成的转型点取用
# (见 recompute_scheduler._update_waiting_for_remote_kv). 用文件而不是 socket 的原因:
# D 可能是 dp>1, 请求落在哪个 dp 组由 D 内部路由决定, proxy 无法按端口寻址到持有
# 该请求的引擎; 而 spool 是共享的, 每个 rank 只认领自己等待的 req_id, 天然免疫错投.
PTD_TOKEN_SPOOL_ROOT = os.getenv("PTD_TOKEN_SPOOL_ROOT", "/tmp/vllm_ptd_tokens")
# 认领/清理的轮询间隔与条目过期时间.
PTD_TOKEN_SPOOL_POLL_S = 0.005
PTD_TOKEN_SPOOL_EXPIRE_S = 300.0
# 缓存条目上限(按插入顺序淘汰, 长跑不涨内存).
PTD_TOKEN_SPOOL_MAX_ENTRIES = 4096
# vLLM 给 EngineCore request_id 追加的序号后缀形如 "-123456"(长度为 9), 见
# get_external_request_id. 写侧(API server 的 header)与读侧(scheduler)用同一套
# 规范化, 保证文件名一致.
_ENGINE_CORE_SUFFIX_RE = re.compile(r"-\d{5}$")


def _spool_key(request_id: str) -> str:
    """把 API 侧或 EngineCore 侧的 request_id 规范化成同一个 spool 文件名."""
    return _ENGINE_CORE_SUFFIX_RE.sub("", request_id)


# 攒批层数: 模型完成 N 层任务后才统一做 D2H flush + 一次传输 + 一次
# LAYER_DONE, 摊薄逐层传输的固定开销(批量 DMA / write 调用 / 控制消息).
# 1 = 逐层传输(旧行为); 层序号断裂(跨步/换批)或到达最后层时立即冲刷.
_LAYER_BATCH = int(os.getenv("MC_TCP_LAYER_BATCH", "8"))

# 性能观测开关: 1=开启后打印 [mooncake][perf] 阶段耗时日志(每批/每请求一条),
# 用于定位 H2H 通路相对 D2D 的 TTFT 增量来源(攒批等待/事件/D2H/写/LAYER_DONE/H2D).
_PERF_LOG = os.getenv("MC_TCP_PERF_LOG", "0") == "1"

# H2H 写线程流水 A/B (仅 protocol=tcp 且非 reshard/量化路径): 1=发送线程只做
# "等事件 + D2H flush", write 交给独立写线程连续占满数据面, LAYER_DONE 在对应
# write 完成后按序发出(与后续批的 write / flush 重叠). 关闭时保持单线程原语义.
_PIPE_WRITER = os.getenv("MC_TCP_PIPE_WRITER", "0") == "1"
# 写队列深度(允许在飞的 write 批数上限, 兼作背压).
_PIPE_DEPTH = int(os.getenv("MC_TCP_PIPE_DEPTH", "2"))


def kv_transfer_params_zmq_port_base(vllm_config: VllmConfig) -> int:
    """参数直连通道端口基址(p_then_d).

    取 side-channel 端口段之后的整段(每 dp/pcp 分组 pcp_size*tp_size 个端口),
    与 handshake 端口段(kv_port .. kv_port+dp*pcp*tp-1)不重叠. 这里只用
    vllm_config 里的并行规模(不依赖分布式 group), 保证 Scheduler 侧(EngineCore)
    也能算出同一个值; 每个 rank 实际绑的端口 = base + pcp_rank*tp_size + tp_rank,
    见 MooncakeLayerwiseConnectorWorker 里的绑定.
    """
    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    tp_size = parallel_config.tensor_parallel_size
    return (
        vllm_config.kv_transfer_config.kv_port
        + parallel_config.data_parallel_size * pcp_size * tp_size
        + parallel_config.data_parallel_rank * pcp_size * tp_size
    )


def _perf_ms(t0: float) -> float:
    """perf_counter 差值的毫秒数."""
    return (time.perf_counter() - t0) * 1e3


@dataclass
class LayerMetadata:
    tensor_group_idx: list[int]
    kv_caches_base_addr: list[int]
    block_len: list[int]
    block_size_scale: list[int]


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    te_rpc_port: int
    layer_metadata: dict[str, LayerMetadata]


@dataclass
class ReqMeta:
    local_block_ids: list[list[int]]
    token_ids: list[int] | None
    # Not None if layer-wise is disabled
    remote_block_ids: list[list[int]]
    remote_block_size: list[list[int]]
    remote_engine_id: str | None
    remote_host: str | None
    remote_port: int | None
    remote_te_rpc_port: int | None
    remote_layer_metadata: dict[str, LayerMetadata] | None
    metaserver: str | None
    remote_tp_size: int | None
    remote_pcp_size: int | None
    remote_dcp_size: int | None
    chunk_finish: bool = False
    prompt_len: int = 0
    trans_count: list[int] | None = None
    remote_cache_tokens: int = 0
    local_computed_tokens: int = 0
    local_transed_tokens: int = 0
    do_virtual: bool = False
    # 多 peer: 同一请求的不同 kv cache group 可能落在不同 D rank 上
    # (hybrid 模型里 attn 组按 head/cp-group 选 peer, mamba 组按连续分片选
    # peer, 不等 TP 下两者可能不一致). 键为 (host, port), 值为该 peer 的
    # 按 group 索引的 block ids / 期望 DONE 数.
    peer_transfer: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    # (host, port) -> 对端各层元数据 / TE rpc 端口
    peer_layer_metadata: dict[tuple[str, int], dict[str, LayerMetadata]] = field(default_factory=dict)
    peer_te_rpc_port: dict[tuple[str, int], int] = field(default_factory=dict)
    # p_then_d: 请求在 P 侧先于 D 派发, 到达时请求里没有 D 参数(块表/侧信道地址),
    # D 会通过直连参数通道事后推来 -> 本 ReqMeta 的 peer 解析推迟到发送线程.
    # 注意 "peer_transfer 为空" 不等于 pending(cp 分组外的 rank 正常就是空),
    # 必须用本显式标记.
    awaiting_params: bool = False
    # 延迟释放标记. **p_then_d 下恒为 False**: D 由 proxy 提前派发, 发送线程不再
    # 长时间等参数, request_finished 也不再延迟释放块 —— 于是发送完成的收尾
    # (add_done_sending_request → finished_sending)整条路径不再启用, 那正是
    # worker.get_finished 里被丢弃、从来没能真正释放过块的那条路(块泄漏的根因).
    # 保留字段是因为完成判据与发送路径共用它, 置 False 即可整体停用.
    delayed_free: bool = False
    # 该请求在本 step metadata 里的序号: cp 分组选择(cp_ratio>1)会用到, 延迟解析
    # 时无法从迭代顺序重新推得, 故随 ReqMeta 携带.
    req_idx: int = 0


@dataclass
class SendTask:
    send_request: dict[str, ReqMeta] = field(default_factory=dict)
    # pd_head_ratio == 1 use
    wait_event: torch.npu.Event | None = None
    # pd_head_ratio > 1 use
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None
    # kv cache quantization layer use
    k_quant_cache: torch.Tensor | None = None
    v_quant_cache: torch.Tensor | None = None
    layer_idx: int = 0
    layer_name: str = ""
    # trans block info
    group_rearrange_block_ids: list[list[int]] | None = None
    group_num_blocks: list[int] | None = None
    group_num_tokens: list[int] | None = None
    group_block_table: list[torch.Tensor | None] | None = None
    group_block_len_tensor: list[torch.Tensor | None] | None = None
    group_seq_start_tensor: list[torch.Tensor | None] | None = None


@dataclass
class TransferMeta:
    src: list[int]
    dst: list[int]
    length: list[int]
    req_ids: list[str]
    # req_id -> [(start, count), ...] 在 src/dst/length 中的区间列表(跨层攒批时
    # 同一请求在合并列表中有多个不连续段), 用于按请求拆分 LAYER_DONE 通知.
    req_slices: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # req_id -> (remote_host, remote_port): LAYER_DONE 的控制面目的地(侧信道),
    # 攒批合并后不再能从单一任务取到, 在聚合阶段记录.
    req_peer: dict[str, tuple[str, int]] = field(default_factory=dict)
    # (peer_host, peer_port, req_id) -> (is_last, trans_count): 只在含最后层的
    # 批里记录 —— 该 peer 在该请求末轮应回的"请求级完成信息", 随本批 LAYER_DONE
    # 一起下发(D 侧 H2D 成功后本地判定完成, 不再单发一次请求级 DONE 往返).
    req_done: dict[tuple[str, int, str], tuple[bool, int]] = field(default_factory=dict)


@dataclass
class SendReqInfo:
    local_block_ids: list[list[int]]
    local_transferred_tokens: int
    local_computed_tokens: int
    request: "Request"

    def extend_local_block_ids(self, new_block_ids: list[list[int]]) -> None:
        """extend local block ids for this step"""
        for i, new_block_id in enumerate(new_block_ids):
            self.local_block_ids[i].extend(new_block_id)

    def update_computed_tokens(self, computed_tokens: int) -> None:
        """update local computen tokens for this step"""
        self.local_computed_tokens = computed_tokens

    def update_transferred_tokens(self, transferred_tokens: int) -> None:
        """update transferred tokens for this step"""
        self.local_transferred_tokens = transferred_tokens

    def unpack(self):
        return (
            self.local_block_ids,
            self.local_transferred_tokens,
            self.local_computed_tokens,
            self.request,
        )


@dataclass
class SizedDict(OrderedDict):
    def __init__(self, max_size=16000, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            value: dict[int, list[int]] = {}
            self[key] = value
            return value


@dataclass
class _PipeJob:
    """写线程流水 (MC_TCP_PIPE_WRITER=1) 的一批任务.

    发送线程填充前段(事件等待/flush)后入写队列; 写线程完成 write 后填充
    write 计时/失败信息并入完成队列; 发送线程在 drain 阶段按序发 LAYER_DONE
    (含请求级完成信息)并做 perf 收尾.
    """

    batch_id: int
    tasks: list[Any]
    sessions: list[tuple[str, "TransferMeta"]]
    transferred_reqs: set[str]
    contains_last: bool
    t_batch0: float
    batch_wait_ms: float
    event_ms: float
    flush_ms: float
    flush_win0: float | None
    flush_win1: float | None
    # 以下由写线程填充:
    write_ms: float = 0.0
    write_win0: float | None = None
    write_win1: float | None = None
    # session_id -> 该 session 写失败涉及的 req ids (与单线程版 ret<0 语义一致).
    failed: dict[str, list[str]] = field(default_factory=dict)


class KVTransferParamsRecvingThread(threading.Thread):
    """D→P 参数直连通道 (p_then_d): 接收 D 侧推来的 kv_transfer_params.

    P 是第一跳(proxy 先发 P, P 生成首 token 后 proxy 才派发 D), P 前向时 D 还
    不存在, 逐层传输所需的 D 参数(块表/侧信道地址/TP 规模/远端缓存命中数)只能
    由 D 事后推给 P. 本线程每个 TP rank 一个, 绑
    ``kv_port + dp_size*pcp_size*tp_size + dp_rank*pcp_size*tp_size
    + pcp_rank*tp_size + tp_rank``(紧邻 side-channel 端口段之后, 不重叠);
    D 侧按响应里的基址 + rank 序号逐个推.

    收到即回 ACK(D 侧按 ACK 判达; 失败要当请求级失败, 否则没收到的 rank 会一直
    等参数 —— D 侧没有请求级超时). 参数按 external request id 缓存, 发送线程在
    需要时取(见 KVCacheSendingLayerThread.resolve_pending_req), 请求发送完成后
    由 worker 弹出; 超期未用则丢弃, 避免长时间挂死请求把参数/内存留下.
    """

    def __init__(
        self,
        kv_transfer_params_zmq_port: int,
        ready_event: threading.Event,
        timeout: float,
        wait_transfer_params_timeout_sec: float,
        prefill_transfer_params_expire_sec: float,
    ):
        super().__init__(daemon=True, name="KVTransferParamsRecvingThread")
        self.side_channel_host = get_ip()
        self.kv_transfer_params_zmq_port = kv_transfer_params_zmq_port
        self.lock = threading.Lock()
        self.ready_event = ready_event
        self.timeout = timeout
        self.wait_transfer_params_timeout_sec = wait_transfer_params_timeout_sec
        self.prefill_transfer_params_expire_sec = prefill_transfer_params_expire_sec
        # external req id -> (params, expire_time)
        self.kv_transfer_params: dict[str, tuple[dict[str, Any], float]] = {}
        self._min_expire_time = 0.0

    def run(self):
        path = make_zmq_path("tcp", self.side_channel_host, self.kv_transfer_params_zmq_port)
        logger.info("KVTransferParamsRecvingThread listening on %s", path)
        decoder = msgspec.msgpack.Decoder(type=tuple)
        with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
            self.ready_event.set()
            sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))  # type: ignore
            while True:
                frames = None
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        logger.error("Invalid message format: %s", frames)
                        continue
                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        logger.error("Invalid message format: %s", frames)
                        continue
                    msg = decoder.decode(payload[0])
                    if msg[0] != KV_TRANSFER_PARAMS_MSG:
                        logger.error("KV transfer params thread got unexpected message %s", msg[0])
                        continue
                    external_req_id = msg[1]
                    params = msg[2]
                    logger.info(
                        "Received decode transfer params for request %s: remote_host=%s remote_port=%s "
                        "remote_tp_size=%s remote_block_ids=%s remote_cached_tokens=%s",
                        external_req_id,
                        params.get("remote_host"),
                        params.get("remote_port"),
                        params.get("remote_tp_size"),
                        params.get("remote_block_ids"),
                        params.get("remote_cached_tokens"),
                    )
                    self.set_kv_transfer_params(external_req_id, params)
                    sock.send_multipart((identity, b"", b"ACK"))
                except Exception as e:
                    logger.error("Failed to handle kv transfer params message: %s. frames=%s", e, frames)

    def set_kv_transfer_params(self, req_id: str, params: dict[str, Any]) -> None:
        current_time = time.time()
        expire_time = current_time + self.prefill_transfer_params_expire_sec
        with self.lock:
            self.kv_transfer_params[req_id] = (params, expire_time)
            if self._min_expire_time == 0.0 or self._min_expire_time > expire_time:
                self._min_expire_time = expire_time
            self._clear_expired_locked(current_time)

    def _clear_expired_locked(self, current_time: float) -> None:
        """清扫过期参数(仅在持锁时调用, 用 min_expire_time 减少扫描次数)."""
        if current_time < self._min_expire_time:
            return
        min_expire_time = 0.0
        for req_id, (_params, expire_time) in list(self.kv_transfer_params.items()):
            if current_time >= expire_time:
                logger.warning(
                    "KV transfer params for request %s expired after %ss",
                    req_id,
                    self.prefill_transfer_params_expire_sec,
                )
                self.kv_transfer_params.pop(req_id, None)
                continue
            if min_expire_time == 0.0 or min_expire_time > expire_time:
                min_expire_time = expire_time
        self._min_expire_time = min_expire_time

    def wait_for_kv_transfer_params(self, req_id: str, timeout: float | None = None) -> dict[str, Any] | None:
        """等某个请求的 D 参数; 超时返回 None(调用方决定失败语义).

        只在发送线程调用(会阻塞). 返回的是缓存里的同一份 dict 引用, 调用方
        合并进 ReqMeta 时不要再改它.
        """
        external_req_id = get_external_request_id(req_id)
        deadline = time.perf_counter() + (self.wait_transfer_params_timeout_sec if timeout is None else timeout)
        logged = False
        while True:
            current_time = time.time()
            with self.lock:
                self._clear_expired_locked(current_time)
                entry = self.kv_transfer_params.get(external_req_id)
            if entry is not None:
                return entry[0]
            now = time.perf_counter()
            if now >= deadline:
                return None
            if not logged:
                logged = True
                logger.info(
                    "Waiting for decode transfer params of request %s (timeout=%.1fs); "
                    "D has not been dispatched yet (p_then_d flow).",
                    external_req_id,
                    deadline - now,
                )
            time.sleep(WAIT_TRANSFER_PARAMS_POLL_S)

    def pop_kv_transfer_params(self, req_id: str) -> None:
        external_req_id = get_external_request_id(req_id)
        with self.lock:
            self.kv_transfer_params.pop(external_req_id, None)


class KVCacheSendingLayerThread(threading.Thread):
    def __init__(
        self,
        engine: TransferEngine,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        kv_cache_specs: list[KVCacheSpec],
        attn_resharding_group_idx: set,
        total_layers: int,
        ready_event: threading.Event,
        tp_size: int,
        tp_rank: int,
        pd_head_ratio: int,
        num_head_replica: int,
        layer_metadata: dict[str, LayerMetadata],
        group_max_layer_idx: dict[int, int],
        use_mla: bool,
        use_attn_mamba_hybrid: bool,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        enable_kv_quant: bool,
        enable_c8_quant: bool,
        resharding_stream: torch.npu.Stream,
        sender_path: str,
        resolve_pending_req: Callable[[str, Any], bool] | None = None,
        add_done_sending_request: Callable[[str], None] | None = None,
    ):
        super().__init__(daemon=True, name="KVCacheSendingLayerThread")
        self.engine = engine
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.kv_cache_specs = kv_cache_specs
        self.attn_resharding_group_idx = attn_resharding_group_idx
        self.mamba_cache_mode = self.vllm_config.cache_config.mamba_cache_mode
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pd_head_ratio = pd_head_ratio
        self.num_head_replica = num_head_replica
        self.layer_metadata = layer_metadata
        self.group_max_layer_idx = group_max_layer_idx
        self.total_layers = total_layers
        self.use_mla = use_mla
        self.use_attn_mamba_hybrid = use_attn_mamba_hybrid
        self.resharding_stream = resharding_stream
        self.current_layer = -1
        # p_then_d: 发送线程在真正推送前解析等待中的 D 参数(worker.resolve_pending_req);
        # 返回 False 表示该请求本批不可发(超时/解析失败).
        self.resolve_pending_req = resolve_pending_req
        # 上面回调判失败后, 由这里上报请求级发送完成(释放延迟释放的块).
        self.add_done_sending_request = add_done_sending_request

        send_queue_size = 0
        if self.pd_head_ratio != 1 and not envs.P_THEN_D:
            if self.use_attn_mamba_hybrid:
                send_queue_size = len(self.kv_cache_specs)
            else:
                send_queue_size = 1
        # p_then_d + reshard/量化路径: 队列必须无界 —— 发送线程要等 D 参数(可能要
        # 等整个 prefill 结束 + proxy 派发 D), 而有界队列的 put 会阻塞模型前向,
        # 前向发不出响应 -> D 永远不会被派发 -> 死锁. 代价: 等待期间整个 prefill 的
        # 逐层 reshard 结果(keys/values)留在显存里(见 save_kv_layer).
        self.send_queue = queue.Queue[SendTask](maxsize=send_queue_size)
        self.failed_reqs: set[str] = set()
        self.k_buffer = k_buffer
        self.v_buffer = v_buffer
        self.enable_kv_quant = enable_kv_quant
        self.enable_c8_quant = enable_c8_quant
        # reshard/量化路径的传输 src 是单槽 k_buffer/v_buffer(所有层复用同一
        # 偏移), 多层攒批会互相覆盖 → 该路径强制逐层传输.
        self.uses_reshard_buffers = pd_head_ratio != 1 or enable_kv_quant or enable_c8_quant
        self.max_batch_layers = 1 if self.uses_reshard_buffers else _LAYER_BATCH
        # 写线程流水要求「批间 staging 区域互不相同」: 它让下一批的 D2H flush
        # 与上一批的在飞 write 重叠. 单槽 buffer 路径两批 flush 到同一 staging
        # 区域, 会把上一批正在发送的字节覆盖掉, 故该路径禁用流水(退回单线程).
        self.use_pipe_writer = _PIPE_WRITER and global_te.use_tcp and not self.uses_reshard_buffers
        self.ready_event = ready_event
        # 本 rank 的侧信道标识, 随 LAYER_DONE 下发供 D 侧做多路径(P/D TP 不等)计数.
        self.sender_path = sender_path
        # perf: 上一批处理完成的时刻(用于统计批间攒批等待)与批序号.
        self._last_batch_end_at: float | None = None
        self._batch_seq = 0
        # perf: 请求级累计(external req id -> 累计), 在 DONE 回调处打印后清除.
        # 批内共享段(flush/write/layerdone)以批共享口径计入该批全部请求.
        self._perf_req: dict[str, dict[str, float]] = {}
        # 写线程流水 (MC_TCP_PIPE_WRITER=1): write 队列由写线程消费, 完成队列
        # 由发送线程在 drain 阶段按序消费; _pipe_inflight 仅发送线程读写.
        self._write_queue: queue.Queue[Any] | None = None
        self._done_queue: queue.Queue[Any] | None = None
        self._pipe_inflight = 0
        self._pipe_writer: threading.Thread | None = None

    def run(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        # MC_TCP_CPU_BIND=1 时把本线程绑到进程允许核集中负载最低的核
        # (受 vllm-ascend cpu_binding taskset 约束, 在 main 核集内选取).
        bind_current_thread_to_idle_cpu(f"kv-send-rank{local_rank}")
        self.ready_event.set()
        if self.use_pipe_writer:
            # 启动独立写线程; 队列在 __init__ 已按 _PIPE_WRITER 创建.
            self._write_queue = queue.Queue(maxsize=_PIPE_DEPTH)  # type: ignore[assignment]
            self._done_queue = queue.Queue()  # type: ignore[assignment]
            self._pipe_writer = threading.Thread(target=self._pipe_writer_loop, daemon=True, name="KVCachePipeWriter")
            self._pipe_writer.start()
            logger.info("[mooncake][pipe] pipe writer thread started (depth=%d)", _PIPE_DEPTH)
        # 攒批: 攒满 _LAYER_BATCH 层、遇到最后层或层序号断裂(跨步/新任务)
        # 才整批处理. 混合模型 (attn+mamba) 与均匀模型同路径: mamba/GDN 层的
        # conv/ssm 状态在各自层 forward 内已写定 (P 侧无投机, 不存在采样后的
        # 状态重写), 随批 flush 读到的是截至该层的最终值; 多 chunk 时中间
        # chunk 的状态会被后续 chunk 覆写, 请求级完成信息在含最后层的批之后,
        # D 在 decode 前收齐的即最终状态. 批大小上限见 max_batch_layers:
        # reshard/量化路径为 1(单槽 buffer), 其余为 _LAYER_BATCH.
        pending: list[SendTask] = []
        while True:
            if self._pipe_writer is not None:
                # 空闲等待期间顺带消化已完成 write 的批: LAYER_DONE 不再等到
                # 下一次攒批触发才发, 数据写完即通知 D (消除批间 LD 排队延迟).
                while True:
                    try:
                        send_task = self.send_queue.get(timeout=0.005)
                        break
                    except queue.Empty:
                        self._pipe_drain(block=False)
            else:
                send_task = self.send_queue.get()
            is_last = send_task.layer_idx == (self.total_layers - 1)
            if pending and send_task.layer_idx != pending[-1].layer_idx + 1:
                self._handle_batch(pending)
                pending = []
            pending.append(send_task)
            if is_last or len(pending) >= self.max_batch_layers:
                self._handle_batch(pending)
                pending = []

    def _handle_batch(self, tasks: list[SendTask]):
        try:
            self._resolve_pending_requests(tasks)
            if self.use_pipe_writer:
                self._transfer_kv_cache_batch_pipe(tasks)
            else:
                self._transfer_kv_cache_batch(tasks)
        except Exception as e:
            logger.error(
                "Failed to transfer KV cache batch. layer_idx=%s, error=%s. Check transfer engine and memory state.",
                [t.layer_idx for t in tasks],
                e,
            )
        if _PERF_LOG and not self.use_pipe_writer:
            self._last_batch_end_at = time.perf_counter()

    def _resolve_pending_requests(self, tasks: list[SendTask]) -> None:
        """p_then_d: 推送前解析本批里仍在等 D 参数的请求(每个 ReqMeta 只解析一次).

        解析必须在本线程做: 模型线程要先把 P 的响应发出去 D 才会被派发, 所以它
        不能等; 块的延迟释放(delayed_free)保证等待期间 KV 仍有效. 解析失败的请求
        清空 peer 映射 —— 后续批的聚合自然跳过它, 不会再产出传输段 —— 并立即上报
        请求级完成, 把块还给 scheduler(否则请求永远留在 Scheduler.requests 里).
        """
        if self.resolve_pending_req is None:
            return
        for send_task in tasks:
            for req_id, req_meta in send_task.send_request.items():
                if not req_meta.awaiting_params:
                    continue
                if self.resolve_pending_req(req_id, req_meta):
                    # 解析成功即置位, 保证同一 ReqMeta 只解析一次(对齐缩放/兼容字段
                    # 改写都不幂等); worker 侧也会置, 这里兜底.
                    req_meta.awaiting_params = False
                    continue
                req_meta.peer_transfer = {}
                req_meta.awaiting_params = False
                self.failed_reqs.add(req_id)
                if req_meta.delayed_free and self.add_done_sending_request is not None:
                    self.add_done_sending_request(req_id)

    def _release_delayed_requests(self, tasks: list[SendTask], transferred_reqs: set[str]) -> None:
        """p_then_d: 本 rank 上逐层数据已发完的请求 -> 上报请求级发送完成.

        只在「本批的 LAYER_DONE 已 ACK(或写失败已判定)」之后调用, 保证块不会在
        数据面写出前被 scheduler 回收复用.

        判据: 请求是延迟释放的、且是末轮(chunk_finish), 且本 rank 有数据的每个
        group 的末层(group_max_layer_idx)都已处理过 —— 比只看最后一层更准: 本 rank
        在末层所属 group 上没有数据的请求不会出现在末层任务里. 参数解析失败/无
        peer 映射时退回"末层"判据 —— 失败请求同样要上报, 否则块永久泄漏.
        """
        if self.add_done_sending_request is None:
            return
        batch_max_layer = max((t.layer_idx for t in tasks), default=-1)
        for send_task in tasks:
            for req_id, req_meta in send_task.send_request.items():
                if not req_meta.delayed_free or not req_meta.chunk_finish:
                    continue
                if req_id not in transferred_reqs:
                    continue
                groups = {
                    group_idx
                    for group_idx in range(len(self.kv_cache_specs))
                    if any(blocks["local_block_ids"][group_idx] for blocks in req_meta.peer_transfer.values())
                }
                if groups:
                    required_layer = max(
                        self.group_max_layer_idx.get(group_idx, self.total_layers - 1) for group_idx in groups
                    )
                else:
                    required_layer = self.total_layers - 1
                if batch_max_layer < required_layer:
                    continue
                self.add_done_sending_request(req_id)

    def _group_end_of_task(self, send_task: SendTask) -> tuple[bool, int]:
        """本 rank 是否发完了该请求在本层所属 group 上的全部数据, 以及该 group 序号.

        返回 (is_group_end, layer_group_idx): is_group_end 表示本任务是该 group 上
        本 rank 最后一个带数据的层 —— 此刻它要发给各 peer 的该 group 数据已在
        之前各批(同步 LAYER_DONE 握手)写达, 可以下发/触发请求级完成信息.
        """
        layer_group_idx = self.layer_metadata[send_task.layer_name].tensor_group_idx[0]
        return send_task.layer_idx == self.group_max_layer_idx.get(layer_group_idx, self.total_layers - 1), (
            layer_group_idx
        )

    def _request_sender_path_count(self, peer_blocks: dict) -> int:
        """本请求在该 peer 上期望的 P 侧发送方路径总数(各 group 取并集).

        沿用各组自己的 trans_count 口径(attention 组乘过 pd_head_ratio, mamba 组
        即 rank 数), 不做换算 —— 保证等分/reshard 场景的期望值与改动前逐字一致;
        不等分 + hybrid 场景下 mamba 组的 2 个发送方包含 attention 组的那 1 个,
        取 max 即精确并集. 取 max 永不高估(高估 => D 永远等不齐 => 死锁).
        """
        return max(
            (
                peer_blocks["trans_count"][g]
                for g in range(len(self.kv_cache_specs))
                if peer_blocks["local_block_ids"][g]
            ),
            default=0,
        )

    def get_transfer_meta(
        self,
        send_task: SendTask,
        req_id: str,
        req_meta: ReqMeta,
        layer_group_idx: int,
        peer: tuple[str, int],
    ):
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []

        peer_blocks = req_meta.peer_transfer[peer]
        local_block_ids = peer_blocks["local_block_ids"][layer_group_idx]
        remote_block_ids = peer_blocks["remote_block_ids"][layer_group_idx]
        if not local_block_ids:
            # 多 peer 场景: 该 group 不经这个 peer, 本 peer 没有要传的段.
            return (src_list, dst_list, length_list)
        layer_name = send_task.layer_name
        layer_kv_cache_spec = self.kv_cache_specs[layer_group_idx]
        remote_layer_metadata = req_meta.peer_layer_metadata[peer][layer_name]
        local_layer_metadata = self.layer_metadata[layer_name]

        if isinstance(layer_kv_cache_spec, MambaSpec):
            # only support one block transfer for mamba
            # 状态块下标随 mamba_cache_mode 变:
            #   * 非 align(如 mamba_cache_mode=none/"all"): 状态恒存在请求首块 [0];
            #   * align: 每请求块表形如
            #       [占位 null 块 × (cdiv(tokens, bs) - 1), 运行状态块, 投机块 × num_spec]
            #     (vLLM MambaManager.allocate_new_blocks: 首轮只分配 1+num_spec 个真块,
            #      中间用 null 块补齐下标; 运行状态块 = 表格下标 cdiv(tokens,bs)-1,
            #      preprocess/postprocess_mamba 会把它按块边界向后轮转),
            #     故状态块 = len-num_spec-1。**源与目的必须同口径**: D 侧列表已被
            #     _trim_hybrid_remote_block_ids 对齐到与源同长, 取 remote_block_ids[0]
            #     在跨块 prompt 下会指到 null 占位块(状态丢失 → D 读到未初始化状态 → 乱码).
            if self.mamba_cache_mode == "align":
                local_transfer_idx = len(local_block_ids) - self.num_speculative_tokens - 1
                remote_transfer_idx = len(remote_block_ids) - self.num_speculative_tokens - 1
            else:
                local_transfer_idx = 0
                remote_transfer_idx = 0
            local_conv_addr, local_ssm_addr = local_layer_metadata.kv_caches_base_addr
            remote_conv_addr, remote_ssm_addr = remote_layer_metadata.kv_caches_base_addr
            local_conv_len, local_ssm_len = local_layer_metadata.block_len
            tp_ratio = self.tp_size // req_meta.remote_tp_size
            if tp_ratio == 1:
                src_list.extend(
                    [
                        local_conv_addr + local_block_ids[local_transfer_idx] * local_conv_len,
                        local_ssm_addr + local_block_ids[local_transfer_idx] * local_ssm_len,
                    ]
                )
                dst_list.extend(
                    [
                        remote_conv_addr + remote_block_ids[remote_transfer_idx] * local_conv_len,
                        remote_ssm_addr + remote_block_ids[remote_transfer_idx] * local_ssm_len,
                    ]
                )
                length_list.extend([local_conv_len, local_ssm_len])
            else:
                conv_shape, ssm_shape = layer_kv_cache_spec.shapes
                conv_dtype, ssm_dtype = layer_kv_cache_spec.dtypes
                remote_conv_len, remote_ssm_len = remote_layer_metadata.block_len
                # conv
                linear_key_head_dim = self.vllm_config.model_config.hf_text_config.linear_key_head_dim
                linear_num_key_heads = self.vllm_config.model_config.hf_text_config.linear_num_key_heads
                linear_value_head_dim = self.vllm_config.model_config.hf_text_config.linear_value_head_dim
                linear_num_value_heads = self.vllm_config.model_config.hf_text_config.linear_num_value_heads
                local_num_key_heads = linear_num_key_heads // self.tp_size
                local_num_value_heads = linear_num_value_heads // self.tp_size
                local_conv_offsets = [
                    0,
                    local_num_key_heads * linear_key_head_dim,
                    local_num_key_heads * 2 * linear_key_head_dim,
                ]
                local_conv_sizes = [
                    local_num_key_heads * linear_key_head_dim,
                    local_num_key_heads * linear_key_head_dim,
                    local_num_value_heads * linear_value_head_dim,
                ]
                for i in range(conv_shape[0]):
                    for local_conv_offset, local_conv_size in zip(local_conv_offsets, local_conv_sizes):
                        local_addr_offset = (i * conv_shape[1] + local_conv_offset) * get_dtype_size(conv_dtype)
                        remote_addr_offset = (
                            (i * conv_shape[1] + local_conv_offset) * tp_ratio
                            + (self.tp_rank % tp_ratio) * local_conv_size
                        ) * get_dtype_size(conv_dtype)
                        src_list.append(
                            local_conv_addr + local_block_ids[local_transfer_idx] * local_conv_len + local_addr_offset
                        )
                        dst_list.append(
                            remote_conv_addr
                            + remote_block_ids[remote_transfer_idx] * remote_conv_len
                            + remote_addr_offset
                        )
                        length_list.append(local_conv_size * get_dtype_size(conv_dtype))
                # ssm
                remote_addr_offset = (self.tp_rank % tp_ratio) * math.prod(ssm_shape) * get_dtype_size(ssm_dtype)
                src_list.append(local_ssm_addr + local_block_ids[local_transfer_idx] * local_ssm_len)
                dst_list.append(
                    remote_ssm_addr + remote_block_ids[remote_transfer_idx] * remote_ssm_len + remote_addr_offset
                )
                length_list.append(local_ssm_len)
        else:
            if self.pd_head_ratio == 1:
                layer_local_kv_base_addr = local_layer_metadata.kv_caches_base_addr
                layer_remote_kv_base_addr = remote_layer_metadata.kv_caches_base_addr
                block_lens = local_layer_metadata.block_len
                grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                    remote_block_ids, local_block_ids
                )
                # kv cache quantization scenario
                if (self.enable_kv_quant or self.enable_c8_quant) and send_task.k_quant_cache is not None:
                    assert len(block_lens) == 2, "Quantization block length must be 2!"
                    if self.enable_kv_quant:
                        quant_block_lens = [block_lens[0] // 2, block_lens[1]]
                    else:
                        quant_block_lens = [block_lens[0] // 2, block_lens[1] // 2]
                    layer_local_quant_kv_addr = [self.k_buffer.data_ptr(), self.v_buffer.data_ptr()]
                    rearrange_block_ids = send_task.group_rearrange_block_ids[layer_group_idx]
                    # eg:[5,6,7,9] -> {5:0, 6:1, 7:2, 9:3}
                    rearrange_block_dict = {
                        value: index
                        for index, value in enumerate(rearrange_block_ids)  # type:ignore
                    }
                    for block_len, src_layer_base_addr, dst_layer_base_addr in zip(
                        quant_block_lens, layer_local_quant_kv_addr, layer_remote_kv_base_addr
                    ):
                        for group_remote_block_id, group_local_block_id in zip(
                            grouped_remote_block_ids, grouped_local_block_ids
                        ):
                            src = src_layer_base_addr + rearrange_block_dict[group_local_block_id[0]] * block_len
                            dst = dst_layer_base_addr + group_remote_block_id[0] * block_len
                            length = len(group_local_block_id) * block_len
                            src_list.append(src)
                            dst_list.append(dst)
                            length_list.append(length)
                else:
                    for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                        zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)
                    ):
                        block_len = block_lens[k]
                        for group_remote_block_id, group_local_block_id in zip(
                            grouped_remote_block_ids, grouped_local_block_ids
                        ):
                            src = src_layer_base_addr + group_local_block_id[0] * block_len
                            dst = dst_layer_base_addr + group_remote_block_id[0] * block_len
                            length = len(group_local_block_id) * block_len
                            src_list.append(src)
                            dst_list.append(dst)
                            length_list.append(length)
            else:
                rearrange_block_ids = send_task.group_rearrange_block_ids[layer_group_idx]
                rearrange_block_dict = {
                    value: index
                    for index, value in enumerate(rearrange_block_ids)  # type:ignore
                }
                layer_local_kv_base_addr = [self.k_buffer.data_ptr(), self.v_buffer.data_ptr()]
                layer_remote_kv_base_addr = remote_layer_metadata.kv_caches_base_addr
                block_lens = local_layer_metadata.block_len
                remote_block_lens = remote_layer_metadata.block_len
                assert len(layer_remote_kv_base_addr) == 2, (
                    "Layer kv_cache resharding only supports two kv cache tensors."
                )
                src_list, dst_list, length_list = [], [], []
                for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                    zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)
                ):
                    block_len = block_lens[k]
                    if self.enable_c8_quant:
                        block_len = block_len // 2
                    remote_block_len = remote_block_lens[k]
                    for remote_block_id, local_block_id in zip(remote_block_ids, local_block_ids):
                        src = src_layer_base_addr + rearrange_block_dict[local_block_id] * block_len
                        dst = (
                            dst_layer_base_addr
                            + remote_block_id * remote_block_len
                            + block_len * ((self.tp_rank // self.num_head_replica) % self.pd_head_ratio)
                        )
                        src_list.append(src)
                        dst_list.append(dst)
                        length_list.append(block_len)
        return (src_list, dst_list, length_list)

    def _wait_task_ready(self, send_task: SendTask) -> None:
        """等本任务的数据就绪事件(计算流上记录).

        轮询 `Event.query()` 而非 `Event.synchronize()`: 本 CANN 版本的 event
        同步有已知偶发挂死(见下方 _transfer_kv_cache_batch 内注释), 轮询还能
        在超时后打日志, 而不是无声卡死.
        """
        event = send_task.wait_event
        if event is None:
            return
        t0 = time.perf_counter()
        while not event.query():
            time.sleep(1e-4)
            if time.perf_counter() - t0 > LAYER_DONE_TIMEOUT_S:
                logger.error(
                    "Wait task ready timeout: rank=%d layer=%s waited=%.1fs",
                    self.tp_rank,
                    send_task.layer_idx,
                    time.perf_counter() - t0,
                )
                return

    def _transfer_kv_cache_batch(self, tasks: list[SendTask]):
        """批量传输攒齐的多个层任务: 跨任务合并 ranges 后, 每 session 一次
        D2H flush + 一次 sync_write + 每 peer 一次 LAYER_DONE(携带批内全部层
        范围), 摊薄逐层传输的固定开销. tasks 为空时直接返回."""
        if not tasks:
            return
        if self.uses_reshard_buffers and len(tasks) > 1:
            # 该路径各层的 src 都指向同一个 k_buffer/v_buffer 偏移, 多层批会
            # 互相覆盖. max_batch_layers 已把批大小钳到 1, 这里只做不变式兜底.
            raise RuntimeError(
                "Layerwise reshard/quant transfer uses a single-slot k/v buffer; "
                f"multi-layer batching is not supported (layers={[t.layer_idx for t in tasks]})."
            )

        # perf: 批级阶段计时. 阶段分解: wait(批间攒批等待) / event(等NPU事件,
        # 含模型产出该批最后一层的等待) / flush(D2H) / write(TCP) /
        # layerdone(LAYER_DONE REQ-REP 往返) / misc(聚合与地址替换等).
        t_batch0 = time.perf_counter()
        self._batch_seq += 1
        batch_id = self._batch_seq
        batch_wait_ms = 0.0
        if _PERF_LOG and self._last_batch_end_at is not None:
            batch_wait_ms = _perf_ms(self._last_batch_end_at)
        t_event0 = t_batch0
        t_event1 = t_batch0
        flush_ms = 0.0
        write_ms = 0.0
        layerdone_ms = 0.0
        # perf: 各阶段绝对起止窗口(CLOCK_MONOTONIC, P/D 同机可直接对齐).
        # 单 session 时即精确的 D2H / H2H / LAYER_DONE-ACK 边界; 多 session
        # 时取并集窗口, 只能反映整批的占用范围.
        flush_win0 = write_win0 = layerdone_win0 = None
        flush_win1 = write_win1 = layerdone_win1 = None

        # 1) 数据就绪: reshard/量化路径等本任务自己的就绪事件(计算流上记录);
        #    pd==1 无量化路径等 reshape 事件. 两者都只依赖本任务, 不能用整条
        #    resharding_stream.synchronize() —— 那会连带等待主线程为后续层
        #    enqueue 的 reshard 工作, 与对端进度成环(2026-09-09 定位的死锁).
        if self.uses_reshard_buffers:
            for send_task in tasks:
                self._wait_task_ready(send_task)
        elif self.pd_head_ratio == 1:
            """
            Note: Due to a bug in ADXL, calling current_event.synchronize() may occasionally hang.
            This issue will be fixed in CANN version 8.5.rc1.
            You can manually build the master branch of the project at https://gitcode.com/cann/hixl
            to resolve this issue before the 8.5.RC1 release.
            """
            # 逐层等待: 攒批把多层合并成一次 D2H flush, 但批内各层的 KV / GDN 状态
            # 写入不保证都排在同一个流上(自定义 op / 侧流), 只等批内最后一层的 event
            # 会把尚未写定的层一起 flush 出去 —— 参考实现(预填充后解码版 connector)
            # 是"一层一个 task, 各自 wait_event.synchronize()", 攒批必须保留这个不变式.
            for send_task in tasks:
                if send_task.wait_event is not None:
                    send_task.wait_event.synchronize()  # type:ignore

        # 2) reshard/量化结果拷进单槽 k/v buffer(逐任务; 发送线程串行执行,
        #    与随后的 flush 互斥). 拷贝提交在侧流上, 侧流此后只承载本线程
        #    自己的拷贝, 等它排空不会牵扯其他线程/其他 rank 的进度.
        for send_task in tasks:
            key = send_task.k_cache
            value = send_task.v_cache
            if self.pd_head_ratio > 1 and key is not None and value is not None:
                with npu_stream_switch(self.resharding_stream):
                    key = key.view(-1, key.shape[-1])  # type:ignore
                    value = value.view(-1, key.shape[-1])  # type:ignore
                    self.k_buffer[: key.shape[0]].copy_(key)  # [:4, 128] ->
                    self.v_buffer[: value.shape[0]].copy_(value)
            if send_task.k_quant_cache is not None:
                with npu_stream_switch(self.resharding_stream):
                    key_quant = send_task.k_quant_cache
                    key_quant = key_quant.view(-1, key_quant.shape[-1])  # type:ignore
                    self.k_buffer[: key_quant.shape[0]].copy_(key_quant)
                    value_quant = send_task.v_quant_cache
                    value_quant = value_quant.view(-1, value_quant.shape[-1])  # type:ignore
                    self.v_buffer[: value_quant.shape[0]].copy_(value_quant)
        if self.uses_reshard_buffers:
            self.resharding_stream.synchronize()
        t_event1 = time.perf_counter()

        # 3) 跨任务按 (session, req) 聚合 ranges
        session_meta: dict[str, TransferMeta] = {}
        for send_task in tasks:
            layer_group_idx = self.layer_metadata[send_task.layer_name].tensor_group_idx[0]
            for req_id, req_meta in send_task.send_request.items():
                for peer, _peer_blocks in req_meta.peer_transfer.items():
                    peer_host, peer_port = peer
                    session_id = f"{peer_host}:{req_meta.peer_te_rpc_port[peer]}"
                    meta = session_meta.get(session_id)
                    if meta is None:
                        meta = TransferMeta(src=[], dst=[], length=[], req_ids=[])
                        session_meta[session_id] = meta
                    if req_id not in meta.req_ids:
                        meta.req_ids.append(req_id)
                    meta.req_peer[req_id] = (peer_host, peer_port)
                    (src_list, dst_list, length_list) = self.get_transfer_meta(
                        send_task, req_id, req_meta, layer_group_idx, peer
                    )
                    if not src_list:
                        continue
                    start = len(meta.src)
                    meta.src.extend(src_list)
                    meta.dst.extend(dst_list)
                    meta.length.extend(length_list)
                    meta.req_slices.setdefault(req_id, []).append((start, len(src_list)))
                    is_group_end, _ = self._group_end_of_task(send_task)
                    if is_group_end:
                        # 本 rank 在该 group 上对该请求的最后一个带数据层: 它此前各批
                        # 的 LAYER_DONE 握手已完成, 数据全部写达. 记录该 peer 在该请求
                        # 末轮的完成信息(is_last=chunk_finish; trans_count=该请求在该
                        # peer 上的发送方路径总数), 随本批 LAYER_DONE 下发.
                        # 每个参与发送的 rank 各自下发一次, D 侧按 sender_path 计数
                        # 收齐 —— 只由最后一层(MTP, 属 attention 组)那批下发时, mamba
                        # 组另一个发送方的数据可能尚未落地, D 会提前开始解码.
                        meta.req_done[(peer_host, peer_port, req_id)] = (
                            req_meta.chunk_finish,
                            self._request_sender_path_count(req_meta.peer_transfer[peer]),
                        )

        # 4) 每 session: 一次 D2H flush → src 替换 → 一次 sync_write → LAYER_DONE
        for session_id, transfer_meta in session_meta.items():
            if len(transfer_meta.src) > 0:
                if global_te.use_tcp:
                    # H2H: 批内各层各请求的 KV 块一次性从 NPU 刷进 CPU staging
                    # (批量 DMA), 再把 src 换成 staging 地址 —— TCP 只能读写
                    # 注册的 host 内存.
                    t_flush0 = time.perf_counter()
                    global_te.sync_npu_to_cpu_for_npu_addrs(transfer_meta.src, transfer_meta.length)
                    if _PERF_LOG:
                        flush_ms += _perf_ms(t_flush0)
                        # 更新 D2H (flush) 绝对窗口: 起点取最早, 终点取最晚.
                        if flush_win0 is None:
                            flush_win0 = t_flush0
                        flush_win1 = time.perf_counter()
                    staging_src = []
                    for src_addr in transfer_meta.src:
                        cpu_addr = global_te.npu_addr_to_cpu_addr(src_addr)
                        if cpu_addr is None:
                            raise RuntimeError(f"H2H layerwise: NPU addr 0x{src_addr:x} not found in TCP staging map.")
                        staging_src.append(cpu_addr)
                    transfer_meta.src = staging_src
                req_start_time = time.perf_counter()
                ret = self.engine.batch_transfer_sync_write(
                    session_id, transfer_meta.src, transfer_meta.dst, transfer_meta.length
                )
                if _PERF_LOG:
                    write_ms += _perf_ms(req_start_time)
                    # H2H write 窗口: 多 session 取并集(起点最早, 终点最晚),
                    # 与 flush/layerdone 窗口口径一致.
                    if write_win0 is None:
                        write_win0 = req_start_time
                    write_win1 = time.perf_counter()
                if ret < 0:
                    logger.error(
                        "Mooncake transfer failed for send requests. req_ids=%s, destination=%s, ret=%d. ",
                        transfer_meta.req_ids,
                        session_id,
                        ret,
                    )
                    for failed_req_id in transfer_meta.req_ids:
                        self.failed_reqs.add(failed_req_id)
                else:
                    req_end_time = time.perf_counter()
                    total_transfer_size = sum(transfer_meta.length) / 1024
                    req_transfer_elapsed = (req_end_time - req_start_time) * 1000
                    logger.debug(
                        "Layers batch KV cache transfer task %dKB to remote_session_id [%s] took %.3f ms.",
                        total_transfer_size,
                        session_id,
                        req_transfer_elapsed,
                    )
                    if global_te.use_tcp:
                        # 批传输完成后通知 D(该批全部层范围), D 完成 H2D 后回
                        # ACK 才处理下一批(同线程串行). 请求级完成信息(is_last /
                        # 期望路径数 / failed)随含最后层的批的 LAYER_DONE 一起
                        # 下发, D 的请求级 done 必然晚于全部层的 H2D, 满足
                        # "收到完整 KV 后才启动计算"的不变式.
                        layer_done_ok = True
                        peer_layer_msgs: dict[tuple[str, int], tuple[list[int], list[int], list[str]]] = {}
                        for layer_req_id in transfer_meta.req_ids:
                            peer_key = transfer_meta.req_peer[layer_req_id]
                            if peer_key not in peer_layer_msgs:
                                peer_layer_msgs[peer_key] = ([], [], [])
                            peer_layer_msgs[peer_key][2].append(get_external_request_id(layer_req_id))
                            for req_start, req_count in transfer_meta.req_slices[layer_req_id]:
                                peer_layer_msgs[peer_key][0].extend(
                                    transfer_meta.dst[req_start : req_start + req_count]
                                )
                                peer_layer_msgs[peer_key][1].extend(
                                    transfer_meta.length[req_start : req_start + req_count]
                                )
                        for (peer_host, peer_port), (peer_addrs, peer_lengths, peer_req_ids) in peer_layer_msgs.items():
                            done_list = self._req_done_list(transfer_meta, peer_host, peer_port)
                            t_ld0 = time.perf_counter()
                            ok = self._send_layer_done_signal(
                                peer_host, peer_port, peer_req_ids, peer_addrs, peer_lengths, done_list
                            )
                            if _PERF_LOG:
                                layerdone_ms += _perf_ms(t_ld0)
                                # LAYER_DONE REQ-REP 往返窗口(内含 D 侧 H2D
                                # 与 ACK 传输), 与 D 侧 H2D 窗口做交叉校验.
                                if layerdone_win0 is None:
                                    layerdone_win0 = t_ld0
                                layerdone_win1 = time.perf_counter()
                            if not ok:
                                layer_done_ok = False
                                break
                        if not layer_done_ok:
                            for failed_req_id in transfer_meta.req_ids:
                                self.failed_reqs.add(failed_req_id)

        # 5) 请求级完成信号: 含最后层任务时在 session 聚合阶段记入
        #    TransferMeta.req_done, 随该批 LAYER_DONE 的 done_list 下发(见
        #    _req_done_list) —— 保证 D 侧请求级 done 晚于本请求全部层批的 ACK;
        #    无任何传输段的请求不产生条目(与原逐层语义一致).
        #    仅 4) 的 perf 打印仍需要此集合, 故保留计算.
        transferred_reqs = {req_id for meta in session_meta.values() for req_id in meta.req_ids}

        # perf 批级行与请求级累计: 必须在第 5 步 DONE 打印之前执行, 否则最后
        # 一批(含最后层)的耗时与批数不会被计入该请求.
        if _PERF_LOG:
            total_ms = _perf_ms(t_batch0)
            event_ms = (t_event1 - t_event0) * 1e3
            misc_ms = max(
                0.0,
                total_ms - batch_wait_ms - event_ms - flush_ms - write_ms - layerdone_ms,
            )
            batch_ext_reqs = [get_external_request_id(r) for r in transferred_reqs]
            # 本批实际写入对端的 payload 字节(去重后各 session ranges 长度和),
            # 用于吞吐与带宽利用率统计.
            batch_bytes = sum(sum(m.length) for m in session_meta.values())

            def _win_str(a: float | None, b: float | None) -> str:
                # 绝对窗口 [起点,终点] (秒, CLOCK_MONOTONIC); 无该阶段时为 "-".
                return "-" if a is None else f"{a:.6f},{b:.6f}"

            logger.info(
                "[mooncake][perf] P batch=%d layers=%s reqs=%s wait=%.1f event=%.1f "
                "flush=%.1f write=%.1f layerdone=%.1f misc=%.1f total=%.1f ms "
                "t0=%.6f flush_win=%s write_win=%s layerdone_win=%s bytes=%d",
                batch_id,
                [t.layer_idx for t in tasks],
                batch_ext_reqs,
                batch_wait_ms,
                event_ms,
                flush_ms,
                write_ms,
                layerdone_ms,
                misc_ms,
                total_ms,
                t_batch0,
                _win_str(flush_win0, flush_win1),
                _win_str(write_win0, write_win1),
                _win_str(layerdone_win0, layerdone_win1),
                batch_bytes,
            )
            # 请求级累计(批共享口径: flush/write/layerdone 是该批全部请求共享的
            # 墙钟, 每个请求都记全值; 单请求场景下即精确分解).
            for ext_req in batch_ext_reqs:
                acc = self._perf_req.get(ext_req)
                if acc is None:
                    acc = {
                        "t0": t_batch0,
                        "batches": 0.0,
                        "event": 0.0,
                        "flush": 0.0,
                        "write": 0.0,
                        "layerdone": 0.0,
                        "wait": 0.0,
                    }
                    self._perf_req[ext_req] = acc
                acc["batches"] += 1
                acc["event"] += event_ms
                acc["flush"] += flush_ms
                acc["write"] += write_ms
                acc["layerdone"] += layerdone_ms
                acc["wait"] += batch_wait_ms
        for send_task in tasks:
            is_group_end, layer_group_idx = self._group_end_of_task(send_task)
            if is_group_end:
                # 每个参与发送的 rank 都在自己的 group 末层处理请求级信号: D2D 的
                # 显式 DONE, 以及写失败的作废通知(失败可能发生在任一发送方).
                is_last_layer = send_task.layer_idx == (self.total_layers - 1)
                for req_id, req_meta in send_task.send_request.items():
                    if req_id not in transferred_reqs:
                        continue
                    if req_meta.chunk_finish:
                        if _PERF_LOG and is_last_layer:
                            ext_req = get_external_request_id(req_id)
                            acc = self._perf_req.pop(ext_req, None)
                            if acc is not None:
                                # 请求级汇总: t0=发送线程开始处理该请求首任务,
                                # total=到 DONE 发出的墙钟跨度.
                                logger.info(
                                    "[mooncake][perf] P req=%s done batches=%d "
                                    "event=%.1f flush=%.1f write=%.1f layerdone=%.1f "
                                    "wait=%.1f total=%.1f ms",
                                    ext_req,
                                    int(acc["batches"]),
                                    acc["event"],
                                    acc["flush"],
                                    acc["write"],
                                    acc["layerdone"],
                                    acc["wait"],
                                    _perf_ms(acc["t0"]),
                                )
                        if req_id in self.failed_reqs:
                            # 失败仍走独立的请求级信号: 写失败的层批根本没有
                            # LAYER_DONE 到达 D, 只靠路径计数会让请求永远等不到
                            # 完成 —— 必须显式通知 D 作废旧块并重试.
                            self._send_failed_signal(req_id, req_meta, layer_group_idx)
                            self.failed_reqs.discard(req_id)
                        elif global_te.use_tcp:
                            # TCP/H2H: 本请求的请求级完成信息已随末批 LAYER_DONE
                            # 的 done_list 下发 (见 _req_done_list), 不再单独发一次
                            # DONE 往返.
                            pass
                        else:
                            # D2D(protocol=ascend): 数据直达 D 的 NPU, 两端之间
                            # 没有 LAYER_DONE 可搭车 —— 沿用独立的请求级 DONE.
                            self._send_done_signal(req_id, req_meta, layer_group_idx)
        # p_then_d: 数据面已同步写出(LAYER_DONE ACK 后), 可以把延迟释放的块还给
        # scheduler 了.
        self._release_delayed_requests(tasks, transferred_reqs)

    # ---- 写线程流水路径 (MC_TCP_PIPE_WRITER=1, protocol=tcp 且非 reshard/量化) ----
    # 阶段归属: 发送线程 = 等事件 + D2H flush + 攒批; 写线程 = TCP write(数据面
    # 连续占满); 发送线程 drain = LAYER_DONE(与后续批 write 重叠) + perf/请求级
    # 完成信息. 不变式与单线程版一致: LAYER_DONE g 只在 write g 完成后发出;
    # 请求级完成信息由含最后层的批随其 LAYER_DONE 下发 (阻塞 drain 保证);
    # 发送线程空闲时持续 drain, LAYER_DONE 不再等下一次攒批触发.
    def _transfer_kv_cache_batch_pipe(self, tasks: list[SendTask]):
        if not tasks:
            return
        # 先发掉已完成 write 的 LAYER_DONE(与本次 等事件/flush 重叠).
        if self._pipe_inflight > 0:
            self._pipe_drain(block=False)

        t_batch0 = time.perf_counter()
        self._batch_seq += 1
        batch_id = self._batch_seq
        batch_wait_ms = 0.0
        if _PERF_LOG and self._last_batch_end_at is not None:
            batch_wait_ms = _perf_ms(self._last_batch_end_at)
        t_event1 = t_batch0
        flush_ms = 0.0
        flush_win0 = flush_win1 = None

        # 数据就绪: 逐层等各自的事件 —— 批内各层的写入不保证同流, 只等最后一层
        # 会把未写定的层一起 flush(详见单线程路径同名注释).
        if any(t.k_quant_cache is not None for t in tasks):
            self.resharding_stream.synchronize()
        elif self.pd_head_ratio == 1:
            for send_task in tasks:
                if send_task.wait_event is not None:
                    send_task.wait_event.synchronize()  # type:ignore
        t_event1 = time.perf_counter()
        event_ms = (t_event1 - t_batch0) * 1e3

        # 跨任务按 (session, req) 聚合 ranges(与单线程版第 3 步一致).
        session_meta: dict[str, TransferMeta] = {}
        for send_task in tasks:
            layer_group_idx = self.layer_metadata[send_task.layer_name].tensor_group_idx[0]
            for req_id, req_meta in send_task.send_request.items():
                for peer, _peer_blocks in req_meta.peer_transfer.items():
                    peer_host, peer_port = peer
                    session_id = f"{peer_host}:{req_meta.peer_te_rpc_port[peer]}"
                    meta = session_meta.get(session_id)
                    if meta is None:
                        meta = TransferMeta(src=[], dst=[], length=[], req_ids=[])
                        session_meta[session_id] = meta
                    if req_id not in meta.req_ids:
                        meta.req_ids.append(req_id)
                    meta.req_peer[req_id] = (peer_host, peer_port)
                    (src_list, dst_list, length_list) = self.get_transfer_meta(
                        send_task, req_id, req_meta, layer_group_idx, peer
                    )
                    if not src_list:
                        continue
                    start = len(meta.src)
                    meta.src.extend(src_list)
                    meta.dst.extend(dst_list)
                    meta.length.extend(length_list)
                    meta.req_slices.setdefault(req_id, []).append((start, len(src_list)))
                    is_group_end, _ = self._group_end_of_task(send_task)
                    if is_group_end:
                        # 本 rank 在该 group 上对该请求的最后一个带数据层: 它此前各批
                        # 的 LAYER_DONE 握手已完成, 数据全部写达. 记录该 peer 在该请求
                        # 末轮的完成信息(is_last=chunk_finish; trans_count=该请求在该
                        # peer 上的发送方路径总数), 随本批 LAYER_DONE 下发.
                        # 每个参与发送的 rank 各自下发一次, D 侧按 sender_path 计数
                        # 收齐 —— 只由最后一层(MTP, 属 attention 组)那批下发时, mamba
                        # 组另一个发送方的数据可能尚未落地, D 会提前开始解码.
                        meta.req_done[(peer_host, peer_port, req_id)] = (
                            req_meta.chunk_finish,
                            self._request_sender_path_count(req_meta.peer_transfer[peer]),
                        )

        # 每 session: D2H flush 进本端 staging 并把 src 换成 staging 地址.
        # 写线程只读这些区域, flush 与上一批在飞的 write 区域不同, 可重叠.
        for session_id, transfer_meta in session_meta.items():
            if len(transfer_meta.src) <= 0:
                continue
            t_flush0 = time.perf_counter()
            global_te.sync_npu_to_cpu_for_npu_addrs(transfer_meta.src, transfer_meta.length)
            if _PERF_LOG:
                flush_ms += _perf_ms(t_flush0)
                if flush_win0 is None:
                    flush_win0 = t_flush0
                flush_win1 = time.perf_counter()
            staging_src = []
            for src_addr in transfer_meta.src:
                cpu_addr = global_te.npu_addr_to_cpu_addr(src_addr)
                if cpu_addr is None:
                    raise RuntimeError(f"H2H layerwise: NPU addr 0x{src_addr:x} not found in TCP staging map.")
                staging_src.append(cpu_addr)
            transfer_meta.src = staging_src

        transferred_reqs = {req_id for meta in session_meta.values() for req_id in meta.req_ids}
        job = _PipeJob(
            batch_id=batch_id,
            tasks=tasks,
            sessions=list(session_meta.items()),
            transferred_reqs=transferred_reqs,
            contains_last=any(t.layer_idx == (self.total_layers - 1) for t in tasks),
            t_batch0=t_batch0,
            batch_wait_ms=batch_wait_ms,
            event_ms=event_ms,
            flush_ms=flush_ms,
            flush_win0=flush_win0,
            flush_win1=flush_win1,
        )
        self._pipe_push(job)
        if job.contains_last:
            # 请求最后一批: 阻塞 drain 至其 write 完成并发出 LAYER_DONE,
            # 保证下一请求 flush 前数据面清空、请求级完成信息不早于全层 H2D.
            self._pipe_drain(block=True)

    def _pipe_writer_loop(self):
        local_rank = get_world_group().local_rank
        bind_current_thread_to_idle_cpu(f"kv-write-rank{local_rank}")
        while True:
            job = self._write_queue.get()  # type: ignore[union-attr]
            try:
                for session_id, transfer_meta in job.sessions:
                    if len(transfer_meta.src) <= 0:
                        continue
                    t_w0 = time.perf_counter()
                    ret = self.engine.batch_transfer_sync_write(
                        session_id, transfer_meta.src, transfer_meta.dst, transfer_meta.length
                    )
                    if _PERF_LOG:
                        job.write_ms += _perf_ms(t_w0)
                        if job.write_win0 is None:
                            job.write_win0 = t_w0
                        job.write_win1 = time.perf_counter()
                    if ret < 0:
                        logger.error(
                            "Mooncake (pipe) transfer failed for send requests. req_ids=%s, destination=%s, ret=%d. ",
                            transfer_meta.req_ids,
                            session_id,
                            ret,
                        )
                        job.failed[session_id] = list(transfer_meta.req_ids)
            except Exception as e:  # noqa: BLE001
                logger.error("[mooncake][pipe] write exception: %s", e)
                for session_id, transfer_meta in job.sessions:
                    job.failed.setdefault(session_id, []).extend(transfer_meta.req_ids)
            self._done_queue.put(job)  # type: ignore[union-attr]

    def _pipe_push(self, job: _PipeJob):
        self._write_queue.put(job)  # type: ignore[union-attr]  # 队列满时阻塞 = 背压
        self._pipe_inflight += 1

    def _pipe_drain(self, block: bool):
        """按序消费完成队列: 每个完成 write 的批发出 LAYER_DONE 并收尾."""
        while self._pipe_inflight > 0:
            try:
                job = self._done_queue.get(block=block, timeout=0.01 if block else 0.0)  # type: ignore[union-attr]
            except queue.Empty:
                if not block:
                    return
                continue
            self._pipe_inflight -= 1
            try:
                self._pipe_finalize(job)
            except Exception as e:  # noqa: BLE001
                logger.error("[mooncake][pipe] finalize batch=%d failed: %s", job.batch_id, e)

    def _pipe_finalize(self, job: _PipeJob):
        """LAYER_DONE + perf 批行 + 请求级累计/DONE. 语义镜像单线程版 4/5 步."""
        # 写失败的 session 不发 LAYER_DONE, 其 req 计入 failed(镜像 ret<0 分支).
        for session_id, failed_ids in job.failed.items():
            self.failed_reqs.update(failed_ids)
        layerdone_ms = 0.0
        layerdone_win0 = layerdone_win1 = None
        for session_id, transfer_meta in job.sessions:
            if session_id in job.failed or len(transfer_meta.src) <= 0:
                continue
            # 组 LAYER_DONE 消息(与单线程版一致): 携带该批全部层范围.
            peer_layer_msgs: dict[tuple[str, int], tuple[list[int], list[int], list[str]]] = {}
            for layer_req_id in transfer_meta.req_ids:
                peer_key = transfer_meta.req_peer[layer_req_id]
                if peer_key not in peer_layer_msgs:
                    peer_layer_msgs[peer_key] = ([], [], [])
                peer_layer_msgs[peer_key][2].append(get_external_request_id(layer_req_id))
                for req_start, req_count in transfer_meta.req_slices[layer_req_id]:
                    peer_layer_msgs[peer_key][0].extend(transfer_meta.dst[req_start : req_start + req_count])
                    peer_layer_msgs[peer_key][1].extend(transfer_meta.length[req_start : req_start + req_count])
            for (peer_host, peer_port), (peer_addrs, peer_lengths, peer_req_ids) in peer_layer_msgs.items():
                done_list = self._req_done_list(transfer_meta, peer_host, peer_port)
                t_ld0 = time.perf_counter()
                ok = self._send_layer_done_signal(
                    peer_host, peer_port, peer_req_ids, peer_addrs, peer_lengths, done_list
                )
                if _PERF_LOG:
                    layerdone_ms += _perf_ms(t_ld0)
                    if layerdone_win0 is None:
                        layerdone_win0 = t_ld0
                    layerdone_win1 = time.perf_counter()
                if not ok:
                    self.failed_reqs.update(transfer_meta.req_ids)

        # perf 批级行与请求级累计(必须在 DONE 之前打印).
        if _PERF_LOG:
            total_ms = _perf_ms(job.t_batch0)
            misc_ms = max(
                0.0,
                total_ms - job.batch_wait_ms - job.event_ms - job.flush_ms - job.write_ms - layerdone_ms,
            )
            batch_ext_reqs = [get_external_request_id(r) for r in job.transferred_reqs]
            # 本批实际写出的 payload 字节(排除写失败的 session).
            batch_bytes = sum(
                sum(meta.length) for sid, meta in job.sessions if len(meta.src) > 0 and sid not in job.failed
            )

            def _win_str(a: float | None, b: float | None) -> str:
                return "-" if a is None else f"{a:.6f},{b:.6f}"

            logger.info(
                "[mooncake][perf] P batch=%d layers=%s reqs=%s wait=%.1f event=%.1f "
                "flush=%.1f write=%.1f layerdone=%.1f misc=%.1f total=%.1f ms "
                "t0=%.6f flush_win=%s write_win=%s layerdone_win=%s bytes=%d",
                job.batch_id,
                [t.layer_idx for t in job.tasks],
                batch_ext_reqs,
                job.batch_wait_ms,
                job.event_ms,
                job.flush_ms,
                job.write_ms,
                layerdone_ms,
                misc_ms,
                total_ms,
                job.t_batch0,
                _win_str(job.flush_win0, job.flush_win1),
                _win_str(job.write_win0, job.write_win1),
                _win_str(layerdone_win0, layerdone_win1),
                batch_bytes,
            )
            for ext_req in batch_ext_reqs:
                acc = self._perf_req.get(ext_req)
                if acc is None:
                    acc = {
                        "t0": job.t_batch0,
                        "batches": 0.0,
                        "event": 0.0,
                        "flush": 0.0,
                        "write": 0.0,
                        "layerdone": 0.0,
                        "wait": 0.0,
                    }
                    self._perf_req[ext_req] = acc
                acc["batches"] += 1
                acc["event"] += job.event_ms
                acc["flush"] += job.flush_ms
                acc["write"] += job.write_ms
                acc["layerdone"] += layerdone_ms
                acc["wait"] += job.batch_wait_ms
        for send_task in job.tasks:
            if send_task.layer_idx == (self.total_layers - 1):
                layer_group_idx = self.layer_metadata[send_task.layer_name].tensor_group_idx[0]
                for req_id, req_meta in send_task.send_request.items():
                    if req_id not in job.transferred_reqs:
                        continue
                    if req_meta.chunk_finish:
                        if _PERF_LOG:
                            ext_req = get_external_request_id(req_id)
                            acc = self._perf_req.pop(ext_req, None)
                            if acc is not None:
                                logger.info(
                                    "[mooncake][perf] P req=%s done batches=%d "
                                    "event=%.1f flush=%.1f write=%.1f layerdone=%.1f "
                                    "wait=%.1f total=%.1f ms",
                                    ext_req,
                                    int(acc["batches"]),
                                    acc["event"],
                                    acc["flush"],
                                    acc["write"],
                                    acc["layerdone"],
                                    acc["wait"],
                                    _perf_ms(acc["t0"]),
                                )
                        if req_id in self.failed_reqs:
                            # 失败仍走独立的请求级信号: 写失败的层批根本没有
                            # LAYER_DONE 到达 D, 只靠路径计数会让请求永远等不到
                            # 完成 —— 必须显式通知 D 作废旧块并重试.
                            self._send_failed_signal(req_id, req_meta, layer_group_idx)
                            self.failed_reqs.discard(req_id)
                        elif global_te.use_tcp:
                            # TCP/H2H: 本请求的请求级完成信息已随末批 LAYER_DONE
                            # 的 done_list 下发 (见 _req_done_list), 不再单独发一次
                            # DONE 往返.
                            pass
                        else:
                            # D2D(protocol=ascend): 数据直达 D 的 NPU, 两端之间
                            # 没有 LAYER_DONE 可搭车 —— 沿用独立的请求级 DONE.
                            self._send_done_signal(req_id, req_meta, layer_group_idx)
        # p_then_d: 本批的 write 与 LAYER_DONE 都已完成(含最后层时是阻塞 drain 到
        # 这里的), 可以释放延迟释放的块.
        self._release_delayed_requests(job.tasks, job.transferred_reqs)
        if _PERF_LOG:
            # 批"处理结束"以 LAYER_DONE 全部发出计(与单线程版语义对齐).
            self._last_batch_end_at = time.perf_counter()

    def _send_done_signal(self, req_id, req_meta, layer_group_idx: int) -> None:
        """请求级成功通知(D2D 专用).

        TCP/H2H 的请求级完成信息随末批 LAYER_DONE 的 done_list 下发(见
        _req_done_list), 不走本方法; D2D(protocol=ascend)数据直达对端 NPU,
        两端之间没有逐层 LAYER_DONE 通道可搭车, 请求级完成只能显式通知
        (与改版前 callback_func = send_done_send_signal 的成功路径语义一致).
        """
        self._send_req_signal(req_id, req_meta, layer_group_idx, DONE_SENDING_MSG)

    def _send_failed_signal(self, req_id, req_meta, layer_group_idx: int) -> None:
        """请求级失败通知(成功路径不需要).

        TCP 下成功路径的请求级完成信息已随最后层批的 LAYER_DONE 下发; 但 P 侧写失败时
        该请求的对应层批不会产生任何 LAYER_DONE, D 侧的路径计数永远收不齐, 会一直
        等到 abort 超时 —— 所以失败仍显式通知 D 作废旧块并重试(与旧 FAILED 语义
        一致: 只发给承载最后层所属 group 数据的 peer)。D2D 下没有 LAYER_DONE,
        成功与失败都走本通道。
        """
        self._send_req_signal(req_id, req_meta, layer_group_idx, FAILED_SENDING_MSG)

    def _send_req_signal(self, req_id, req_meta, layer_group_idx: int, msg_type: bytes) -> None:
        """P→D 请求级信号: TCP 失败路径 / D2D 成功与失败共用(REQ-REP, 等 ACK)."""
        external_req_id = get_external_request_id(req_id)
        encoder = msgspec.msgpack.Encoder()
        for (remote_host, remote_port), peer_blocks in req_meta.peer_transfer.items():
            # 请求级发送方路径总数(各 group 并集口径): 与 LAYER_DONE done_list 里的
            # trans_count 保持一致 —— D 侧按 sender_path 去重计数, 收齐才算完成.
            # (layer_group_idx 参数保留以兼容调用方, 不再用于取计数.)
            trans_count = self._request_sender_path_count(peer_blocks)
            if trans_count <= 0:
                continue
            try:
                data_bytes = encoder.encode((msg_type, external_req_id, trans_count, self.sender_path))
                with zmq_ctx(zmq.REQ, make_zmq_path("tcp", remote_host, remote_port)) as sock:  # type: ignore
                    timeout_ms = int(LAYER_DONE_TIMEOUT_S * 1000)
                    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)  # type: ignore
                    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)  # type: ignore
                    sock.send(data_bytes)
                    sock.recv()  # 等 D 侧 ACK: 确保完成/作废动作已被受理
            except zmq.ZMQError as e:  # type: ignore
                logger.error(
                    "Failed to send signal %s for request %s to %s:%d: %s",
                    msg_type,
                    external_req_id,
                    remote_host,
                    remote_port,
                    e,
                )

    def _req_done_list(
        self, transfer_meta: TransferMeta, peer_host: str, peer_port: int
    ) -> list[tuple[str, bool, int, bool]]:
        """该 peer 在本批 LAYER_DONE 里携带的请求级完成信息.

        [(ext_req_id, is_last, trans_count, failed), ...]: 只有末轮(chunk_finish)的
        请求才有条目; failed 表示该请求在 P 侧已判失败, D 侧收到即作废并重试.
        通常为空列表 —— 只有含最后层的批(即该请求的末轮)才会填.
        """
        return [
            (get_external_request_id(rid), is_last, trans_count, rid in self.failed_reqs)
            for (d_host, d_port, rid), (is_last, trans_count) in transfer_meta.req_done.items()
            if is_last and d_host == peer_host and d_port == peer_port
        ]

    def _send_layer_done_signal(
        self,
        remote_host: str,
        remote_port: int,
        req_ids: list[str],
        dst_addrs: list[int],
        lengths: list[int],
        done_list: list[tuple[str, bool, int, bool]] | None = None,
    ) -> bool:
        """H2H: 通知 D 本层 KV 已写入其 staging, 等待 D 完成该层 H2D 后的 ACK.

        REQ-REP 同步往返同时充当流控: ACK 未回前不进入下一层, 保证 D 侧请求级
        done 严格晚于全层 H2D。请求级完成信息(末轮/期望路径数/失败)随本批
        payload 一起下发: D 侧 H2D 成功后就地判定完成, 不再单发一次请求级
        DONE 往返(见 _req_done_list)。
        """
        path = make_zmq_path("tcp", remote_host, remote_port)
        encoder = msgspec.msgpack.Encoder()
        # payload: (LAYER_DONE_SENDING_MSG, req_ids, dst_addrs, lengths,
        #           sender_path, done_list)
        # req_ids 为 external id, 供 D 侧把每批 H2D 归到请求做请求级累计;
        # sender_path 为 P 侧本 rank 的侧信道标识, D 侧按它做多路径(不等 TP)计数;
        # done_list 见 _req_done_list, 非空仅出现在含最后层的批.
        data_bytes = encoder.encode(
            (
                LAYER_DONE_SENDING_MSG,
                req_ids,
                dst_addrs,
                lengths,
                self.sender_path,
                done_list or [],
            )
        )
        try:
            with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
                timeout_ms = int(LAYER_DONE_TIMEOUT_S * 1000)
                sock.setsockopt(zmq.SNDTIMEO, timeout_ms)  # type: ignore
                sock.setsockopt(zmq.RCVTIMEO, timeout_ms)  # type: ignore
                sock.send(data_bytes)
                resp = sock.recv()
                return resp == b"ACK"
        except zmq.ZMQError as e:  # type: ignore
            logger.error(
                "Layer done signal to %s:%d failed: %s",
                remote_host,
                remote_port,
                e,
            )
            return False


class KVCacheRecvingLayerThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        side_channel_port: int,
        tp_size: int,
        pd_head_ratio: int,
        local_engine_id: str,
        metadata: MooncakeAgentMetadata,
        ready_event: threading.Event,
    ):
        super().__init__(daemon=True, name="KVCacheRecvingLayerThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pd_head_ratio = pd_head_ratio
        self.local_engine_id = local_engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = side_channel_port
        self.lock = threading.Lock()
        self.done_requests = set[str]()
        self.failed_requests = set[str]()
        self.task_tracker = dict[str, int]()
        self.ready_event = ready_event
        self.metadata = metadata
        # perf: D 侧请求级累计(external req id -> 累计), 会计完成时打印后清除.
        # 仅在 recv 线程内访问, 无需额外锁.
        self._perf_d_req: dict[str, dict[str, float]] = {}

    def get_and_clear_done_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.lock:
            finished_requests = self.done_requests
            self.done_requests = set()
        return finished_requests

    def get_and_clear_failed_requests(self) -> set[str]:
        """
        Get and clear the requests that have failed.
        Returns:
            A set of request IDs that have failed.
        """
        with self.lock:
            failed_requests = self.failed_requests
            self.failed_requests = set()
        return failed_requests

    def update_failed_task(self, req_id: str) -> None:
        """
        Handle a failed task by adding it to the failed_requests set and removing it from the task tracker.
        Args:
            req_id: The ID of the request that has failed.
        """
        with self.lock:
            if req_id not in self.task_tracker:
                self.task_tracker[req_id] = set()
            self.task_tracker.pop(req_id, None)
            self.failed_requests.add(req_id)

    def apply_layer_done_meta(self, done_list, sender_path: str) -> None:
        """就地应用随 LAYER_DONE 下发的请求级完成信息 (替代独立的 DONE/FAILED 往返).

        done_list = [(ext_req_id, is_last, trans_count, failed), ...], 由 P 侧在
        含最后层的批里填充:
          - failed=True: 该请求在 P 侧已判失败 → 作废重试(与旧 FAILED 语义一致);
          - is_last=True: 该请求末轮数据已全部 H2D 完成 → 按 sender_path 累计
            路径, 收齐 trans_count 条即认为请求完成(不等 TP 时同一请求来自多个
            P rank, 必须按路径去重计数);
          - is_last=False(中间 chunk): 不触发完成.
        """
        for ext_req, is_last, trans_count, failed in done_list:
            if failed:
                # 同批内先到 done 条目、后到 failed 时(多 peer 且后一个 session 写
                # 失败): 撤回已置的完成标记, 避免 scheduler 抢在作废前接纳.
                with self.lock:
                    self.done_requests.discard(ext_req)
                self.update_failed_task(ext_req)
            elif is_last:
                self.update_done_task(ext_req, trans_count, sender_path)

    def update_done_task(self, req_id, trans_count, side_channel_path):
        """
        Handle a completed task by adding it to the done_requests set and removing it from the task tracker.
        Args:
            req_id: The ID of the request that has completed.
        """
        with self.lock:
            if req_id not in self.task_tracker:
                self.task_tracker[req_id] = set()
            self.task_tracker[req_id].add(side_channel_path)
            if len(self.task_tracker[req_id]) == trans_count:
                self.task_tracker.pop(req_id)
                self.done_requests.add(req_id)
                if _PERF_LOG:
                    # 请求级汇总: first2done = 首个 LAYER_DONE 到达 → 会计完成.
                    acc = self._perf_d_req.pop(req_id, None)
                    if acc is not None:
                        logger.info(
                            "[mooncake][perf] D req=%s done layerdone=%d h2d_sum=%.1f first2done=%.1f ms",
                            req_id,
                            int(acc["n"]),
                            acc["h2d"],
                            _perf_ms(acc["t_first"]),
                        )

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        # LAYER_DONE H2D 在本线程执行 (sync_cpu_to_npu_for_transfer), 必须显式
        # 绑定本 rank 的 NPU 设备: torch_npu 的设备/流上下文是线程级的, 未
        # set_device 的线程在自定义 op 内 aclrtGetDevice 会拿到非预期设备,
        # 导致 aclrtMemcpyBatchAsync 参数校验失败 (107000). 与发送线程
        # (KVCacheSendingLayerThread.run) 的 set_device 保持一致.
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        # MC_TCP_CPU_BIND=1 时把本线程绑到进程允许核集中负载最低的核.
        bind_current_thread_to_idle_cpu(f"kv-recv-rank{local_rank}")
        handshake_port = self.side_channel_port + self.tp_rank
        path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
        logger.info("KVCacheRecvingLayerThread listening on %s, tp_rank=%d", path, self.tp_rank)
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
            self.ready_event.set()
            decoder = msgspec.msgpack.Decoder(type=tuple)
            while True:
                try:
                    frames = sock.recv_multipart()
                    if len(frames) < 2:
                        logger.error(
                            "Invalid message format. expected>=2 frames, got %d. frames=%s", len(frames), frames
                        )
                        continue

                    identity = frames[0]
                    payload = [f for f in frames[1:] if f != b""]
                    if len(payload) != 1:
                        logger.error("Invalid payload count. expected=1, got %d. frames=%s", len(payload), frames)
                        continue

                    msg = decoder.decode(payload[0])
                    if msg[0] == GET_META_MSG:
                        logger.info("Got GET META INFO for request %s", msg[0])
                        sock.send_multipart((identity, b"", encoded_data))
                    elif msg[0] == DONE_SENDING_MSG:
                        # 兼容旧 P 端(请求级 DONE 已改为随 LAYER_DONE 的 done_list
                        # 下发, 新 P 不再单发本条; 保留以便灰度/回滚时混部不炸).
                        logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
                        request_id = msg[1]
                        trans_count = msg[2]
                        side_channel_path = msg[3]
                        self.update_done_task(request_id, trans_count, side_channel_path)
                        sock.send_multipart((identity, b"", b"ACK"))
                    elif msg[0] == FAILED_SENDING_MSG:
                        request_id = msg[1]
                        logger.error("Got FAILED_SENDING_MSG for request. request_id=%s. ", msg[1])
                        self.update_failed_task(request_id)
                        sock.send_multipart((identity, b"", b"ACK"))
                    elif msg[0] == LAYER_DONE_SENDING_MSG:
                        # H2H: P 已完成本层推送, 数据落在本节点 CPU staging 中.
                        # 把该层字节区间 H2D 拷回 NPU KV cache 后回 ACK; P 收到
                        # ACK 才推进下一层, 保证请求级 done 严格晚于全层 H2D.
                        # payload: (LAYER_DONE_SENDING_MSG, req_ids, dst_addrs, lengths,
                        #           sender_path, done_list)
                        # done_list = [(ext_req_id, is_last, trans_count, failed), ...]:
                        # 只有含最后层的批(该请求末轮)非空 —— 请求级完成信息随本批
                        # 下发, D 侧 H2D 成功后就地判定完成(不再单发请求级 DONE).
                        layer_req_ids = msg[1]
                        layer_dst_addrs = msg[2]
                        layer_lengths = msg[3]
                        sender_path = msg[4] if len(msg) > 4 else ""
                        done_list = msg[5] if len(msg) > 5 else ()
                        logger.debug("Got LAYER_DONE_SENDING_MSG for %d ranges", len(layer_dst_addrs))
                        t_recv0 = time.perf_counter()
                        layer_reply = b"ACK"
                        try:
                            global_te.sync_cpu_to_npu_for_transfer(layer_dst_addrs, layer_lengths)
                            h2d_ms = _perf_ms(t_recv0)
                        except Exception as e:
                            logger.error(
                                "LAYER_DONE_SENDING_MSG H2D failed: %s. msg=%s",
                                e,
                                msg,
                            )
                            layer_reply = b"NAK"
                            h2d_ms = _perf_ms(t_recv0)
                        sock.send_multipart((identity, b"", layer_reply))
                        if layer_reply == b"ACK":
                            # 就地判定请求级完成/失败: 与 H2D 同一处理点, 省掉 P 侧
                            # 单独一次请求级 DONE/FAILED 往返. 失败时 P 侧不再补发,
                            # 由这里(NAK 分支)直接作废重试.
                            self.apply_layer_done_meta(done_list, sender_path)
                        else:
                            for ext_req in layer_req_ids:
                                self.update_failed_task(ext_req)
                        if _PERF_LOG:
                            # total 覆盖 收到消息 → H2D 完成 → ACK 发出 全程;
                            # t0 为收到 LAYER_DONE 的绝对时刻(CLOCK_MONOTONIC),
                            # H2D 阶段窗口 = [t0, t0 + h2d/1000], 供与 P 侧
                            # write/layerdone 窗口在同机时间线上对齐. H2D 单独
                            # 计时, 不计入 P 侧 H2H(write)统计.
                            logger.info(
                                "[mooncake][perf] D batch layerdone reqs=%s ranges=%d h2d=%.1f total=%.1f ms t0=%.6f",
                                layer_req_ids,
                                len(layer_dst_addrs),
                                h2d_ms,
                                _perf_ms(t_recv0),
                                t_recv0,
                            )
                            # 请求级累计(批共享口径): 该批 h2d 计入消息内全部请求.
                            for ext_req in layer_req_ids:
                                acc = self._perf_d_req.get(ext_req)
                                if acc is None:
                                    acc = {"n": 0.0, "h2d": 0.0, "t_first": t_recv0}
                                    self._perf_d_req[ext_req] = acc
                                acc["n"] += 1
                                acc["h2d"] += h2d_ms
                    else:
                        logger.error(
                            "Unexpected message type: %s. expected GET_META_MSG, "
                            "DONE_SENDING_MSG, FAILED_SENDING_MSG or "
                            "LAYER_DONE_SENDING_MSG. msg=%s",
                            msg[0] if msg else "empty",
                            msg,
                        )
                except Exception as e:
                    logger.error(
                        "Failed to decode message. type=%s, error=%s. context=decoding payload", type(e).__name__, e
                    )


class MooncakeLayerwiseConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.send_task: SendTask = SendTask()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        token_ids: list[int] | None = None,
        chunk_finish: bool = False,
        prompt_len: int = 0,
        remote_cache_tokens: int = 0,
        local_computed_tokens: int = 0,
        local_transed_tokens: int = 0,
        awaiting_params: bool = False,
        delayed_free: bool = False,
    ):
        self.requests[request_id] = ReqMeta(
            token_ids=token_ids or [],
            local_block_ids=local_block_ids,
            awaiting_params=awaiting_params,
            delayed_free=delayed_free,
            remote_block_ids=kv_transfer_params.get("remote_block_ids", []),
            remote_block_size=kv_transfer_params.get("remote_block_size", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id"),
            remote_host=kv_transfer_params.get("remote_host"),
            remote_port=kv_transfer_params.get("remote_port"),
            remote_te_rpc_port=kv_transfer_params.get("remote_te_rpc_port"),
            remote_layer_metadata=kv_transfer_params.get("remote_layer_metadata"),
            metaserver=kv_transfer_params.get("metaserver"),
            remote_tp_size=kv_transfer_params.get("remote_tp_size"),
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size"),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size"),
            do_virtual=kv_transfer_params.get("do_virtual"),
            chunk_finish=chunk_finish,
            remote_cache_tokens=remote_cache_tokens,
            local_computed_tokens=local_computed_tokens,
            prompt_len=prompt_len,
            local_transed_tokens=local_transed_tokens,
            trans_count=[],
        )


class MooncakeLayerwiseConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeLayerwiseConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeLayerwiseConnectorScheduler | None = MooncakeLayerwiseConnectorScheduler(
                vllm_config, kv_cache_config, str(self.engine_id)
            )
            self.connector_worker: MooncakeLayerwiseConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeLayerwiseConnectorWorker(vllm_config, kv_cache_config, str(self.engine_id))

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get the block ids that have load errors."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeLayerwiseConnector does not do layerwise saving."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self, layer_name: str, kv_layer: list[torch.Tensor], attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata, self._connector_metadata)

    def wait_for_save(self):
        """MooncakeLayerwiseConnector does not save explicitly."""
        pass


class MooncakeLayerwiseConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.block_size = [group_spec.kv_cache_spec.block_size for group_spec in kv_cache_config.kv_cache_groups]
        self.engine_id = engine_id
        logger.info("Initializing Mooncake Scheduler %s", engine_id)

        self.side_channel_host = get_ip()

        # disable prefill context parallel on decoder nodes
        if vllm_config.kv_transfer_config.is_kv_consumer:
            assert vllm_config.parallel_config.prefill_context_parallel_size == 1, (
                "Prefill context parallel is not support on decoder nodes"
            )

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
        )

        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int], list[list[int]]]] = {}
        self._reqs_need_send_layerwise: dict[str, SendReqInfo] = {}
        self.need_truncate = self._has_attn_mamba_hybrid_cache(kv_cache_config)
        self.executor = ThreadPoolExecutor(32)
        tls_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("tls_config", {})
        ssl_keyfile = tls_config.get("ssl_keyfile")
        ssl_certfile = tls_config.get("ssl_certfile")
        ssl_ca_certs = tls_config.get("ssl_ca_certs", False)
        ssl_keyfile_password = tls_config.get("ssl_keyfile_password")
        self.cert_path = (ssl_certfile, ssl_keyfile, ssl_keyfile_password)
        self.ssl_enable = tls_config.get("ssl_enable", False)
        self.ca_path = ssl_ca_certs
        if self.ssl_enable:
            self.metaserver_client = httpx.Client(
                limits=httpx.Limits(max_connections=100000), timeout=None, cert=self.cert_path, verify=self.ca_path
            )
        else:
            self.metaserver_client = httpx.Client(limits=httpx.Limits(max_connections=100000), timeout=None)

    @staticmethod
    def _iter_kv_cache_specs(kv_cache_config: KVCacheConfig):
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                yield from kv_cache_spec.kv_cache_specs.values()
            else:
                yield kv_cache_spec

    @classmethod
    def _has_attn_mamba_hybrid_cache(cls, kv_cache_config: KVCacheConfig) -> bool:
        has_attn = False
        has_mamba = False
        for kv_cache_spec in cls._iter_kv_cache_specs(kv_cache_config):
            has_attn = has_attn or isinstance(kv_cache_spec, AttentionSpec)
            has_mamba = has_mamba or isinstance(kv_cache_spec, MambaSpec)
        return has_attn and has_mamba

    def _hybrid_prefill_token_count(self, num_prompt_tokens: int) -> int:
        if self.need_truncate and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_request_for_hybrid_prefill(self, request: "Request") -> None:
        params = request.kv_transfer_params
        if (
            params is None
            or not self.need_truncate
            or params.get("_p_side_truncated")
            or getattr(request, "num_prompt_tokens", len(request.prompt_token_ids or [])) <= 1
        ):
            return

        if request.prompt_token_ids is not None:
            request.prompt_token_ids.pop()
        elif request.prompt_embeds is not None:
            request.prompt_embeds = request.prompt_embeds[:-1]
        else:
            return

        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        request.max_tokens = 1
        params["_p_side_truncated"] = True

    def _trim_hybrid_remote_block_ids(self, block_ids: tuple[list[int], ...], prompt_len: int) -> tuple[list[int], ...]:
        if not self.need_truncate or prompt_len <= 1:
            return block_ids

        trimmed_block_ids: list[list[int]] = []
        for group_block_ids, block_size in zip(block_ids, self.block_size):
            if prompt_len % block_size == 1:
                trimmed_block_ids.append(list(group_block_ids[:-1]))
            else:
                trimmed_block_ids.append(list(group_block_ids))
        return tuple(trimmed_block_ids)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeLayerwiseConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert num_computed_tokens % min(self.block_size) == 0
            count = max(self._hybrid_prefill_token_count(len(request.prompt_token_ids)) - num_computed_tokens, 0)
            return count, count > 0

        if params is not None and params.get("do_remote_decode"):
            self._truncate_request_for_hybrid_prefill(request)

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        # p_then_d 诊断: 从"请求被调度"到"块表推给 P"这一段是提前派发能否与前向重叠的
        # 关键路径, 用 epoch 秒打点便于与 P 侧 "Waiting for decode transfer params" 对齐.
        t_perf = time.perf_counter()
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeLayerwiseConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            do_virtual = params.get("do_virtual", False)
            local_block_ids = (blocks.get_block_ids()) if num_external_tokens > 0 else []
            remote_block_ids = self._trim_hybrid_remote_block_ids(local_block_ids, len(request.prompt_token_ids))
            remote_cached_tokens = request.num_computed_tokens
            # Get unhashed blocks to pull from remote.
            logger.debug(
                "MooncakeLayerwiseConnector update_state_after_alloc: add %s to need recv queue", request.request_id
            )
            self._reqs_need_recv[request.request_id] = (
                request,
                [],  # request._all_token_ids,
                local_block_ids,
            )

            params["do_remote_prefill"] = False

            logger.info("Send request: %s to proxy metaserver: %s", request.request_id, params.get("metaserver", None))
            # All parameters here should appear in the returned dict of
            # request_finished in the scheduler side except "request_id".
            # change the format of request_id if vllm-version >= 0.14.0
            external_req_id = get_external_request_id(request.request_id)
            kv_transfer_params = dict(
                token_ids=[],
                request_id=external_req_id,
                do_remote_prefill=False,
                do_remote_decode=True,
                remote_block_ids=remote_block_ids,
                remote_block_size=self.block_size,
                remote_engine_id=self.engine_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                remote_tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                remote_pcp_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
                remote_dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
                remote_cached_tokens=remote_cached_tokens,
            )
            if not do_virtual:
                p_params_zmq_port = params.get("kv_transfer_params_zmq_port")
                remote_host = params.get("remote_host")
                if p_params_zmq_port is not None and remote_host is None:
                    logger.error(
                        "Request %s carries kv_transfer_params_zmq_port but no remote_host; the prefill "
                        "engine address is required to push decode params (p_then_d).",
                        request.request_id,
                    )
                if p_params_zmq_port is not None and remote_host is not None:
                    # p_then_d: 不经过 proxy metaserver, 直接把 D 的参数推给 P 的直连
                    # 参数通道(每个 P rank 一份). P 是先被派发的一跳, 正等着这些参数
                    # 才能开始逐层推送(见 KVTransferParamsRecvingThread).
                    remote_pcp_size = params.get("remote_pcp_size", 1) or 1
                    remote_tp_size = params.get("remote_tp_size", 1) or 1
                    logger.debug(
                        "[p_then_d] D push block table for %s: alloc->push=%.1f ms",
                        request.request_id,
                        (time.perf_counter() - t_perf) * 1e3,
                    )
                    for rank_idx in range(remote_pcp_size * remote_tp_size):
                        # 每个 P rank 一份; 失败由 _push_kv_transfer_params 内部重试
                        # 并打 error(它自己吞异常, 不往外抛).
                        self.executor.submit(
                            self._push_kv_transfer_params,
                            remote_host=remote_host,
                            remote_port=p_params_zmq_port + rank_idx,
                            message=kv_transfer_params,
                            req_id=request.request_id,
                        )
                else:
                    future = self.executor.submit(
                        self._access_metaserver, url=params.get("metaserver", None), message=kv_transfer_params
                    )

                    def handle_exception(future):
                        if future.exception():
                            logger.error("Access metaserver fail. error=%s. ", future.exception())

                    future.add_done_callback(handle_exception)

        # Layerwise prefiller add request need send
        if params is not None and params.get("do_remote_decode"):
            local_block_ids = list(blocks.get_block_ids())
            logger.debug(
                "MooncakeLayerwiseConnector update_state_after_alloc: add %s to need send queue", request.request_id
            )
            # p_then_d: 请求里没有 D 的参数(块表/侧信道地址), 由 D 事后直连推来 ->
            # 打标记, 逐层映射推迟到发送线程解析(worker.resolve_pending_req).
            # 不变式: 标记与 _reqs_need_send_layerwise 在同一处设置, 所以打了标记的
            # 请求一定会有 ReqMeta 进入 worker, 发送线程总有机会解析/超时后释放块.
            pending_params = params.get("remote_block_ids") is None
            if pending_params and not envs.P_THEN_D:
                logger.error(
                    "Request %s arrived without decoder kv_transfer_params but P_THEN_D is disabled; "
                    "enable P_THEN_D on prefill engines for the p_then_d flow.",
                    request.request_id,
                )
            if pending_params:
                params["_p_then_d"] = True
            remote_cache_tokens = params.get("remote_cached_tokens", 0) or 0
            local_transferred_tokens = remote_cache_tokens
            local_computed_tokens = 0
            self._reqs_need_send_layerwise[request.request_id] = SendReqInfo(
                local_block_ids=local_block_ids,
                local_transferred_tokens=local_transferred_tokens,
                local_computed_tokens=local_computed_tokens,
                request=request,
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeLayerwiseConnectorMetadata()

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            # Loop through scheduled reqs and convert to ReqMeta.
            for req_id, (req, token_ids, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                # For the case where there are no remote blocks to pull
                # (block_ids is empty), we don't need to schedule
                # an async read on the worker side.
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    token_ids=token_ids,
                )

            # Clear the list once workers start the transfers
            self._reqs_need_recv.clear()
        else:
            cached_reqs = scheduler_output.scheduled_cached_reqs
            new_reqs = scheduler_output.scheduled_new_reqs
            scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
            for req_id, new_blocks in zip(cached_reqs.req_ids, cached_reqs.new_block_ids):
                if req_id in self._reqs_need_send_layerwise and new_blocks is not None:
                    self._reqs_need_send_layerwise[req_id].extend_local_block_ids(new_blocks)
            computed_tokens = dict(
                list(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens))
                + [(x.req_id, x.num_computed_tokens) for x in new_reqs]
            )
            for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
                if req_id in self._reqs_need_send_layerwise:
                    send_req_info = self._reqs_need_send_layerwise[req_id]
                    # update local transferred tokens
                    send_req_info.update_transferred_tokens(
                        round_down(send_req_info.local_computed_tokens, min(self.block_size))
                    )
                    # update local computed tokens, not transfer spec decode tokens
                    spec_decode_tokens = (
                        len(scheduled_spec_decode_tokens[req_id]) if (req_id in scheduled_spec_decode_tokens) else 0
                    )
                    send_req_info.update_computed_tokens(
                        computed_tokens.get(req_id, 0) + scheduled_tokens - spec_decode_tokens
                    )

                    def add_transfer_task(req_id, send_req_info: SendReqInfo, chunk_finish=False):
                        (
                            local_block_ids,
                            local_transed_tokens,
                            local_computed_tokens,
                            request,
                        ) = send_req_info.unpack()
                        # remote_cached_tokens: p_then_d 下请求里还没有 D 的参数
                        # (D 事后直连推来), 这里只能是 0; 真正的值在发送线程解析时
                        # 合并进去(否则 None 会进 max()/除法直接 TypeError).
                        meta.add_new_req(
                            request_id=req_id,
                            local_block_ids=local_block_ids,
                            kv_transfer_params=request.kv_transfer_params,
                            token_ids=[],
                            chunk_finish=chunk_finish,
                            remote_cache_tokens=request.kv_transfer_params.get("remote_cached_tokens") or 0,
                            prompt_len=len(request.all_token_ids),
                            local_computed_tokens=local_computed_tokens,
                            local_transed_tokens=local_transed_tokens,
                            awaiting_params=bool(request.kv_transfer_params.get("_p_then_d")),
                            # delayed_free 恒为 False: p_then_d 由 proxy 提前派发 D,
                            # 发送线程不再长时间等参数(request_finished 已不延迟释放).
                            delayed_free=False,
                        )
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "MooncakeLayerwiseConnector build_connector_meta: req_id=%r prompt_len=%s "
                                "local_computed_tokens=%r local_transed_tokens=%r remote_cache_tokens=%s "
                                "chunk_finish=%r local_block_ids=%r remote_block_ids=%s",
                                req_id,
                                len(request.all_token_ids),
                                local_computed_tokens,
                                local_transed_tokens,
                                request.kv_transfer_params.get("remote_cached_tokens"),
                                chunk_finish,
                                local_block_ids,
                                request.kv_transfer_params.get("remote_block_ids"),
                            )

                    # whether chunk finish
                    chunk_finish = send_req_info.local_computed_tokens >= len(send_req_info.request.all_token_ids)

                    add_transfer_task(req_id, send_req_info, chunk_finish=chunk_finish)
                    if chunk_finish:
                        self._reqs_need_send_layerwise.pop(req_id)
        return meta

    def _access_metaserver(self, url, message):
        success = False
        retry = 0
        while retry < 3 and success is False:
            retry += 1
            try:
                self.metaserver_client.post(url, json=message)
                success = True
            except Exception as e:
                logger.error("Failed to connect to metaserver. url=%s, retry=%d. ", url, retry)
                if retry == 3:
                    raise e

    def _push_kv_transfer_params(self, remote_host: str, remote_port: int, message: dict, req_id: str) -> None:
        """p_then_d: 把本 D 的参数推给某个 P rank 的直连参数通道(REQ→ROUTER + ACK).

        P 是先派发的一跳, 正等这些参数(块表/侧信道地址)才能开始逐层推 KV;
        每个 P rank 都要推一份 —— 某一 rank 收不到就不会回 LAYER_DONE, 而 D 侧
        没有请求级超时, 该请求会一直停在 WAITING_FOR_REMOTE_KVS(只能等客户端
        abort), 所以这里失败要重试并打 error, 让问题可见.
        """
        external_req_id = get_external_request_id(req_id)
        path = make_zmq_path("tcp", remote_host, remote_port)
        encoded_data = msgspec.msgpack.Encoder().encode((KV_TRANSFER_PARAMS_MSG, external_req_id, message))
        timeout_sec = self.vllm_config.kv_transfer_config.get_from_extra_config("zmq_timeout_sec", 10)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Sending decode transfer params for request %s to prefill %s:%d (attempt %d/%d)",
                    external_req_id,
                    remote_host,
                    remote_port,
                    attempt,
                    max_retries,
                )
                with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
                    timeout_ms = int(timeout_sec * 1000)
                    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)  # type: ignore
                    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)  # type: ignore
                    sock.send(encoded_data)
                    ack = sock.recv()
                    if ack != b"ACK":
                        raise RuntimeError(f"Unexpected ACK response: {ack}")
                return
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        "Failed to send decode transfer params for request %s to %s:%d on attempt %d/%d: %s. "
                        "Retrying...",
                        external_req_id,
                        remote_host,
                        remote_port,
                        attempt,
                        max_retries,
                        e,
                    )
                    time.sleep(0.1)
                else:
                    logger.error(
                        "Failed to send decode transfer params for request %s to prefill %s:%d after %d attempts: %s. "
                        "The request will stay in WAITING_FOR_REMOTE_KVS until the client aborts.",
                        external_req_id,
                        remote_host,
                        remote_port,
                        max_retries,
                        e,
                    )

    def _p_then_d_send_params(self, request: "Request") -> dict[str, Any]:
        """p_then_d: P 在自己响应里回给 proxy 的参数(proxy 再转给 D).

        除了 entrypoint 补丁追加的 prefilled_token/stop_reasons/prompt_token_ids
        (REUSE_PREFILLED_TOKENS/SKIP_DECODE_TOKENIZE), 这里要给出 D 找到本 P 并
        把 D 自己参数推过来所需的信息: 主机、TP/PCP 规模、参数直连通道基址
        (D 按 base+rank_idx 逐个推, 见 KVTransferParamsRecvingThread)、引擎 id.
        必须非空: entrypoint 补丁用 ``if final_res.kv_transfer_params:`` 区分 P/D
        分支, 空 dict 会让 P 侧走进"复用 D 首 token"的分支.
        """
        parallel_config = self.vllm_config.parallel_config
        return dict(
            do_remote_prefill=False,
            do_remote_decode=True,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_tp_size=parallel_config.tensor_parallel_size,
            remote_pcp_size=parallel_config.prefill_context_parallel_size,
            remote_dcp_size=parallel_config.decode_context_parallel_size,
            kv_transfer_params_zmq_port=kv_transfer_params_zmq_port_base(self.vllm_config),
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        params = request.kv_transfer_params
        if params is not None and params.get("_p_then_d"):
            # p_then_d: 参数要回给 proxy 转发(proxy 用它派发 D / 兜底发现 P 的身份),
            # 但**块不再延迟释放**: D 由 proxy 提前派发(块表在 P 前向期间就推来),
            # 发送线程不再长时间阻塞, 滞后回到 D-first 的量级; 而数据面读的是
            # CPU staging, 一旦 D2H flush 落地, NPU 块被复用不影响已发出的数据.
            # 延迟释放的前提(等参数期间块必须有效)已不成立, 而它依赖的
            # finished_sending 上报路径在 worker.get_finished 里被丢弃 —— 继续延迟
            # 只会让请求永久留在 Scheduler.requests 里(块泄漏).
            return False, self._p_then_d_send_params(request)
        # layer_wise push, not need delay_free_blocks
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        params = request.kv_transfer_params
        if params is not None and params.get("_p_then_d"):
            # 同 request_finished: HMA 路径(本连接器实现 SupportsHMA)走这里.
            return False, self._p_then_d_send_params(request)
        # layer_wise push, not need delay_free_blocks
        return False, None

    def prefilled_token_spool_dir(self, request: "Request") -> str | None:
        """p_then_d: 首 token 的 spool 目录(请求没带 kv_port 时返回 None).

        目录按 P 的 kv_port 分: 同机多实例各自一份; 写侧(API server)与读侧
        (EngineCore)都用请求里同一份 kv_port 推出来, 不需要额外的环境约定。
        """
        params = request.kv_transfer_params
        kv_port = params.get("kv_port") if params else None
        if kv_port is None:
            return None
        return os.path.join(PTD_TOKEN_SPOOL_ROOT, str(kv_port))

    def deliver_prefilled_token(self, request: "Request", timeout: float) -> dict[str, Any] | None:
        """p_then_d: 等 proxy 把首 token 投递到 spool(超时返回 None).

        只在 D 的 scheduler 线程调用, 且只在 KV 接收完成、请求即将离开
        WAITING_FOR_REMOTE_KVS 时 —— 此时 P 早已出首 token, 正常情况下 spool 里
        已经有记录(proxy 在拿到首 token 后、驱动 D 的响应流之前就投递了)。
        """
        spool_dir = self.prefilled_token_spool_dir(request)
        if spool_dir is None:
            return None
        return get_prefilled_token_spool(spool_dir).wait_for_prefilled_token(request.request_id, timeout)


class MooncakeLayerwiseConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())

        if TransferEngine is None:
            raise RuntimeError("mooncake is not available")
        logger.info("Initializing Mooncake work %s", engine_id)

        # Metadata.
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        self.kv_cache_specs: list[KVCacheSpec] = [spec.kv_cache_spec for spec in self.kv_cache_config.kv_cache_groups]
        self.local_engine_id: str = " "
        self.engine_id = engine_id
        self.dp_rank: int = vllm_config.parallel_config.data_parallel_rank
        self.tp_rank: int = get_tensor_model_parallel_rank()
        self.tp_size: int = vllm_config.parallel_config.tensor_parallel_size
        self.pcp_size: int = vllm_config.parallel_config.prefill_context_parallel_size
        self.pcp_rank: int = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size: int = vllm_config.parallel_config.decode_context_parallel_size
        self.dcp_rank: int = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        self.tp_group = get_tp_group()
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.use_mla = self.vllm_config.model_config.use_mla
        self.request_map = dict[str, str]()
        self.use_attn_mamba_hybrid = False
        if self.use_mla:
            self.total_num_kv_heads = 1
        else:
            self.total_num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + self.dp_rank * self.pcp_size * self.tp_size
            + self.pcp_rank * self.tp_size
        )
        self.handshake_port = self.side_channel_port + self.tp_rank
        self.sockets: dict = {}
        logger.info("Initializing Mooncake work %s", engine_id)
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_recv_layer_thread: KVCacheRecvingLayerThread | None = None
        self.kv_send_layer_thread: KVCacheSendingLayerThread | None = None

        # p_then_d: D→P 参数直连通道(P 侧监听; D 派发后把块表/侧信道地址推过来).
        self.params_recv_thread: KVTransferParamsRecvingThread | None = None
        self.wait_transfer_params_timeout_sec = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {"wait_transfer_params_timeout_sec": DEFAULT_WAIT_TRANSFER_PARAMS_TIMEOUT_SEC}
            ).get("wait_transfer_params_timeout_sec", DEFAULT_WAIT_TRANSFER_PARAMS_TIMEOUT_SEC)
        )
        self.prefill_transfer_params_expire_sec = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {"prefill_transfer_params_expire_sec": DEFAULT_PREFILL_TRANSFER_PARAMS_EXPIRE_SEC}
            ).get("prefill_transfer_params_expire_sec", DEFAULT_PREFILL_TRANSFER_PARAMS_EXPIRE_SEC)
        )
        # 延迟释放块的上报集合(delayed_free 请求): get_finished 报给 scheduler,
        # 由它走 finished_sending 释放(见 _free_blocks).
        self.done_sending: set[str] = set()
        # 已上报过的请求(去重): 一个请求可能在多个 step 满足"末轮 + group 末层"判据
        # (prefill 步与随后的 decode 步都会), 重复上报会让 scheduler 的
        # ``assert req_id in self.requests`` 直接炸掉引擎. 有界(长跑不涨内存).
        self._reported_done_sending: dict[str, bool] = SizedDict(max_size=16000)
        self.done_sending_lock = threading.Lock()
        # 对端元数据/GET_META 的互斥: p_then_d 下发送线程也会拉远端元数据,
        # 与模型线程(save_kv_layer->update_decoder_info)共享 socket/缓存.
        self.remote_meta_lock = threading.Lock()

        self.block_size: list[int] = [spec.block_size for spec in self.kv_cache_specs]
        self.kernel_block_size_scale: list[int] = [1 for _ in range(self.num_kv_cache_groups)]
        self.layer_metadata: dict[str, LayerMetadata] = {}
        self.attn_resharding_group_idx = set[int]()

        self.enable_kv_quant = (
            vllm_config.quant_config.enable_fa_quant if vllm_config.quant_config is not None else False
        )
        self.enable_c8_quant = (
            vllm_config.quant_config.enable_c8_quant if vllm_config.quant_config is not None else False
        )
        self.pd_head_ratio = get_ascend_config().pd_head_ratio
        self.num_head_replica = get_ascend_config().num_head_replica
        self.resharding_stream = None
        if self.pd_head_ratio > 1 or self.enable_kv_quant or self.enable_c8_quant:
            self.resharding_stream = torch.npu.Stream()

        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.encoder = msgspec.msgpack.Encoder()

        self.index_to_name = defaultdict(list)
        self.remote_layer_metadata: dict[str, dict[int, dict[str, LayerMetadata]]] = SizedDict()
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.timeout = 1.0  # seconds
        self.k_buffer: torch.Tensor | None = None
        self.v_buffer: torch.Tensor | None = None
        self.virtual_request: set[str] = set()
        self._invalid_block_ids: set[int] = set()
        self._recving_metadata: dict[str, ReqMeta] = {}

    def create_kv_buffer(self, first_kv_cache_tuple):
        alignment = 2 * 1024 * 1024
        first_k_cache = first_kv_cache_tuple[0]
        first_v_cache = first_kv_cache_tuple[1]
        if self.pd_head_ratio > 1 or self.enable_kv_quant or self.enable_c8_quant:
            # regesit kv buffer
            if self.enable_kv_quant:
                k_dtype = torch.int8
                v_dtype = first_v_cache.dtype
            elif self.enable_c8_quant:
                k_dtype = torch.int8
                v_dtype = torch.int8
            else:
                k_dtype = first_k_cache.dtype
                v_dtype = first_v_cache.dtype
            self.k_buffer = torch.zeros(first_k_cache.numel() + alignment, dtype=k_dtype, device=first_k_cache.device)
            self.k_buffer = align_memory(self.k_buffer, alignment)[: first_k_cache.numel()].view(
                -1, first_k_cache.shape[-1]
            )
            self.v_buffer = torch.zeros(first_v_cache.numel() + alignment, dtype=v_dtype, device=first_k_cache.device)
            self.v_buffer = align_memory(self.v_buffer, alignment)[: first_v_cache.numel()].view(
                -1, first_v_cache.shape[-1]
            )
        for tensor in (self.k_buffer, self.v_buffer):
            assert tensor.data_ptr() % alignment == 0, "The address of the registered kv cache should be aligned to 2M"
            if global_te.use_tcp:
                # TCP 数据面只能读写注册过的 host 内存: 设备内存不注册, reshard/
                # 量化 buffer 的 CPU staging 由 register_kv_caches 随 kv cache
                # 一并登记(见 register_tcp_staging 的 extra_tensors).
                continue
            ret_value = self.engine.register_memory(tensor.data_ptr(), tensor.numel() * tensor.element_size())
            logger.info("Register memory buffer for transfer, buffer size:%s", tensor.numel() * tensor.element_size())
            if ret_value != 0:
                raise RuntimeError("Mooncake memory registration failed. ")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""
        layer2group_ids: dict[str, int] = {}
        kv_cache_groups = self.kv_cache_config.kv_cache_groups
        for i, kv_cache_group_spec in enumerate(kv_cache_groups):
            for layer_name in kv_cache_group_spec.layer_names:
                layer2group_ids[layer_name] = i

        use_mamba, use_attn = False, False
        conv_total_padding_size = 0
        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            for layer_name in kv_cache_tensor.shared_by:
                layer_kv_cache_spec = self.kv_cache_specs[layer2group_ids[layer_name]]
                if isinstance(layer_kv_cache_spec, MambaSpec):
                    use_mamba = True
                    conv_shape, conv_dtype = layer_kv_cache_spec.shapes[0], layer_kv_cache_spec.dtypes[0]
                    conv_total_padding_size = (
                        self.kv_cache_config.num_blocks * math.prod(conv_shape) * get_dtype_size(conv_dtype)
                    )
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    use_attn = True
            if use_mamba and use_attn:
                self.use_attn_mamba_hybrid = True
                break

        ptrs = []
        lengths = []
        use_kv_buffer = False
        kv_buffer = None
        for layer_name, kv_cache_tuple in kv_caches.items():
            if isinstance(kv_cache_tuple, (list, tuple)) is False:
                kv_cache_tuple = [kv_cache_tuple]
            layer_kv_group_id = layer2group_ids[layer_name]
            layer_kv_cache_spec = kv_cache_groups[layer_kv_group_id].kv_cache_spec
            if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
            if (
                (self.pd_head_ratio > 1 and (isinstance(layer_kv_cache_spec, (FullAttentionSpec, SlidingWindowSpec))))
                or self.enable_kv_quant
                or self.enable_c8_quant
            ):
                self.attn_resharding_group_idx.add(layer_kv_group_id)
                if use_kv_buffer is False:
                    use_kv_buffer = True
                    kv_buffer = kv_cache_tuple
            single_layer_meta = LayerMetadata([], [], [], [])
            for single_kv_cache in kv_cache_tuple:
                block_start_rank = 1
                num_blocks = self.kv_cache_config.num_blocks
                tensor_num_blocks = single_kv_cache.shape[0]
                assert tensor_num_blocks % num_blocks == 0, (
                    "The external block size must be an integer multiple of the kernel block size."
                )
                block_size_scale = tensor_num_blocks // num_blocks
                block_shape = single_kv_cache.shape[block_start_rank:]
                single_layer_meta.tensor_group_idx.append(layer_kv_group_id)
                single_layer_meta.kv_caches_base_addr.append(single_kv_cache.data_ptr())
                single_layer_meta.block_len.append(single_kv_cache.element_size() * math.prod(block_shape))
                single_layer_meta.block_size_scale.append(block_size_scale)
                self.kernel_block_size_scale[layer2group_ids[layer_name]] = block_size_scale
                if single_kv_cache.data_ptr() not in ptrs and not self.use_attn_mamba_hybrid:
                    ptrs.append(single_kv_cache.data_ptr())
                    lengths.append(
                        num_blocks * single_kv_cache.element_size() * math.prod(block_shape) * block_size_scale
                    )
                logger.info("layer: %s, num_blocks: %s, block_shape: %s", layer_name, num_blocks, block_shape)
            self.layer_metadata[layer_name] = single_layer_meta

        if self.use_attn_mamba_hybrid:
            for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
                tensor_addrs = []
                for layer_name in kv_cache_tensor.shared_by:
                    tensor_addrs.extend(self.layer_metadata[layer_name].kv_caches_base_addr)
                    if "mtp" in layer_name:
                        tensor_addrs.append(min(tensor_addrs) - conv_total_padding_size)
                assert min(set(tensor_addrs)) % (2 * 1024 * 1024) == 0, "Tensor start addr is not align with 2M."
                ptrs.append(min(set(tensor_addrs)))
                lengths.append(kv_cache_tensor.size)

        if self.use_attn_mamba_hybrid:
            register_regions = RegisterRegions(ptrs=ptrs, lengths=lengths)
        else:
            # For normal attention / sparse-c8 KV cache, register merged memory
            # ranges while keeping layer metadata at logical tensor addresses.
            register_regions = collect_storage_merged_register_regions(kv_caches)

        validate_register_region_count(register_regions)

        if use_kv_buffer and self.vllm_config.kv_transfer_config.is_kv_producer:
            # reshard/量化 buffer 只服务于发送路径(get_transfer_meta 的 src 基址、
            # save_kv_layer 的 reshard 落点), consumer 侧从不使用 → 不分配.
            # 该 buffer 是「一整层物理 cache」大小: 1024-token page 下 D 侧单块
            # 就有 3+ GiB, 而 decode 侧 0.95 util 下余量只有几百 MB, 分配即 OOM.
            # 必须在 staging 构建前分配: TCP 下它也要镜像进 CPU staging.
            self.create_kv_buffer(kv_buffer)

        # TCP 下 reshard/量化 buffer 也要走 staging; 只有 producer 侧会传输,
        # consumer 侧不必为其付出 CPU 镜像开销.
        extra_staging_tensors: list[torch.Tensor] = []
        if global_te.use_tcp and self.vllm_config.kv_transfer_config.is_kv_producer and self.k_buffer is not None:
            extra_staging_tensors = [self.k_buffer, self.v_buffer]

        if global_te.use_tcp:
            if self.use_attn_mamba_hybrid:
                # Hybrid (attn + mamba/GDN linear-attn) 模型: 各层 cache 是共享
                # 物理 tensor 上的视图(含页填充/对齐空隙), 按物理 tensor 整体
                # 镜像到 CPU staging (region mode), 保持任意字节地址的线性映射,
                # 与 D2D 的 use_attn_mamba_hybrid 注册语义一致.
                hybrid_regions = []
                for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
                    tensor_addrs = []
                    for layer_name in kv_cache_tensor.shared_by:
                        tensor_addrs.extend(self.layer_metadata[layer_name].kv_caches_base_addr)
                        if "mtp" in layer_name:
                            tensor_addrs.append(min(tensor_addrs) - conv_total_padding_size)
                    hybrid_regions.append((min(tensor_addrs), kv_cache_tensor.size))
                # reshard/量化 buffer 是独立物理分配, 作为额外 region 追加
                # (region 模式要求各 region 互不重叠).
                hybrid_regions.extend(
                    (tensor.data_ptr(), tensor.numel() * tensor.element_size()) for tensor in extra_staging_tensors
                )
                global_te.register_tcp_staging_regions(hybrid_regions)
            else:
                global_te.register_tcp_staging(kv_caches, extra_tensors=extra_staging_tensors)
            if self.vllm_config.kv_transfer_config.is_kv_consumer:
                # consumer 发布的是自己 CPU staging 的地址: P 推写的目的地.
                # producer 侧 layer_metadata 保留 NPU 地址供内部 src 计算.
                for layer_name, layer_meta in self.layer_metadata.items():
                    replaced = []
                    for base_addr in layer_meta.kv_caches_base_addr:
                        cpu_addr = global_te.npu_addr_to_cpu_addr(base_addr)
                        if cpu_addr is None:
                            raise RuntimeError(
                                f"H2H layerwise: layer {layer_name} tensor "
                                f"0x{base_addr:x} not found in TCP staging map."
                            )
                        replaced.append(cpu_addr)
                    layer_meta.kv_caches_base_addr = replaced
                logger.info(
                    "TCP mode: consumer published CPU staging addrs for %d layers.",
                    len(self.layer_metadata),
                )
        else:
            global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        num_attn_module = 2 if self.vllm_config.model_config.hf_text_config.model_type == "longcat_flash" else 1
        mtp_layer_name = ""
        for layer_name in kv_caches:
            if "mtp" in layer_name:
                mtp_layer_name = layer_name
                continue
            self.index_to_name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)
            assert len(self.index_to_name[extract_layer_index(layer_name, num_attn_module)]) == 1, (
                "Mooncake Layerwise Connector does not support multiple `attn_module` in one layer now."
            )
        if mtp_layer_name != "":
            self.index_to_name[max(self.index_to_name.keys()) + 1].append(mtp_layer_name)
        if self.total_layers < len(self.layer_metadata.keys()):
            self.total_layers = len(self.layer_metadata.keys())

        # 每个 kv cache group 的"末层"序号: 某 rank 发到该层时, 它在这个 group 上
        # 要发的数据已全部发完. 不等分 P/D 切分下 mamba(linear-attn) 组每个 D rank
        # 有多个 P 发送方, 且这些层永远不是 total_layers-1 —— 只有让每个发送方在
        # 自己的 group 末层下发请求级完成信息, D 侧才可能等到全部发送方的数据落地
        # (否则 D 会只凭最后一层那批(MTP 层, 属 attention 组)就放行解码).
        self.group_max_layer_idx: dict[int, int] = {}
        for layer_idx, layer_names in self.index_to_name.items():
            for layer_name in layer_names:
                group_idx = self.layer_metadata[layer_name].tensor_group_idx[0]
                self.group_max_layer_idx[group_idx] = max(self.group_max_layer_idx.get(group_idx, -1), layer_idx)

        # After KV Caches registered, start the sending or receiving thread.
        metadata = MooncakeAgentMetadata(
            te_rpc_port=self.te_rpc_port,
            layer_metadata=self.layer_metadata,
        )
        if self.vllm_config.kv_transfer_config.is_kv_producer:
            # p_then_d: 参数直连通道线程(D 派发后把块表/侧信道地址推过来). 无论是否
            # 用 p_then_d 都起(端口固定, 不占资源); 只有"请求里没有 D 参数"的请求会
            # 真的用到它.
            params_recv_ready = threading.Event()
            self.params_recv_thread = KVTransferParamsRecvingThread(
                kv_transfer_params_zmq_port=kv_transfer_params_zmq_port_base(self.vllm_config)
                + self.pcp_rank * self.tp_size
                + self.tp_rank,
                ready_event=params_recv_ready,
                timeout=self.timeout,
                wait_transfer_params_timeout_sec=self.wait_transfer_params_timeout_sec,
                prefill_transfer_params_expire_sec=self.prefill_transfer_params_expire_sec,
            )
            self.params_recv_thread.start()
            params_recv_ready.wait()
            ready_event = threading.Event()
            self.kv_send_layer_thread = KVCacheSendingLayerThread(
                engine=self.engine,
                vllm_config=self.vllm_config,
                kv_cache_config=self.kv_cache_config,
                kv_cache_specs=self.kv_cache_specs,
                attn_resharding_group_idx=self.attn_resharding_group_idx,
                total_layers=self.total_layers,
                ready_event=ready_event,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
                pd_head_ratio=self.pd_head_ratio,
                num_head_replica=self.num_head_replica,
                layer_metadata=self.layer_metadata,
                group_max_layer_idx=self.group_max_layer_idx,
                use_mla=self.use_mla,
                use_attn_mamba_hybrid=self.use_attn_mamba_hybrid,
                k_buffer=self.k_buffer,
                v_buffer=self.v_buffer,
                enable_kv_quant=self.enable_kv_quant,
                enable_c8_quant=self.enable_c8_quant,
                resharding_stream=self.resharding_stream,
                sender_path=f"{self.side_channel_host}:{self.handshake_port}",
                resolve_pending_req=self.resolve_pending_req,
                add_done_sending_request=self.add_done_sending_request,
            )
            self.kv_send_layer_thread.start()
            ready_event.wait()

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            ready_event = threading.Event()
            self.kv_recv_layer_thread = KVCacheRecvingLayerThread(
                self.tp_rank,
                self.side_channel_port,
                self.tp_size,
                self.pd_head_ratio,
                self.engine_id,
                metadata,
                ready_event,
            )
            self.kv_recv_layer_thread.start()
            ready_event.wait()

    def add_done_sending_request(self, req_id: str) -> None:
        """上报"本 rank 已发完该请求" -> scheduler 走 finished_sending 释放块.

        只对 delayed_free 的请求调用; p_then_d 下 delayed_free 恒为 False(块由
        request_finished 直接释放), 因此本函数当前无调用者, 保留是为了将来若某条
        路径需要延迟释放时能直接用. 注意: **报一个没有延迟释放的请求 ID 会让
        scheduler 的 ``assert req_id in self.requests`` 直接炸掉引擎**, 而
        worker.get_finished 目前也并未把 done_sending 返回给 scheduler.
        """
        with self.done_sending_lock:
            if req_id in self._reported_done_sending:
                return
            self._reported_done_sending[req_id] = True
            self.done_sending.add(req_id)
        if self.params_recv_thread is not None:
            # 请求发完, 参数缓存不再需要(防长时间挂着的内存/状态).
            self.params_recv_thread.pop_kv_transfer_params(req_id)

    def get_and_clear_done_sending(self) -> set[str]:
        with self.done_sending_lock:
            done_sending = self.done_sending
            self.done_sending = set()
        return done_sending

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.get_and_clear_done_sending() if self.vllm_config.kv_transfer_config.is_kv_producer else set()
        )
        if len(done_sending) > 0:
            logger.info("Number of completed KV cache send requests: %s", done_sending)
        done_recving = (
            self.kv_recv_layer_thread.get_and_clear_done_requests(  # type: ignore[union-attr]
            )
            if self.vllm_config.kv_transfer_config.is_kv_consumer
            else set()
        )
        done_recving = {self.request_map[s] for s in done_recving if s in self.request_map}
        done_recving.update(self.virtual_request)
        self.virtual_request = set()

        failed_recving = (
            self.kv_recv_layer_thread.get_and_clear_failed_requests()
            if self.vllm_config.kv_transfer_config.is_kv_consumer and self.kv_recv_layer_thread is not None
            else set()
        )
        failed_recving = {self.request_map[s] for s in failed_recving if s in self.request_map}
        for req_id in failed_recving:
            if meta := self._recving_metadata.get(req_id):
                self._invalid_block_ids.update(block_id for group in meta.local_block_ids for block_id in group)
        for req_id in done_recving.union(failed_recving):
            org_req_id = req_id[:-9]
            self.request_map.pop(org_req_id, None)
            self._recving_metadata.pop(req_id, None)
        if len(done_recving) > 0:
            logger.info(
                "Number of completed KV cache recv requests: %s, receive requests: %s", len(done_recving), done_recving
            )
        # 注意: done_sending 只用于日志, **不返回给 scheduler**. scheduler 的
        # finished_sending 分支有 assert req_id in self.requests, 而无主上报
        # (请求早已释放)会直接把 EngineCore 打挂; layerwise push 的块不延迟释放,
        # 也就不需要这条上报. p_then_d 下 delayed_free 同样恒为 False, 无调用者.
        return set(), done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Return and clear the set of block IDs that failed to load.
        This is called by the scheduler to identify blocks that need
        to be retried after a transfer failure.
        """
        result = self._invalid_block_ids
        self._invalid_block_ids = set()
        return result

    # {(ip, port)]: {local_block_ids: [], remote_block_ids: {}}}
    def _get_kv_split_metadata(self, req_meta: ReqMeta, req_idx: int, req_id: str, group_idx: int):
        remote_pcp_size = req_meta.remote_pcp_size
        remote_dcp_size = req_meta.remote_dcp_size
        remote_tp_size = req_meta.remote_tp_size
        remote_hosts = [req_meta.remote_host]
        remote_port = req_meta.remote_port
        local_transed_tokens = max(req_meta.remote_cache_tokens, req_meta.local_transed_tokens)
        # local_transed_tokens tokens that have already been transmitted on the local side
        local_computed_tokens = req_meta.local_computed_tokens
        prompt_len = req_meta.prompt_len
        p_parallel_info = parallel_info(
            tp_size=self.tp_size,
            pcp_size=self.pcp_size,
            dcp_size=self.dcp_size,
            pd_head_ratio=self.pd_head_ratio,
            use_mla=self.use_mla,
        )
        d_parallel_info = parallel_info(
            tp_size=remote_tp_size,
            pcp_size=remote_pcp_size,
            dcp_size=remote_dcp_size,
            pd_head_ratio=self.pd_head_ratio,
            use_mla=self.use_mla,
        )
        cp_size = self.pcp_size * self.dcp_size
        # to_trans_idx all tokens that have been processed up to the current step
        if req_meta.chunk_finish:
            to_trans_idx = math.ceil(local_computed_tokens / self.block_size[group_idx])
        else:
            to_trans_idx = math.floor(local_computed_tokens / self.block_size[group_idx])
        prompt_block_size = math.ceil(prompt_len / self.block_size[group_idx])
        #
        num_local_blocks = prompt_block_size // cp_size + int(
            (prompt_block_size % cp_size) > (self.pcp_rank * self.dcp_size + self.dcp_rank)
        )
        already_send_blocks = to_trans_idx // cp_size + int(
            (to_trans_idx % cp_size) > (self.pcp_rank * self.dcp_size + self.dcp_rank)
        )
        if num_local_blocks == already_send_blocks:
            req_meta.chunk_finish = True
        transed_idx = math.floor(local_transed_tokens / self.block_size[group_idx])

        p_cp_group = get_cp_group(self.tp_size, self.total_num_kv_heads, self.dcp_size)
        d_cp_group = get_cp_group(remote_tp_size, self.total_num_kv_heads, remote_dcp_size)
        logger.debug("Compute cp group for P&D req_id=%r p_cp_group=%r d_cp_group=%r", req_id, p_cp_group, d_cp_group)

        cp_ratio = len(p_cp_group) // len(d_cp_group)
        if cp_ratio == 0:
            selected_p_cp_groups = p_cp_group
            selected_d_cp_groups = d_cp_group
        else:
            x = req_idx % cp_ratio
            start = x * len(d_cp_group)
            selected_p_cp_groups = p_cp_group[start : (start + len(d_cp_group))]
            selected_d_cp_groups = d_cp_group
        assert len(selected_p_cp_groups) == len(selected_d_cp_groups)

        p_head_group_rank = (self.tp_rank - self.dcp_rank) // self.dcp_size
        selected_p_cp_group = []
        selected_d_cp_group = []
        for idx, cp_group in enumerate(selected_p_cp_groups):
            if p_head_group_rank in cp_group:  # Check whether the rank is in selected_p_cp_groups
                selected_p_cp_group = cp_group
                selected_d_cp_group = selected_d_cp_groups[idx]
        if len(selected_p_cp_group) == 0:
            return {}

        logger.debug(
            "MooncakeLayerwiseConnector _get_kv_split_metadata req_id=%r "
            "P-side selected head_group cp group: %s, D-side selected head_group cp group: %s",
            req_id,
            selected_p_cp_group,
            selected_d_cp_group,
        )

        context_parallel_parameters_check(
            remote_pcp_size, remote_dcp_size, p_parallel_info, d_parallel_info, self.total_num_kv_heads
        )
        p_rank_block_mapping, d_block_rank_mapping, pd_head_mapping, d_trans_count_mapping = (
            get_local_remote_block_port_mappings(
                to_trans_idx,
                p_parallel_info,
                d_parallel_info,
                remote_hosts,
                remote_port,
                selected_p_cp_group,
                selected_d_cp_group,
                prompt_len,
                self.block_size[group_idx],
                req_meta,
                self.total_num_kv_heads,
                req_id,
            )
        )
        transfer_mappings = get_transfer_mappings(
            p_rank_block_mapping,
            d_block_rank_mapping,
            pd_head_mapping,
            d_trans_count_mapping,
            req_meta,
            group_idx,
            p_parallel_info,
            req_id,
            transed_idx,
            to_trans_idx,
            self.tp_rank,
            self.pcp_rank,
            self.dcp_rank,
        )
        return transfer_mappings

    def _get_kv_split_metadata_for_mamba(self, req_meta: ReqMeta, req_idx: int, req_id: str, group_idx: int):
        assert self.tp_size >= req_meta.remote_tp_size, (
            "Mamba group prefill TP_size must equal or larger than decode TP_size."
        )
        remote_tp_size = req_meta.remote_tp_size
        tp_raito = self.tp_size // remote_tp_size
        remote_host = req_meta.remote_host
        remote_port = req_meta.remote_port + self.tp_rank // tp_raito
        transfer_mappings: dict[tuple[str, int], dict[str, Any]] = {}
        if req_meta.chunk_finish:
            transfer_mappings[(remote_host, remote_port)] = {
                "local_block_ids": req_meta.local_block_ids[group_idx],
                "remote_block_ids": req_meta.remote_block_ids[group_idx],
                "trans_count": self.tp_size // remote_tp_size,
            }

        return transfer_mappings

    def _align_remote_block_ids(self, req_meta: ReqMeta):
        remote_block_size = req_meta.remote_block_size
        remote_block_ids = req_meta.remote_block_ids
        for i in range(self.num_kv_cache_groups):
            if isinstance(self.kv_cache_specs[i], MambaSpec):
                continue
            if remote_block_size[i] != self.block_size[i] and len(req_meta.remote_block_ids[i]) > 0:
                assert remote_block_size[i] > self.block_size[i] and remote_block_size[i] % self.block_size[i] == 0, (
                    "Remote block size must be divisible by local block size."
                )
                assert self.pcp_size * self.dcp_size * req_meta.remote_pcp_size * req_meta.remote_dcp_size == 1, (
                    "Context parallel does not support different P/D block size now."
                )
                pd_block_size_ratio = remote_block_size[i] // self.block_size[i]
                remtote_block_ids_with_scale = [
                    block_id * pd_block_size_ratio + j
                    for block_id in remote_block_ids[i]
                    for j in range(pd_block_size_ratio)
                ]
                req_meta.remote_block_ids[i] = remtote_block_ids_with_scale

    def _get_kernel_block_ids(self, block_ids):
        for i in range(self.num_kv_cache_groups):
            if isinstance(self.kv_cache_specs[i], MambaSpec):
                continue
            if len(block_ids[i]) > 0:
                block_ids[i] = [
                    block_id * self.kernel_block_size_scale[i] + j
                    for block_id in block_ids[i]
                    for j in range(self.kernel_block_size_scale[i])
                ]
        return block_ids

    def _check_pending_params_supported(self, req_id: str) -> None:
        """p_then_d 延迟解析的支持范围检查(只对等待参数的请求).

        cp 开启时本 rank 在请求上的块只是全量块表的一个子集, 兜底块表(见
        start_load_kv 的 reshard 部分)会多算 —— 未验证组合, 直接报错.
        """
        if self.pcp_size > 1 or self.dcp_size > 1:
            logger.error(
                "p_then_d with prefill/decode context parallel is not supported (request %s); "
                "use pcp_size=dcp_size=1 for the deferred send path.",
                req_id,
            )

    def _resolve_send_mapping(self, req_id: str, req_meta: ReqMeta) -> None:
        """按 ReqMeta 里的 D 参数算出逐 peer 传输映射(peer_transfer).

        抽自 start_load_kv 的生产者分支: p_then_d 下 D 参数后到, 同一段逻辑要由发送
        线程在参数到达后调用(见 resolve_pending_req), 因此不能再依赖 start_load_kv
        的局部变量.

        注意(每个 ReqMeta 只能解析一次):
        * ``_align_remote_block_ids`` 就地缩放 remote_block_ids 且不幂等;
        * 末尾的兼容字段会把 remote_host/remote_port 改写成第一个 peer, 二次解析
          会让 mamba 分支(取 remote_port + tp_rank//tp_ratio)指到错误的侧信道端口.
        peer_transfer 整体换入(不就地改), 避免模型线程迭代到半成品 dict.
        """
        transfer_mappings: dict[tuple[str, int], dict[str, Any]] = {}
        self._align_remote_block_ids(req_meta)
        for i, kv_cache_spec in enumerate(self.kv_cache_specs):
            if isinstance(kv_cache_spec, MambaSpec):
                single_group_transfer_mappings = self._get_kv_split_metadata_for_mamba(
                    req_meta, req_meta.req_idx, req_id, i
                )
            else:
                single_group_transfer_mappings = self._get_kv_split_metadata(req_meta, req_meta.req_idx, req_id, i)
            for (host, port), block_dict in single_group_transfer_mappings.items():
                if (host, port) not in transfer_mappings:
                    transfer_mappings[(host, port)] = {
                        "local_block_ids": [[] for _ in range(self.num_kv_cache_groups)],
                        "remote_block_ids": [[] for _ in range(self.num_kv_cache_groups)],
                        "trans_count": [0 for _ in range(self.num_kv_cache_groups)],
                    }
                transfer_mappings[(host, port)]["local_block_ids"][i].extend(
                    single_group_transfer_mappings[(host, port)]["local_block_ids"]
                )
                transfer_mappings[(host, port)]["remote_block_ids"][i].extend(
                    single_group_transfer_mappings[(host, port)]["remote_block_ids"]
                )
                transfer_mappings[(host, port)]["trans_count"][i] = single_group_transfer_mappings[(host, port)][
                    "trans_count"
                ]
        # 一个请求的 KV 可能按 group 落到多个 D rank(见 ReqMeta.peer_transfer 注释),
        # 这里全部记下来, 发送线程按 (peer, group) 分别推送.
        peer_transfer: dict[tuple[str, int], dict[str, Any]] = {}
        for (host, port), block_dict in transfer_mappings.items():
            peer_transfer[(host, port)] = {
                "local_block_ids": self._get_kernel_block_ids(block_dict["local_block_ids"]),
                "remote_block_ids": self._get_kernel_block_ids(block_dict["remote_block_ids"]),
                "trans_count": block_dict["trans_count"],
            }
        req_meta.peer_transfer = peer_transfer
        # 兼容字段(日志/旧路径): 指向第一个 peer.
        first_peer = next(iter(peer_transfer), None)
        if first_peer is not None:
            req_meta.remote_host, req_meta.remote_port = first_peer
            req_meta.local_block_ids = peer_transfer[first_peer]["local_block_ids"]
            req_meta.remote_block_ids = peer_transfer[first_peer]["remote_block_ids"]
            req_meta.trans_count = peer_transfer[first_peer]["trans_count"]

    def _merge_decoder_params(self, req_meta: ReqMeta, params: dict[str, Any]) -> None:
        """把 D 直连推来的参数合并进 ReqMeta(p_then_d).

        键名与"proxy 转发 D 参数"路径完全一致(见 scheduler.update_state_after_alloc
        里构造的 kv_transfer_params), 因此后续映射逻辑无需区分来源.
        """
        req_meta.remote_block_ids = list(params.get("remote_block_ids", []) or [])
        req_meta.remote_block_size = params.get("remote_block_size", [])
        req_meta.remote_engine_id = params.get("remote_engine_id")
        req_meta.remote_host = params.get("remote_host")
        req_meta.remote_port = params.get("remote_port")
        req_meta.remote_te_rpc_port = params.get("remote_te_rpc_port")
        # 层元数据/te 端口通常不在参数里(D 侧 GET_META 现取), 但允许内联.
        if params.get("remote_layer_metadata"):
            req_meta.remote_layer_metadata = params["remote_layer_metadata"]
        req_meta.metaserver = params.get("metaserver")
        req_meta.remote_tp_size = params.get("remote_tp_size")
        req_meta.remote_pcp_size = params.get("remote_pcp_size")
        req_meta.remote_dcp_size = params.get("remote_dcp_size")
        req_meta.do_virtual = params.get("do_virtual")
        req_meta.remote_cache_tokens = params.get("remote_cached_tokens", 0) or 0

    def resolve_pending_req(self, req_id: str, req_meta: ReqMeta) -> bool:
        """p_then_d: 发送线程在推送前解析等待中的 D 参数. 返回是否解析成功.

        失败语义(调用方负责把该请求的层任务丢弃 + 上报请求级完成): 只 log 不抛,
        否则异常会冒到 _handle_batch 的兜底 catch, 把整批所有请求的层任务一起丢掉,
        而 D 侧可能凭 group-end 的完成信息提前放行 —— 静默乱码.
        """
        assert self.params_recv_thread is not None
        start_time = time.perf_counter()
        params = self.params_recv_thread.wait_for_kv_transfer_params(req_id)
        if params is None:
            logger.error(
                "Timeout waiting %ss for decode transfer params of request %s; drop its KV transfer. "
                "The decode engine was never dispatched (or failed to push params).",
                self.wait_transfer_params_timeout_sec,
                get_external_request_id(req_id),
            )
            return False
        try:
            self._merge_decoder_params(req_meta, params)
            self._resolve_send_mapping(req_id, req_meta)
            # GET_META: 取 D 的 CPU staging 层元数据与 TE 端口(按 engine+port 缓存).
            self.update_decoder_info(req_id, req_meta)
        except Exception as e:
            logger.error(
                "Failed to resolve decode transfer params for request %s: %s", get_external_request_id(req_id), e
            )
            req_meta.peer_transfer = {}
            return False
        # 换入完成后再清等待标记: 模型线程以该标记决定是否跳过 peer 判断.
        req_meta.awaiting_params = False
        logger.info(
            "Resolved decode transfer params for request %s in %.1f ms (peers=%s)",
            get_external_request_id(req_id),
            _perf_ms(start_time),
            list(req_meta.peer_transfer.keys()),
        )
        return True

    def start_load_kv(self, metadata: MooncakeLayerwiseConnectorMetadata):
        """Start loading KV blocks from remote engine."""
        self.current_layer = 0
        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            for req_id, meta in metadata.requests.items():
                if meta.do_virtual:
                    self.virtual_request.add(req_id)
                    continue
                external_req_id = get_external_request_id(req_id)
                assert self.kv_recv_layer_thread is not None
                self.request_map[external_req_id] = req_id
                self._recving_metadata[req_id] = meta
        elif self.vllm_config.kv_transfer_config.is_kv_producer:
            # update trans info
            update_metadata = {}
            for req_idx, (req_id, req_meta) in enumerate(metadata.requests.items()):
                # 冻结本 step 的块表快照: ReqMeta.local_block_ids 在 scheduler 侧是
                # SendReqInfo 的引用且会随 step 增长, 而逐层任务与(延迟)解析都要用
                # 本 step 当时的口径 —— align 模式下 mamba 状态块下标
                # (len-num_spec-1) 与 D 侧固定长度的块表必须严格对齐.
                update_req_meta = copy.deepcopy(req_meta)
                update_req_meta.req_idx = req_idx
                if req_meta.awaiting_params:
                    # p_then_d: D 参数还没到(P 先于 D 派发), peer 解析推迟到发送线程
                    # (见 resolve_pending_req).
                    self._check_pending_params_supported(req_id)
                else:
                    self._resolve_send_mapping(req_id, update_req_meta)
                update_metadata[req_id] = update_req_meta
            metadata.requests = update_metadata

            # update send task trans block info
            if self.pd_head_ratio != 1 or self.enable_kv_quant or self.enable_c8_quant:
                send_task = metadata.send_task
                send_task.group_rearrange_block_ids = [[] for _ in range(self.num_kv_cache_groups)]
                send_task.group_num_blocks = [0 for _ in range(self.num_kv_cache_groups)]
                send_task.group_num_tokens = [0 for _ in range(self.num_kv_cache_groups)]
                send_task.group_block_table = [None for _ in range(self.num_kv_cache_groups)]
                send_task.group_block_len_tensor = [None for _ in range(self.num_kv_cache_groups)]
                send_task.group_seq_start_tensor = [None for _ in range(self.num_kv_cache_groups)]
                device = self.k_buffer.device  # type: ignore
                for i in self.attn_resharding_group_idx:
                    # 多 peer 下同一 group 的 block 可能记在不同 peer 名下(兼容字段
                    # local_block_ids 只指向第一个 peer), 这里取所有 peer 的并集.
                    rearrange_block_ids: set[int] = set()
                    for req_meta in metadata.requests.values():
                        if req_meta.awaiting_params:
                            # p_then_d: peer 还没解析, 但 reshard 必须在模型前向里
                            # 当场做完(gather + alltoall, 结果落在本层任务自己的
                            # keys/values 张量上) —— 于是用本请求自己的本地块(kernel
                            # 口径)兜底: 本 rank 在该 group 要发的块必是它的子集,
                            # 多算只是白做 reshard, 索引与张量同源仍自洽, 传输侧由
                            # 映射(peer_transfer)过滤.
                            local_block_ids = self._get_kernel_block_ids(copy.deepcopy(req_meta.local_block_ids))
                            rearrange_block_ids.update(local_block_ids[i])
                            continue
                        rearrange_block_ids.update(
                            block_id
                            for peer_blocks in req_meta.peer_transfer.values()
                            for block_id in peer_blocks["local_block_ids"][i]
                        )
                    send_task.group_rearrange_block_ids[i].extend(sorted(rearrange_block_ids))
                    flat_block_ids = send_task.group_rearrange_block_ids[i]
                    block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
                    send_task.group_num_blocks[i] = len(flat_block_ids)
                    send_task.group_num_tokens[i] = send_task.group_num_blocks[i] * (
                        self.block_size[i] // self.kernel_block_size_scale[i]
                    )

                    send_task.group_block_table[i] = block_ids_tensor.view(1, -1)
                    send_task.group_block_len_tensor[i] = torch.tensor(
                        [send_task.group_num_tokens[i]], dtype=torch.int32, device=device
                    )
                    send_task.group_seq_start_tensor[i] = torch.tensor([0], dtype=torch.int32, device=device)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: list[torch.Tensor],
        attn_metadata: "AttentionMetadata",
        connector_metadata: MooncakeLayerwiseConnectorMetadata,
        **kwargs,
    ) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        if self.vllm_config.kv_transfer_config.is_kv_producer and connector_metadata.requests.keys():
            if self.current_layer >= self.total_layers:
                self.current_layer += 1
                return
            # get reshape and cache event
            if layer_name == "":
                layer_name = self.index_to_name[self.current_layer][0]
            if (
                isinstance(attn_metadata, dict)
                and hasattr(attn_metadata[layer_name], "reshape_cache_event")
                and attn_metadata[layer_name].reshape_cache_event is not None
            ):
                reshape_cache_event = attn_metadata[layer_name].reshape_cache_event
            elif (
                attn_metadata
                and hasattr(attn_metadata, "reshape_cache_event")
                and attn_metadata.reshape_cache_event is not None
            ):
                reshape_cache_event = attn_metadata.reshape_cache_event
            else:
                reshape_cache_event = torch.npu.Event()
                reshape_cache_event.record()
            send_task = connector_metadata.send_task
            layer_group_idx = self.layer_metadata[layer_name].tensor_group_idx[0]
            keys = None
            values = None
            quant_keys = None
            quant_values = None
            # reshard/量化路径: 计算流上记录的"本任务数据就绪"事件, 供发送线程
            # 按任务等待(替代 resharding_stream 整条流 synchronize).
            reshard_ready_event: torch.npu.Event | None = None
            if (
                (
                    self.pd_head_ratio != 1
                    and (isinstance(self.kv_cache_specs[layer_group_idx], (FullAttentionSpec, SlidingWindowSpec)))
                    and send_task.group_num_blocks[layer_group_idx] > 0
                )
                or (self.enable_c8_quant and self.current_layer in self.vllm_config.quant_config.c8_quant_layers)
                or (self.enable_kv_quant and self.current_layer in self.vllm_config.quant_config.kvcache_quant_layers)
            ):
                assert self.resharding_stream is not None
                # reshard 计算跑在计算流上(与 attention 同流): 顺序天然确定,
                # 既不需要跨流 event.wait, 也不在侧流上发集合通信. 2026-09-09
                # 实测侧流方案下设备侧 alltoall 会偶发永久等待(发送线程卡在
                # resharding_stream.synchronize()), 队列随即堵死模型前向.
                # 侧流此后只承载发送线程自己的 k/v buffer 拷贝.
                reshape_cache_event.wait()
                device = self.k_buffer.device  # type: ignore
                # Initialize buffers
                keys = torch.empty(
                    (send_task.group_num_tokens[layer_group_idx], *kv_layer[0].size()[-2:]),
                    dtype=kv_layer[0].dtype,
                    device=device,
                )
                values = torch.empty(
                    (send_task.group_num_tokens[layer_group_idx], *kv_layer[1].size()[-2:]),
                    dtype=kv_layer[1].dtype,
                    device=device,
                )

                # Load cache data into buffers
                torch_npu.atb.npu_paged_cache_load(
                    kv_layer[0],
                    kv_layer[1],
                    send_task.group_block_table[layer_group_idx],
                    send_task.group_block_len_tensor[layer_group_idx],
                    seq_starts=send_task.group_seq_start_tensor[layer_group_idx],
                    key=keys,
                    value=values,
                )
                if self.pd_head_ratio != 1:
                    # sort kv caches for each block
                    keys = (
                        keys.view(send_task.group_num_blocks[layer_group_idx], self.pd_head_ratio, -1, *keys.shape[1:])
                        .transpose(0, 1)
                        .reshape_as(keys)
                    )
                    values = (
                        values.view(
                            send_task.group_num_blocks[layer_group_idx], self.pd_head_ratio, -1, *values.shape[1:]
                        )
                        .transpose(0, 1)
                        .reshape_as(values)
                    )
                    # reshard kv cache
                    keys = keys.reshape(-1, *kv_layer[0].shape[2:])
                    values = values.reshape(-1, *kv_layer[1].shape[2:])

                    (keys, values) = kv_alltoall_and_rearrange(self.pd_head_ratio, keys, values)
                if self.enable_c8_quant:
                    layer = self.vllm_config.compilation_config.static_forward_context[layer_name]
                    quant_keys = torch.clamp(
                        torch.round(keys * layer._c8_k_inv_scale + layer._c8_k_offset),
                        -128,
                        127,
                    ).to(torch.int8)
                    quant_values = torch.clamp(
                        torch.round(values * layer._c8_v_inv_scale + layer._c8_v_offset),
                        -128,
                        127,
                    ).to(torch.int8)
                    quant_keys = self.get_nz_cache(quant_keys, layer_group_idx)
                    quant_values = self.get_nz_cache(quant_values, layer_group_idx)
                if self.enable_kv_quant and self.current_layer in self.vllm_config.quant_config.kvcache_quant_layers:
                    layer = self.vllm_config.compilation_config.static_forward_context[layer_name]
                    keys = torch.ops.vllm.quantize(
                        keys, layer.fak_descale, layer.fak_descale_reciprocal, layer.fak_offset
                    )
                    quant_keys = self.get_nz_cache(keys, layer_group_idx)
                    quant_values = self.get_nz_cache(values, layer_group_idx)
                # 本任务的数据就绪事件(记录在计算流上): 发送线程按任务等待它,
                # 而不是等整条 resharding stream —— 后者会连带等待后续层的
                # reshard 工作, 与对端进度成环.
                reshard_ready_event = torch.npu.Event()
                reshard_ready_event.record()

            assert self.kv_send_layer_thread is not None
            assert reshape_cache_event is not None
            # 每层任务随批发送: mamba/GDN 状态在层 forward 内已写定 (P 侧无
            # 投机, 无采样后状态重写), 与 full-attention 层同路径攒批传输.
            layer_send_task = SendTask(
                wait_event=reshard_ready_event if reshard_ready_event is not None else reshape_cache_event,
                k_cache=keys,
                v_cache=values,
                k_quant_cache=quant_keys,
                v_quant_cache=quant_values,
                layer_idx=self.current_layer,
                layer_name=layer_name,
                group_rearrange_block_ids=send_task.group_rearrange_block_ids,
            )
            for req_id, req_meta in connector_metadata.requests.items():
                if req_meta.awaiting_params:
                    # p_then_d: D 参数还没到, peer_transfer 还是空的 —— 只挂任务,
                    # 不判断 group/不 GET_META(发送线程解析后再按映射过滤本层).
                    # 这里刻意不迭代 peer_transfer: 发送线程稍后会整体换入新 dict.
                    layer_send_task.send_request[req_id] = req_meta
                    continue
                # 多 peer 下本层的 group 可能只落在其中某个 peer 上, 要按所有 peer 判断.
                if not any(blocks["local_block_ids"][layer_group_idx] for blocks in req_meta.peer_transfer.values()):
                    continue
                try:
                    req_meta_update = self.update_decoder_info(req_id, req_meta)
                except Exception as e:
                    logger.warning(
                        "MooncakeLayerwiseConnector transfer fail. req_id=%s, layer_idx=%s, error=%s. ",
                        req_id,
                        self.current_layer,
                        e,
                    )
                    continue
                logger.debug("Add request %s to kv send layer thread. req_meta_update=%r", req_id, req_meta_update)
                layer_send_task.send_request[req_id] = req_meta_update

            t_put0 = time.perf_counter()
            self.kv_send_layer_thread.send_queue.put(layer_send_task)
            if _PERF_LOG:
                put_block_ms = _perf_ms(t_put0)
                if put_block_ms > 2.0:
                    # 队列满导致模型前向被钳制(发送线程处理慢于模型产出)的信号.
                    logger.info(
                        "[mooncake][perf] P save_kv_layer put blocked layer=%s %.1f ms",
                        layer_name,
                        put_block_ms,
                    )
            self.current_layer += 1

    # NOTE: Due to the FIA operator constraints, the expected kv cache is ND format, NZ shape,
    # while the npu_format_cast method only modifies the memory layout, we manually convert it to NZ shape here
    def get_nz_cache(self, cache_tensor: torch.Tensor, layer_group_idx: int):
        head_num, head_dim = cache_tensor.shape[-2], cache_tensor.shape[-1]
        cache_tensor = cache_tensor.view(-1, self.block_size[layer_group_idx], head_num * head_dim)
        cache_tensor = trans_nd_to_nz(cache_tensor)
        cache_tensor = cache_tensor.reshape(-1, head_num, head_dim)
        return cache_tensor

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False,
            )
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            # 不再注册到共享 poller: GET_META 的收包用调用方自己的 poller
            # (zmq.Poller 非线程安全), 共享 poller 只会在每次建连时白记一项.
            return sock

    def update_decoder_info(self, req_id, req_meta: ReqMeta):
        # 一个请求可能对应多个 D peer(见 ReqMeta.peer_transfer): 每个 peer 的
        # KV base / 层元数据都要单独拉取.
        for remote_host, remote_port in req_meta.peer_transfer:
            self._ensure_peer_metadata(req_id, req_meta, remote_host, remote_port)
        first_peer = next(iter(req_meta.peer_transfer), None)
        if first_peer is not None:
            req_meta.remote_host, req_meta.remote_port = first_peer
            req_meta.remote_te_rpc_port = req_meta.peer_te_rpc_port[first_peer]
            req_meta.remote_layer_metadata = req_meta.peer_layer_metadata[first_peer]
        return req_meta

    def _ensure_peer_metadata(self, req_id, req_meta: ReqMeta, remote_host: str, remote_port: int):
        """拉取某个对端(host, port)的 KV 元信息(带缓存), 不等 TP 时预建链路.

        整段(查缓存 → GET_META → 写缓存 → 预建链路)持 remote_meta_lock: p_then_d
        下发送线程也会走这里, 与模型线程共享 remote_layer_metadata(SizedDict 会
        插入/淘汰)与 socket. GET_META 用本次调用私有的 poller(并把本次的 socket
        注册进去)—— zmq.Poller 的 register/poll 不是线程安全的, 不能共享.
        """
        peer = (remote_host, remote_port)
        with self.remote_meta_lock:
            if remote_port not in self.remote_layer_metadata[req_meta.remote_engine_id]:
                try:
                    encoded_data = self.encoder.encode((GET_META_MSG, req_id))
                    sock = self._get_remote_socket(remote_host, remote_port)
                    path = f"{remote_host}:{remote_port}"
                    ensure_zmq_send(sock, encoded_data, path)
                    poller: zmq.Poller = zmq.Poller()  # type: ignore
                    poller.register(sock, zmq.POLLIN)  # type: ignore
                    metadata_bytes = ensure_zmq_recv(sock, poller, path)
                    agent_meta: MooncakeAgentMetadata = self.decoder.decode(metadata_bytes)
                except Exception as e:
                    logger.error(
                        "Query to port and kv base addr for request fail. req_id=%s, source=%s:%s, error=%s. ",
                        req_id,
                        remote_host,
                        remote_port,
                        e,
                    )
                    raise e
                assert req_meta.remote_engine_id != self.engine_id, (
                    f"Conflict engine id {req_meta.remote_engine_id} with local engine id {self.local_engine_id}."
                )
                self.remote_layer_metadata[req_meta.remote_engine_id][remote_port] = agent_meta.layer_metadata
                self.remote_te_port[req_meta.remote_engine_id][remote_port] = agent_meta.te_rpc_port
                logger.debug(
                    "Query to port and kv base addr for request %s from %s:%s success "
                    "agent_meta.layer_metadata=%r agent_meta.te_rpc_port=%r",
                    req_id,
                    remote_host,
                    remote_port,
                    agent_meta.layer_metadata,
                    agent_meta.te_rpc_port,
                )
                if self.pd_head_ratio > 1:
                    # for tp inequal, pre-create link to prevent alltoall out of memory
                    session_id = f"{remote_host}:{agent_meta.te_rpc_port}"
                    first_layer_name = next(iter(self.layer_metadata.keys()))
                    local_base_addr = self.layer_metadata[first_layer_name].kv_caches_base_addr[0]
                    if global_te.use_tcp:
                        # TCP 数据面只能读注册过的 host 内存: 先把这 128B flush 到
                        # CPU staging, 再用 staging 地址做 src(与逐层传输的 src
                        # 替换语义一致). 目的地址是 consumer 发布的 staging 地址.
                        global_te.sync_npu_to_cpu_for_npu_addrs([local_base_addr], [128])
                        staging_base_addr = global_te.npu_addr_to_cpu_addr(local_base_addr)
                        if staging_base_addr is None:
                            raise RuntimeError(
                                f"H2H layerwise: pre-create link src 0x{local_base_addr:x} not in TCP staging map."
                            )
                        local_base_addr = staging_base_addr
                    ret = self.engine.batch_transfer_sync_write(
                        session_id,
                        [local_base_addr],
                        [agent_meta.layer_metadata[first_layer_name].kv_caches_base_addr[0]],
                        [128],
                    )
                    if ret < 0:
                        logger.error(
                            "Mooncake transfer failed to create link. session_id=%s, ret=%d. ", session_id, ret
                        )
            req_meta.peer_layer_metadata[peer] = self.remote_layer_metadata[req_meta.remote_engine_id][remote_port]
            req_meta.peer_te_rpc_port[peer] = self.remote_te_port[req_meta.remote_engine_id][remote_port]

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:  # type: ignore
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def group_concurrent_contiguous(
    src: list[int], dst: list[int] | None = None
) -> tuple[list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    if dst is None:
        dst = []
    if not dst:
        src_only_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)

        if src_only_indices.size == 0:
            return [], []

        brk = np.where(np.diff(src_only_indices) != 1)[0] + 1
        src_groups = np.split(src_only_indices, brk)
        src_groups = [g.tolist() for g in src_groups]

        return src_groups, []

    else:
        src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
        dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

        if src_indices.size == 0:
            return [], []

        brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
        src_groups = np.split(src_indices, brk)
        dst_groups = np.split(dst_indices, brk)

        src_groups = [g.tolist() for g in src_groups]
        dst_groups = [g.tolist() for g in dst_groups]

        return src_groups, dst_groups


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(
    socket: zmq.Socket,  # type: ignore
    data: bytes,
    path: str,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Send failed after all retries. error=%s. ", e)
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def ensure_zmq_recv(
    socket: zmq.Socket,  # type: ignore
    poller: zmq.Poller,  # type: ignore
    path: str,
    timeout: float = 1.0,
    max_retries: int = 3,
) -> bytes:
    retries_left = max_retries
    while True:
        try:
            if dict(poller.poll(int(timeout * 1000))):  # milliseconds
                data = socket.recv()
                return data
            else:
                raise zmq.ZMQError("Receive timeout")  # type: ignore
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Receive failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Receive failed after all retries. error=%s. ", e)
                raise RuntimeError(f"Failed to receive data from {path} after {max_retries} retries: {e}")


def get_external_request_id(request_id: str):
    # NOTE(zxr): vLLM PR #27987 add additional suffix
    # to EngineCore request_id with len(suffix) == 9
    return request_id[:-9]


class PrefilledTokenSpool:
    """p_then_d: 首 token 的落盘通道(写侧在 API server 进程, 读侧在 EngineCore).

    背景: proxy 提前派发 D(不等 P 的响应), 于是 D 的请求体里没有 `prefilled_token`
    —— 而 entrypoint 所有"复用 P 首 token"的分支都在请求构造时读它, 中途补不进去。
    首 token 改由 proxy 在拿到后 POST 到 D 的 API server, 落到本机 spool; D 侧在
    KV 接收完成的转型点取用(recompute_scheduler._update_waiting_for_remote_kv)。

    为什么不用 socket: D 可能 dp>1, 请求落在哪个 dp 组由 D 内部路由决定, proxy 无法
    按端口寻址; spool 是共享目录, 每个 rank 只认领自己等待的 req_id, 错投也不会丢。

    文件协议: 文件名 = 规范化后的 external request id, 内容 = JSON
    ``{"prefilled_token": [...], "stop_reasons": [...], "ts": <写入时刻>}``;
    写入用 ``.tmp`` + ``os.rename`` 原子发布, 读侧 ``os.unlink`` 认领(只有一个 rank
    能成功删除, 于是只有它消费)。
    """

    def __init__(self, spool_dir: str):
        self.spool_dir = spool_dir
        self._entries: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._poller: threading.Thread | None = None
        self._poller_started = False

    # ---- 读侧(EngineCore) ------------------------------------------------
    def _start_poller(self) -> None:
        if self._poller_started:
            return
        self._poller_started = True
        os.makedirs(self.spool_dir, exist_ok=True)
        self._poller = threading.Thread(target=self._poll_loop, daemon=True, name="p-then-d-token-spool")
        self._poller.start()
        logger.info("[p_then_d] prefilled-token spool polling %s", self.spool_dir)

    def _poll_loop(self) -> None:
        while True:
            try:
                names = os.listdir(self.spool_dir)
            except FileNotFoundError:
                # 目录被清掉了(进程收尾/测试清理): 本线程是 daemon, 直接退出.
                return
            except OSError as e:
                logger.warning("[p_then_d] token spool poll failed on %s: %s", self.spool_dir, e)
                time.sleep(PTD_TOKEN_SPOOL_POLL_S)
                continue
            try:
                for name in names:
                    if name.endswith(".tmp"):
                        continue
                    path = os.path.join(self.spool_dir, name)
                    try:
                        with open(path, encoding="utf-8") as f:
                            payload = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        # 半截文件(理论上 rename 已保证原子)或已被别的 rank 认领.
                        continue
                    try:
                        os.unlink(path)
                    except OSError:
                        # 已被别的 rank 认领: 本次不缓存, 让认领者消费.
                        continue
                    with self._lock:
                        self._entries[name] = (time.time(), payload)
                        self._evict_locked()
                self._sweep_expired()
            except OSError as e:
                logger.warning("[p_then_d] token spool poll failed on %s: %s", self.spool_dir, e)
            time.sleep(PTD_TOKEN_SPOOL_POLL_S)

    def _evict_locked(self) -> None:
        while len(self._entries) > PTD_TOKEN_SPOOL_MAX_ENTRIES:
            self._entries.pop(next(iter(self._entries)), None)

    def _sweep_expired(self) -> None:
        now = time.time()
        with self._lock:
            for key in [k for k, (ts, _) in self._entries.items() if now - ts > PTD_TOKEN_SPOOL_EXPIRE_S]:
                self._entries.pop(key, None)
        # 停写/崩溃留下的孤儿文件也要清, 否则目录会一直涨.
        try:
            for name in os.listdir(self.spool_dir):
                path = os.path.join(self.spool_dir, name)
                if now - os.path.getmtime(path) > PTD_TOKEN_SPOOL_EXPIRE_S:
                    os.unlink(path)
        except OSError:
            pass

    def wait_for_prefilled_token(self, request_id: str, timeout: float) -> dict[str, Any] | None:
        """等某个请求的首 token; 超时返回 None(调用方落回普通生成路径).

        只在 D 的 scheduler 线程调用(KV 接收完成、请求即将离开
        WAITING_FOR_REMOTE_KVS 时)。
        """
        self._start_poller()
        key = _spool_key(request_id)
        deadline = time.perf_counter() + timeout
        while True:
            with self._lock:
                entry = self._entries.pop(key, None)
            if entry is not None:
                return entry[1]
            if time.perf_counter() >= deadline:
                return None
            time.sleep(PTD_TOKEN_SPOOL_POLL_S)

    # ---- 写侧(API server) ------------------------------------------------
    @classmethod
    def write_entry(cls, spool_dir: str, request_id: str, payload: dict[str, Any]) -> None:
        """原子发布一条首 token 记录(写侧进程调用)."""
        os.makedirs(spool_dir, exist_ok=True)
        key = _spool_key(request_id)
        tmp_path = os.path.join(spool_dir, f"{key}.tmp")
        payload = {**payload, "ts": time.time()}
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.rename(tmp_path, os.path.join(spool_dir, key))


# 单例: 同一个进程里 API server 与 EngineCore 共享(见 PrefilledTokenSpool 的说明).
_TOKEN_SPOOLS: dict[str, PrefilledTokenSpool] = {}
_TOKEN_SPOOLS_LOCK = threading.Lock()


def get_prefilled_token_spool(spool_dir: str) -> PrefilledTokenSpool:
    with _TOKEN_SPOOLS_LOCK:
        spool = _TOKEN_SPOOLS.get(spool_dir)
        if spool is None:
            spool = PrefilledTokenSpool(spool_dir)
            _TOKEN_SPOOLS[spool_dir] = spool
        return spool
