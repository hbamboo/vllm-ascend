import os
import threading
import time
import zlib
from bisect import bisect_right

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.utils.utils import iter_kv_cache_tensors

_BG_SYNC_INTERVAL = float(os.getenv("MC_TCP_BG_SYNC_INTERVAL", "1.0"))

# 传输线程 CPU 绑核开关: 1=开启. 开启后发送/接收线程自动绑定到当前进程
# 允许核集(Cpus_allowed_list, 已受 vllm-ascend cpu_binding 的 taskset 约束)中
# 负载最低的核, 外层无需感知具体核号. 默认关闭, 避免与模型计算意外争抢.
_CPU_BIND_ENABLED = os.getenv("MC_TCP_CPU_BIND", "0") == "1"

# 性能观测开关(与 layerwise connector 同 env): 1=热路径(flush/H2D)耗时以
# INFO 输出供逐批观测; 否则降为 DEBUG 避免每批刷屏.
_PERF_LOG = os.getenv("MC_TCP_PERF_LOG", "0") == "1"
_cpu_bind_lock = threading.Lock()
_cpu_bind_used: set[int] = set()


def _parse_cpus_allowed() -> list[int]:
    """解析 /proc/self/status 的 Cpus_allowed_list(容器 cpuset + 已绑核的交集)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list"):
                    cpus: list[int] = []
                    for part in line.split(":", 1)[1].strip().split(","):
                        part = part.strip()
                        if "-" in part:
                            lo, hi = part.split("-")
                            cpus.extend(range(int(lo), int(hi) + 1))
                        elif part:
                            cpus.append(int(part))
                    return cpus
    except Exception:  # noqa: BLE001
        pass
    return []


def _cpu_busy_ratios() -> dict[int, float]:
    """两次采样 /proc/stat 计算各 CPU 的忙碌比例(近似负载)."""

    def sample() -> dict[int, tuple[int, int]]:
        out: dict[int, tuple[int, int]] = {}
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if not line.startswith("cpu"):
                        continue
                    parts = line.split()
                    if not parts[0][3:].isdigit():
                        continue
                    cpu = int(parts[0][3:])
                    vals = [int(x) for x in parts[1:]]
                    idle = vals[3] + vals[4]  # idle + iowait
                    total = sum(vals)
                    out[cpu] = (idle, total)
        except Exception:  # noqa: BLE001
            pass
        return out

    s1 = sample()
    time.sleep(0.05)
    s2 = sample()
    ratios: dict[int, float] = {}
    for cpu, (idle2, total2) in s2.items():
        if cpu in s1:
            idle1, total1 = s1[cpu]
            delta_total = total2 - total1
            if delta_total > 0:
                ratios[cpu] = 1.0 - (idle2 - idle1) / delta_total
    return ratios


def bind_current_thread_to_idle_cpu(tag: str) -> bool:
    """把当前线程绑定到进程允许核集中负载最低、且未被本进程其它传输线程
    占用的核. 失败(无权限/无可用核)只告警, 不影响功能. 返回是否绑定成功."""
    if not _CPU_BIND_ENABLED:
        return False
    allowed = _parse_cpus_allowed()
    if not allowed:
        logger.warning("[mooncake][TCP] cpu bind skipped for %s: no allowed cpus", tag)
        return False
    ratios = _cpu_busy_ratios()
    with _cpu_bind_lock:
        candidates = [c for c in allowed if c not in _cpu_bind_used]
        if not candidates:
            logger.warning("[mooncake][TCP] cpu bind skipped for %s: allowed cpus all in use", tag)
            return False
        candidates.sort(key=lambda c: ratios.get(c, 0.0))
        chosen = candidates[0]
        _cpu_bind_used.add(chosen)
    try:
        os.sched_setaffinity(0, {chosen})
        logger.info("[mooncake][TCP] thread %s bound to cpu %d (allowed=%s)", tag, chosen, allowed)
        return True
    except OSError as e:
        with _cpu_bind_lock:
            _cpu_bind_used.discard(chosen)
        logger.warning("[mooncake][TCP] cpu bind failed for %s on cpu %d: %s", tag, chosen, e)
        return False


# Direction codes shared with csrc/torch_binding.cpp::swap_blocks_batch
# (see vllm_ascend/simple_kv_offload/npu_mem_ops.py).
_DIRECTION_H2D = 0
_DIRECTION_D2H = 1

try:
    _HAS_SWAP_BLOCKS_BATCH = hasattr(torch.ops, "_C_ascend") and hasattr(torch.ops._C_ascend, "swap_blocks_batch")
except Exception:  # noqa: BLE001 - ops namespace may fail to import on some envs
    _HAS_SWAP_BLOCKS_BATCH = False


def _safe_copy_npu_to_cpu(
    cpu_view: torch.Tensor,
    npu_tensor: torch.Tensor,
    byte_size: int,
    tensor_byte_offset: int,
):
    """D2H: 单区间拷贝, 经批量 DMA 后端在专用拷贝流上执行并等待完成.

    Reference: ``vllm_ascend/simple_kv_offload/copy_backend.py`` —
    copies are issued via ``aclrtMemcpyBatchAsync``
    (``torch.ops._C_ascend.swap_blocks_batch``) on a dedicated transfer
    stream instead of per-range ``.cpu()``/``.copy_()`` on the compute
    stream. Falls back to the old torch path when the op is unavailable.
    """
    global_te.submit_dma_copy(
        [(npu_tensor, tensor_byte_offset, cpu_view, byte_size)],
        direction=_DIRECTION_D2H,
        sync=True,
    )


def _safe_copy_cpu_to_npu(
    npu_tensor: torch.Tensor,
    cpu_view: torch.Tensor,
    byte_size: int,
    tensor_byte_offset: int,
):
    """H2D: 单区间拷贝, 经批量 DMA 后端在专用拷贝流上执行并等待完成."""
    global_te.submit_dma_copy(
        [(npu_tensor, tensor_byte_offset, cpu_view, byte_size)],
        direction=_DIRECTION_H2D,
        sync=True,
    )


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()
        self._protocol: str = ""
        self._cpu_tensors: list[torch.Tensor] = []
        self._npu_to_cpu_offset: dict[int, tuple[int, torch.Tensor]] = {}
        self._region_offsets: list[int] = []
        self._regions_by_offset: list[tuple[int, int, torch.Tensor]] = []
        self._npu_region_bases: list[int] = []
        self._npu_regions_by_base: list[tuple[int, int, torch.Tensor]] = []
        # Region-mode staging (attn-mamba hybrid / shared-tensor layouts):
        # mirrors whole physical NPU regions (incl. inter-view padding) 1:1
        # into CPU staging, so any NPU byte address inside a region maps
        # linearly to its staging address. Populated by
        # ``register_tcp_staging_regions``.
        self._regions_npu_base: list[int] = []
        self._regions_npu: list[tuple[int, int, int]] = []  # (npu_base, len, cpu_off)
        self._regions_cpu_off: list[int] = []
        self._regions_cpu: list[tuple[int, int, int]] = []  # (cpu_off, len, npu_base)
        self._use_tcp: bool | None = None
        self._bg_sync_thread: threading.Thread | None = None
        self._bg_sync_stop = threading.Event()
        self._copy_stream_lock = threading.Lock()
        self._d2h_stream: torch.npu.Stream | None = None
        self._h2d_stream: torch.npu.Stream | None = None

    @property
    def protocol(self) -> str:
        if not self._protocol:
            self._protocol = os.getenv("VLLM_ASCEND_MOONCAKE_PROTOCOL", "ascend").strip().lower()
        return self._protocol

    @property
    def use_tcp(self) -> bool:
        if self._use_tcp is None:
            self._use_tcp = self.protocol == "tcp"
        return self._use_tcp

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the "
                            "instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/"
                            "main/doc/en/build.md "
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    protocol = self.protocol
                    t0 = time.perf_counter()
                    logger.debug(
                        "[mooncake] TransferEngine.initialize BEGIN protocol=%s hostname=%s",
                        protocol,
                        hostname,
                    )
                    ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", protocol, device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
                    logger.debug(
                        "[mooncake] TransferEngine.initialize DONE elapsed=%.3fs",
                        time.perf_counter() - t0,
                    )
                    if self.use_tcp:
                        logger.info("TCP mode active: NPU<->CPU staging will be used for KV cache transfers")
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            if self.use_tcp:
                logger.warning(
                    "register_buffer called in TCP mode. "
                    "Use register_tcp_staging(kv_caches) to register "
                    "with CPU staging."
                )
            self._register_buffer_native(ptrs, sizes)
            self.is_register_buffer = True

    def register_tcp_staging(
        self,
        kv_caches: dict[str, torch.Tensor],
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        """Per-view TCP staging: 把 kv cache 张量(以及可选 extra_tensors, 如
        layerwise 的 k/v reshard buffer)镜像进同一块注册到 mooncake 的 CPU buffer.

        extra_tensors 一并进入 NPU->staging 地址映射, 使 ``npu_addr_to_cpu_addr``
        与 ``sync_npu_to_cpu_for_npu_addrs`` 能覆盖它们 (pd_head_ratio>1 或量化
        时数据面 src 落在这些 buffer 上).
        """
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            self._build_tcp_staging_from_tensors(kv_caches, extra_tensors)
            self.is_register_buffer = True

    def register_tcp_staging_regions(self, npu_regions: list[tuple[int, int]]) -> None:
        """Region-mode TCP staging: mirror whole physical NPU regions into CPU staging.

        ``npu_regions`` is a list of ``(npu_base, byte_len)`` covering every byte
        range that may ever be flushed / transferred (the physical KV cache
        tensors, with the same [min(shared addr) - mamba conv padding, size]
        semantics as the D2D ``use_attn_mamba_hybrid`` registration).  Layout
        inside each region is preserved 1:1 (including padding between layer
        views of a shared buffer), so any NPU byte address inside a region maps
        to a unique staging address via ``npu_addr_to_cpu_addr`` and back via
        ``sync_cpu_to_npu_for_transfer``.

        Required for attn-mamba hybrid models whose per-layer caches are views
        of a shared physical tensor with non-contiguous page / padding layout;
        the per-view staging built by ``register_tcp_staging`` drops those gaps
        and cannot address them consistently.
        """
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            assert not self._cpu_tensors, "staging buffer already built"
            regions = sorted((base, length) for base, length in npu_regions if length > 0)
            prev_end = 0
            for base, length in regions:
                if base < prev_end:
                    raise RuntimeError(
                        f"Overlapping TCP staging regions: 0x{base:x} < previous end 0x{prev_end:x}. "
                        "Region-mode staging requires disjoint physical KV cache regions."
                    )
                prev_end = base + length
            total_bytes = sum(length for _, length in regions)
            cpu_buffer = torch.empty(total_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
            cpu_base = cpu_buffer.data_ptr()
            cpu_off = 0
            self._regions_npu = []
            self._regions_cpu = []
            for base, length in regions:
                self._regions_npu.append((base, length, cpu_off))
                self._regions_cpu.append((cpu_off, length, base))
                cpu_off += length
            self._regions_npu_base = [base for base, _, _ in self._regions_npu]
            self._regions_cpu_off = [off for off, _, _ in self._regions_cpu]
            ret_value = self.transfer_engine.register_memory(cpu_base, total_bytes)
            if ret_value != 0:
                logger.error(
                    "Mooncake TCP register_memory failed: ptr=0x%x size=%d ret=%d",
                    cpu_base,
                    total_bytes,
                    ret_value,
                )
                raise RuntimeError("Mooncake memory registration failed.")
            self._cpu_tensors.append(cpu_buffer)
            self.is_register_buffer = True
            logger.info(
                "TCP region-mode staging ready: CPU buffer at 0x%x (%.2f GiB), %d regions.",
                cpu_base,
                total_bytes / (1024**3),
                len(regions),
            )

    def _npu_to_cpu_addr_in_regions(self, npu_addr: int) -> int | None:
        """Region-mode: map an arbitrary NPU byte address to its staging address."""
        if not self._regions_npu:
            return None
        idx = bisect_right(self._regions_npu_base, npu_addr) - 1
        if idx < 0:
            return None
        base, length, cpu_off = self._regions_npu[idx]
        if npu_addr >= base + length:
            return None
        return self._cpu_tensors[0].data_ptr() + cpu_off + (npu_addr - base)

    def _cpu_to_npu_addr_in_regions(self, cpu_addr: int) -> int | None:
        """Region-mode: map a staging address back to its NPU address."""
        if not self._regions_cpu:
            return None
        cpu_base = self._cpu_tensors[0].data_ptr()
        rel = cpu_addr - cpu_base
        idx = bisect_right(self._regions_cpu_off, rel) - 1
        if idx < 0:
            return None
        off, length, base = self._regions_cpu[idx]
        if rel >= off + length:
            return None
        return base + (rel - off)

    @property
    def _region_mode(self) -> bool:
        return bool(self._regions_npu)

    def _flush_cpu_addrs_region_mode(self, cpu_addrs: list[int], lengths: list[int]) -> int:
        """Region-mode FLUSH (pull path): copy the byte ranges a remote reader
        is about to read back from NPU into this node's CPU staging."""
        t0 = time.perf_counter()
        copied_bytes = 0
        skipped = 0
        # 前置同步: 确保 NPU cache 数据(计算流)已就绪; 拷贝在专用 DMA 流.
        torch.npu.synchronize()
        src_ptrs: list[int] = []
        dst_ptrs: list[int] = []
        sizes: list[int] = []
        for cpu_addr, byte_size in zip(cpu_addrs, lengths):
            npu_addr = self._cpu_to_npu_addr_in_regions(cpu_addr)
            if npu_addr is None:
                skipped += 1
                continue
            src_ptrs.append(npu_addr)
            dst_ptrs.append(cpu_addr)
            sizes.append(byte_size)
            copied_bytes += byte_size
        self.submit_dma_copy_ptrs(src_ptrs, dst_ptrs, sizes, _DIRECTION_D2H)
        log_fn = logger.info if _PERF_LOG else logger.debug
        log_fn(
            "[mooncake][TCP] region flush for pull: ranges=%d bytes=%d skipped=%d elapsed=%.3fs",
            len(cpu_addrs),
            copied_bytes,
            skipped,
            time.perf_counter() - t0,
        )
        return copied_bytes

    def start_bg_sync(self):
        self._start_bg_sync()

    def _register_buffer_native(self, ptrs: list[int], sizes: list[int]):
        for ptr, size in zip(ptrs, sizes):
            ret_value = self.transfer_engine.register_memory(ptr, size)
            if ret_value != 0:
                logger.error(
                    "Mooncake register_memory failed: ptr=0x%x size=%d ret=%d",
                    ptr,
                    size,
                    ret_value,
                )
                raise RuntimeError("Mooncake memory registration failed.")
        total = sum(sizes)
        logger.info(
            "Registered %d native buffers (%.2f GiB) with mooncake",
            len(ptrs),
            total / (1024**3),
        )

    def _build_tcp_staging_from_tensors(
        self,
        kv_caches: dict[str, torch.Tensor],
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        t0 = time.perf_counter()
        # cache_list: list[torch.Tensor] = []
        # for cache_or_caches in kv_caches.values():
        #     for cache in cache_or_caches:
        #         cache_list.append(cache)

        cache_list = list(iter_kv_cache_tensors(kv_caches))
        # 额外 NPU 张量(reshard/量化 buffer): 与 kv cache 张量同等地建 NPU->CPU
        # 映射, 逐层传输的 src 才能被 flush/地址替换覆盖.
        cache_list.extend(t for t in (extra_tensors or []) if t is not None)

        total_bytes = sum(c.numel() * c.element_size() for c in cache_list)
        logger.info(
            "TCP mode: allocating CPU staging (%.2f GiB) for %d tensors",
            total_bytes / (1024**3),
            len(cache_list),
        )
        # 需要这么大吗? 需要完整拷贝所有NPU tensor?
        cpu_buffer = torch.empty(total_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        cpu_base = cpu_buffer.data_ptr()

        offset = 0
        self._npu_to_cpu_offset.clear()
        for cache in cache_list:
            npu_ptr = cache.data_ptr()
            byte_size = cache.numel() * cache.element_size()
            self._npu_to_cpu_offset[npu_ptr] = (offset, cache)
            offset += byte_size

        # 按 staging 偏移升序建索引, 供按 CPU 地址反查 region (flush 用).
        self._regions_by_offset = sorted(
            (off, npu_ptr, tensor) for npu_ptr, (off, tensor) in self._npu_to_cpu_offset.items()
        )
        self._region_offsets = [off for off, _, _ in self._regions_by_offset]
        # 按 NPU 首地址升序建索引, 供按任意 NPU 字节地址换算 staging 地址
        # (逐层 H2H: flush 与 src 地址替换).
        self._npu_regions_by_base = sorted(
            (npu_ptr, off, tensor) for npu_ptr, (off, tensor) in self._npu_to_cpu_offset.items()
        )
        self._npu_region_bases = [base for base, _, _ in self._npu_regions_by_base]

        ret_value = self.transfer_engine.register_memory(cpu_base, total_bytes)
        if ret_value != 0:
            logger.error(
                "Mooncake TCP register_memory failed: ptr=0x%x size=%d ret=%d",
                cpu_base,
                total_bytes,
                ret_value,
            )
            raise RuntimeError("Mooncake memory registration failed.")

        self._cpu_tensors.append(cpu_buffer)
        logger.info(
            "TCP staging ready: CPU buffer at 0x%x (%.2f GiB), "
            "%d NPU->CPU mappings registered. "
            "Background sync interval: %.1fs",
            cpu_base,
            total_bytes / (1024**3),
            len(self._npu_to_cpu_offset),
            _BG_SYNC_INTERVAL,
        )
        logger.debug(
            "[mooncake] _build_tcp_staging elapsed=%.3fs tensors=%d",
            time.perf_counter() - t0,
            len(cache_list),
        )

    def _start_bg_sync(self):
        if self._bg_sync_thread is not None and self._bg_sync_thread.is_alive():
            return
        self._bg_sync_stop.clear()
        self._bg_sync_thread = threading.Thread(target=self._bg_sync_worker, daemon=True, name="mooncake-tcp-bg-sync")
        self._bg_sync_thread.start()
        logger.info("TCP background sync thread started (interval=%.1fs)", _BG_SYNC_INTERVAL)

    def _bg_sync_worker(self):
        sync_count = 0
        while not self._bg_sync_stop.is_set():
            try:
                self._sync_npu_to_cpu_all(first=sync_count == 0)
                sync_count += 1
            except Exception as e:
                logger.warning("Background NPU->CPU sync error: %s", e)
            self._bg_sync_stop.wait(timeout=_BG_SYNC_INTERVAL)

    def _sync_npu_to_cpu_all(self, first: bool = False):
        if not self._cpu_tensors:
            return
        t0 = time.perf_counter()
        cpu_tensor = self._cpu_tensors[0]
        total_bytes = 0
        # 前置同步: 确保 NPU cache 数据(计算流)就绪; 拷贝在专用 DMA 流执行.
        torch.npu.synchronize()
        if self._region_mode:
            src_ptrs: list[int] = []
            dst_ptrs: list[int] = []
            sizes: list[int] = []
            for npu_base, byte_size, cpu_off in self._regions_npu:
                total_bytes += byte_size
                src_ptrs.append(npu_base)
                dst_ptrs.append(self._cpu_tensors[0].data_ptr() + cpu_off)
                sizes.append(byte_size)
            self.submit_dma_copy_ptrs(src_ptrs, dst_ptrs, sizes, _DIRECTION_D2H)
            if first:
                logger.info(
                    "[mooncake][TCP] bg_sync D2H (NPU->CPU) first pass: %.2f MiB in %.3fs, regions=%d",
                    total_bytes / (1024**2),
                    time.perf_counter() - t0,
                    len(self._regions_npu),
                )
            return
        items: list[tuple[torch.Tensor, int, torch.Tensor, int]] = []
        for npu_ptr, (offset, npu_tensor) in self._npu_to_cpu_offset.items():
            byte_size = npu_tensor.numel() * npu_tensor.element_size()
            total_bytes += byte_size
            items.append((npu_tensor, 0, cpu_tensor[offset : offset + byte_size], byte_size))
        # 一次 aclrtMemcpyBatchAsync 提交全部 72 个 region; 只同步专用拷贝流.
        self.submit_dma_copy(items, direction=_DIRECTION_D2H)
        if first:
            # 首次周期同步: 确认 producer 侧 D2H (NPU->CPU) staging 在工作;
            # 后续同步保持 debug 避免刷屏
            logger.info(
                "[mooncake][TCP] bg_sync D2H (NPU->CPU) first pass: %.2f MiB in %.3fs, tensors=%d",
                total_bytes / (1024**2),
                time.perf_counter() - t0,
                len(self._npu_to_cpu_offset),
            )
        logger.debug(
            "[mooncake] bg_sync NPU->CPU elapsed=%.3fs tensors=%d",
            time.perf_counter() - t0,
            len(self._npu_to_cpu_offset),
        )

    def get_cpu_address_for_npu(self, npu_ptr: int) -> int | None:
        if not self._cpu_tensors:
            return None
        if self._region_mode:
            return self._npu_to_cpu_addr_in_regions(npu_ptr)
        entry = self._npu_to_cpu_offset.get(npu_ptr)
        if entry is None:
            return None
        offset, _ = entry
        return self._cpu_tensors[0].data_ptr() + offset

    def sync_npu_to_cpu_for_region(self, npu_ptr: int, byte_size: int):
        if not self._cpu_tensors:
            return
        entry = self._npu_to_cpu_offset.get(npu_ptr)
        if entry is None:
            return
        offset, npu_tensor = entry
        cpu_tensor = self._cpu_tensors[0]
        actual_size = min(byte_size, npu_tensor.numel() * npu_tensor.element_size())
        cpu_view = cpu_tensor[offset : offset + actual_size]
        torch.npu.synchronize()
        _safe_copy_npu_to_cpu(cpu_view, npu_tensor, actual_size, 0)
        torch.npu.synchronize()

    def sync_npu_to_cpu_for_addrs(self, cpu_addrs: list[int], lengths: list[int]) -> int:
        """P 侧: 收到 D 的 FLUSH 请求后, 按 staging 地址反查 NPU region,
        把 D 即将读取的字节区间从 NPU 拷回 CPU staging (D2H), 保证 TCP
        读到的不是陈旧快照. 返回实际拷贝的字节数."""
        if not self._cpu_tensors:
            return 0
        if self._region_mode:
            return self._flush_cpu_addrs_region_mode(cpu_addrs, lengths)
        if not self._regions_by_offset:
            return 0
        cpu_tensor = self._cpu_tensors[0]
        cpu_base = cpu_tensor.data_ptr()
        t0 = time.perf_counter()
        copied_bytes = 0
        skipped = 0
        # 前置同步: 确保 NPU cache 数据(计算流)已就绪; 拷贝本身在专用
        # DMA 流上批量执行, 不占用计算流.
        torch.npu.synchronize()
        items: list[tuple[torch.Tensor, int, torch.Tensor, int]] = []
        for cpu_addr, byte_size in zip(cpu_addrs, lengths):
            rel = cpu_addr - cpu_base
            if rel < 0:
                skipped += 1
                continue
            idx = bisect_right(self._region_offsets, rel) - 1
            if idx < 0:
                skipped += 1
                continue
            np_offset, _, npu_tensor = self._regions_by_offset[idx]
            npu_byte_size = npu_tensor.numel() * npu_tensor.element_size()
            if rel >= np_offset + npu_byte_size:
                skipped += 1
                continue
            inner_offset = rel - np_offset
            actual_size = min(byte_size, npu_byte_size - inner_offset)
            if actual_size <= 0:
                skipped += 1
                continue
            items.append((npu_tensor, inner_offset, cpu_tensor[rel : rel + actual_size], actual_size))
            copied_bytes += actual_size
        # 一次 aclrtMemcpyBatchAsync 批量 D2H; 内部只同步专用拷贝流.
        self.submit_dma_copy(items, direction=_DIRECTION_D2H)
        # 热路径日志门控: MC_TCP_PERF_LOG=1 时 INFO(供逐批观测), 否则 DEBUG 防刷屏.
        log_fn = logger.info if _PERF_LOG else logger.debug
        log_fn(
            "[mooncake][TCP] flush for pull: ranges=%d bytes=%d skipped=%d elapsed=%.3fs",
            len(cpu_addrs),
            copied_bytes,
            skipped,
            time.perf_counter() - t0,
        )
        return copied_bytes

    def sync_npu_to_cpu_for_npu_addrs(self, npu_addrs: list[int], lengths: list[int]) -> int:
        """P 侧: 把任意 NPU 字节区间(层内 block 地址)拷到 CPU staging (D2H).
        逐层 H2H 时在 batch_transfer_sync_write 前调用, 保证 push 出去的是
        最新 KV 而非陈旧快照. 返回实际拷贝的字节数."""
        if not self._cpu_tensors:
            return 0
        if self._region_mode:
            return self._sync_npu_to_cpu_regions(npu_addrs, lengths)
        if not self._npu_regions_by_base:
            return 0
        cpu_tensor = self._cpu_tensors[0]
        t0 = time.perf_counter()
        copied_bytes = 0
        skipped = 0
        # 前置同步: 确保本层 KV(计算流)已写完; 拷贝在专用 DMA 流上批量执行.
        # TODO: push模式已通过wait_event保证数据ready；pull无前序保证，需完善
        # torch.npu.synchronize()
        items: list[tuple[torch.Tensor, int, torch.Tensor, int]] = []
        for npu_addr, byte_size in zip(npu_addrs, lengths):
            idx = bisect_right(self._npu_region_bases, npu_addr) - 1
            if idx < 0:
                skipped += 1
                continue
            npu_base, np_offset, npu_tensor = self._npu_regions_by_base[idx]
            npu_byte_size = npu_tensor.numel() * npu_tensor.element_size()
            if npu_addr >= npu_base + npu_byte_size:
                skipped += 1
                continue
            inner_offset = npu_addr - npu_base
            actual_size = min(byte_size, npu_byte_size - inner_offset)
            if actual_size <= 0:
                skipped += 1
                continue
            items.append(
                (
                    npu_tensor,
                    inner_offset,
                    cpu_tensor[np_offset + inner_offset : np_offset + inner_offset + actual_size],
                    actual_size,
                )
            )
            copied_bytes += actual_size
        # 一次 aclrtMemcpyBatchAsync 批量 D2H; 内部只同步专用拷贝流.
        self.submit_dma_copy(items, direction=_DIRECTION_D2H)
        log_fn = logger.info if _PERF_LOG else logger.debug
        log_fn(
            "[mooncake][TCP] layerwise flush (NPU->CPU): ranges=%d bytes=%d skipped=%d elapsed=%.5f ms",
            len(npu_addrs),
            copied_bytes,
            skipped,
            (time.perf_counter() - t0) * 1000,
        )
        return copied_bytes

    def _sync_npu_to_cpu_regions(self, npu_addrs: list[int], lengths: list[int]) -> int:
        """Region-mode D2H flush: map NPU ranges to staging addrs and batch-copy."""
        t0 = time.perf_counter()
        copied_bytes = 0
        skipped = 0
        src_ptrs: list[int] = []
        dst_ptrs: list[int] = []
        sizes: list[int] = []
        for npu_addr, byte_size in zip(npu_addrs, lengths):
            if byte_size <= 0:
                continue
            cpu_addr = self._npu_to_cpu_addr_in_regions(npu_addr)
            if cpu_addr is None:
                skipped += 1
                continue
            src_ptrs.append(npu_addr)
            dst_ptrs.append(cpu_addr)
            sizes.append(byte_size)
            copied_bytes += byte_size
        self.submit_dma_copy_ptrs(src_ptrs, dst_ptrs, sizes, _DIRECTION_D2H)
        log_fn = logger.info if _PERF_LOG else logger.debug
        log_fn(
            "[mooncake][TCP] region flush (NPU->CPU): ranges=%d bytes=%d skipped=%d elapsed=%.5f ms",
            len(npu_addrs),
            copied_bytes,
            skipped,
            (time.perf_counter() - t0) * 1000,
        )
        return copied_bytes

    def _sync_cpu_to_npu_regions(self, cpu_addrs: list[int], lengths: list[int]) -> None:
        """Region-mode H2D: map staging addrs back to NPU ranges and batch-copy."""
        t0 = time.perf_counter()
        # H2D 写入的是 D 端 KV cache 的块, 必须与计算流排序 —— 该排序已由拷贝提交处
        # 的 stream.wait_stream(default_stream) 精确保证(见 submit_dma_copy_ptrs),
        # 不再需要整设备同步.
        total_bytes = 0
        src_ptrs: list[int] = []
        dst_ptrs: list[int] = []
        sizes: list[int] = []
        for cpu_addr, byte_size in zip(cpu_addrs, lengths):
            if byte_size <= 0:
                continue
            npu_addr = self._cpu_to_npu_addr_in_regions(cpu_addr)
            if npu_addr is None:
                raise RuntimeError(
                    f"H2H layerwise: CPU staging addr 0x{cpu_addr:x} not found in region map. "
                    "Remote producer wrote to an address outside the registered staging regions."
                )
            src_ptrs.append(cpu_addr)
            dst_ptrs.append(npu_addr)
            sizes.append(byte_size)
            total_bytes += byte_size
        self.submit_dma_copy_ptrs(src_ptrs, dst_ptrs, sizes, _DIRECTION_H2D)
        log_fn = logger.info if _PERF_LOG else logger.debug
        log_fn(
            "[mooncake][TCP] region H2D (CPU->NPU): ranges=%d bytes=%d elapsed=%.5f ms",
            len(cpu_addrs),
            total_bytes,
            (time.perf_counter() - t0) * 1000,
        )

    def npu_addr_to_cpu_addr(self, npu_addr: int) -> int | None:
        """把任意 NPU 字节地址换算成 staging 中的对应 CPU 地址.
        用于把逐层传输的 src (NPU 块地址) 整体替换为 TCP 可读的 staging 地址."""
        if not self._cpu_tensors:
            return None
        if self._region_mode:
            return self._npu_to_cpu_addr_in_regions(npu_addr)
        if not self._npu_regions_by_base:
            return None
        idx = bisect_right(self._npu_region_bases, npu_addr) - 1
        if idx < 0:
            return None
        npu_base, np_offset, npu_tensor = self._npu_regions_by_base[idx]
        npu_byte_size = npu_tensor.numel() * npu_tensor.element_size()
        if npu_addr >= npu_base + npu_byte_size:
            return None
        return self._cpu_tensors[0].data_ptr() + np_offset + (npu_addr - npu_base)

    def sync_cpu_to_npu_for_region(self, npu_ptr: int, byte_size: int):
        if not self._cpu_tensors:
            return
        entry = self._npu_to_cpu_offset.get(npu_ptr)
        if entry is None:
            return
        offset, npu_tensor = entry
        cpu_tensor = self._cpu_tensors[0]
        actual_size = min(byte_size, npu_tensor.numel() * npu_tensor.element_size())
        cpu_view = cpu_tensor[offset : offset + actual_size]
        torch.npu.synchronize()
        _safe_copy_cpu_to_npu(npu_tensor, cpu_view, actual_size, 0)
        torch.npu.synchronize()

    def sync_cpu_to_npu_for_transfer(self, src_addrs: list[int], lengths: list[int]):
        if not self._cpu_tensors:
            return
        if self._region_mode:
            self._sync_cpu_to_npu_regions(src_addrs, lengths)
            return
        cpu_tensor = self._cpu_tensors[0]
        cpu_base = cpu_tensor.data_ptr()
        t0 = time.perf_counter()
        synced: set[int] = set()
        total_bytes = 0
        items: list[tuple[torch.Tensor, int, torch.Tensor, int]] = []
        # 与计算流的排序由拷贝提交处的 stream.wait_stream(default_stream) 保证
        # (见 submit_dma_copy), 拷贝本身仍在专用 DMA 流上执行、不占用计算流.
        for cpu_addr, byte_size in zip(src_addrs, lengths):
            cpu_offset = cpu_addr - cpu_base
            if cpu_offset < 0:
                continue
            for npu_ptr, (np_offset, npu_tensor) in self._npu_to_cpu_offset.items():
                npu_byte_size = npu_tensor.numel() * npu_tensor.element_size()
                if np_offset <= cpu_offset < np_offset + npu_byte_size:
                    inner_offset = cpu_offset - np_offset
                    actual_size = min(byte_size, npu_byte_size - inner_offset)
                    items.append(
                        (
                            npu_tensor,
                            inner_offset,
                            cpu_tensor[cpu_offset : cpu_offset + actual_size],
                            actual_size,
                        )
                    )
                    total_bytes += actual_size
                    synced.add(npu_ptr)
                    break
        # 一次 aclrtMemcpyBatchAsync 批量 H2D; 内部只同步专用拷贝流.
        self.submit_dma_copy(items, direction=_DIRECTION_H2D)
        if synced:
            # consumer 侧: TCP get 完成后把 staging 数据拷回 NPU (H2D)
            log_fn = logger.info if _PERF_LOG else logger.debug
            log_fn(
                "[mooncake][TCP] H2D (CPU->NPU) after TCP get: regions=%d bytes=%d elapsed=%.5f ms tensors=%d",
                len(src_addrs),
                total_bytes,
                (time.perf_counter() - t0) * 1000,
                len(synced),
            )


    # ------------------------------------------------------------------
    # 内容链路自检 (MC_XCHK=1, 调试用): 对 staging 区间取 crc, 用于
    # 「P staging -> TCP -> D staging -> H2D -> D NPU」逐段比对 —— 只有带上
    # 由调用方给定的 (批/序号) 键才不会像早期版本那样被"同地址被后续批复用"
    # 搞成配对错位.
    # ------------------------------------------------------------------
    def crc_for_cpu_addrs(self, cpu_addrs: list[int], lengths: list[int], sample: int = 512) -> list[int]:
        if not self._cpu_tensors:
            return []
        cpu_tensor = self._cpu_tensors[0]
        base = cpu_tensor.data_ptr()
        nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
        flat = cpu_tensor.view(torch.uint8).reshape(-1)
        out: list[int] = []
        for addr, ln in zip(cpu_addrs, lengths):
            off = addr - base
            if off < 0 or off >= nbytes or ln <= 0:
                out.append(0)
                continue
            take = min(sample, ln, nbytes - off)
            out.append(zlib.crc32(flat[off : off + take].numpy().tobytes()))
        return out

    def poke_cpu_addr(self, cpu_addr: int, nbytes: int = 4) -> None:
        """MC_XCHK_FAULT 反向对照: 故意改坏 staging 头 nbytes 字节."""
        if not self._cpu_tensors:
            return
        cpu_tensor = self._cpu_tensors[0]
        base = cpu_tensor.data_ptr()
        off = cpu_addr - base
        flat = cpu_tensor.view(torch.uint8).reshape(-1)
        if 0 <= off < flat.numel():
            flat[off : off + min(nbytes, flat.numel() - off)] = 0xAB

    def refetch_npu_to_staging(self, cpu_addrs: list[int], lengths: list[int]) -> None:
        """把 staging 区间对应的 NPU 内容反向拷回 staging(crc 自检用)."""
        if not self._cpu_tensors or not self._region_mode:
            return
        src: list[int] = []
        dst: list[int] = []
        sizes: list[int] = []
        for addr, ln in zip(cpu_addrs, lengths):
            npu_addr = self._cpu_to_npu_addr_in_regions(addr)
            if npu_addr is None or ln <= 0:
                continue
            src.append(npu_addr)
            dst.append(addr)
            sizes.append(ln)
        if src:
            self.submit_dma_copy_ptrs(src, dst, sizes, _DIRECTION_D2H)


    def stop_bg_sync(self):
        self._bg_sync_stop.set()
        if self._bg_sync_thread is not None:
            self._bg_sync_thread.join(timeout=5.0)

    def _get_copy_stream(self, direction: int) -> torch.npu.Stream:
        """惰性创建 D2H/H2D 专用拷贝流.

        拷贝在专用流上执行, 与计算流(默认流)解耦: 第 i 层数据的 DMA
        拷贝可与模型第 i+1 层计算并行, 互不排队; 等待只同步拷贝流.
        """
        with self._copy_stream_lock:
            if direction == _DIRECTION_D2H:
                if self._d2h_stream is None:
                    self._d2h_stream = torch.npu.Stream()
                return self._d2h_stream
            if self._h2d_stream is None:
                self._h2d_stream = torch.npu.Stream()
            return self._h2d_stream

    def submit_dma_copy(
        self,
        items: list[tuple[torch.Tensor, int, torch.Tensor, int]],
        direction: int,
        sync: bool = True,
    ) -> None:
        """批量提交 NPU<->CPU staging 区间拷贝.

        参考 ``vllm_ascend/simple_kv_offload/copy_backend.py``: 把一批
        (npu_tensor, npu内字节偏移, cpu_view, 字节数) 一次性经
        ``aclrtMemcpyBatchAsync`` (``torch.ops._C_ascend.swap_blocks_batch``)
        提交到专用拷贝流, 避免逐区间 ``.cpu()``/``.copy_()`` 的小拷贝与
        流竞争. 无该自定义 op 时回退为逐区间 torch 拷贝.

        Args:
            items: (npu_tensor, tensor_byte_offset, cpu_view, byte_size).
            direction: _DIRECTION_D2H 或 _DIRECTION_H2D.
            sync: 完成后同步等待(只等拷贝流).
        """
        if not items:
            return
        if _HAS_SWAP_BLOCKS_BATCH:
            try:
                stream = self._get_copy_stream(direction)
                # 排序: 拷贝流先等待计算流(默认流)当前已入队的全部工作 —— 拷贝会写入
                # KV cache 的块, 异步调度下上一个请求的 decode 可能仍在飞, 不排序会与
                # 在飞计算相互覆盖(表现为新请求首 token 就错/随批大小时序随机复现);
                # 只等"当前点", 后续层计算仍可与拷贝并行, 不牺牲流水.
                stream.wait_stream(torch.npu.default_stream())
                sizes = np.asarray([n for _, _, _, n in items], dtype=np.int64)
                if direction == _DIRECTION_D2H:
                    # D2H: NPU 为源, CPU staging 为目的.
                    src_ptrs = np.asarray([t.data_ptr() + off for t, off, _, _ in items], dtype=np.int64)
                    dst_ptrs = np.asarray([c.data_ptr() for _, _, c, _ in items], dtype=np.int64)
                else:
                    # H2D: CPU staging 为源, NPU 为目的.
                    src_ptrs = np.asarray([c.data_ptr() for _, _, c, _ in items], dtype=np.int64)
                    dst_ptrs = np.asarray([t.data_ptr() + off for t, off, _, _ in items], dtype=np.int64)
                with torch.npu.stream(stream):
                    torch.ops._C_ascend.swap_blocks_batch(
                        torch.from_numpy(src_ptrs),
                        torch.from_numpy(dst_ptrs),
                        torch.from_numpy(sizes),
                        direction,
                    )
                if sync:
                    stream.synchronize()
                return
            except Exception as e:  # noqa: BLE001
                # aclrtMemcpyBatchAsync 在部分设备/场景下参数校验失败
                # (107000). 自动降级为逐区间 torch 拷贝, 保证正确性;
                # 批量 DMA 的根治在 csrc 侧(attr 构造).
                logger.warning(
                    "[mooncake][TCP] batch DMA failed (%s), falling back to torch copy for %d ranges (direction=%d)",
                    e,
                    len(items),
                    direction,
                )
        # 回退: 逐区间旧式 torch 拷贝 (默认流).
        for npu_tensor, npu_off, cpu_view, byte_size in items:
            flat = npu_tensor.detach().reshape(-1)
            if direction == _DIRECTION_D2H:
                npu_bytes = flat.view(torch.uint8)
                cpu_view.copy_(npu_bytes[npu_off : npu_off + byte_size].cpu())
            else:
                flat.view(torch.uint8)[npu_off : npu_off + byte_size].copy_(cpu_view)
        if sync:
            torch.npu.synchronize()

    def submit_dma_copy_ptrs(
        self,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        sizes: list[int],
        direction: int,
        sync: bool = True,
    ) -> None:
        """Region-mode batch NPU<->CPU staging copy over raw byte addresses.

        Same batch-DMA backend as ``submit_dma_copy``, but takes raw source /
        destination byte pointers instead of tensors, so region ranges that do
        not map to a single torch tensor (shared-buffer padding offsets) can be
        flushed / H2D'd in one ``aclrtMemcpyBatchAsync`` call. Region mode
        requires the custom batch op: there is no safe per-range torch fallback
        for arbitrary byte ranges.
        """
        if not src_ptrs:
            return
        if not _HAS_SWAP_BLOCKS_BATCH:
            raise RuntimeError(
                "[mooncake][TCP] region-mode staging requires the swap_blocks_batch "
                "custom op for batch DMA; it is unavailable in this environment."
            )
        stream = self._get_copy_stream(direction)
        # 排序: 拷贝流先等待计算流(默认流)当前已入队的全部工作 —— 拷贝会写入
        # KV cache 的块, 异步调度下上一个请求的 decode 可能仍在飞, 不排序会与
        # 在飞计算相互覆盖(表现为新请求首 token 就错/随批大小时序随机复现);
        # 只等"当前点", 后续层计算仍可与拷贝并行, 不牺牲流水.
        stream.wait_stream(torch.npu.default_stream())
        src_t = torch.from_numpy(np.asarray(src_ptrs, dtype=np.int64))
        dst_t = torch.from_numpy(np.asarray(dst_ptrs, dtype=np.int64))
        size_t = torch.from_numpy(np.asarray(sizes, dtype=np.int64))
        with torch.npu.stream(stream):
            torch.ops._C_ascend.swap_blocks_batch(src_t, dst_t, size_t, direction)
        if sync:
            stream.synchronize()


global_te = GlobalTE()
