import os
import threading
import time
from bisect import bisect_right

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.utils.utils import iter_kv_cache_tensors

_BG_SYNC_INTERVAL = float(os.getenv("MC_TCP_BG_SYNC_INTERVAL", "1.0"))

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

    def register_tcp_staging(self, kv_caches: dict[str, torch.Tensor]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            self._build_tcp_staging_from_tensors(kv_caches)
            self.is_register_buffer = True

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

    def _build_tcp_staging_from_tensors(self, kv_caches: dict[str, torch.Tensor]):
        t0 = time.perf_counter()
        # cache_list: list[torch.Tensor] = []
        # for cache_or_caches in kv_caches.values():
        #     for cache in cache_or_caches:
        #         cache_list.append(cache)

        cache_list = list(iter_kv_cache_tensors(kv_caches))

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
        if not self._cpu_tensors or not self._regions_by_offset:
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
        logger.info(
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
        if not self._cpu_tensors or not self._npu_regions_by_base:
            return 0
        cpu_tensor = self._cpu_tensors[0]
        t0 = time.perf_counter()
        copied_bytes = 0
        skipped = 0
        # 前置同步: 确保本层 KV(计算流)已写完; 拷贝在专用 DMA 流上批量执行.
        torch.npu.synchronize()
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
        logger.info(
            "[mooncake][TCP] layerwise flush (NPU->CPU): ranges=%d bytes=%d skipped=%d elapsed=%.3fs",
            len(npu_addrs),
            copied_bytes,
            skipped,
            time.perf_counter() - t0,
        )
        return copied_bytes

    def npu_addr_to_cpu_addr(self, npu_addr: int) -> int | None:
        """把任意 NPU 字节地址换算成 staging 中的对应 CPU 地址.
        用于把逐层传输的 src (NPU 块地址) 整体替换为 TCP 可读的 staging 地址."""
        if not self._cpu_tensors or not self._npu_regions_by_base:
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
        cpu_tensor = self._cpu_tensors[0]
        cpu_base = cpu_tensor.data_ptr()
        t0 = time.perf_counter()
        synced: set[int] = set()
        total_bytes = 0
        items: list[tuple[torch.Tensor, int, torch.Tensor, int]] = []
        # 前置同步: H2D 写 NPU cache 前确保计算流不再读写这些块; 拷贝本身
        # 在专用 DMA 流上批量执行, 不占用计算流.
        torch.npu.synchronize()
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
            logger.info(
                "[mooncake][TCP] H2D (CPU->NPU) after TCP get: regions=%d bytes=%d elapsed=%.3fs tensors=%d",
                len(src_addrs),
                total_bytes,
                time.perf_counter() - t0,
                len(synced),
            )

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


global_te = GlobalTE()
