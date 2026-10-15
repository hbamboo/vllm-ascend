# SPDX-License-Identifier: Apache-2.0
"""D 侧 LAYER_DONE 执行池(MC_TCP_RECV_POOL>0)单测.

不依赖 NPU/引擎: 用假 engine(记录调用区间)与假 H2D 驱动, 验证
- 不同 session 的两笔 LAYER_DONE 在池内并行(调用区间重叠, 而非串行相加);
- 完成项经 socketpair 唤醒 + 队列回收, 收线程据此回 ACK/NAK;
- ACK 路径置请求完成、NAK 路径作废请求(簿记语义与内联路径一致, 两者共用实现).
"""

import queue
import threading
import time
import unittest
from unittest import mock

import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_prefill_then_decode_connector as conn

WAIT_S = 5.0


class _FakeEngine:
    """记录每次 sync_read 的起止时刻, 便于断言"并行而非串行"."""

    def __init__(self, delay: float = 0.2, ret: int = 0):
        self.delay = delay
        self.ret = ret
        self.calls: list[tuple[float, float, str]] = []
        self._lock = threading.Lock()

    def batch_transfer_sync_read(self, peer: str, dst, src, lengths) -> int:
        t0 = time.perf_counter()
        time.sleep(self.delay)
        t1 = time.perf_counter()
        with self._lock:
            self.calls.append((t0, t1, peer))
        return self.ret


class _FakeGlobalTe:
    """只实现被调用到的 H2D 入口."""

    def __init__(self):
        self.h2d_calls: list[tuple[list, list]] = []
        self._lock = threading.Lock()

    def sync_cpu_to_npu_for_transfer(self, dst_addrs, lengths) -> None:
        time.sleep(0.02)
        with self._lock:
            self.h2d_calls.append((list(dst_addrs), list(lengths)))


def _make_recv_thread(engine) -> conn.KVCacheRecvingLayerThread:
    """绕过 __init__, 只装配收线程处理一条 LAYER_DONE 用到的状态."""
    thread = object.__new__(conn.KVCacheRecvingLayerThread)
    thread.tp_rank = 0
    thread.lock = threading.Lock()
    thread.done_requests = set()
    thread.failed_requests = set()
    thread.task_tracker = {}
    thread._perf_d_req = {}
    thread.engine = engine
    thread._recv_jobs = None
    thread._recv_done = queue.Queue()
    thread._wake_r = None
    thread._wake_w = None
    thread._recv_workers = []
    thread._bind_worker_device = lambda: None  # type: ignore[method-assign]
    return thread


def _job(req_id: str, te_path: str = "127.0.0.1:17000", identity: bytes = b"id-1"):
    return conn._RecvJob(
        identity=identity,
        sender_path="127.0.0.1:16000",
        req_ids=[req_id],
        dst_addrs=[1000],
        lengths=[2048],
        done_list=[(req_id, True, 1, False)],
        src_addrs=[4096],
        sender_te_path=te_path,
        t_recv0=time.perf_counter(),
    )


def _recv_router_msg(router):
    """等 ROUTER 上的一条消息, 返回 (identity, payload)."""
    poller = zmq.Poller()
    poller.register(router, zmq.POLLIN)
    assert dict(poller.poll(int(WAIT_S * 1000))), "ROUTER 未收到客户端消息"
    frames = router.recv_multipart()
    return frames[0], frames[-1]


def _req_round_trip(thread, router, port: int, req_id: str) -> bytes:
    """一次真实 REQ↔ROUTER 往返: 客户端线程发, 收线程执行数据面并回 reply."""
    out: dict = {}
    done = threading.Event()
    path = f"tcp://127.0.0.1:{port}"

    def client():
        # REQ socket 的创建/收发全部收在这个线程内(ZMQ socket 不跨线程).
        cctx = zmq.Context()
        sock = cctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, int(WAIT_S * 1000))
        sock.connect(path)
        sock.send(b"payload")
        try:
            out["reply"] = sock.recv()
        except Exception as e:  # noqa: BLE001
            out["error"] = e
        finally:
            sock.close(linger=0)
            cctx.destroy(linger=0)
        done.set()

    worker = threading.Thread(target=client, daemon=True)
    worker.start()
    identity, payload = _recv_router_msg(router)
    assert payload == b"payload"
    job = _job(req_id, identity=identity)
    thread._run_recv_job(job)
    thread._finish_recv_job(router, job)
    assert done.wait(WAIT_S), "客户端未收到 reply"
    if "error" in out:
        raise out["error"]
    return out["reply"]


class TestRecvPool(unittest.TestCase):
    def test_two_sessions_overlap_in_pool(self):
        """两笔 LAYER_DONE 在池内并行: 假 engine 的两次调用区间必须重叠."""
        engine = _FakeEngine(delay=0.2)
        fake_te = _FakeGlobalTe()
        thread = _make_recv_thread(engine)
        with mock.patch.object(conn, "_RECV_POOL", 2), mock.patch.object(conn, "global_te", fake_te):
            thread._start_recv_pool()
            try:
                self.assertIsNotNone(thread._recv_jobs)
                self.assertEqual(len(thread._recv_workers), 2)
                thread._recv_jobs.put(_job("req-a", "127.0.0.1:17001", b"id-a"))
                thread._recv_jobs.put(_job("req-b", "127.0.0.1:17002", b"id-b"))

                # 真实收线程的等待方式: poll 在唤醒端(worker 完成即写字节),
                # 再按顺序先清唤醒、后清完成队列.
                poller = zmq.Poller()
                poller.register(thread._wake_r, zmq.POLLIN)
                finished = []
                deadline = time.perf_counter() + WAIT_S
                while len(finished) < 2 and time.perf_counter() < deadline:
                    self.assertTrue(dict(poller.poll(500)), "worker 完成应唤醒收线程")
                    thread._drain_wakeup()
                    while True:
                        try:
                            finished.append(thread._recv_done.get_nowait())
                        except queue.Empty:
                            break

                self.assertEqual(len(finished), 2, "两笔都应完成")
                self.assertTrue(all(j.ok for j in finished))
                self.assertEqual(len(engine.calls), 2)
                (s0, e0, _), (s1, e1, _) = engine.calls
                self.assertLess(max(s0, s1), min(e0, e1), "两笔 pull 必须时间上重叠(池内并行)")
                self.assertEqual(len(fake_te.h2d_calls), 2)
            finally:
                thread._recv_jobs.put(None)  # 哨兵: 让 worker 退出
                thread._recv_jobs.put(None)
                for w in thread._recv_workers:
                    w.join(timeout=2.0)
                thread._wake_r.close()
                thread._wake_w.close()

    def test_inline_ack_completes_request(self):
        """内联路径(池关闭): 成功回 ACK, 且把该请求计入完成."""
        thread = _make_recv_thread(_FakeEngine(delay=0.0))
        ctx = zmq.Context()
        router = ctx.socket(zmq.ROUTER)
        port = router.bind_to_random_port("tcp://127.0.0.1")
        try:
            with mock.patch.object(conn, "global_te", _FakeGlobalTe()):
                self.assertEqual(_req_round_trip(thread, router, port, "req-ok"), b"ACK")
            self.assertIn("req-ok", thread.done_requests)
            self.assertNotIn("req-ok", thread.task_tracker)
            self.assertNotIn("req-ok", thread.failed_requests)
        finally:
            router.close(linger=0)
            ctx.destroy(linger=0)

    def test_inline_nak_marks_request_failed(self):
        """内联路径: sync_read 失败 → NAK + 作废该请求(不留 task_tracker 残渣)."""
        thread = _make_recv_thread(_FakeEngine(delay=0.0, ret=-1))
        ctx = zmq.Context()
        router = ctx.socket(zmq.ROUTER)
        port = router.bind_to_random_port("tcp://127.0.0.1")
        try:
            with mock.patch.object(conn, "global_te", _FakeGlobalTe()):
                self.assertEqual(_req_round_trip(thread, router, port, "req-bad"), b"NAK")
            self.assertIn("req-bad", thread.failed_requests)
            self.assertNotIn("req-bad", thread.done_requests)
            self.assertNotIn("req-bad", thread.task_tracker)
        finally:
            router.close(linger=0)
            ctx.destroy(linger=0)


if __name__ == "__main__":
    unittest.main()
