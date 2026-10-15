# SPDX-License-Identifier: Apache-2.0
"""P 侧逐层 LAYER_DONE 的"刷完即发 + 统一收 ACK"路径(MC_TCP_LD_CONCURRENT=1)单测.

不依赖 NPU/引擎: ROUTER 假扮 D 侧侧信道接收线程, 只验证发送侧的协议与调度行为
—— post 不等 ACK(因此 session 之间的发送与等待互相并行)、连接常驻复用、
payload 线格式、失败后弃用重建。

线程约定: **ZMQ socket 绝不跨线程使用**。假 D 端(ROUTER)全程在本用例线程创建、
收发、关闭; 发送侧(REQ)在其自己的子线程里创建、收发 —— 每次调用都在同一个子
线程内完成, 因此 `_peer_socks` 的常驻连接不会跨线程复用。
"""

import threading
import time
import unittest

import msgspec
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_prefill_then_decode_connector import (
    LAYER_DONE_SENDING_MSG,
    KVCacheSendingLayerThread,
)


class _FakeDecoderPeer:
    """最小 D 侧: ROUTER 收 LAYER_DONE, 由用例线程决定何时回 b"ACK"."""

    def __init__(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.identities: list[bytes] = []
        self._identity: bytes | None = None

    def recv(self, timeout_ms: int = 3000) -> bytes | None:
        """收一条消息; 超时返回 None(供"必须不超时"的断言使用)."""
        if not self.sock.poll(timeout_ms, zmq.POLLIN):
            return None
        frames = self.sock.recv_multipart()
        self._identity = frames[0]
        self.identities.append(frames[0])
        return frames[-1]

    def ack(self) -> None:
        assert self._identity is not None
        self.sock.send_multipart((self._identity, b"", b"ACK"))

    def close(self) -> None:
        self.sock.close(linger=0)
        self.ctx.destroy(linger=0)


def _make_sender() -> KVCacheSendingLayerThread:
    """绕过 __init__(需要引擎/NPU 配置), 只装配发送路径用到的状态."""
    thread = object.__new__(KVCacheSendingLayerThread)
    thread.sender_path = "127.0.0.1:16000"  # type: ignore[attr-defined]
    thread.sender_te_path = "127.0.0.1:17000"  # type: ignore[attr-defined]
    thread._peer_ctx = None
    thread._peer_socks = {}
    thread._ld_encoder = msgspec.msgpack.Encoder()
    return thread


def _msg(peer: _FakeDecoderPeer, req_ids=None):
    return ("127.0.0.1", peer.port, req_ids or ["req-a"], [1000], [2048], [], [4096])


def _run_in_thread(fn):
    """在子线程里执行发送并捕获结果/异常(发送侧的 ZMQ 使用全部收敛在这一个线程)."""
    box: dict = {}

    def _target():
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread, box


def _join(thread, box, timeout=5.0):
    thread.join(timeout)
    assert not thread.is_alive(), "发送线程未在预期时间内结束"
    if "error" in box:
        raise box["error"]
    return box["result"]


def _post_then_collect(sender, msgs):
    """镜像调用点的顺序: 逐个 post(不等 ACK), 最后统一 collect."""
    posted = [(m, sender._post_layer_done(m)) for m in msgs]
    return [sender._collect_layer_done_ack(m, sock) for m, sock in posted]


class TestLayerDonePostAndCollect(unittest.TestCase):
    def test_post_does_not_wait_for_ack(self):
        """post 不等 ACK: A 的 ACK 尚未回时, B 的信号必须已经发出.

        这正是调用点依赖的性质 —— 每个 session 刷完自己的 staging 就发它的
        LAYER_DONE, 不被别的 session 挡住(攒齐全批再发会退化成一道同步屏障)。
        串行实现(发 A → 等 A 的 ACK)下, B 的消息在 A 回 ACK 前根本发不出来,
        本用例的 `peer_b.recv()` 就会超时失败 —— 判据是确定性的, 不依赖掐表.
        """
        peer_a, peer_b = _FakeDecoderPeer(), _FakeDecoderPeer()
        try:
            sender = _make_sender()
            thread, box = _run_in_thread(lambda: _post_then_collect(sender, [_msg(peer_a), _msg(peer_b)]))
            self.assertIsNotNone(peer_a.recv(3000), "A 应在 3s 内收到 LAYER_DONE")
            self.assertIsNotNone(peer_b.recv(2000), "B 的消息必须不依赖 A 的 ACK 到达")
            peer_a.ack()
            peer_b.ack()
            self.assertEqual(_join(thread, box), [True, True])
        finally:
            peer_a.close()
            peer_b.close()

    def test_connection_persists_across_batches(self):
        """常驻连接: 两批之间同一 peer 的 socket 复用, ROUTER 侧 identity 不变."""
        peer = _FakeDecoderPeer()
        try:
            sender = _make_sender()

            def two_batches():
                # 两批必须在同一线程内跑: 常驻 socket 不可跨线程复用.
                return [
                    _post_then_collect(sender, [_msg(peer)]),
                    _post_then_collect(sender, [_msg(peer)]),
                ]

            thread, box = _run_in_thread(two_batches)
            for _ in range(2):
                self.assertIsNotNone(peer.recv(3000))
                peer.ack()
            self.assertEqual(_join(thread, box), [[True], [True]])
            self.assertEqual(len(sender._peer_socks), 1, "两批应复用同一条常驻连接")
            self.assertEqual(len(set(peer.identities)), 1, "两批应走同一条连接(identity 不变)")
        finally:
            peer.close()

    def test_payload_wire_format(self):
        """payload 口径必须是 8 元组且顺序不变(两端唯一的契约)."""
        peer = _FakeDecoderPeer()
        try:
            sender = _make_sender()
            thread, box = _run_in_thread(lambda: _post_then_collect(sender, [_msg(peer, ["ext-1"])]))
            payload = peer.recv(3000)
            self.assertIsNotNone(payload)
            peer.ack()
            self.assertEqual(_join(thread, box), [True])
            decoded = msgspec.msgpack.Decoder(type=tuple).decode(payload)
            self.assertEqual(decoded[0], LAYER_DONE_SENDING_MSG)
            self.assertEqual(decoded[1], ["ext-1"])
            self.assertEqual(decoded[2], [1000])
            self.assertEqual(decoded[3], [2048])
            self.assertEqual(decoded[4], "127.0.0.1:16000")
            self.assertEqual(decoded[5], [])
            self.assertEqual(decoded[6], [4096])
            self.assertEqual(decoded[7], "127.0.0.1:17000")
        finally:
            peer.close()

    def test_timeout_drops_socket_and_next_batch_rebuilds(self):
        """已发未收后 socket 不可再用: 超时 → 弃用 → 下批按需重建新连接."""
        peer = _FakeDecoderPeer()
        try:
            sender = _make_sender()
            stale = sender._peer_req_socket("127.0.0.1", peer.port)
            stale.setsockopt(zmq.RCVTIMEO, 100)  # 免等 LAYER_DONE_TIMEOUT_S
            msg = _msg(peer)
            sock = sender._post_layer_done(msg)  # 同一线程内直接调用
            self.assertIsNotNone(sock)
            t0 = time.perf_counter()
            self.assertFalse(sender._collect_layer_done_ack(msg, sock))
            self.assertLess(time.perf_counter() - t0, 3.0)
            self.assertIsNotNone(peer.recv(1000), "消息仍应已送达对端")
            self.assertNotIn(("127.0.0.1", peer.port), sender._peer_socks)
            self.assertIsNot(sender._peer_req_socket("127.0.0.1", peer.port), stale)
            # 发送就没成功(sock=None)时直接判失败, 不需要也不允许再收一次.
            self.assertFalse(sender._collect_layer_done_ack(msg, None))
        finally:
            peer.close()


if __name__ == "__main__":
    unittest.main()
