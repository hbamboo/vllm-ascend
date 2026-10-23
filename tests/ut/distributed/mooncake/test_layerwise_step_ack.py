# SPDX-License-Identifier: Apache-2.0
"""step 级 ACK（MC_TCP_STEP_ACK=1）的 P/D 协议单测.

覆盖:
- P 连续 post 多条 LAYER_DONE，不逐条等 ACK；
- D 只有在收到 step_last、且该 step 全部 job 完成后才回唯一 ACK；
- D 侧乱序完成时 ACK 不会提前；
- 任一 job 失败时 D 回 NAK（带失败的 external req id），P 映射回内部 req id。

用真实 ZMQ DEALER/ROUTER，不依赖 NPU/引擎。
"""

import threading
import unittest

import msgspec
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_prefill_then_decode_connector as conn


def _make_sender() -> conn.KVCacheSendingLayerThread:
    thread = object.__new__(conn.KVCacheSendingLayerThread)
    thread.sender_path = "127.0.0.1:16000"  # type: ignore[attr-defined]
    thread.sender_te_path = "127.0.0.1:17000"  # type: ignore[attr-defined]
    thread._ld_ctx = None
    thread._peer_dealers = {}
    thread._ld_outstanding = {}
    thread._step_id = 0
    thread._step_seq = {}
    thread._ld_encoder = msgspec.msgpack.Encoder()
    thread._ld_send_failed = set()
    thread.failed_reqs = set()
    return thread


def _make_recv() -> conn.KVCacheRecvingLayerThread:
    thread = object.__new__(conn.KVCacheRecvingLayerThread)
    thread._ld_step_state = {}
    thread._ld_reply_encoder = msgspec.msgpack.Encoder()
    thread._failed_reqs_step = set()
    return thread


def _job(payload, identity: bytes, ok: bool = True) -> conn._RecvJob:
    msg = msgspec.msgpack.Decoder(type=tuple).decode(payload)
    assert msg[0] == conn.LAYER_DONE_SENDING_MSG
    assert len(msg) == 11, "step ACK payload 应为 11 元组"
    return conn._RecvJob(
        identity=identity,
        sender_path=msg[4],
        req_ids=list(msg[1]),
        dst_addrs=list(msg[2]),
        lengths=list(msg[3]),
        done_list=msg[5],
        src_addrs=msg[6],
        sender_te_path=msg[7],
        t_recv0=0.0,
        seq=msg[8],
        step_id=msg[9],
        step_last=msg[10],
        ok=ok,
    )


def _run_sender(fn):
    box: dict = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


class TestStepAck(unittest.TestCase):
    def _peer(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.ROUTER)
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        return ctx, sock, port

    def test_single_ack_after_all_jobs_with_out_of_order_completion(self):
        ctx, peer, port = self._peer()
        try:
            sender = _make_sender()
            recv = _make_recv()
            req_maps = [
                {"ext-a": "int-a"},
                {"ext-b": "int-b"},
                {"ext-c": "int-c"},
            ]
            replies: list[tuple] = []
            enc = msgspec.msgpack.Encoder()

            def record_reply(sock, identity, reply):
                replies.append(reply)
                sock.send_multipart((identity, b"", enc.encode(reply)))

            recv._send_ld_reply = record_reply  # type: ignore[method-assign]

            def send_all():
                for idx in range(3):
                    ok = sender._post_layer_done_async(
                        "127.0.0.1",
                        port,
                        list(req_maps[idx]),
                        [1000 + idx],
                        [2048],
                        None,
                        [4096 + idx],
                        step_last=(idx == 2),
                        req_id_map=req_maps[idx],
                    )
                    assert ok
                # 不逐条等：这里才收 step ACK.
                sender._drain_step_acks()
                return list(sender.failed_reqs)

            thread, box = _run_sender(send_all)

            jobs = []
            identities = set()
            for _ in range(3):
                frames = peer.recv_multipart()
                identities.add(frames[0])
                job = _job([f for f in frames[1:] if f != b""][0], frames[0])
                jobs.append(job)
                recv._ld_note_last(job)
            self.assertEqual(len(identities), 1, "同一 peer 应复用同一条 DEALER")
            self.assertEqual([j.seq for j in jobs], [0, 1, 2])
            self.assertTrue(jobs[0].step_last is False)
            self.assertTrue(jobs[2].step_last is True)

            # 乱序完成：最后一条先完成，不能提前 ACK.
            recv._ld_finish_step(peer, jobs[2])
            self.assertEqual(replies, [])
            recv._ld_finish_step(peer, jobs[0])
            self.assertEqual(replies, [])
            recv._ld_finish_step(peer, jobs[1])
            self.assertEqual(replies, [(b"ACK", 0, 2)])

            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(box.get("result"), [], box.get("error"))
        finally:
            peer.close(0)
            ctx.destroy(0)
            if sender._ld_ctx is not None:
                sender._ld_ctx.destroy(0)

    def test_nak_reports_failed_external_ids_and_p_maps_to_internal(self):
        ctx, peer, port = self._peer()
        try:
            sender = _make_sender()
            recv = _make_recv()
            replies: list[tuple] = []
            enc = msgspec.msgpack.Encoder()

            def record_reply(sock, identity, reply):
                replies.append(reply)
                sock.send_multipart((identity, b"", enc.encode(reply)))

            recv._send_ld_reply = record_reply  # type: ignore[method-assign]

            def send_all():
                sender._post_layer_done_async(
                    "127.0.0.1", port, ["ext-a"], [1], [2], None, [3], step_last=False, req_id_map={"ext-a": "int-a"}
                )
                sender._post_layer_done_async(
                    "127.0.0.1", port, ["ext-b"], [4], [5], None, [6], step_last=True, req_id_map={"ext-b": "int-b"}
                )
                sender._drain_step_acks()
                return sorted(sender.failed_reqs)

            thread, box = _run_sender(send_all)

            jobs = []
            for _ in range(2):
                frames = peer.recv_multipart()
                job = _job([f for f in frames[1:] if f != b""][0], frames[0])
                jobs.append(job)
                recv._ld_note_last(job)

            jobs[0].ok = False
            jobs[0].req_ids = ["ext-a"]
            recv._ld_finish_step(peer, jobs[0])
            self.assertEqual(replies, [])
            recv._ld_finish_step(peer, jobs[1])
            self.assertEqual(replies, [(b"NAK", 0, 1, ["ext-a"])])

            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(box.get("result"), ["int-a"], box.get("error"))
        finally:
            peer.close(0)
            ctx.destroy(0)
            if sender._ld_ctx is not None:
                sender._ld_ctx.destroy(0)

    def test_empty_step_marker_is_noop_and_acks(self):
        """step 末补的空 LAYER_DONE(无 dst/length/src)不触发 pull/H2D, 只回 ACK."""
        recv = _make_recv()
        job = conn._RecvJob(
            identity=b"id",
            sender_path="p",
            req_ids=[],
            dst_addrs=[],
            lengths=[],
            done_list=[],
            src_addrs=[],
            sender_te_path="",
            t_recv0=0.0,
            seq=0,
            step_id=0,
            step_last=True,
        )
        recv._ld_note_last(job)
        replies: list[tuple] = []
        recv._send_ld_reply = lambda sock, identity, reply: replies.append(reply)  # type: ignore[method-assign]
        recv._run_recv_job(job)
        self.assertTrue(job.ok)
        recv._ld_finish_step(None, job)
        self.assertEqual(replies, [(b"ACK", 0, 0)])

    def test_old_protocol_path_is_untouched(self):
        """旧 8 元组消息 seq<0: 不做 step 状态机, 由 _finish_recv_job 逐条回."""
        recv = _make_recv()
        job = conn._RecvJob(
            identity=b"id",
            sender_path="p",
            req_ids=["ext"],
            dst_addrs=[1],
            lengths=[2],
            done_list=[],
            src_addrs=[],
            sender_te_path="",
            t_recv0=0.0,
        )
        self.assertEqual(job.seq, -1)
        recv._ld_note_last(job)
        self.assertEqual(recv._ld_step_state, {})


if __name__ == "__main__":
    unittest.main()
