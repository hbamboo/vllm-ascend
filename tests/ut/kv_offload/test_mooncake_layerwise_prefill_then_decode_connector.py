# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""先 P 后 D (p_then_d) 的 connector 行为测试.

围绕两件事: P 侧把采样的首 token 与 KV 末轮任务凑齐后投给 D; D 侧把首 token 与
"接收完成"配对交给 scheduler (缺一不可 —— 少了首 token 就补不进 prompt, 晚一步
到达则请求已被放行, 注入落空)。
"""

import importlib.util
import sys
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine
fake_torch_npu = types.ModuleType("torch_npu")
fake_torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
fake_torch_npu.npu = MagicMock()  # type: ignore[attr-defined]
fake_torch_npu.npu.current_device = MagicMock(return_value=0)  # type: ignore[attr-defined]
fake_torch_npu.npu.Stream = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("torch_npu", fake_torch_npu)
torch.npu = fake_torch_npu.npu  # type: ignore[attr-defined]
fake_uvloop = types.ModuleType("uvloop")
fake_uvloop.__spec__ = importlib.util.spec_from_loader("uvloop", loader=None)
sys.modules.setdefault("uvloop", fake_uvloop)

# 清理其它测试文件塞进来的同名 mock 模块, 用完再放回去 (与
# test_mooncake_layerwise_connector.py 的处理保持一致)。
_kv_xfer = "vllm_ascend.distributed.kv_transfer"
_saved_modules: dict[str, types.ModuleType] = {}
for _k in list(sys.modules):
    if _k.startswith(_kv_xfer):
        _suffix = _k[len(_kv_xfer) :]
        if _suffix == "" or _suffix.startswith(".utils") or _suffix.startswith(".kv_p2p"):
            _saved_modules[_k] = sys.modules.pop(_k)

from vllm_ascend.distributed.kv_transfer.kv_p2p import (  # noqa: E402
    mooncake_layerwise_prefill_then_decode_connector as ptd,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import GET_META_MSG  # noqa: E402

for _k, _v in _saved_modules.items():
    sys.modules[_k] = _v

FIRST_TOKEN_MSG = ptd.FIRST_TOKEN_MSG


def _make_agent_metadata():
    return ptd.MooncakeAgentMetadata(
        te_rpc_port=6000,
        layer_metadata={"layer0": ptd.LayerMetadata([0], [1000, 2000], [1024], [1])},
    )


def _make_recv_thread(tp_rank=0):
    return ptd.KVCacheRecvingLayerThread(
        tp_rank=tp_rank,
        side_channel_port=5555,
        tp_size=1,
        pd_head_ratio=1,
        local_engine_id="engineD",
        metadata=_make_agent_metadata(),
        ready_event=threading.Event(),
    )


def _make_worker(**attrs):
    worker = object.__new__(ptd.MooncakeLayerwisePrefillThenDecodeConnectorWorker)
    worker.tp_rank = attrs.pop("tp_rank", 0)
    worker.timeout = 1
    worker.req_send_done_tasks = {}
    worker.req_send_done_tasks_lock = threading.Lock()
    worker.vllm_config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(is_kv_consumer=True),
    )
    worker.kv_recv_layer_thread = None
    worker.request_map = {}
    worker._recving_metadata = {}
    worker._expect_first_token = set()
    worker.virtual_request = set()
    worker._invalid_block_ids = set()
    for key, value in attrs.items():
        setattr(worker, key, value)
    return worker


def _make_req_meta(prompt_len=8, reuse=True, host="127.0.0.1", port=6000, recv_ports=None):
    return types.SimpleNamespace(
        prompt_len=prompt_len,
        reuse_prefilled_tokens=reuse,
        remote_host=host,
        remote_port=port,
        remote_recv_ports=[(host, port)] if recv_ports is None else recv_ports,
    )


class TestFirstTokenMessage(unittest.TestCase):
    def test_store_and_clear(self):
        th = _make_recv_thread()
        self.assertIsNone(th.get_first_token("reqA-00000000"))

        with th.lock:
            th.first_tokens["reqA"] = 42
        self.assertEqual(th.get_first_token("reqA-00000000"), 42)

        th.clear_first_token("reqA-00000000")
        self.assertIsNone(th.get_first_token("reqA-00000000"))

    def test_requeue_puts_requests_back(self):
        th = _make_recv_thread()
        with th.lock:
            th.done_requests.add("reqA")
        th.get_and_clear_done_requests()

        th.requeue_done_requests({"reqA"})
        self.assertEqual(th.get_and_clear_done_requests(), {"reqA"})

    def test_requeue_empty_is_noop(self):
        th = _make_recv_thread()
        th.requeue_done_requests(set())
        self.assertEqual(th.get_and_clear_done_requests(), set())

    @patch.object(ptd, "bind_current_thread_to_idle_cpu")
    @patch.object(ptd, "get_world_group", return_value=types.SimpleNamespace(local_rank=0))
    @patch.object(ptd, "logger")
    @patch.object(ptd, "get_ip", return_value="127.0.0.1")
    @patch.object(ptd.msgspec.msgpack, "Decoder")
    @patch.object(ptd.msgspec.msgpack, "Encoder")
    @patch.object(ptd, "zmq_ctx")
    def test_run_loop_stores_first_token_and_acks(
        self, mock_zmq_ctx, mock_encoder, mock_decoder, _ip, _logger, _wg, _bind
    ):
        enc_inst = MagicMock()
        enc_inst.encode.return_value = b"ENCODED"
        mock_encoder.return_value = enc_inst

        dec_inst = MagicMock()
        dec_inst.decode.side_effect = [
            (GET_META_MSG,),
            (FIRST_TOKEN_MSG, "reqA", 77),
            SystemExit,
        ]
        mock_decoder.return_value = dec_inst

        sock = MagicMock()
        sock.recv_multipart.side_effect = [
            [b"ID", b"P1"],
            [b"ID", b"P2"],
            SystemExit,
        ]
        cm = MagicMock()
        cm.__enter__.return_value = sock
        mock_zmq_ctx.return_value = cm

        th = _make_recv_thread()
        with self.assertRaises(SystemExit):
            th.run()

        self.assertEqual(th.get_first_token("reqA-00000000"), 77)
        replies = [c.args[0][2] for c in sock.send_multipart.call_args_list]
        # 首条回 GET_META 的元数据本体, 第二条回 ACK
        self.assertEqual(replies[0], b"ENCODED")
        self.assertEqual(replies[1], b"ACK")


class TestConnectorGetFinished(unittest.TestCase):
    def _make_connector(self, finished, first_token_map, producer=False):
        connector = object.__new__(ptd.MooncakeLayerwisePrefillThenDecodeConnector)
        worker = MagicMock()
        worker.get_finished.return_value = finished
        worker.get_first_token.side_effect = lambda req_id: first_token_map.get(req_id)
        connector.connector_worker = worker
        connector._is_kv_producer = producer
        return connector, worker

    def test_returns_first_tokens_for_consumer(self):
        connector, _ = self._make_connector((set(), {"reqA"}), {"reqA": 42})
        with patch.object(ptd, "REUSE_PREFILLED_TOKENS", True):
            result = connector.get_finished(set())

        self.assertEqual(result, (set(), {"reqA"}, {"reqA": 42}))

    def test_request_without_token_yields_empty_mapping(self):
        # 没在等首 token 的请求(非流式)照样正常完成
        connector, _ = self._make_connector((set(), {"reqB"}), {})
        with patch.object(ptd, "REUSE_PREFILLED_TOKENS", True):
            result = connector.get_finished(set())

        self.assertEqual(result, (set(), {"reqB"}, {}))

    def test_returns_two_tuple_when_env_disabled(self):
        connector, _ = self._make_connector((set(), {"reqA"}), {"reqA": 42})
        with patch.object(ptd, "REUSE_PREFILLED_TOKENS", False):
            result = connector.get_finished(set())

        self.assertEqual(result, (set(), {"reqA"}))

    def test_producer_side_returns_two_tuple(self):
        connector, _ = self._make_connector(({"reqA"}, set()), {"reqA": 42}, producer=True)
        with patch.object(ptd, "REUSE_PREFILLED_TOKENS", True):
            result = connector.get_finished(set())

        self.assertEqual(result, ({"reqA"}, set()))


def _make_scheduler_output(req_ids, computed_tokens, num_scheduled, spec_tokens=None):
    cached = MagicMock()
    cached.req_ids = list(req_ids)
    cached.num_computed_tokens = list(computed_tokens)
    return types.SimpleNamespace(
        scheduled_cached_reqs=cached,
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens=spec_tokens or {},
        num_scheduled_tokens=num_scheduled,
    )


class TestFirstTokenRendezvous(unittest.TestCase):
    """首 token 与 KV 末轮任务谁先到都要能配上对, 且只在 prompt 算完后才发。"""

    def _run_with_sender(self, worker, action):
        sock = MagicMock()
        sock.poll.return_value = True
        sock.recv.return_value = b"ACK"
        cm = MagicMock()
        cm.__enter__.return_value = sock
        encoder = MagicMock()
        with (
            patch.object(ptd, "zmq_ctx", return_value=cm),
            patch.object(ptd, "make_zmq_path", return_value="tcp://127.0.0.1:6000"),
            patch.object(ptd.msgspec.msgpack, "Encoder", return_value=encoder),
            patch.object(ptd, "REUSE_PREFILLED_TOKENS", True),
        ):
            action()
        return encoder

    def test_token_then_transfer_completes_sends_once(self):
        worker = _make_worker()
        req_meta = _make_req_meta()
        sched = _make_scheduler_output(["r1"], [0], {"r1": 8})

        # 采样结果先到: req_meta 还没有, 发不出去
        self._run_with_sender(worker, lambda: worker.send_prefilled_tokens(sched, ["r1"], [[42]]))
        self.assertEqual(worker.req_send_done_tasks["r1"]["token"], 42)

        # 末轮任务到达 -> 立即投递
        encoder = self._run_with_sender(worker, lambda: worker.register_transfer_done("r1", req_meta))
        payloads = [c.args[0] for c in encoder.encode.call_args_list]
        self.assertEqual(payloads[0][0], FIRST_TOKEN_MSG)
        self.assertEqual(payloads[0][1], ptd.get_external_request_id("r1"))
        self.assertEqual(payloads[0][2], 42)
        self.assertEqual(worker.req_send_done_tasks, {})

    def test_transfer_then_token_sends_once(self):
        worker = _make_worker()
        req_meta = _make_req_meta()
        sched = _make_scheduler_output(["r1"], [0], {"r1": 8})

        self._run_with_sender(worker, lambda: worker.register_transfer_done("r1", req_meta))
        self.assertIsNone(worker.req_send_done_tasks["r1"]["token"])

        encoder = self._run_with_sender(worker, lambda: worker.send_prefilled_tokens(sched, ["r1"], [[42]]))
        payloads = [c.args[0] for c in encoder.encode.call_args_list]
        self.assertEqual(payloads[0][0], FIRST_TOKEN_MSG)
        self.assertEqual(worker.req_send_done_tasks, {})

    def test_latest_sample_wins_and_sends_after_transfer_done(self):
        """采样结果与末轮任务都到齐后才发, 且发的是最新那次采样。

        每次调用只带"本步真的采样出来的 token": prefill 分 chunk 时模型 runner
        给的是空列表, 会被直接跳过(见 test_ignores_request_without_sampled_token),
        因此不需要再拿 computed_tokens 去判是不是末 chunk。
        """
        worker = _make_worker()
        req_meta = _make_req_meta(prompt_len=8)

        # 采样先到, 末轮任务还没来 -> 不发
        encoder = self._run_with_sender(
            worker,
            lambda: worker.send_prefilled_tokens(_make_scheduler_output(["r1"], [0], {"r1": 4}), ["r1"], [[41]]),
        )
        encoder.encode.assert_not_called()

        # 又采样一次 -> 覆盖为最新值, 仍不发
        encoder = self._run_with_sender(
            worker,
            lambda: worker.send_prefilled_tokens(_make_scheduler_output(["r1"], [4], {"r1": 4}), ["r1"], [[42]]),
        )
        encoder.encode.assert_not_called()
        self.assertEqual(worker.req_send_done_tasks["r1"]["token"], 42)

        # 末轮任务到达 -> 发最新那次
        encoder = self._run_with_sender(worker, lambda: worker.register_transfer_done("r1", req_meta))
        payloads = [c.args[0] for c in encoder.encode.call_args_list]
        self.assertEqual(payloads[0][2], 42)
        self.assertEqual(worker.req_send_done_tasks, {})

    def test_skips_when_reuse_disabled(self):
        worker = _make_worker()
        req_meta = _make_req_meta(reuse=False)

        self._run_with_sender(worker, lambda: worker.register_transfer_done("r1", req_meta))
        encoder = self._run_with_sender(
            worker,
            lambda: worker.send_prefilled_tokens(_make_scheduler_output(["r1"], [0], {"r1": 8}), ["r1"], [[42]]),
        )
        encoder.encode.assert_not_called()
        self.assertEqual(worker.req_send_done_tasks, {})

    def test_non_zero_tp_rank_does_not_send(self):
        worker = _make_worker(tp_rank=1)
        req_meta = _make_req_meta()

        with patch.object(ptd, "REUSE_PREFILLED_TOKENS", True):
            worker.register_transfer_done("r1", req_meta)
            worker.send_prefilled_tokens(_make_scheduler_output(["r1"], [0], {"r1": 8}), ["r1"], [[42]])

        self.assertEqual(worker.req_send_done_tasks, {})

    def test_ignores_request_without_sampled_token(self):
        worker = _make_worker()
        with patch.object(ptd, "REUSE_PREFILLED_TOKENS", True):
            worker.send_prefilled_tokens(_make_scheduler_output(["r1"], [0], {"r1": 8}), ["r1"], [[]])
        self.assertNotIn("r1", worker.req_send_done_tasks)

    def test_sends_to_every_decode_rank(self):
        """每个 D worker 都各自等首 token, 必须逐 rank 投递, 否则其余 rank 卡死。"""
        worker = _make_worker()
        ports = [("127.0.0.1", 6000), ("127.0.0.1", 6001), ("127.0.0.1", 6002), ("127.0.0.1", 6003)]
        req_meta = _make_req_meta(recv_ports=ports)
        sock = MagicMock()
        sock.poll.return_value = True
        sock.recv.return_value = b"ACK"
        cm = MagicMock()
        cm.__enter__.return_value = sock

        with (
            patch.object(ptd, "zmq_ctx", return_value=cm),
            patch.object(ptd, "make_zmq_path", side_effect=lambda _s, h, p: f"tcp://{h}:{p}") as mock_path,
            patch.object(ptd.msgspec.msgpack, "Encoder", return_value=MagicMock()),
            patch.object(ptd, "REUSE_PREFILLED_TOKENS", True),
        ):
            worker.register_transfer_done("r1", req_meta)
            worker.send_prefilled_tokens(_make_scheduler_output(["r1"], [0], {"r1": 8}), ["r1"], [[42]])

        self.assertEqual([c.args[1:] for c in mock_path.call_args_list], [("127.0.0.1", p) for _, p in ports])
        self.assertEqual(sock.poll.call_count, len(ports))


class TestGetFinishedHoldsBackMissingToken(unittest.TestCase):
    """D 侧: 还在等首 token 的请求不能被报成"接收完成"。

    recv 线程用的是 external req id (去掉末尾 9 位), 对外报的是内部 id。
    """

    def _make_gated_worker(self, done_ext, failed_ext=(), token_map=None):
        worker = _make_worker()
        recv = MagicMock()
        recv.get_and_clear_done_requests.return_value = set(done_ext)
        recv.get_and_clear_failed_requests.return_value = set(failed_ext)
        worker.kv_recv_layer_thread = recv
        worker.request_map = {ext: f"{ext}-00000000" for ext in list(done_ext) + list(failed_ext)}
        worker.get_first_token = lambda rid: (token_map or {}).get(rid)
        return worker, recv

    def test_holds_back_until_token_arrives(self):
        worker, recv = self._make_gated_worker({"reqA"}, token_map={})
        worker._expect_first_token = {"reqA-00000000"}

        with patch.object(ptd, "REUSE_PREFILLED_TOKENS", True):
            _, recving = worker.get_finished()

        self.assertEqual(recving, set())
        recv.requeue_done_requests.assert_called_once_with({"reqA"})
        # 压回的请求必须保留 id 映射, 否则下个 step 再也映射不回内部 id
        self.assertIn("reqA", worker.request_map)

    def test_reports_once_token_arrives(self):
        worker, recv = self._make_gated_worker({"reqA"}, token_map={"reqA-00000000": 42})
        worker._expect_first_token = {"reqA-00000000"}

        with patch.object(ptd, "REUSE_PREFILLED_TOKENS", True):
            _, recving = worker.get_finished()

        self.assertEqual(recving, {"reqA-00000000"})
        recv.requeue_done_requests.assert_not_called()
        self.assertNotIn("reqA-00000000", worker._expect_first_token)

    def test_request_not_expecting_token_reports_immediately(self):
        # 非流式请求: P 不会投首 token, 不能把它压住
        worker, recv = self._make_gated_worker({"reqB"}, token_map={})
        worker._expect_first_token = set()

        with patch.object(ptd, "REUSE_PREFILLED_TOKENS", True):
            _, recving = worker.get_finished()

        self.assertEqual(recving, {"reqB-00000000"})
        recv.requeue_done_requests.assert_not_called()

    def test_gating_disabled_when_env_off(self):
        worker, recv = self._make_gated_worker({"reqA"}, token_map={})
        worker._expect_first_token = {"reqA-00000000"}

        with patch.object(ptd, "REUSE_PREFILLED_TOKENS", False):
            _, recving = worker.get_finished()

        self.assertEqual(recving, {"reqA-00000000"})
        recv.requeue_done_requests.assert_not_called()


class TestDecoderParamsChannel(unittest.TestCase):
    """P 先受理时手里没有对端信息, 靠 D 直投的 block table 建 peer 映射。"""

    def _make_req_meta(self):
        meta = ptd.MooncakeLayerwisePrefillThenDecodeConnectorMetadata()
        meta.add_new_req(
            request_id="r1-00000000",
            local_block_ids=[[1, 2]],
            kv_transfer_params={"do_remote_decode": True},
            chunk_finish=True,
            prompt_len=8,
        )
        return meta.requests["r1-00000000"]

    def test_build_remote_recv_ports_covers_every_decode_rank(self):
        ports = ptd.MooncakeLayerwisePrefillThenDecodeConnectorMetadata._build_remote_recv_ports(
            {"remote_host": "10.0.0.1", "remote_port": 16583, "remote_tp_size": 4}
        )
        self.assertEqual(ports, [("10.0.0.1", 16583 + r) for r in range(4)])

    def test_build_remote_recv_ports_empty_without_peer(self):
        self.assertEqual(ptd.MooncakeLayerwisePrefillThenDecodeConnectorMetadata._build_remote_recv_ports({}), [])

    def test_apply_decoder_params_fills_req_meta(self):
        worker = _make_worker()
        req_meta = self._make_req_meta()
        pull = MagicMock()
        pull.get_kv_transfer_params.return_value = (
            {
                "remote_block_ids": [[7, 8]],
                "remote_block_size": [[16]],
                "remote_host": "10.0.0.1",
                "remote_port": 16579,
                "remote_tp_size": 2,
                "remote_cached_tokens": 3,
                "reuse_prefilled_tokens": False,
            },
            False,
        )
        worker.pull_thread = pull

        self.assertTrue(worker._apply_decoder_params("r1-00000000", req_meta, True))

        self.assertEqual(req_meta.remote_block_ids, [[7, 8]])
        self.assertEqual(req_meta.remote_host, "10.0.0.1")
        self.assertEqual(req_meta.remote_port, 16579)
        self.assertEqual(req_meta.remote_cache_tokens, 3)
        self.assertFalse(req_meta.reuse_prefilled_tokens)
        self.assertEqual(req_meta.remote_recv_ports, [("10.0.0.1", 16579), ("10.0.0.1", 16580)])

    def test_apply_decoder_params_reports_failure(self):
        worker = _make_worker()
        req_meta = self._make_req_meta()
        pull = MagicMock()
        pull.get_kv_transfer_params.return_value = (None, True)
        worker.pull_thread = pull

        self.assertFalse(worker._apply_decoder_params("r1-00000000", req_meta, True))

    def test_apply_decoder_params_noop_without_channel(self):
        worker = _make_worker()
        worker.pull_thread = None
        self.assertTrue(worker._apply_decoder_params("r1-00000000", self._make_req_meta(), True))

    def test_push_fans_out_to_every_prefill_rank(self):
        scheduler = object.__new__(ptd.MooncakeLayerwisePrefillThenDecodeConnectorScheduler)
        scheduler.executor = MagicMock()
        scheduler.executor.submit.return_value = MagicMock()
        params = {"kv_transfer_params_zmq_port": 16583, "remote_tp_size": 4, "remote_host": "10.0.0.5"}

        scheduler._push_kv_transfer_params_to_prefiller(params, {"remote_block_ids": [[1]]}, "r1")

        ports = [c.kwargs["remote_port"] for c in scheduler.executor.submit.call_args_list]
        self.assertEqual(ports, [16583, 16584, 16585, 16586])
        self.assertTrue(all(c.kwargs["remote_host"] == "10.0.0.5" for c in scheduler.executor.submit.call_args_list))

    def test_push_skipped_without_channel(self):
        scheduler = object.__new__(ptd.MooncakeLayerwisePrefillThenDecodeConnectorScheduler)
        scheduler.executor = MagicMock()

        scheduler._push_kv_transfer_params_to_prefiller({}, {"remote_block_ids": [[1]]}, "r1")

        scheduler.executor.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
