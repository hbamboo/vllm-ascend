import contextlib
import importlib.util
import os
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import zmq

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine
fake_torch_npu = types.ModuleType("torch_npu")
fake_torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
fake_torch_npu.npu = MagicMock()  # type: ignore[attr-defined]
fake_torch_npu.npu.current_device = MagicMock(return_value=0)  # type: ignore[attr-defined]
fake_torch_npu.npu.Stream = MagicMock  # type: ignore[attr-defined]
fake_torch_npu.npu_fusion_attention = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("torch_npu", fake_torch_npu)
torch.npu = fake_torch_npu.npu  # type: ignore[attr-defined]
fake_uvloop = types.ModuleType("uvloop")
fake_uvloop.__spec__ = importlib.util.spec_from_loader("uvloop", loader=None)
sys.modules.setdefault("uvloop", fake_uvloop)

# Clean up stale mock modules installed by other test files
# (e.g., ascend_store/_mock_deps.py) that replace real kv_transfer
# subpackages with MagicMock/fake modules, breaking our imports.
# We save the removed modules so we can restore them after our imports
# complete, so other test files (ascend_store) still see their mocks.
_kv_xfer = "vllm_ascend.distributed.kv_transfer"
_vllm_kv_xfer = "vllm.distributed.kv_transfer"
_saved_modules: dict[str, types.ModuleType] = {}
_to_remove = []
for k in list(sys.modules):
    if k.startswith(_kv_xfer):
        suffix = k[len(_kv_xfer) :]
        if suffix == "" or suffix.startswith(".utils") or suffix.startswith(".kv_p2p"):
            _to_remove.append(k)
    elif k.startswith(_vllm_kv_xfer):
        _to_remove.append(k)
for _m in _to_remove:
    _saved_modules[_m] = sys.modules.pop(_m)

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (  # noqa: E402
    _LAYER_BATCH,
    KVCacheRecvingLayerThread,
    KVCacheSendingLayerThread,
    KVConnectorRole,
    LayerMetadata,
    MooncakeAgentMetadata,
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
    ReqMeta,
    SendReqInfo,
    SendTask,
    TransferMeta,
    ensure_zmq_recv,
    ensure_zmq_send,
    group_concurrent_contiguous,
    string_to_int64_hash,
    zmq_ctx,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te  # noqa: E402

# Restore the mocked modules so other test files still work correctly.
# For keys that our real import loaded, overwrite with the saved mock.
for _k, _v in _saved_modules.items():
    sys.modules[_k] = _v

GET_META_MSG = b"get_meta_msg"
DONE_SENDING_MSG = b"done_sending_msg"


def _make_layer_metadata(**overrides):
    defaults = dict(
        tensor_group_idx=[0],
        kv_caches_base_addr=[1000, 2000],
        block_len=[1024],
        block_size_scale=[1],
    )
    defaults.update(overrides)
    return LayerMetadata(**defaults)


def _set_single_peer(req_meta):
    """把单 peer 兼容字段搬进 peer_transfer(start_load_kv 之后发送路径按 peer 走)."""
    peer = (req_meta.remote_host, req_meta.remote_port)
    req_meta.peer_transfer[peer] = {
        "local_block_ids": req_meta.local_block_ids,
        "remote_block_ids": req_meta.remote_block_ids,
        "trans_count": [1] * len(req_meta.local_block_ids),
    }
    req_meta.peer_layer_metadata[peer] = req_meta.remote_layer_metadata
    req_meta.peer_te_rpc_port[peer] = req_meta.remote_te_rpc_port
    return req_meta


def _make_mock_kv_cache_config(block_size=16):
    kv_cache_spec = MagicMock()
    kv_cache_spec.block_size = block_size
    group_spec = MagicMock()
    group_spec.kv_cache_spec = kv_cache_spec
    group_spec.layer_names = ["layer0"]
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [group_spec]
    return kv_cache_config


class TestKVCacheSendingLayerThread(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.engine.register_memory.return_value = 0
        self.engine.batch_transfer_sync_write.return_value = 1
        fake_stream = MagicMock(name="FakeStream")
        fake_stream.synchronize = MagicMock()

        self.first_kv_cache = torch.zeros((2, 2, 2, 8), dtype=torch.float32, device="cpu")

        self.ready_event = threading.Event()

        self.fake_k_buffer = MagicMock()
        self.fake_v_buffer = MagicMock()
        fake_resharding_stream = MagicMock()

        self.layer_metadata = {
            "layer0": _make_layer_metadata(
                tensor_group_idx=[0],
                kv_caches_base_addr=[1000, 2000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
            "layer1": _make_layer_metadata(
                tensor_group_idx=[0],
                kv_caches_base_addr=[3000, 4000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
            "layer2": _make_layer_metadata(
                tensor_group_idx=[0],
                kv_caches_base_addr=[5000, 6000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
        }

        self.vllm_config = MagicMock()
        self.vllm_config.cache_config.mamba_cache_mode = None
        self.vllm_config.speculative_config = None

        self.kv_cache_config = _make_mock_kv_cache_config()
        self.kv_cache_specs = [MagicMock(block_size=16)]

        self.key = torch.zeros((4, 8), dtype=torch.float32)
        self.value = torch.zeros((4, 8), dtype=torch.float32)
        self.thread = KVCacheSendingLayerThread(
            engine=self.engine,
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
            kv_cache_specs=self.kv_cache_specs,
            attn_resharding_group_idx=set(),
            total_layers=3,
            ready_event=self.ready_event,
            tp_size=1,
            tp_rank=0,
            pd_head_ratio=1,
            num_head_replica=1,
            layer_metadata=self.layer_metadata,
            use_mla=True,
            use_attn_mamba_hybrid=False,
            k_buffer=self.fake_k_buffer,
            v_buffer=self.fake_v_buffer,
            enable_kv_quant=False,
            enable_c8_quant=False,
            resharding_stream=fake_resharding_stream,
            sender_path="127.0.0.1:9999",
        )

        self.req_meta_base = ReqMeta(
            local_block_ids=[[5, 8]],
            token_ids=[1, 2, 3],
            remote_block_ids=[[10, 20]],
            remote_block_size=[[16]],
            remote_engine_id="remote_engine",
            remote_host="127.0.0.1",
            remote_port=7777,
            remote_te_rpc_port=6000,
            remote_layer_metadata={
                "layer0": _make_layer_metadata(
                    kv_caches_base_addr=[4000, 8000],
                    block_len=[64, 64],
                    block_size_scale=[1, 1],
                ),
            },
            metaserver="http://dummy",
            remote_tp_size=8,
            remote_pcp_size=1,
            remote_dcp_size=1,
            chunk_finish=False,
        )

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.npu_stream_switch",
        side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.torch.Tensor.data_ptr",
        autospec=True,
        return_value=0x200000,
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.align_memory",
        side_effect=lambda x, _align: x,
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.torch.npu.synchronize")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.group_concurrent_contiguous")
    def test_transfer_pd_gt1_uses_buffers_and_calls_engine(
        self, mock_group, _mock_sync, _mock_align, _mock_dataptr, mock_stream_switch
    ):
        fake_resharding_stream = MagicMock()

        layer_metadata = {
            "layer0": _make_layer_metadata(
                tensor_group_idx=[0],
                kv_caches_base_addr=[1111, 2222],
                block_len=[64, 64],
                block_size_scale=[1, 1],
            ),
        }

        vllm_config = MagicMock()
        vllm_config.cache_config.mamba_cache_mode = None
        vllm_config.speculative_config = None

        kv_cache_config = _make_mock_kv_cache_config()
        kv_cache_specs = [MagicMock(block_size=16)]

        thread = KVCacheSendingLayerThread(
            engine=self.engine,
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            kv_cache_specs=kv_cache_specs,
            attn_resharding_group_idx=set(),
            total_layers=2,
            ready_event=self.ready_event,
            tp_size=1,
            tp_rank=0,
            pd_head_ratio=2,
            num_head_replica=1,
            layer_metadata=layer_metadata,
            use_mla=False,
            use_attn_mamba_hybrid=False,
            k_buffer=self.fake_k_buffer,
            v_buffer=self.fake_v_buffer,
            enable_kv_quant=False,
            enable_c8_quant=False,
            resharding_stream=fake_resharding_stream,
            sender_path="127.0.0.1:9999",
        )

        req_meta = self.req_meta_base
        req_meta.remote_block_ids = [[10, 20]]
        req_meta.remote_layer_metadata = {
            "layer0": _make_layer_metadata(
                kv_caches_base_addr=[4000, 8000],
                block_len=[64, 64],
                block_size_scale=[1, 1],
            ),
        }

        mock_group.return_value = ([[10, 11], [20, 21]], [])
        key = torch.zeros((1, 8), dtype=torch.float32)
        value = torch.zeros((1, 8), dtype=torch.float32)
        _set_single_peer(req_meta)

        send_task = SendTask(
            send_request={"req1": req_meta},
            wait_event=MagicMock(),
            k_cache=key,
            v_cache=value,
            layer_idx=0,
            layer_name="layer0",
            group_rearrange_block_ids=[[5, 8]],
        )

        thread._transfer_kv_cache_batch([send_task])

        self.engine.batch_transfer_sync_write.assert_called_once()
        session_id, src_list, dst_list, length_list = self.engine.batch_transfer_sync_write.call_args[0]
        self.assertEqual(session_id, "127.0.0.1:6000")

        self.assertEqual(len(src_list), 4)
        self.assertEqual(len(dst_list), 4)
        self.assertEqual(len(length_list), 4)

        for L in length_list:
            self.assertGreater(L, 0)
            self.assertEqual(L % 64, 0)

        remote_block_len = 64
        expected_offsets = [10 * remote_block_len, 20 * remote_block_len]
        self.assertEqual(dst_list[0] - 4000, expected_offsets[0])
        self.assertEqual(dst_list[1] - 4000, expected_offsets[1])
        self.assertEqual(dst_list[2] - 8000, expected_offsets[0])
        self.assertEqual(dst_list[3] - 8000, expected_offsets[1])

    def test_transfer_skips_when_no_local_blocks(self):
        req_meta = self.req_meta_base
        req_meta.local_block_ids = [[]]
        send_task = SendTask(
            send_request={"req2": req_meta},
            wait_event=MagicMock(),
            k_cache=torch.zeros((1, 8)),
            v_cache=torch.zeros((1, 8)),
            layer_idx=0,
            layer_name="layer0",
            group_rearrange_block_ids=[[]],
        )
        self.thread._transfer_kv_cache_batch([send_task])
        self.engine.batch_transfer_sync_write.assert_not_called()

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.group_concurrent_contiguous",
        side_effect=group_concurrent_contiguous,
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.torch.npu.synchronize")
    def test_layer_done_carries_req_done_on_final_layer(self, _mock_sync, _mock_group):
        """末轮(chunk_finish)的请求级完成信息随含最后层的批 LAYER_DONE 一起下发."""
        req_meta = self.req_meta_base
        req_meta.chunk_finish = True
        req_meta.local_block_ids = [[5, 6]]
        req_meta.remote_block_ids = [[10, 11]]
        req_meta.remote_layer_metadata = {
            "layer0": _make_layer_metadata(
                kv_caches_base_addr=[7000, 8000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
            "layer1": _make_layer_metadata(
                kv_caches_base_addr=[9000, 10000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
            "layer2": _make_layer_metadata(
                kv_caches_base_addr=[11000, 12000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
        }

        key = torch.zeros((1, 8), dtype=torch.float32)
        value = torch.zeros((1, 8), dtype=torch.float32)
        _set_single_peer(req_meta)

        send_task = SendTask(
            send_request={"req5abcdefghi": req_meta},
            wait_event=MagicMock(),
            k_cache=key,
            v_cache=value,
            layer_idx=2,
            layer_name="layer2",
            group_rearrange_block_ids=[[]],
        )
        with (
            patch.object(global_te, "_use_tcp", True),
            patch.object(global_te, "npu_addr_to_cpu_addr", return_value=0x1234),
            patch.object(self.thread, "_send_layer_done_signal", return_value=True) as mock_ld,
        ):
            self.thread._transfer_kv_cache_batch([send_task])

        mock_ld.assert_called_once()
        done_list = mock_ld.call_args[0][5]
        # (external_req_id, is_last, trans_count, failed): 末轮 → is_last=True;
        # trans_count 取该 peer 在最后层所属 group 的期望路径数(_set_single_peer → 1).
        self.assertEqual(done_list, [("req5", True, 1, False)])

    def _final_layer_send_task(self, req_id: str) -> SendTask:
        """构造"含最后层 + 末轮(chunk_finish)"的单任务, 供完成/失败信号测试复用."""
        req_meta = self.req_meta_base
        req_meta.chunk_finish = True
        req_meta.local_block_ids = [[5, 6]]
        req_meta.remote_block_ids = [[10, 11]]
        req_meta.remote_layer_metadata = {
            "layer2": _make_layer_metadata(
                kv_caches_base_addr=[11000, 12000], block_len=[1024, 2048], block_size_scale=[1, 1]
            ),
        }
        _set_single_peer(req_meta)
        return SendTask(
            send_request={req_id: req_meta},
            wait_event=MagicMock(),
            k_cache=torch.zeros((1, 8), dtype=torch.float32),
            v_cache=torch.zeros((1, 8), dtype=torch.float32),
            layer_idx=2,
            layer_name="layer2",
            group_rearrange_block_ids=[[]],
        )

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.group_concurrent_contiguous",
        side_effect=group_concurrent_contiguous,
    )
    def test_d2d_final_layer_sends_request_level_done(self, _mock_group):
        """D2D(protocol=ascend) 没有 LAYER_DONE 可搭车 → 末轮发独立请求级 DONE."""
        send_task = self._final_layer_send_task("req6abcdefghi")
        with (
            patch.object(global_te, "_use_tcp", False),
            patch.object(self.thread, "_send_layer_done_signal") as mock_ld,
            patch.object(self.thread, "_send_done_signal") as mock_done,
        ):
            self.thread._transfer_kv_cache_batch([send_task])

        mock_ld.assert_not_called()
        mock_done.assert_called_once()
        req_id, _req_meta, group_idx = mock_done.call_args[0]
        self.assertEqual(req_id, "req6abcdefghi")
        self.assertEqual(group_idx, 0)

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.group_concurrent_contiguous",
        side_effect=group_concurrent_contiguous,
    )
    def test_tcp_final_layer_skips_request_level_done(self, _mock_group):
        """TCP: 完成信息随末批 LAYER_DONE 的 done_list 下发, 不再单发 DONE 往返."""
        send_task = self._final_layer_send_task("req7abcdefghi")
        with (
            patch.object(global_te, "_use_tcp", True),
            patch.object(global_te, "npu_addr_to_cpu_addr", return_value=0x1234),
            patch.object(self.thread, "_send_layer_done_signal", return_value=True) as mock_ld,
            patch.object(self.thread, "_send_done_signal") as mock_done,
        ):
            self.thread._transfer_kv_cache_batch([send_task])

        mock_ld.assert_called_once()
        self.assertEqual(mock_ld.call_args[0][5], [("req7", True, 1, False)])
        mock_done.assert_not_called()

    def test_req_done_entry_skips_peer_without_last_layer_data(self):
        """只承载了其它层(如 mamba)数据的 peer 既不产生请求级完成条目也不发 LAYER_DONE."""
        req_meta = self.req_meta_base
        req_meta.chunk_finish = True
        peer_attn = ("127.0.0.1", 7777)
        peer_mamba = ("127.0.0.1", 7778)
        req_meta.peer_transfer = {
            # peer_attn 承载本 group(layer2 → group 0)数据; peer_mamba 该 group 为空.
            peer_attn: {"local_block_ids": [[5]], "remote_block_ids": [[10]], "trans_count": [2]},
            peer_mamba: {"local_block_ids": [[]], "remote_block_ids": [[]], "trans_count": [0]},
        }
        req_meta.peer_layer_metadata = {
            peer_attn: {
                "layer2": _make_layer_metadata(
                    kv_caches_base_addr=[11000, 12000], block_len=[1024, 2048], block_size_scale=[1, 1]
                )
            },
            peer_mamba: {
                "layer2": _make_layer_metadata(
                    kv_caches_base_addr=[13000, 14000], block_len=[1024, 2048], block_size_scale=[1, 1]
                )
            },
        }
        req_meta.peer_te_rpc_port = {peer_attn: 6000, peer_mamba: 6001}

        send_task = SendTask(
            send_request={"reqAabcdefghi": req_meta},
            wait_event=MagicMock(),
            k_cache=torch.zeros((1, 8), dtype=torch.float32),
            v_cache=torch.zeros((1, 8), dtype=torch.float32),
            layer_idx=2,
            layer_name="layer2",
            group_rearrange_block_ids=[[5]],
        )
        with (
            patch.object(global_te, "_use_tcp", True),
            patch.object(global_te, "npu_addr_to_cpu_addr", return_value=0x1234),
            patch.object(self.thread, "_send_layer_done_signal", return_value=True) as mock_ld,
        ):
            self.thread._transfer_kv_cache_batch([send_task])

        # peer_mamba 本批无数据 → 不建 LAYER_DONE(其 D 侧期望计数为 0).
        self.assertEqual(len(mock_ld.call_args_list), 1)
        host, port, _ids, _addrs, _lens, done_list = mock_ld.call_args[0]
        self.assertEqual((host, port), peer_attn)
        self.assertEqual(done_list, [("reqA", True, 2, False)])

    def test_send_failed_signal_only_reaches_last_group_peers(self):
        """失败通知只发给承载最后层所属 group 数据的 peer(期望计数为 0 的 peer 跳过)."""
        req_meta = self.req_meta_base
        peer_attn = ("127.0.0.1", 7777)
        peer_mamba = ("127.0.0.1", 7778)
        req_meta.peer_transfer = {
            peer_attn: {"local_block_ids": [[5]], "remote_block_ids": [[10]], "trans_count": [2]},
            peer_mamba: {"local_block_ids": [[], [7]], "remote_block_ids": [[], [11]], "trans_count": [0]},
        }
        with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.zmq_ctx") as mock_ctx:
            sock = mock_ctx.return_value.__enter__.return_value
            sock.recv.return_value = b"ACK"
            self.thread._send_failed_signal("reqAabcdefghi", req_meta, 0)

        self.assertEqual(mock_ctx.call_count, 1)
        self.assertIn("7777", mock_ctx.call_args[0][1])

    def test_send_done_signal_tags_done_sending_msg(self):
        """D2D 成功信号复用同一通道, 但消息类型必须是 DONE_SENDING_MSG(不是 FAILED)."""
        req_meta = self.req_meta_base
        peer_attn = ("127.0.0.1", 7777)
        req_meta.peer_transfer = {
            peer_attn: {"local_block_ids": [[5]], "remote_block_ids": [[10]], "trans_count": [2]},
        }
        with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.zmq_ctx") as mock_ctx:
            sock = mock_ctx.return_value.__enter__.return_value
            sock.recv.return_value = b"ACK"
            self.thread._send_done_signal("reqBabcdefghi", req_meta, 0)

        self.assertEqual(mock_ctx.call_count, 1)
        payload = sock.send.call_args[0][0]
        self.assertIn(DONE_SENDING_MSG, payload)
        self.assertIn(b"reqB", payload)

    def test_req_done_list_filters_peer_and_marks_failed(self):
        """请求级完成条目按 peer 过滤, 并携带 P 侧的失败标记."""
        meta = TransferMeta(src=[], dst=[], length=[], req_ids=[])
        meta.req_done[("127.0.0.1", 7777, "reqAabcdefghi")] = (True, 2)
        meta.req_done[("127.0.0.1", 7778, "reqBabcdefghi")] = (False, 2)

        self.assertEqual(self.thread._req_done_list(meta, "127.0.0.1", 7777), [("reqA", True, 2, False)])
        # 另一 peer / 非末轮(is_last=False, 如中间 chunk)不产生条目.
        self.assertEqual(self.thread._req_done_list(meta, "127.0.0.1", 7779), [])

        self.thread.failed_reqs.add("reqAabcdefghi")
        self.assertEqual(self.thread._req_done_list(meta, "127.0.0.1", 7777), [("reqA", True, 2, True)])

    def test_transfer_multi_peer_splits_sessions(self):
        """同一请求的不同 peer: 拆成多个 session, 各自用自己的 dst 与 rpc 端口."""
        req_meta = self.req_meta_base
        peer_a = ("127.0.0.1", 7777)
        peer_b = ("127.0.0.1", 7778)
        req_meta.peer_transfer = {
            peer_a: {"local_block_ids": [[5]], "remote_block_ids": [[10]], "trans_count": [1]},
            peer_b: {"local_block_ids": [[8]], "remote_block_ids": [[20]], "trans_count": [1]},
        }
        req_meta.peer_layer_metadata = {
            peer_a: {
                "layer0": _make_layer_metadata(
                    kv_caches_base_addr=[4000, 8000], block_len=[64, 64], block_size_scale=[1, 1]
                )
            },
            peer_b: {
                "layer0": _make_layer_metadata(
                    kv_caches_base_addr=[9000, 9000], block_len=[64, 64], block_size_scale=[1, 1]
                )
            },
        }
        req_meta.peer_te_rpc_port = {peer_a: 6000, peer_b: 6001}

        send_task = SendTask(
            send_request={"req1": req_meta},
            wait_event=MagicMock(),
            layer_idx=0,
            layer_name="layer0",
            group_rearrange_block_ids=[[5, 8]],
        )
        self.thread._transfer_kv_cache_batch([send_task])

        self.assertEqual(self.engine.batch_transfer_sync_write.call_count, 2)
        sessions = {call[0][0] for call in self.engine.batch_transfer_sync_write.call_args_list}
        self.assertEqual(sessions, {"127.0.0.1:6000", "127.0.0.1:6001"})
        # block_len 取自本端 layer_metadata(1024/2048), 目的地址用各 peer 自己的 base.
        dst_by_session = {
            call[0][0]: call[0][2][0] for call in self.engine.batch_transfer_sync_write.call_args_list
        }
        self.assertEqual(dst_by_session["127.0.0.1:6000"], 4000 + 10 * 1024)
        self.assertEqual(dst_by_session["127.0.0.1:6001"], 9000 + 20 * 1024)

    def _make_send_thread(self, **overrides):
        """构造发送线程(默认与 setUp 一致), 便于按 pd_head_ratio/量化开关定制."""
        kwargs = dict(
            engine=self.engine,
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
            kv_cache_specs=self.kv_cache_specs,
            attn_resharding_group_idx=set(),
            total_layers=3,
            ready_event=self.ready_event,
            tp_size=1,
            tp_rank=0,
            pd_head_ratio=1,
            num_head_replica=1,
            layer_metadata=self.layer_metadata,
            use_mla=True,
            use_attn_mamba_hybrid=False,
            k_buffer=self.fake_k_buffer,
            v_buffer=self.fake_v_buffer,
            enable_kv_quant=False,
            enable_c8_quant=False,
            resharding_stream=MagicMock(),
            sender_path="127.0.0.1:9999",
        )
        kwargs.update(overrides)
        return KVCacheSendingLayerThread(**kwargs)

    def test_reshard_path_caps_layer_batch(self):
        """pd_head_ratio>1 的传输 src 落在单槽 k_buffer/v_buffer 上, 必须逐层传."""
        reshard_thread = self._make_send_thread(pd_head_ratio=2)
        self.assertTrue(reshard_thread.uses_reshard_buffers)
        self.assertEqual(reshard_thread.max_batch_layers, 1)
        self.assertFalse(self.thread.uses_reshard_buffers)
        self.assertEqual(self.thread.max_batch_layers, _LAYER_BATCH)

    def test_quant_path_caps_layer_batch(self):
        for override in ({"enable_kv_quant": True}, {"enable_c8_quant": True}):
            thread = self._make_send_thread(**override)
            self.assertTrue(thread.uses_reshard_buffers)
            self.assertEqual(thread.max_batch_layers, 1)

    def test_wait_task_ready_polls_until_event_fires(self):
        """就绪等待按任务轮询 event.query(), 不用整条流 synchronize()."""
        thread = self._make_send_thread(pd_head_ratio=2)
        event = MagicMock()
        event.query.side_effect = [False, False, True]
        task = SendTask(layer_idx=0, layer_name="layer0", wait_event=event)
        thread._wait_task_ready(task)
        self.assertEqual(event.query.call_count, 3)

    def test_wait_task_ready_times_out_without_raising(self):
        thread = self._make_send_thread(pd_head_ratio=2)
        event = MagicMock()
        event.query.return_value = False
        task = SendTask(layer_idx=3, layer_name="layer0", wait_event=event)
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.LAYER_DONE_TIMEOUT_S",
            0.001,
        ):
            thread._wait_task_ready(task)

    def test_transfer_batch_rejects_multilayer_reshard(self):
        """兜底不变式: 单槽 buffer 路径收到多层批时显式报错, 而不是静默覆盖."""
        reshard_thread = self._make_send_thread(pd_head_ratio=2)
        tasks = [SendTask(layer_idx=i, layer_name="layer0", group_rearrange_block_ids=[[]]) for i in range(2)]
        with self.assertRaises(RuntimeError):
            reshard_thread._transfer_kv_cache_batch(tasks)

    def test_pipe_writer_disabled_for_reshard_path(self):
        """流水让批间 flush 重叠, 单槽 buffer 会被后一批覆盖 → 两路径互斥."""
        with (
            patch.object(global_te, "_use_tcp", True),
            patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector._PIPE_WRITER", True),
        ):
            self.assertTrue(self._make_send_thread(pd_head_ratio=1).use_pipe_writer)
            self.assertFalse(self._make_send_thread(pd_head_ratio=2).use_pipe_writer)
            self.assertFalse(self._make_send_thread(enable_kv_quant=True).use_pipe_writer)


class TestKVCacheRecvingLayerThread(unittest.TestCase):
    def setUp(self):
        self.meta = MooncakeAgentMetadata(
            te_rpc_port=6000,
            layer_metadata={"layer0": _make_layer_metadata()},
        )
        self.ready_event = threading.Event()

    def test_get_and_clear_done_requests(self):
        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=1,
            local_engine_id="engineA",
            metadata=self.meta,
            ready_event=self.ready_event,
        )

        with th.lock:
            th.done_requests.update({"r1", "r2"})
        got = th.get_and_clear_done_requests()
        self.assertEqual(got, {"r1", "r2"})

        got2 = th.get_and_clear_done_requests()
        self.assertEqual(got2, set())

    def test_get_and_clear_failed_requests(self):
        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=1,
            local_engine_id="engineA",
            metadata=self.meta,
            ready_event=self.ready_event,
        )

        with th.lock:
            th.failed_requests.update({"r1", "r2"})
        got = th.get_and_clear_failed_requests()
        self.assertEqual(got, {"r1", "r2"})

        got2 = th.get_and_clear_failed_requests()
        self.assertEqual(got2, set())

    def test_update_failed_task_aggregates_by_pd_head_ratio(self):
        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=2,
            local_engine_id="engineA",
            metadata=self.meta,
            ready_event=self.ready_event,
        )

        with th.lock:
            th.task_tracker["reqX"] = set()
            th.request_map = MagicMock()

        th.update_failed_task("reqX")
        with th.lock:
            self.assertNotIn("reqX", th.task_tracker)
            self.assertIn("reqX", th.failed_requests)

    def test_update_done_task_aggregates_by_pd_head_ratio(self):
        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=2,
            local_engine_id="engineA",
            metadata=self.meta,
            ready_event=self.ready_event,
        )

        with th.lock:
            th.task_tracker["reqX"] = set()

        th.update_done_task("reqX", 2, "path1")
        with th.lock:
            self.assertIn("reqX", th.task_tracker)
            self.assertNotIn("reqX", th.done_requests)

        th.update_done_task("reqX", 2, "path2")
        with th.lock:
            self.assertNotIn("reqX", th.task_tracker)
            self.assertIn("reqX", th.done_requests)

    def test_apply_layer_done_meta_marks_done_and_failed(self):
        """请求级完成信息随 LAYER_DONE 下发: is_last 计入完成(按路径收齐), failed 就地作废."""
        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=2,
            local_engine_id="engineA",
            metadata=self.meta,
            ready_event=self.ready_event,
        )
        # P/D TP 不等: 同一请求的两条 P 侧路径都报齐才算完成.
        th.apply_layer_done_meta([("reqA", True, 2, False)], "10.0.0.1:1234")
        with th.lock:
            self.assertNotIn("reqA", th.done_requests)
        th.apply_layer_done_meta([("reqA", True, 2, False)], "10.0.0.2:1234")
        with th.lock:
            self.assertIn("reqA", th.done_requests)

        # 中间 chunk(is_last=False)不触发完成; failed 直接作废重试.
        th.apply_layer_done_meta([("reqB", False, 1, False), ("reqC", True, 1, True)], "10.0.0.1:1234")
        with th.lock:
            self.assertNotIn("reqB", th.done_requests)
            self.assertIn("reqC", th.failed_requests)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_ip", return_value="127.0.0.1")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.make_zmq_socket")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.make_zmq_path",
        side_effect=lambda proto, host, port: f"{proto}://{host}:{port}",
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.msgspec.msgpack.Decoder")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.msgspec.msgpack.Encoder")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.zmq_ctx")
    def test_run_loop_handles_meta_done_invalid_unexpected_and_ack(
        self, mock_zmq_ctx, mock_Encoder, mock_Decoder, _mock_make_path, _mock_make_sock, _mock_get_ip, mock_logger
    ):
        enc_inst = MagicMock()
        enc_inst.encode.return_value = b"ENCODED_META"
        mock_Encoder.return_value = enc_inst

        dec_inst = MagicMock()
        dec_inst.decode.side_effect = [
            (GET_META_MSG,),
            (DONE_SENDING_MSG, "reqA", 1, "path1"),
            (b"weird_msg",),
        ]
        mock_Decoder.return_value = dec_inst

        sock = MagicMock()

        sock.recv_multipart.side_effect = [
            [b"ID", b"SOME_PAYLOAD"],
            [b"ID", b"SOME_PAYLOAD2"],
            [b"ONLY_ID"],
            [b"ID", b"SOME_PAYLOAD3"],
            SystemExit,
        ]

        cm = MagicMock()
        cm.__enter__.return_value = sock
        mock_zmq_ctx.return_value = cm

        ready_event = threading.Event()
        th = KVCacheRecvingLayerThread(
            tp_rank=1,
            side_channel_port=6000,
            tp_size=2,
            pd_head_ratio=1,
            local_engine_id="engineZ",
            metadata=self.meta,
            ready_event=ready_event,
        )

        with th.lock:
            th.task_tracker["reqA"] = set()

        with self.assertRaises(SystemExit):
            th.run()

        self.assertTrue(ready_event.is_set())

        self.assertGreaterEqual(sock.send_multipart.call_count, 2)
        calls = [c.args for c in sock.send_multipart.call_args_list]

        meta_call = calls[0]
        self.assertEqual(meta_call[0][0], b"ID")
        self.assertEqual(meta_call[0][1], b"")
        self.assertEqual(meta_call[0][2], b"ENCODED_META")

        ack_call = calls[1]
        self.assertEqual(ack_call[0][0], b"ID")
        self.assertEqual(ack_call[0][1], b"")
        self.assertEqual(ack_call[0][2], b"ACK")

        self.assertTrue(mock_logger.error.called)

        finished = th.get_and_clear_done_requests()
        self.assertIn("reqA", finished)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_ip", return_value="127.0.0.1")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.msgspec.msgpack.Decoder")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.msgspec.msgpack.Encoder")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.zmq_ctx")
    def test_run_loop_pd_head_ratio_gt1_requires_multiple_done(
        self, mock_zmq_ctx, mock_Encoder, mock_Decoder, _mock_get_ip, _mock_logger
    ):
        enc_inst = MagicMock()
        enc_inst.encode.return_value = b"ENC"
        mock_Encoder.return_value = enc_inst

        dec_inst = MagicMock()
        dec_inst.decode.side_effect = [
            (DONE_SENDING_MSG, "reqB", 2, "path1"),
            (DONE_SENDING_MSG, "reqB", 2, "path2"),
        ]
        mock_Decoder.return_value = dec_inst

        sock = MagicMock()
        sock.recv_multipart.side_effect = [
            [b"ID", b"PAY1"],
            [b"ID", b"PAY2"],
            SystemExit,
        ]
        cm = MagicMock()
        cm.__enter__.return_value = sock
        mock_zmq_ctx.return_value = cm

        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=2,
            local_engine_id="engineY",
            metadata=self.meta,
            ready_event=self.ready_event,
        )
        with th.lock:
            th.task_tracker["reqB"] = set()
        with self.assertRaises(SystemExit):
            th.run()
        finished = th.get_and_clear_done_requests()
        self.assertIn("reqB", finished)


class MockVllmConfig:
    def __init__(self):
        self.model_config = MagicMock()
        self.parallel_config = MagicMock()
        self.cache_config = MagicMock()
        self.kv_transfer_config = MagicMock()
        self.speculative_config = None
        self.quant_config = None
        self.model_config.use_mla = True
        self.parallel_config.tensor_parallel_size = 2
        self.parallel_config.data_parallel_rank_local = 0
        self.parallel_config.data_parallel_size_local = 1
        self.parallel_config.data_parallel_size = 1
        self.parallel_config.data_parallel_rank = 0
        self.parallel_config.prefill_context_parallel_size = 1
        self.parallel_config.decode_context_parallel_size = 1
        self.cache_config.block_size = 16
        self.cache_config.mamba_cache_mode = None
        self.model_config.hf_config.num_key_value_heads = 1
        self.model_config.get_num_layers = MagicMock(return_value=1)
        self.model_config.get_total_num_kv_heads = MagicMock(return_value=1)
        self.model_config.hf_text_config = MagicMock()
        self.model_config.hf_text_config.model_type = "default"

        self.kv_transfer_config.engine_id = "test_engine"
        self.kv_transfer_config.kv_port = 5000
        self.kv_transfer_config.is_kv_producer = True
        self.kv_transfer_config.is_kv_consumer = False
        self.kv_transfer_config.get_from_extra_config = MagicMock()
        self.kv_transfer_config.get_from_extra_config.side_effect = lambda k, d: {
            "prefill": {"tp_size": 2, "dp_size": 1},
            "decode": {"tp_size": 2, "dp_size": 1},
        }.get(k, d)


class MockKVCacheConfig:
    def __init__(self, block_size=16):
        kv_cache_spec = MagicMock()
        kv_cache_spec.block_size = block_size
        group_spec = MagicMock()
        group_spec.kv_cache_spec = kv_cache_spec
        group_spec.layer_names = ["encoder.layer.0"]
        self.kv_cache_groups = [group_spec]
        self.kv_cache_tensors = []
        self.num_blocks = 10


class MockRequest:
    def __init__(self, request_id, prompt_token_ids=None, kv_transfer_params=None, status=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids or [1, 2, 3, 4]
        self.prompt_embeds = None
        self.kv_transfer_params = kv_transfer_params or {}
        self.status = status or "running"
        self.output_token_ids = [101, 102]
        self.num_computed_tokens = 0
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self.max_tokens = 16

        self.all_token_ids = list(self.prompt_token_ids)
        self._all_token_ids = list(self.prompt_token_ids)


class TestMooncakeLayerwiseConnectorMetadata(unittest.TestCase):
    def test_add_new_req(self):
        meta = MooncakeLayerwiseConnectorMetadata()
        self.assertEqual(len(meta.requests), 0)

        meta.add_new_req(
            request_id="req1",
            local_block_ids=[[1, 2, 3]],
            kv_transfer_params={
                "remote_block_ids": [[4, 5, 6]],
                "remote_block_size": [[16]],
                "remote_engine_id": "remote_engine",
                "remote_host": "localhost",
                "remote_port": 5000,
            },
        )

        self.assertEqual(len(meta.requests), 1)
        req_meta = meta.requests["req1"]
        self.assertIsInstance(req_meta, ReqMeta)
        self.assertEqual(req_meta.local_block_ids, [[1, 2, 3]])
        self.assertEqual(req_meta.remote_block_ids, [[4, 5, 6]])
        self.assertEqual(req_meta.remote_engine_id, "remote_engine")
        self.assertEqual(req_meta.remote_host, "localhost")
        self.assertEqual(req_meta.remote_port, 5000)


class TestMooncakeLayerwiseConnectorSchedulerMatchedTokens(unittest.TestCase):
    def setUp(self):
        config = MockVllmConfig()
        kv_cache_config = MockKVCacheConfig()
        self.scheduler = MooncakeLayerwiseConnectorScheduler(config, kv_cache_config, "test_engine")

    def test_get_num_new_matched_tokens(self):
        request = MockRequest("req1")
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)

        request.kv_transfer_params = {"do_remote_prefill": True}
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(tokens, 4)
        self.assertTrue(async_flag)

    def test_get_num_new_matched_tokens_hybrid_excludes_last_token(self):
        self.scheduler.need_truncate = True
        request = MockRequest("req1", prompt_token_ids=list(range(17)), kv_transfer_params={"do_remote_prefill": True})

        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)

        self.assertEqual(tokens, 16)
        self.assertTrue(async_flag)

    def test_get_num_new_matched_tokens_hybrid_truncates_prefill_request(self):
        self.scheduler.need_truncate = True
        request = MockRequest("req1", prompt_token_ids=list(range(4)), kv_transfer_params={"do_remote_decode": True})

        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)

        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)
        self.assertEqual(request.prompt_token_ids, [0, 1, 2])
        self.assertEqual(request._all_token_ids, [0, 1, 2])
        self.assertEqual(request.num_prompt_tokens, 3)
        self.assertEqual(request.max_tokens, 1)
        self.assertTrue(request.kv_transfer_params["_p_side_truncated"])

    def test_build_connector_meta(self):
        self.scheduler.vllm_config.kv_transfer_config.is_kv_consumer = True
        request = MockRequest("req1")

        self.scheduler._reqs_need_recv["req1"] = (request, [], [[4, 5, 6]])
        request.kv_transfer_params = {
            "remote_block_ids": [[1, 2, 3]],
            "remote_block_size": [[16]],
            "remote_engine_id": "remote",
            "remote_host": "localhost",
            "remote_port": 5000,
        }

        meta = self.scheduler.build_connector_meta(MagicMock())
        self.assertIsInstance(meta, MooncakeLayerwiseConnectorMetadata)
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests["req1"].local_block_ids, [[4, 5, 6]])
        self.assertEqual(meta.requests["req1"].remote_block_ids, [[1, 2, 3]])
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)

    def test_update_state_after_alloc_hybrid_trims_remote_block_with_only_last_token(self):
        self.scheduler.need_truncate = True
        request = MockRequest(
            "req1",
            prompt_token_ids=list(range(17)),
            kv_transfer_params={"do_remote_prefill": True, "metaserver": "http://meta"},
        )
        blocks = _MockBlocks(unhashed=[], block_ids_tuple=([4, 5],))
        self.scheduler.executor.submit = MagicMock()

        self.scheduler.update_state_after_alloc(request, blocks, num_external_tokens=16)

        _, kwargs = self.scheduler.executor.submit.call_args
        self.assertEqual(kwargs["message"]["remote_block_ids"], ([4],))


class _MockBlocks:
    def __init__(self, unhashed, block_ids_tuple=None):
        self._unhashed = list(unhashed)
        self._block_ids_tuple = block_ids_tuple if block_ids_tuple is not None else ([1, 2],)

    def get_unhashed_block_ids(self):
        return list(self._unhashed)

    def get_block_ids(self):
        return self._block_ids_tuple


class _MockSchedulerOutput:
    def __init__(
        self,
        cached_req_ids=None,
        cached_new_block_ids=None,
        cached_num_computed=None,
        new_reqs=None,
        num_sched=None,
        scheduled_spec_decode_tokens=None,
    ):
        self.scheduled_cached_reqs = SimpleNamespace(
            req_ids=cached_req_ids or [],
            new_block_ids=cached_new_block_ids or [],
            num_computed_tokens=cached_num_computed or [],
        )
        self.scheduled_spec_decode_tokens = scheduled_spec_decode_tokens or {}
        self.scheduled_new_reqs = new_reqs or []
        self.num_scheduled_tokens = num_sched or {}


class TestMooncakeLayerwiseConnectorScheduler_More(unittest.TestCase):
    def setUp(self):
        self.config = MockVllmConfig()
        self.kv_cache_config = MockKVCacheConfig()
        self.scheduler = MooncakeLayerwiseConnectorScheduler(self.config, self.kv_cache_config, "test_engine")

    def test_get_num_new_matched_tokens_with_prefill_block_aligned(self):
        req = MockRequest(
            "req_prefill", prompt_token_ids=list(range(32)), kv_transfer_params={"do_remote_prefill": True}
        )
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(req, num_computed_tokens=16)
        self.assertEqual(tokens, 16)
        self.assertTrue(async_flag)

    def test_update_state_after_alloc_prefill_records_and_resets_flag(self):
        req = MockRequest("req_u1", prompt_token_ids=list(range(24)), kv_transfer_params={"do_remote_prefill": True})
        req.num_computed_tokens = 0
        blocks = _MockBlocks(unhashed=[4, 5, 6], block_ids_tuple=([[4, 5, 6]],))

        self.scheduler.update_state_after_alloc(req, blocks, num_external_tokens=8)
        self.assertIn("req_u1", self.scheduler._reqs_need_recv)
        record = self.scheduler._reqs_need_recv["req_u1"]
        self.assertIs(record[0], req)
        self.assertEqual(record[1], [])
        self.assertEqual(record[2], ([[4, 5, 6]],))
        self.assertFalse(req.kv_transfer_params.get("do_remote_prefill", True))

    def test_update_state_after_alloc_decode_records_send_layerwise(self):
        req = MockRequest(
            "req_u2",
            prompt_token_ids=list(range(10)),
            kv_transfer_params={"do_remote_decode": True, "remote_block_ids": [], "remote_cached_tokens": 0},
        )
        blocks = _MockBlocks(unhashed=[], block_ids_tuple=([[7, 8, 9]],))
        self.scheduler.update_state_after_alloc(req, blocks, num_external_tokens=0)
        self.assertIn("req_u2", self.scheduler._reqs_need_send_layerwise)
        info = self.scheduler._reqs_need_send_layerwise["req_u2"]
        self.assertEqual(info.local_block_ids, [[[7, 8, 9]]])
        self.assertIs(info.request, req)

    def test_build_connector_meta_consumes_reqs_need_recv_and_clears(self):
        self.scheduler.vllm_config.kv_transfer_config.is_kv_consumer = True
        req = MockRequest(
            "req_b1",
            kv_transfer_params={
                "remote_block_ids": [[1, 2]],
                "remote_block_size": [[16]],
                "remote_engine_id": "E",
                "remote_host": "H",
                "remote_port": 5555,
                "remote_te_rpc_port": 6000,
                "remote_layer_metadata": {"layer0": _make_layer_metadata()},
            },
        )
        self.scheduler._reqs_need_recv["req_b1"] = (req, [], [[100, 101]])
        meta = self.scheduler.build_connector_meta(_MockSchedulerOutput())
        self.assertIsInstance(meta, MooncakeLayerwiseConnectorMetadata)
        self.assertIn("req_b1", meta.requests)
        self.assertEqual(meta.requests["req_b1"].local_block_ids, [[100, 101]])
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)

    def test_build_connector_meta_accumulates_cached_blocks(self):
        req_meta = MagicMock(spec=SendReqInfo)
        req_meta.local_block_ids = [[1, 2, 3]]
        req_meta.local_transferred_tokens = 50
        req_meta.local_computed_tokens = 75
        req_meta.request = MagicMock()
        req_meta.extend_local_block_ids = MagicMock()
        req_meta.update_computed_tokens = MagicMock()
        req_meta.update_transferred_tokens = MagicMock()
        req_meta.unpack = MagicMock(
            return_value=(
                req_meta.local_block_ids,
                req_meta.local_transferred_tokens,
                req_meta.local_computed_tokens,
                req_meta.request,
            )
        )

        self.scheduler._reqs_need_send_layerwise["req_b2"] = req_meta

        out = _MockSchedulerOutput(
            cached_req_ids=["req_b2"],
            cached_new_block_ids=[([[3, 4]],)],
            cached_num_computed=[4],
            new_reqs=[],
            num_sched={},
        )
        meta = self.scheduler.build_connector_meta(out)
        self.assertEqual(len(meta.requests), 0)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.group_concurrent_contiguous")
    def test_build_connector_meta_emits_when_tokens_reach_total(self, mock_group_concurrent_contiguous):
        send_req_info = MagicMock(spec=SendReqInfo)
        send_req_info.local_block_ids = [[1, 2, 3]]
        send_req_info.local_transferred_tokens = 50
        send_req_info.local_computed_tokens = 75
        send_req_info.request = MagicMock()
        send_req_info.request.kv_transfer_params = {
            "remote_block_ids": [[4, 5]],
            "remote_block_size": [[16]],
            "remote_cached_tokens": 100,
        }
        send_req_info.request.all_token_ids = list(range(80))
        send_req_info.extend_local_block_ids = MagicMock()
        send_req_info.update_computed_tokens = MagicMock()
        send_req_info.update_transferred_tokens = MagicMock()
        send_req_info.unpack = MagicMock(
            return_value=(
                send_req_info.local_block_ids,
                send_req_info.local_transferred_tokens,
                send_req_info.local_computed_tokens,
                send_req_info.request,
            )
        )

        self.scheduler._reqs_need_send_layerwise["req_b3"] = send_req_info
        out = _MockSchedulerOutput(
            cached_req_ids=["req_b3"],
            cached_new_block_ids=[([[50]],)],
            cached_num_computed=[8],
            new_reqs=[MagicMock(req_id="other", num_computed_tokens=0)],
            num_sched={"req_b3": 4},
        )
        meta = self.scheduler.build_connector_meta(out)
        send_req_info.extend_local_block_ids.assert_called_once_with(([[50]],))
        self.assertIn("req_b3", meta.requests)

    def test_request_finished_returns_false_none(self):
        ok, params = self.scheduler.request_finished(MockRequest("req_fin"), [1, 2])
        self.assertFalse(ok)
        self.assertIsNone(params)


class TestHelperFunctions(unittest.TestCase):
    def test_group_concurrent_contiguous(self):
        src: list[int] = [1, 2, 3, 5, 6]
        dst: list[int] = [10, 11, 12, 14, 15]
        src_groups, dst_groups = group_concurrent_contiguous(src, dst)
        self.assertEqual(len(src_groups), 2)
        self.assertEqual(src_groups[0], [1, 2, 3])
        self.assertEqual(src_groups[1], [5, 6])
        self.assertEqual(dst_groups[0], [10, 11, 12])
        self.assertEqual(dst_groups[1], [14, 15])

    def test_group_concurrent_contiguous_empty(self):
        src: list[int] = []
        dst: list[int] = []
        src_groups, dst_groups = group_concurrent_contiguous(src, dst)
        self.assertEqual(src_groups, [])
        self.assertEqual(dst_groups, [])

    def test_string_to_int64_hash(self):
        hash1 = string_to_int64_hash("test_string")
        hash2 = string_to_int64_hash("test_string")
        self.assertEqual(hash1, hash2)

        hash3 = string_to_int64_hash("different_string")
        self.assertNotEqual(hash1, hash3)

    def test_zmq_ctx_invalid_type(self):
        with self.assertRaises(ValueError), zmq_ctx("INVALID", "tcp://127.0.0.1:5555"):
            pass

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.make_zmq_socket")
    def test_zmq_ctx_ok(self, mock_make_socket):
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        with zmq_ctx(zmq.REQ, "tcp://localhost:1234") as s:  # type: ignore
            self.assertEqual(s, mock_socket)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_send_success(self, _):
        mock_socket = MagicMock()
        path = "127.0.0.1:12345"
        ensure_zmq_send(mock_socket, b"hello", path)
        mock_socket.send.assert_called_once_with(b"hello")

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_send_retry_and_fail(self, _):
        mock_socket = MagicMock()
        path = "127.0.0.1:12345"
        mock_socket.send.side_effect = zmq.ZMQError(  # type: ignore
            "send failed"
        )
        with self.assertRaises(RuntimeError):
            ensure_zmq_send(mock_socket, b"hello", path, max_retries=2)
        self.assertEqual(mock_socket.send.call_count, 2)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_recv_success(self, _):
        mock_socket = MagicMock()
        mock_socket.recv.return_value = b"response"
        mock_poller = MagicMock()
        mock_poller.poll.return_value = [
            (mock_socket, zmq.POLLIN)  # type: ignore
        ]
        path = "127.0.0.1:12345"
        data = ensure_zmq_recv(mock_socket, mock_poller, path)
        self.assertEqual(data, b"response")

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_recv_timeout_and_fail(self, _):
        mock_socket = MagicMock()
        mock_poller = MagicMock()
        mock_poller.poll.return_value = []
        path = "127.0.0.1:12345"
        with self.assertRaises(RuntimeError):
            ensure_zmq_recv(mock_socket, mock_poller, path, timeout=0.01, max_retries=2)


class TestMooncakeLayerwiseConnectorForScheduler(unittest.TestCase):
    def _make_config(self):
        config = MockVllmConfig()
        kv_cache_config = MockKVCacheConfig()
        return config, kv_cache_config

    def test_scheduler_role(self):
        config, kv_cache_config = self._make_config()
        connector = MooncakeLayerwiseConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "get_num_new_matched_tokens")
    def test_scheduler_methods(self, mock_method):
        config, kv_cache_config = self._make_config()
        connector = MooncakeLayerwiseConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
        request = MockRequest("req1")
        connector.get_num_new_matched_tokens(request, 0)
        mock_method.assert_called_once_with(request, 0)


class MockKVCacheBlocks:
    def get_unhashed_block_ids(self):
        return [4, 5, 6]


class MockSchedulerOutput:
    pass


class MockForwardContext:
    pass


class TestMooncakeLayerwiseConnector(unittest.TestCase):
    def setUp(self):
        self.config = MockVllmConfig()
        self.kv_cache_config = MockKVCacheConfig()
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

    def test_scheduler_initialization(self):
        connector = MooncakeLayerwiseConnector(self.config, KVConnectorRole.SCHEDULER, self.kv_cache_config)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "get_num_new_matched_tokens")
    def test_get_num_new_matched_tokens(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config, KVConnectorRole.SCHEDULER, self.kv_cache_config)
        request = MockRequest("req1")
        connector.get_num_new_matched_tokens(request, 0)
        mock_method.assert_called_once_with(request, 0)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "update_state_after_alloc")
    def test_update_state_after_alloc(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config, KVConnectorRole.SCHEDULER, self.kv_cache_config)
        request = MockRequest("req1")
        blocks = MockKVCacheBlocks()
        connector.update_state_after_alloc(request, blocks, 3)
        mock_method.assert_called_once_with(request, blocks, 3)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "build_connector_meta")
    def test_build_connector_meta(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config, KVConnectorRole.SCHEDULER, self.kv_cache_config)
        scheduler_output = MockSchedulerOutput()
        connector.build_connector_meta(scheduler_output)
        mock_method.assert_called_once_with(scheduler_output)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "request_finished")
    def test_request_finished(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config, KVConnectorRole.SCHEDULER, self.kv_cache_config)
        request = MockRequest("req1")
        connector.request_finished(request, [1, 2, 3])
        mock_method.assert_called_once_with(request, [1, 2, 3])


class TestMooncakeLayerwiseConnectorWorker(unittest.TestCase):
    def setUp(self):
        self.mock_transfer_engine = MagicMock()
        self.mock_transfer_engine.get_rpc_port.return_value = 9090
        self.mock_transfer_engine.initialize.return_value = 0
        self.mock_transfer_engine.register_memory.return_value = 0

        self.patches = [
            patch("torch.Tensor.size", return_value=(10, 16, 8, 16)),
            patch("torch.Tensor.element_size", return_value=4),
            patch("torch.Tensor.data_ptr", return_value=0x1000),
            patch("math.prod", return_value=128),
            patch("random.Random"),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_tp_group",
                return_value=None,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_ip",
                return_value="127.0.0.1",
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.string_to_int64_hash",
                side_effect=lambda s: hash(s),
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.global_te.get_transfer_engine",
                return_value=self.mock_transfer_engine,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.global_te.register_buffer",
                return_value=None,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.KVCacheSendingLayerThread",
                MagicMock(),
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.KVCacheRecvingLayerThread",
                MagicMock(),
            ),
            patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger", MagicMock()),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.threading.Event", MagicMock()
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_ascend_config",
                return_value=SimpleNamespace(pd_tp_ratio=1, num_head_replica=1, pd_head_ratio=1),
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_pcp_group",
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
        ]

        for p in self.patches:
            p.start()  # type: ignore

        self.vllm_config = MockVllmConfig()
        self.engine_id = "test_engine"
        mock_k = MagicMock()
        mock_k.shape = (10, 16, 8, 16)
        mock_k.data_ptr.return_value = 0x1000
        mock_k.element_size.return_value = 4
        mock_v = MagicMock()
        mock_v.shape = (10, 16, 8, 16)
        mock_v.data_ptr.return_value = 0x2000
        mock_v.element_size.return_value = 4
        self.kv_caches = {"encoder.layer.0": (mock_k, mock_v)}
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.decode_context_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.kv_transfer_config.kv_port = 1234

        self.kv_cache_config = MockKVCacheConfig()

    def tearDown(self):
        for p in self.patches:
            p.stop()  # type: ignore

    def test_register_kv_caches_producer(self):
        self.vllm_config.kv_transfer_config.is_kv_producer = True
        self.vllm_config.kv_transfer_config.is_kv_consumer = False
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config, self.kv_cache_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertEqual(len(worker.layer_metadata), 1)
        self.assertIsNotNone(worker.kv_send_layer_thread)
        self.assertIsNone(worker.kv_recv_layer_thread)

    @staticmethod
    def _install_reshard_buffers(worker):
        """让 worker 走 reshard/量化路径并装上假的 k/v buffer(跳过真实 NPU 分配)."""
        worker.pd_head_ratio = 2
        worker.enable_c8_quant = True

        def _fake_create_kv_buffer(_first_kv_cache_tuple):
            worker.k_buffer = torch.zeros(8192, dtype=torch.uint8)
            worker.v_buffer = torch.zeros(8192, dtype=torch.uint8)

        worker.create_kv_buffer = _fake_create_kv_buffer

    def test_register_kv_caches_tcp_producer_stages_reshard_buffers(self):
        """TCP + reshard 路径: k/v buffer 是数据面 src, 必须作为 extra 进 staging."""
        self.vllm_config.kv_transfer_config.is_kv_producer = True
        self.vllm_config.kv_transfer_config.is_kv_consumer = False
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config, self.kv_cache_config, self.engine_id)
        self._install_reshard_buffers(worker)
        with (
            patch.object(global_te, "_use_tcp", True),
            patch.object(global_te, "register_tcp_staging") as mock_staging,
        ):
            worker.register_kv_caches(self.kv_caches)
        self.assertEqual(mock_staging.call_args[1]["extra_tensors"], [worker.k_buffer, worker.v_buffer])

    def test_register_kv_caches_consumer_skips_reshard_buffers(self):
        """consumer 只收不发: 既不分配 reshard buffer, 也不为其付 CPU 镜像开销."""
        self.vllm_config.kv_transfer_config.is_kv_producer = False
        self.vllm_config.kv_transfer_config.is_kv_consumer = True
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config, self.kv_cache_config, self.engine_id)
        worker.pd_head_ratio = 2
        worker.enable_c8_quant = True  # 让 use_kv_buffer 为真, 验证仍被 producer 判定挡住
        worker.create_kv_buffer = MagicMock()
        with (
            patch.object(global_te, "_use_tcp", True),
            patch.object(global_te, "register_tcp_staging") as mock_staging,
            patch.object(global_te, "npu_addr_to_cpu_addr", return_value=0xABCD0000),
        ):
            worker.register_kv_caches(self.kv_caches)
        worker.create_kv_buffer.assert_not_called()
        self.assertIsNone(worker.k_buffer)
        self.assertEqual(mock_staging.call_args[1]["extra_tensors"], [])

    def test_register_kv_caches_consumer(self):
        self.vllm_config.kv_transfer_config.is_kv_producer = False
        self.vllm_config.kv_transfer_config.is_kv_consumer = True
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config, self.kv_cache_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertEqual(len(worker.layer_metadata), 1)
        self.assertIsNone(worker.kv_send_layer_thread)
        self.assertIsNotNone(worker.kv_recv_layer_thread)

    def test_register_kv_caches_mla_case(self):
        mla_cache1 = MagicMock()
        mla_cache1.size.return_value = (10, 16, 1, 16)
        mla_cache1.shape = (10, 16, 1, 16)
        mla_cache1.data_ptr.return_value = 0x1000
        mla_cache1.element_size.return_value = 4
        mla_cache2 = MagicMock()
        mla_cache2.size.return_value = (10, 16, 1, 8)
        mla_cache2.shape = (10, 16, 1, 8)
        mla_cache2.data_ptr.return_value = 0x2000
        mla_cache2.element_size.return_value = 4
        mla_caches = {"encoder.layer.0": (mla_cache1, mla_cache2)}
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config, self.kv_cache_config, self.engine_id)
        worker.register_kv_caches(mla_caches)
        self.assertTrue(worker.use_mla)
        self.assertEqual(len(worker.layer_metadata["encoder.layer.0"].block_len), 2)


class TestGlobalTEStagingExtras(unittest.TestCase):
    """TCP staging 必须覆盖 k/v reshard buffer(pd_head_ratio>1 / 量化的数据面 src)."""

    def _make_te(self):
        from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE

        te = GlobalTE()
        te.transfer_engine = MagicMock()
        te.transfer_engine.register_memory.return_value = 0
        return te

    def test_extra_tensors_join_per_tensor_staging(self):
        te = self._make_te()
        kv_caches = {"layer0": torch.zeros(4096, dtype=torch.uint8)}
        k_buffer = torch.zeros(8192, dtype=torch.uint8)
        v_buffer = torch.zeros(8192, dtype=torch.uint8)

        te.register_tcp_staging(kv_caches, extra_tensors=[k_buffer, v_buffer])

        # staging 布局: kv(4096) + k_buffer(8192) + v_buffer(8192)
        cpu_base = te._cpu_tensors[0].data_ptr()
        self.assertEqual(te.npu_addr_to_cpu_addr(k_buffer.data_ptr() + 4096), cpu_base + 4096 + 4096)
        self.assertEqual(te.npu_addr_to_cpu_addr(v_buffer.data_ptr()), cpu_base + 4096 + 8192)

        te.submit_dma_copy = MagicMock()
        te.sync_npu_to_cpu_for_npu_addrs([k_buffer.data_ptr(), v_buffer.data_ptr()], [8192, 8192])
        items = te.submit_dma_copy.call_args[0][0]
        self.assertEqual([size for _, _, _, size in items], [8192, 8192])

    def test_extra_regions_join_region_staging(self):
        """hybrid 模型走 region 模式: 追加 region 后任意字节地址仍线性映射."""
        te = self._make_te()
        te.register_tcp_staging_regions([(0x100000, 4096), (0x200000, 8192)])

        cpu_base = te._cpu_tensors[0].data_ptr()
        self.assertEqual(te.npu_addr_to_cpu_addr(0x200000 + 512), cpu_base + 4096 + 512)

        te.submit_dma_copy_ptrs = MagicMock()
        te.sync_npu_to_cpu_for_npu_addrs([0x200000], [8192])
        self.assertEqual(te.submit_dma_copy_ptrs.call_args[0][2], [8192])
