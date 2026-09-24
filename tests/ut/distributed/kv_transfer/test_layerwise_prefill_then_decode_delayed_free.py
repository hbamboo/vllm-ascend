# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the delayed-free handshake of the p_then_d layerwise connector.

P 侧逐层 push 下, 请求结束时其末 chunk 的层任务可能还在发送队列里 —— 块一旦回到
free queue 就可能被别的请求覆写, D 会读到脏 KV。为此 scheduler 对"末 chunk 已进
发送流水"的请求延迟释放, 等本 rank 的发送线程在该请求最后一批数据上拿到 D 的
LAYER_DONE ACK 后经 finished_sending 上报。

这里覆盖两端的簿记逻辑(scheduler 侧判定 / worker 侧登记与上报 / 发送线程的完成
判据), 不建立真实传输。
"""

import threading
import time
import types
import unittest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_prefill_then_decode_connector import (
    KVCacheSendingLayerThread,
    MooncakeLayerwisePrefillThenDecodeConnectorScheduler,
    MooncakeLayerwisePrefillThenDecodeConnectorWorker,
    SizedDict,
)


def _make_worker() -> MooncakeLayerwisePrefillThenDecodeConnectorWorker:
    """只装配延迟释放相关的状态, 绕开需要 NPU / 传输引擎的 __init__。"""
    worker = MooncakeLayerwisePrefillThenDecodeConnectorWorker.__new__(
        MooncakeLayerwisePrefillThenDecodeConnectorWorker
    )
    worker._send_obligations = {}
    worker._send_completed = SizedDict()
    worker._send_done_lock = threading.Lock()
    worker._send_has_data = SizedDict()
    return worker


def _make_scheduler() -> MooncakeLayerwisePrefillThenDecodeConnectorScheduler:
    scheduler = MooncakeLayerwisePrefillThenDecodeConnectorScheduler.__new__(
        MooncakeLayerwisePrefillThenDecodeConnectorScheduler
    )
    scheduler._reqs_send_inflight = set()
    scheduler._reqs_need_send = {}
    return scheduler


def _make_thread(on_send_done) -> KVCacheSendingLayerThread:
    """只装配完成判据需要的属性: group 末层线 + 两个 group(attention / mamba)。"""
    thread = KVCacheSendingLayerThread.__new__(KVCacheSendingLayerThread)
    thread.total_layers = 32
    thread.kv_cache_specs = [object(), object()]
    # group 0(attention) 末层 31, group 1(mamba) 末层 15
    thread.group_max_layer_idx = {0: 31, 1: 15}
    thread.on_send_done = on_send_done
    return thread


def _req_meta(peer_blocks: dict | None = None):
    """ReqMeta 的替身: 完成判据只读 peer_transfer 的 (local_block_ids, trans_count)。"""
    return types.SimpleNamespace(peer_transfer=peer_blocks or {})


def _send_task(layer_idx: int):
    return types.SimpleNamespace(layer_idx=layer_idx)


class TestWorkerSendObligations(unittest.TestCase):
    """worker 侧: 登记 -> 完成 -> 上报, 每次只上报一次。"""

    def test_completed_obligations_are_reported_once(self):
        worker = _make_worker()
        worker.register_send_obligations({"req-done": 0.0, "req-pending": 0.0})

        worker.mark_send_done("req-done")
        self.assertEqual(worker.get_and_clear_send_done(), {"req-done"})
        # 已上报的请求不再重复上报(重复上报会让 core 对已归还的请求再 free 一次)。
        self.assertEqual(worker.get_and_clear_send_done(), set())

    def test_early_completion_before_registration_is_reported(self):
        """末批可能在调度侧判定延迟之前就 ACK 回来 —— 完成先到, 登记后要能了结。"""
        worker = _make_worker()
        worker.mark_send_done("req-early")
        self.assertEqual(worker.get_and_clear_send_done(), set())

        worker.register_send_obligations({"req-early": 0.0})
        self.assertEqual(worker.get_and_clear_send_done(), {"req-early"})

    def test_no_data_request_is_completed_at_registration(self):
        """本 rank 没有该请求的数据要发: 没有发送线程的完成信号可等, 直接了结。"""
        worker = _make_worker()
        worker._send_has_data["req-no-data"] = False
        worker.register_send_obligations({"req-no-data": 0.0})
        self.assertEqual(worker.get_and_clear_send_done(), {"req-no-data"})

    def test_registration_keeps_first_timestamp(self):
        """metadata 每步重发: 已登记的不能刷新超时起点。"""
        worker = _make_worker()
        worker.register_send_obligations({"req-1": 0.0})
        registered_at = worker._send_obligations["req-1"]
        worker.register_send_obligations({"req-1": 0.0})
        self.assertEqual(worker._send_obligations["req-1"], registered_at)

    def test_timeout_reports_stuck_request(self):
        """等不到完成信号的请求按超时兜底上报, 否则块一直被占着。"""
        worker = _make_worker()
        worker.register_send_obligations({"req-stuck": 0.0})
        worker._send_obligations["req-stuck"] = time.time() - 10_000.0

        self.assertEqual(worker.get_and_clear_send_done(), {"req-stuck"})
        self.assertEqual(worker.get_and_clear_send_done(), set())

    def test_has_send_data_predicate(self):
        """各 peer 的 block 表里只要有一个非空 group, 本 rank 就有数据要发。"""
        no_data = _req_meta()
        empty_groups = _req_meta({("h", 1): {"local_block_ids": [[], []], "trans_count": [0, 0]}})
        attn_only = _req_meta({("h", 1): {"local_block_ids": [[7], []], "trans_count": [1, 0]}})
        mamba_only = _req_meta({("h", 1): {"local_block_ids": [[], [9]], "trans_count": [0, 1]}})

        self.assertFalse(MooncakeLayerwisePrefillThenDecodeConnectorWorker._req_has_send_data(no_data))
        self.assertFalse(MooncakeLayerwisePrefillThenDecodeConnectorWorker._req_has_send_data(empty_groups))
        self.assertTrue(MooncakeLayerwisePrefillThenDecodeConnectorWorker._req_has_send_data(attn_only))
        self.assertTrue(MooncakeLayerwisePrefillThenDecodeConnectorWorker._req_has_send_data(mamba_only))


class TestSendEndLayer(unittest.TestCase):
    """发送线程侧: 只有"有数据的最高层组"的末层才算本 rank 发完了。"""

    def test_send_end_layer_is_highest_group_with_data(self):
        thread = _make_thread(on_send_done=None)
        attn_data = _req_meta({("h", 1): {"local_block_ids": [[7], []], "trans_count": [1, 0]}})
        mamba_data = _req_meta({("h", 1): {"local_block_ids": [[], [9]], "trans_count": [0, 1]}})
        both = _req_meta({("h", 1): {"local_block_ids": [[7], [9]], "trans_count": [1, 1]}})

        self.assertEqual(thread._req_send_end_layer_idx(attn_data), 31)
        # 只有 mamba 组有数据时, 末层是 mamba 组的 —— 不能等 attention 组的 31。
        self.assertEqual(thread._req_send_end_layer_idx(mamba_data), 15)
        self.assertEqual(thread._req_send_end_layer_idx(both), 31)
        self.assertEqual(thread._req_send_end_layer_idx(_req_meta()), -1)

    def test_notify_only_on_last_data_layer(self):
        """mamba 组末层早于 attention 组, 提前通知会让还没 flush 的层被当成已发完。"""
        notified: list[str] = []
        thread = _make_thread(on_send_done=notified.append)
        req_meta = _req_meta({("h", 1): {"local_block_ids": [[7], [9]], "trans_count": [1, 1]}})

        thread._notify_send_done(_send_task(15), "req-1", req_meta)
        self.assertEqual(notified, [])
        thread._notify_send_done(_send_task(31), "req-1", req_meta)
        self.assertEqual(notified, ["req-1"])

    def test_notify_mamba_only_request_on_mamba_end(self):
        notified: list[str] = []
        thread = _make_thread(on_send_done=notified.append)
        req_meta = _req_meta({("h", 1): {"local_block_ids": [[], [9]], "trans_count": [0, 1]}})

        thread._notify_send_done(_send_task(15), "req-1", req_meta)
        self.assertEqual(notified, ["req-1"])

    def test_notify_without_callback_is_noop(self):
        thread = _make_thread(on_send_done=None)
        req_meta = _req_meta({("h", 1): {"local_block_ids": [[7], []], "trans_count": [1, 0]}})
        thread._notify_send_done(_send_task(31), "req-1", req_meta)  # 不应抛异常


class TestSchedulerDelayFree(unittest.TestCase):
    """scheduler 侧: 末 chunk 调度过的请求延迟释放, worker 上报后了结。"""

    @staticmethod
    def _request(req_id: str):
        return types.SimpleNamespace(request_id=req_id)

    def test_request_without_inflight_send_is_freed_now(self):
        scheduler = _make_scheduler()
        self.assertEqual(scheduler.request_finished(self._request("req-1"), []), (False, None))

    def test_request_with_inflight_send_is_delayed(self):
        scheduler = _make_scheduler()
        scheduler._reqs_send_inflight.add("req-1")

        self.assertEqual(scheduler.request_finished(self._request("req-1"), []), (True, None))
        # 判定后移交 _reqs_need_send 等待上报, 不再算"在飞"。
        self.assertEqual(scheduler._reqs_send_inflight, set())
        self.assertIn("req-1", scheduler._reqs_need_send)

    def test_all_groups_variant_matches(self):
        scheduler = _make_scheduler()
        scheduler._reqs_send_inflight.add("req-1")

        self.assertEqual(scheduler.request_finished_all_groups(self._request("req-1"), ([],)), (True, None))
        self.assertEqual(scheduler.request_finished_all_groups(self._request("req-2"), ([],)), (False, None))

    def test_update_connector_output_releases_reported_request(self):
        scheduler = _make_scheduler()
        scheduler._reqs_send_inflight.update({"req-1", "req-2"})
        scheduler.request_finished(self._request("req-1"), [])
        scheduler.request_finished(self._request("req-2"), [])

        scheduler.update_connector_output(types.SimpleNamespace(finished_sending={"req-1"}, finished_recving=None))
        self.assertEqual(set(scheduler._reqs_need_send), {"req-2"})

    def test_update_connector_output_without_sending_is_noop(self):
        scheduler = _make_scheduler()
        scheduler._reqs_send_inflight.add("req-1")
        scheduler.request_finished(self._request("req-1"), [])

        scheduler.update_connector_output(types.SimpleNamespace(finished_sending=None, finished_recving=None))
        self.assertEqual(set(scheduler._reqs_need_send), {"req-1"})


if __name__ == "__main__":
    unittest.main()
