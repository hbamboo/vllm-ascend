# SPDX-License-Identifier: Apache-2.0
"""P 侧 finished_sending 上报的过期条目过滤单测.

背景(线上打挂过一次 EngineCore): scheduler 的 `requests_to_send` 是**每步重发**的
(防丢单), 而 async scheduling 下"下一步的新批"排在本步 worker 上报被应用之前 ——
一个已经上报过的 req_id 会再被下发一次, worker 重新登记后它的完成信号永远不会来,
只能等 480s 超时强杀; 那次强杀上报的恰是 scheduler 已经释放掉的请求,
`_update_from_kv_xfer_finished` 的 `assert req_id in self.requests` 直接打死引擎。

这里锁住两件事:
- 本 connector 的过期上报在 update_connector_output 被滤掉(不进 scheduler);
- 其它 connector 的条目、以及"还没上报过"的条目原样放行(不能误伤 MultiConnector).
"""

import threading
import time
import unittest
from unittest import mock

from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_prefill_then_decode_connector as conn


class _FakeScheduler:
    """复刻 vLLM scheduler 对上报的处理: 未知 req_id 直接断言(del self.requests)."""

    def __init__(self, requests: dict[str, object]):
        # 持有引用而非拷贝: 用例要断言"请求表被真的动过"
        self.requests = requests

    def apply(self, out: KVConnectorOutput) -> None:
        for req_id in out.finished_sending or ():
            assert req_id in self.requests, req_id
            del self.requests[req_id]


def _make_scheduler_connector(need_send: dict[str, float], promised: list[str]):
    """绕过 __init__, 只装配过滤逻辑用到的两本账."""
    sched = object.__new__(conn.MooncakeLayerwisePrefillThenDecodeConnectorScheduler)
    sched._reqs_need_send = dict(need_send)
    sched._send_promised = conn.SizedDict()
    for req_id in promised:
        sched._send_promised[req_id] = True
    return sched


class TestStaleSendReportFilter(unittest.TestCase):
    def test_stale_report_is_dropped_before_scheduler(self):
        """已了结的请求再被上报(超时强杀/复活登记): 必须被滤掉, 否则引擎被打死."""
        sched = _make_scheduler_connector(need_send={}, promised=["req-freed"])
        out = KVConnectorOutput(finished_sending={"req-freed"})
        sched.update_connector_output(out)

        self.assertEqual(out.finished_sending, set())
        # 真 scheduler 拿到这个 out 不会断言(请求早已释放)
        _FakeScheduler(requests={}).apply(out)

    def test_fresh_report_passes_and_clears_ledger(self):
        """正常上报: 原样放行, 并摘掉"等上报"的账."""
        sched = _make_scheduler_connector(need_send={"req-a": 1.0}, promised=["req-a"])
        out = KVConnectorOutput(finished_sending={"req-a"})
        sched.update_connector_output(out)

        self.assertEqual(out.finished_sending, {"req-a"})
        self.assertEqual(sched._reqs_need_send, {})
        _FakeScheduler(requests={"req-a": object()}).apply(out)

    def test_other_connectors_entries_are_left_alone(self):
        """别的 connector(MultiConnector 组合)的条目与本 connector 无关, 不碰."""
        sched = _make_scheduler_connector(need_send={"req-a": 1.0}, promised=["req-a"])
        out = KVConnectorOutput(finished_sending={"req-a", "req-other"})
        sched.update_connector_output(out)

        self.assertEqual(out.finished_sending, {"req-a", "req-other"})
        self.assertEqual(sched._reqs_need_send, {})

    def test_lost_report_heals_through_watchdog_reregistration(self):
        """上报半路丢了(never applied): 该 id 仍在等上报, 再来的上报要放行.

        这条是"重新登记 => 超时强杀 => 再上报"恢复路径的接收端前提, 与上面的过滤
        并不矛盾: 判据是"还在不在等上报", 不是"有没有上报过".
        """
        # 第一次上报丢了: _reqs_need_send 里还留着
        sched = _make_scheduler_connector(need_send={"req-a": 1.0}, promised=["req-a"])
        # 之后 worker 靠超时强杀补了一次上报
        out = KVConnectorOutput(finished_sending={"req-a"})
        sched.update_connector_output(out)
        self.assertEqual(out.finished_sending, {"req-a"})
        _FakeScheduler(requests={"req-a": object()}).apply(out)

        # 彻底了结之后再冒出来的同 id 上报(异步调度下的错位重发)则被丢掉
        out2 = KVConnectorOutput(finished_sending={"req-a"})
        sched.update_connector_output(out2)
        self.assertEqual(out2.finished_sending, set())


class TestResurrectionCycle(unittest.TestCase):
    """把现场的完整时序跑一遍: 重发 => 复活登记 => 超时强杀 => 过期上报 => 不再打死引擎.

    时序(async scheduling, 见 vllm/v1/engine/core.py step_with_batch_queue:
    先 schedule() 排新批, 之后才 update_from_output 应用本步上报):
      t0 请求发送完成, worker 上报 req (进 finished_sending)
      t1 scheduler 排下一步的新批 —— 上报还没应用, requests_to_send 里仍有 req
      t2 worker 收到重发, 重新登记成一个永远等不到信号的义务
      t3 上报被应用: scheduler 释放请求(此后 self.requests 里没有 req)
      t4 义务超时强杀, 又上报一次 req —— 这一次是"已了结"的, 修复前直接打死引擎
    """

    def _make_worker(self):
        worker = object.__new__(conn.MooncakeLayerwisePrefillThenDecodeConnectorWorker)
        worker._send_obligations = {}
        worker._send_completed = conn.SizedDict()
        worker._send_done_lock = threading.Lock()
        worker._send_has_data = conn.SizedDict()
        return worker

    def test_full_cycle_survives_and_never_reports_twice(self):
        sched = _make_scheduler_connector(need_send={}, promised=[])
        sched._reqs_send_inflight = {"req-a"}
        worker = self._make_worker()
        request = mock.Mock()
        request.request_id = "req-a"
        live = {"req-a": object()}  # scheduler 手里的请求表

        # t0: 请求结束 -> 延迟释放判定(记账 + 墓碑)
        self.assertEqual(sched._request_finished_delay_free(request), (True, None))

        # t0.5: 重发到 worker, worker 登记; 发送线程完成 -> 上报
        worker.register_send_obligations(dict(sched._reqs_need_send))
        worker.mark_send_done("req-a")
        first_report = worker.get_and_clear_send_done()
        self.assertEqual(first_report, {"req-a"})

        # t1: async scheduling 先排下一步的新批 —— 上报尚未应用, 于是 req 又被下发
        resent = dict(sched._reqs_need_send)
        self.assertIn("req-a", resent)
        # t2: worker 重新登记(死等)
        worker.register_send_obligations(resent)
        self.assertIn("req-a", worker._send_obligations)

        # t3: 应用第一次上报(修复后的过滤不动它), scheduler 释放该请求
        out = KVConnectorOutput(finished_sending=set(first_report))
        sched.update_connector_output(out)
        self.assertEqual(out.finished_sending, {"req-a"})
        _FakeScheduler(live).apply(out)
        self.assertEqual(live, {})

        # 复现修复前的行为: 这条上报会撞上 scheduler 的未知 req_id 断言
        with self.assertRaises(AssertionError):
            _FakeScheduler(live).apply(KVConnectorOutput(finished_sending={"req-a"}))

        # t4: 义务超时(把登记时刻拨老) -> 强杀上报 -> 必须被滤掉
        worker._send_obligations["req-a"] = time.time() - 10**6
        late_report = worker.get_and_clear_send_done()
        self.assertEqual(late_report, {"req-a"})
        out2 = KVConnectorOutput(finished_sending=set(late_report))
        sched.update_connector_output(out2)
        self.assertEqual(out2.finished_sending, set())
        _FakeScheduler(live).apply(out2)  # 不再断言 -> 引擎活着


class TestPromiseLedger(unittest.TestCase):
    def test_request_finished_records_promise(self):
        """延迟释放的判定要留墓碑: 过滤只看本 connector 承诺过的 req_id."""
        sched = _make_scheduler_connector(need_send={}, promised=[])
        sched._reqs_send_inflight = {"req-a"}
        request = mock.Mock()
        request.request_id = "req-a"

        delay, params = sched._request_finished_delay_free(request)

        self.assertTrue(delay)
        self.assertIsNone(params)
        self.assertIn("req-a", sched._reqs_need_send)
        self.assertIn("req-a", sched._send_promised)

    def test_no_delay_for_request_never_scheduled(self):
        """末 chunk 没进过发送流水: 按时归还, 不记账."""
        sched = _make_scheduler_connector(need_send={}, promised=[])
        sched._reqs_send_inflight = set()
        request = mock.Mock()
        request.request_id = "req-b"

        delay, params = sched._request_finished_delay_free(request)

        self.assertFalse(delay)
        self.assertIsNone(params)
        self.assertEqual(sched._reqs_need_send, {})
        self.assertNotIn("req-b", sched._send_promised)


if __name__ == "__main__":
    unittest.main()
