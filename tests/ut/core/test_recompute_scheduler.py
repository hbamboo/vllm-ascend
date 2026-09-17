# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend import envs
from vllm_ascend.core.recompute_scheduler import RecomputeScheduler


def test_pd_consumer_first_step_injects_placeholder_spec_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.is_kv_producer = False
    scheduler.is_hybrid_model = False
    scheduler.is_mtp_kv_consumer = True
    scheduler.num_spec_tokens = 1
    scheduler.max_model_len = 1024
    scheduler.log_stats = False
    scheduler.connector = None

    enqueued_requests = []

    def enqueue_waiting_request(self, request):
        enqueued_requests.append(request)

    scheduler._enqueue_waiting_request = MethodType(enqueue_waiting_request, scheduler)

    request = Request(
        request_id="pd-consumer-first-step",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )

    scheduler.add_request(request)

    assert enqueued_requests == [request]
    assert scheduler.requests[request.request_id] is request
    assert request.spec_token_ids == [PLACEHOLDER_TOKEN_ID]
    assert request.num_tokens_with_spec == request.num_tokens + 1


def test_update_from_output_settles_finished_request_in_flight_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    request = SimpleNamespace(
        num_in_flight_tokens=1,
        is_finished=lambda: True,
    )
    scheduler.requests = {"request": request}
    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.enable_return_routed_experts = False
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.finished_req_ids_dict = {}
    scheduler.make_stats = MagicMock(return_value=None)

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request": 1},
        recomputed_reqs=None,
    )
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        routed_experts=None,
    )

    assert scheduler.update_from_output(scheduler_output, model_runner_output) == {}
    assert request.num_in_flight_tokens == 0


def _make_remote_kv_scheduler(request_id: str, pending_first_tokens: dict) -> RecomputeScheduler:
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = MagicMock()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = {request_id}
    scheduler.kv_cache_manager = MagicMock()
    scheduler.is_mtp_kv_consumer = False
    scheduler._pending_first_tokens = pending_first_tokens
    return scheduler


def _make_ptd_request(reuse_prefilled_tokens: bool | None = None) -> Request:
    request = Request(
        request_id="ptd-req",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    if reuse_prefilled_tokens is not None:
        request.kv_transfer_params = {"reuse_prefilled_tokens": reuse_prefilled_tokens}
    return request


def test_p_then_d_first_token_is_appended_to_prompt(monkeypatch):
    """先 P 后 D: D 侧把 P 传来的首 token 补成最后一个 prompt token。"""
    monkeypatch.setattr(envs, "REUSE_PREFILLED_TOKENS", True)
    request = _make_ptd_request()
    scheduler = _make_remote_kv_scheduler(request.request_id, {request.request_id: 99})

    scheduler._update_waiting_for_remote_kv(request)

    assert request.prompt_token_ids == [1, 2, 3, 4, 99]
    # 该 token 只重算一次 KV: num_computed_tokens 直接把它算作已计算
    assert request.num_computed_tokens == 1
    # 并且计入 output, D 后续生成的 token 从第 2 个开始
    assert list(request.output_token_ids) == [99]
    assert request.request_id not in scheduler._pending_first_tokens


def test_p_then_d_first_token_skipped_when_reuse_disabled(monkeypatch):
    monkeypatch.setattr(envs, "REUSE_PREFILLED_TOKENS", True)
    request = _make_ptd_request(reuse_prefilled_tokens=False)
    scheduler = _make_remote_kv_scheduler(request.request_id, {request.request_id: 99})

    scheduler._update_waiting_for_remote_kv(request)

    assert request.prompt_token_ids == [1, 2, 3, 4]
    assert list(request.output_token_ids) == []
    # 没消费掉, 留给同一请求的后续 chunk
    assert scheduler._pending_first_tokens == {request.request_id: 99}


def test_p_then_d_first_token_skipped_when_env_disabled(monkeypatch):
    monkeypatch.setattr(envs, "REUSE_PREFILLED_TOKENS", False)
    request = _make_ptd_request()
    scheduler = _make_remote_kv_scheduler(request.request_id, {request.request_id: 99})

    scheduler._update_waiting_for_remote_kv(request)

    assert request.prompt_token_ids == [1, 2, 3, 4]
    assert list(request.output_token_ids) == []


def test_p_then_d_first_token_ignored_for_other_request(monkeypatch):
    monkeypatch.setattr(envs, "REUSE_PREFILLED_TOKENS", True)
    request = _make_ptd_request()
    scheduler = _make_remote_kv_scheduler(request.request_id, {})

    scheduler._update_waiting_for_remote_kv(request)

    assert request.prompt_token_ids == [1, 2, 3, 4]


def test_invalid_blocks_with_multi_group_block_ids():
    """多 group KV cache (如 MLA) 下 get_block_ids 返回多张表, 不能直接解包。"""
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.recompute_kv_load_failures = False
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.get_block_ids.return_value = ([1, 2], [3, 4])

    request = SimpleNamespace(request_id="r1")
    affected, recomputed_tokens, evict = scheduler._update_requests_with_invalid_blocks([request], {3}, {})

    assert affected == {"r1"}
    assert recomputed_tokens == 0
    assert evict == set()


def test_invalid_blocks_ignores_unaffected_requests():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.recompute_kv_load_failures = False
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.get_block_ids.return_value = ([1, 2], [3, 4])

    request = SimpleNamespace(request_id="r1")
    affected, _, _ = scheduler._update_requests_with_invalid_blocks([request], {7}, {})

    assert affected == set()
