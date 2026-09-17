# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""先 P 后 D 的三个 vLLM 侧 patch 的行为测试。"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

import vllm_ascend.patch.platform.patch_kv_connector  # noqa: F401
import vllm_ascend.patch.platform.patch_kv_utils  # noqa: F401
from vllm_ascend import envs
from vllm_ascend.patch.platform import patch_render_serving


class TestKVConnectorOutputPatch:
    def test_first_tokens_is_a_real_dataclass_field(self):
        # worker -> scheduler 的 ModelRunnerOutput 走 msgspec, 只搬运
        # dataclasses.fields() 里的字段, 所以它必须是声明过的字段。
        assert "first_tokens" in [f.name for f in dataclasses.fields(KVConnectorOutput)]

    def test_default_is_none_and_constructible(self):
        assert KVConnectorOutput().first_tokens is None
        assert KVConnectorOutput(first_tokens={"r1": 7}).first_tokens == {"r1": 7}

    def test_other_fields_keep_their_defaults(self):
        output = KVConnectorOutput()
        assert output.invalid_block_ids == set()
        assert output.expected_finished_count == 0
        assert output.finished_sending is None


class TestKVOutputAggregatorPatch:
    def _make_output(self, finished_recving, first_tokens):
        return SimpleNamespace(
            kv_connector_output=KVConnectorOutput(
                finished_recving=finished_recving,
                first_tokens=first_tokens,
            )
        )

    def test_first_tokens_survives_aggregation(self):
        aggregator = KVOutputAggregator(expected_finished_count=1)
        outputs = [
            self._make_output({"r1"}, {"r1": 42}),
            self._make_output(None, None),
        ]

        result = aggregator.aggregate(outputs, output_rank=0)

        assert result is not None
        assert result.kv_connector_output.first_tokens == {"r1": 42}
        assert result.kv_connector_output.finished_recving == {"r1"}

    def test_last_non_empty_first_tokens_wins(self):
        aggregator = KVOutputAggregator(expected_finished_count=1)
        outputs = [
            self._make_output({"r1"}, {"r1": 42}),
            self._make_output(None, {"r2": 7}),
        ]

        result = aggregator.aggregate(outputs, output_rank=0)

        assert result.kv_connector_output.first_tokens == {"r2": 7}


class TestKVConnectorSendPrefilledTokensPatch:
    def test_base_connector_gets_a_noop_default(self):
        connector = MagicMock(spec=KVConnectorBase_V1)
        assert KVConnectorBase_V1.send_prefilled_tokens(connector, MagicMock(), [], []) is None

    def test_mixin_forwards_to_the_transfer_group(self):
        group = MagicMock()
        with patch(
            "vllm_ascend.patch.platform.patch_kv_connector.get_kv_transfer_group",
            return_value=group,
        ):
            KVConnectorModelRunnerMixin.send_prefilled_tokens("sched_out", ["r1"], [[1]])

        group.send_prefilled_tokens.assert_called_once_with("sched_out", ["r1"], [[1]])


class TestRenderServingPatch:
    def _make_request(self, stream, kv_transfer_params=None):
        return SimpleNamespace(stream=stream, kv_transfer_params=kv_transfer_params)

    def test_wraps_preprocess_methods(self):
        for name in ("preprocess_completion", "preprocess_cmpl", "preprocess_chat"):
            assert hasattr(getattr(OpenAIServingRender, name), "__wrapped__")

    def test_marks_reuse_for_streaming_request(self, monkeypatch):
        monkeypatch.setattr(envs, "REUSE_PREFILLED_TOKENS", True)
        request = self._make_request(stream=True, kv_transfer_params={"do_remote_prefill": True})

        patch_render_serving._mark_reuse_prefilled_tokens(request)

        assert request.kv_transfer_params["reuse_prefilled_tokens"] is True

    def test_forces_false_for_non_streaming_request(self, monkeypatch):
        # 非流式没法把 P 的首 token 和 D 的续写拼起来, D 必须自己重算。
        monkeypatch.setattr(envs, "REUSE_PREFILLED_TOKENS", True)
        request = self._make_request(stream=False, kv_transfer_params={"do_remote_prefill": True})

        patch_render_serving._mark_reuse_prefilled_tokens(request)

        assert request.kv_transfer_params["reuse_prefilled_tokens"] is False

    def test_respects_disabled_env(self, monkeypatch):
        monkeypatch.setattr(envs, "REUSE_PREFILLED_TOKENS", False)
        request = self._make_request(stream=True, kv_transfer_params={"do_remote_prefill": True})

        patch_render_serving._mark_reuse_prefilled_tokens(request)

        assert request.kv_transfer_params["reuse_prefilled_tokens"] is False

    def test_ignores_requests_without_kv_transfer_params(self, monkeypatch):
        monkeypatch.setattr(envs, "REUSE_PREFILLED_TOKENS", True)
        request = self._make_request(stream=True, kv_transfer_params=None)

        patch_render_serving._mark_reuse_prefilled_tokens(request)

        assert request.kv_transfer_params is None

    def test_ignores_missing_request(self):
        patch_render_serving._mark_reuse_prefilled_tokens(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
