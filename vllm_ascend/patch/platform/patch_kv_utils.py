#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
# 先 P 后 D: 给上游的 KVConnectorOutput 增加 first_tokens 字段, 并让
# KVOutputAggregator 在汇聚各 worker 输出时把它带上, 使 worker 侧 connector
# 收到的 "P 侧首 token" 能一路传到 scheduler。
#
# first_tokens 必须是真正的 dataclass 字段: worker -> scheduler 的
# ModelRunnerOutput 走 msgspec 序列化, 只搬运 dataclasses.fields() 里的字段,
# 普通类属性会被丢掉。
#

import dataclasses

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

logger = init_logger(__name__)

if "first_tokens" not in KVConnectorOutput.__annotations__:
    KVConnectorOutput.__annotations__["first_tokens"] = "dict[str, int] | None"
    KVConnectorOutput.first_tokens = None
    # dataclass() 处理 default_factory 字段时会把类属性删掉, 再次处理时就会把
    # invalid_block_ids 当成无默认值字段, 因此这里先把它的默认值补回去。
    if "invalid_block_ids" not in KVConnectorOutput.__dict__:
        KVConnectorOutput.invalid_block_ids = dataclasses.field(default_factory=set)
    # dataclass() 只在类里还没有同名的魔法方法时才生成, 所以要先摘掉旧的
    # __init__/__repr__/__eq__, 否则新字段进不了构造参数与打印结果。
    for _magic in ("__init__", "__repr__", "__eq__"):
        if _magic in KVConnectorOutput.__dict__:
            delattr(KVConnectorOutput, _magic)
    dataclasses.dataclass(KVConnectorOutput)


def aggregate(self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0) -> ModelRunnerOutput | None:
    if not outputs[output_rank]:
        return None

    # Aggregate kv_connector_output from all workers

    def update_finished_set(
        req_ids: set[str] | None,
        remaining_count_dict: dict[str, int],
        finished_set: set[str],
    ) -> None:
        for req_id in req_ids or ():
            remaining_count = remaining_count_dict.get(req_id, self._expected_finished_count)
            remaining_count_dict[req_id] = remaining_count - 1
            if remaining_count_dict[req_id] == 0:
                finished_set.add(req_id)
                del remaining_count_dict[req_id]

    finished_sending = set[str]()
    finished_recving = set[str]()
    aggregated_kv_connector_stats = None
    aggregated_kv_connector_worker_meta = None
    combined_kv_cache_events = None
    invalid_block_ids = set[int]()
    # 先 P 后 D: 首 token 由 P 侧随 KV 一起投来, 只有实际收到它的 worker
    # 会带 first_tokens, 这里取最后一个非空值。
    first_tokens = None
    for model_runner_output in outputs:
        assert model_runner_output is not None
        kv_output = model_runner_output.kv_connector_output
        if kv_output is not None and kv_output.first_tokens:
            first_tokens = kv_output.first_tokens
        if not kv_output:
            continue
        # Allow the worker to dynamically update the expected number of
        # finished sending/recving for new requests.
        if kv_output.expected_finished_count > 0 and kv_output.expected_finished_count != self._expected_finished_count:
            logger.debug(
                "Expected finished requests updated from %d to %d",
                self._expected_finished_count,
                kv_output.expected_finished_count,
            )
            self._expected_finished_count = kv_output.expected_finished_count

        update_finished_set(kv_output.finished_sending, self._send_remaining_count, finished_sending)
        update_finished_set(kv_output.finished_recving, self._recv_remaining_count, finished_recving)

        # Aggregate kv_connector_stats from all workers.
        if aggregated_kv_connector_stats is None:
            # Use the first worker's kv_connector_stats as accumulator.
            aggregated_kv_connector_stats = kv_output.kv_connector_stats
        elif kv_connector_stats := kv_output.kv_connector_stats:
            assert isinstance(aggregated_kv_connector_stats, type(kv_connector_stats))
            aggregated_kv_connector_stats = aggregated_kv_connector_stats.aggregate(kv_connector_stats)

        # Aggregate kv_connector_worker_meta from all workers.
        if aggregated_kv_connector_worker_meta is None:
            # Use the first worker's kv_connector_worker_meta as accumulator.
            aggregated_kv_connector_worker_meta = kv_output.kv_connector_worker_meta
        elif kv_connector_worker_meta := kv_output.kv_connector_worker_meta:
            aggregated_kv_connector_worker_meta = aggregated_kv_connector_worker_meta.aggregate(
                kv_connector_worker_meta
            )

        # Combine kv_cache_events from all workers.
        if combined_kv_cache_events is None:
            # Use the first worker's kv_cache events as start event list.
            combined_kv_cache_events = kv_output.kv_cache_events
        elif kv_cache_events := kv_output.kv_cache_events:
            assert isinstance(
                combined_kv_cache_events,
                type(kv_cache_events),
            )
            worker_kv_cache_events = kv_cache_events.get_all_events()
            combined_kv_cache_events.add_events(worker_kv_cache_events)
            combined_kv_cache_events.increment_workers(1)

        invalid_block_ids |= kv_output.invalid_block_ids

    # select output of the worker specified by output_rank
    output = outputs[output_rank]

    assert output is not None
    output.kv_connector_output = KVConnectorOutput(
        finished_sending=finished_sending or None,
        finished_recving=finished_recving or None,
        kv_connector_stats=aggregated_kv_connector_stats or None,
        kv_cache_events=combined_kv_cache_events or None,
        kv_connector_worker_meta=aggregated_kv_connector_worker_meta or None,
        invalid_block_ids=invalid_block_ids,
        expected_finished_count=self._expected_finished_count,
        first_tokens=first_tokens,
    )

    return output


KVOutputAggregator.aggregate = aggregate
