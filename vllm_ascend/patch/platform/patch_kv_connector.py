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
from collections.abc import Generator
from contextlib import contextmanager

from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.forward_context import get_forward_context
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin


def send_prefilled_tokens_interface(self, scheduler_output: SchedulerOutput, req_ids, token_ids):
    return None


def static_send_prefilled_tokens(scheduler_output: SchedulerOutput, req_ids: list, token_ids):
    kv_connector = get_kv_transfer_group()
    kv_connector.send_prefilled_tokens(scheduler_output, req_ids, token_ids)


@contextmanager
def _get_kv_connector_output(
    scheduler_output: "SchedulerOutput",
    wait_for_save: bool = True,
    defer_finalize: bool = False,
) -> Generator[KVConnectorOutput, None, None]:
    output = KVConnectorOutput()

    # Update KVConnector with the KVConnector metadata forward().
    kv_connector = get_kv_transfer_group()
    assert isinstance(kv_connector, KVConnectorBase)
    assert scheduler_output.kv_connector_metadata is not None
    kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

    # Background KV cache transfers happen here.
    # These transfers are designed to be async and the requests
    # involved may be disjoint from the running requests.
    # Do this here to save a collective_rpc.
    kv_connector.start_load_kv(get_forward_context())
    try:
        yield output
    finally:
        if wait_for_save and not defer_finalize:
            kv_connector.wait_for_save()

        res = kv_connector.get_finished(scheduler_output.finished_req_ids)
        if len(res) == 3:
            output.finished_sending = res[0]
            output.finished_recving = res[1]
            output.first_tokens = res[2]
        else:
            output.finished_sending = res[0]
            output.finished_recving = res[1]

        output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

        output.kv_connector_stats = kv_connector.get_kv_connector_stats()
        output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

        if not defer_finalize:
            kv_connector.clear_connector_metadata()


KVConnectorBase_V1.send_prefilled_tokens = send_prefilled_tokens_interface
KVConnectorModelRunnerMixin.send_prefilled_tokens = staticmethod(static_send_prefilled_tokens)
KVConnectorModelRunnerMixin._get_kv_connector_output = staticmethod(_get_kv_connector_output)
