# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_c8_mxfp import (
    AscendC8MXFPAttentionBackendImpl,
    scatter_mxfp_k_scale_cache,
)


class TestKScaleScatterDispatch(TestBase):
    def test_nonempty_scatter_delegates_without_accessing_the_cache(self):
        module = ModuleType("vllm_ascend.ops.triton.mxfp_k_scale")
        scatter = MagicMock()
        vars(module)["scatter_k_scales"] = scatter
        cache = object()
        values = torch.ones((1, 1, 4, 2), dtype=torch.uint8)
        slots = torch.tensor([1], dtype=torch.int32)
        with patch.dict("sys.modules", {module.__name__: module}):
            scatter_mxfp_k_scale_cache(values, cache, slots)
        scatter.assert_called_once_with(values, cache, slots)

    def test_empty_scatter_does_not_import_the_kernel_or_access_the_cache(self):
        slots = torch.empty(0, dtype=torch.int32)
        with patch.dict("sys.modules", {"vllm_ascend.ops.triton.mxfp_k_scale": None}):
            scatter_mxfp_k_scale_cache(torch.empty((0, 1, 4, 2), dtype=torch.uint8), None, slots)

    def test_shared_metadata_keeps_raw_group_slot_mappings_independent(self):
        impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
        impl.q_quant_mode = "per_token_group"
        shared = {}
        mappings = torch.tensor([[2, 99, 513, 99, -1, 99], [514, 99, 1025, 99, -7, 99]], dtype=torch.int32)[:, ::2]
        key = value = torch.ones((3, 1, 128), dtype=torch.uint8)
        key_scale = torch.ones((3, 1, 2, 2), dtype=torch.uint8)
        cache = (object(), object(), object(), object())
        with (
            patch("vllm_ascend.attention.attention_c8_mxfp.scatter_mxfp_pa_nz_kv_cache") as kv_scatter,
            patch("vllm_ascend.attention.attention_c8_mxfp.scatter_mxfp_k_scale_cache") as scale_scatter,
            patch("vllm_ascend.attention.attention_c8_mxfp.notify_kv_cache_written"),
        ):
            for slots in mappings:
                metadata = SimpleNamespace(num_actual_tokens=3, slot_mapping=slots, qfa_shared_metadata_cache=shared)
                impl.reshape_and_cache(key, value, key_scale, cache, metadata)
        for index, (kv_call, scale_call) in enumerate(zip(kv_scatter.call_args_list, scale_scatter.call_args_list)):
            slots = scale_call.args[2]
            self.assertEqual(slots.data_ptr(), mappings[index].data_ptr())
            self.assertEqual(slots.dtype, torch.int32)
            self.assertEqual(slots.stride(), (2,))
            self.assertEqual(slots.tolist(), mappings[index].tolist())
            self.assertEqual(kv_call.args[4].data_ptr(), slots.data_ptr())
        self.assertEqual(scale_scatter.call_count, 2)
        self.assertEqual(shared, {})
