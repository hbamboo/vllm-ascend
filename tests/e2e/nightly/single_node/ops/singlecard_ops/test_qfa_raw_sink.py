# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Raw BF16 QFA sink numerics and graph replay on one Ascend 950 NPU."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# The sibling triton/ test directory shadows the installed compiler in direct runs.
if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import torch
import torch_npu
import triton
import triton.language as tl


class TestQFARawSinkNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        if torch_npu.npu.get_soc_version() != 260:
            raise RuntimeError("This integration test requires an Ascend 950 NPU.")
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        spec = importlib.util.spec_from_file_location("qfa_raw_sink_npu_test", root / "vllm_ascend/ops/triton/sink.py")
        cls.sink = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.sink
        with patch.dict(
            sys.modules,
            {
                "vllm.triton_utils": SimpleNamespace(triton=triton, tl=tl),
                "vllm_ascend.ops.triton.triton_utils": SimpleNamespace(init_device_properties_triton=lambda: None),
            },
        ):
            spec.loader.exec_module(cls.sink)

    def make_case(self, tokens, heads=6, kv_heads=1, *, strided=False, dim_stride=1):
        torch.manual_seed(91)
        dim = 128
        query = torch.randn(tokens, heads + int(strided), dim * dim_stride, device="npu", dtype=torch.bfloat16)[
            :, :heads, ::dim_stride
        ]
        attn = torch.randn(tokens, heads, dim, device="npu", dtype=torch.bfloat16)
        lse = torch.randn(heads, tokens + 3, device="npu")[:, :tokens]
        sink_key = torch.randn(kv_heads, dim, device="npu", dtype=torch.float32)
        destination = torch.full((tokens + 3, heads, dim), 23, dtype=torch.bfloat16, device="npu")
        output = destination[:tokens]
        return SimpleNamespace(**locals())

    def run_case(self, case):
        self.sink.qfa_sink_apply_triton(
            case.query,
            None,
            case.attn,
            case.lse,
            case.sink_key,
            case.dim**-0.5,
            case.tokens,
            case.heads,
            case.kv_heads,
            case.dim,
            case.output,
        )

    def check_case(self, case, before):
        sink = case.sink_key.cpu().float().repeat_interleave(case.heads // case.kv_heads, dim=0)
        logit = (case.query.cpu().float() * sink[None]).sum(-1) * case.dim**-0.5
        factor = torch.sigmoid(case.lse.cpu().t() - logit)
        expected = (case.attn.cpu().float() * factor[..., None]).bfloat16()
        torch.testing.assert_close(case.output.cpu(), expected, rtol=0.008, atol=0.002)
        torch.testing.assert_close(case.attn, before, rtol=0, atol=0)
        self.assertTrue(torch.all(case.destination[case.tokens :].cpu() == 23))

    def test_raw_query_head_major_lse_and_direct_output(self):
        for tokens, heads, kv_heads, strided, dim_stride in (
            (1, 6, 1, False, 1),
            (12, 6, 1, True, 1),
            (12, 6, 1, False, 2),
            (12, 6, 1, True, 2),
            (129, 8, 2, True, 1),
            (6144, 6, 1, False, 1),
        ):
            with self.subTest(tokens=tokens, heads=heads, kv_heads=kv_heads, strided=strided, dim_stride=dim_stride):
                case = self.make_case(tokens, heads, kv_heads, strided=strided, dim_stride=dim_stride)
                before = case.attn.clone()
                self.run_case(case)
                torch.npu.synchronize()
                self.check_case(case, before)

    def test_sink_and_factor_stay_fp32_until_output_store(self):
        # These inputs distinguish FP32 arithmetic from the FIA sink/dot and
        # factor BF16 roundings by one full BF16 output quantum.
        for query, lse, attn in ((1.0, 1.0, 1.0), (0.0, 0.5, 1.5)):
            with self.subTest(query=query, lse=lse, attn=attn):
                case = self.make_case(12)
                case.query.fill_(query)
                case.sink_key.fill_(0.12549)
                case.lse.fill_(lse)
                case.attn.fill_(attn)
                q = case.query.cpu().float()
                sink = case.sink_key.cpu().float()
                lse_cpu = case.lse.cpu().t()
                attn_cpu = case.attn.cpu().float()
                logit = (q * sink).sum(-1) * case.dim**-0.5
                expected = (attn_cpu * torch.sigmoid(lse_cpu - logit)[..., None]).bfloat16()
                rounded_dot = (q * sink.bfloat16().float()).sum(-1).bfloat16().float()
                rounded_factor = torch.sigmoid(lse_cpu - rounded_dot * case.dim**-0.5).bfloat16().float()
                rounded = (attn_cpu * rounded_factor[..., None]).bfloat16()
                self.assertTrue(torch.any(expected != rounded))
                self.run_case(case)
                torch.npu.synchronize()
                torch.testing.assert_close(case.output.cpu(), expected, rtol=0, atol=0)

    def test_graph_replay_uses_updated_raw_query_and_lse(self):
        case = self.make_case(12, strided=True, dim_stride=2)
        before = case.attn.clone()
        for _ in range(3):
            self.run_case(case)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            self.run_case(case)
        case.query.mul_(2)
        case.lse.add_(1)
        graph.replay()
        torch.npu.synchronize()
        self.check_case(case, before)


if __name__ == "__main__":
    unittest.main()
