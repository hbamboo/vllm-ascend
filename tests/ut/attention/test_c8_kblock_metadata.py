# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_c8_kblock import (
    AscendC8KBlockMetadataBuilder,
    KBlockFIAParamProvider,
)
from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPMetadataBuilder


class TestKBlockMetadata(TestBase):
    def test_regrouping_preserves_pointers_and_updates_active_requests(self):
        def init_parent(builder, *args):
            builder.max_num_blocks_per_req = 3

        config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=3))
        with patch.object(AscendC8MXFPMetadataBuilder, "__init__", init_parent):
            builder = AscendC8KBlockMetadataBuilder(None, ["swa"], config, torch.device("cpu"))
        sources = [
            SimpleNamespace(
                actual_seq_lengths_q=[1, 2, 3],
                query_start_loc_gpu=torch.tensor([0, 1, 2, 3]),
                seq_lens_gpu=torch.tensor([100, 700, 1200]),
                block_tables=torch.tensor([[1, 0, 0], [2, 3, 0], [4, 5, 6]]),
            ),
            SimpleNamespace(
                actual_seq_lengths_q=[3],
                query_start_loc_gpu=torch.tensor([0, 3]),
                seq_lens_gpu=torch.tensor([40]),
                block_tables=torch.tensor([[7, 0, 0]]),
            ),
        ]
        pointers = []
        for source in sources:
            with patch.object(AscendC8MXFPMetadataBuilder, "build", return_value=source):
                metadata = builder.build(0, None)
            pointers.append(
                tuple(
                    x.data_ptr()
                    for x in (
                        metadata.query_start_loc_gpu,
                        metadata.seq_lens_gpu,
                        metadata.block_tables,
                        metadata.kblock_num_reqs,
                    )
                )
            )
            self.assertEqual(metadata.kblock_num_reqs.tolist(), [len(source.actual_seq_lengths_q)])
        self.assertEqual(pointers[0], pointers[1])
        self.assertEqual(metadata.query_start_loc_gpu[:2].tolist(), [0, 3])
        self.assertEqual(metadata.seq_lens_gpu[0].item(), 40)
        self.assertEqual(metadata.block_tables[0].tolist(), [7, 0, 0])
        # The old tail intentionally remains: kernels must consume active_B.
        self.assertEqual(metadata.seq_lens_gpu[1].item(), 700)

    def test_provider_preserves_inclusive_window_and_identity_pages(self):
        identity = torch.arange(36, dtype=torch.int32).view(4, 9)
        provider = KBlockFIAParamProvider("swa", 4096, identity)
        for query_len in (1, 6):
            for seq_len in (4096 + query_len - 1, 4096 + query_len, 4096 + query_len + 1):
                metadata = SimpleNamespace(actual_seq_lengths_q=[query_len], seq_lens_list=[seq_len])
                params = provider.resolve({"swa": metadata})
                earliest_visible = max(0, seq_len - query_len - 4096)
                self.assertEqual(params["actual_seq_kvlen"], [seq_len - earliest_visible])
                self.assertEqual(params["actual_seq_qlen"], [query_len])
                self.assertEqual(params["block_table"].shape, (1, 9))
                self.assertEqual(params["block_table"].data_ptr(), identity.data_ptr())
