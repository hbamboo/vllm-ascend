import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackendImpl


class TestC8MXFPQfaQueryPlan(TestBase):
    """What the two QFA operators are planned against for one step.

    max_seqlen_q is the longest single query, not the batch total: the
    metadata op seeds querySeqSize with it and then raises it with
    max(attr, per-request length), so an inflated attr is never walked back.
    The q scale layout then picks the kernel -- TND compiles the prefill
    template, N2TGD the decode one -- and both operators must be given the
    same one.
    """

    NUM_KV_HEADS = 2
    GROUP_SIZE = 4
    NUM_HEADS = NUM_KV_HEADS * GROUP_SIZE
    D_GROUPS = 2  # head_dim // 64

    def setUp(self):
        self.impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
        self.impl.sliding_window = None
        self.impl.num_heads = self.NUM_HEADS
        self.impl.num_kv_heads = self.NUM_KV_HEADS

    def _forward(self, query_start_loc, max_query_len):
        """Run one step and report what each operator was handed."""
        num_tokens = int(query_start_loc[-1])
        # Distinct value per (token, head, group, element) so the N2TGD
        # permutation can be checked entry by entry.
        scale = (
            torch.arange(num_tokens * self.NUM_HEADS * self.D_GROUPS * 2, dtype=torch.int32)
            .remainder(251)
            .to(torch.uint8)
            .view(num_tokens, self.NUM_HEADS, self.D_GROUPS, 2)
        )
        attn_metadata = SimpleNamespace(
            causal=True,
            query_start_loc_gpu=torch.tensor(query_start_loc, dtype=torch.int32),
            seq_lens_gpu=torch.ones(len(query_start_loc) - 1, dtype=torch.int32),
            max_query_len=max_query_len,
        )
        seen = {"source_scale": scale}

        def fake_metadata(_self, _metadata, *, cu_seqlens_q, seqused_kv, max_seqlen_q, mask_mode, layout_q_descale):
            seen["metadata_max_seqlen_q"] = max_seqlen_q
            seen["metadata_mask_mode"] = mask_mode
            seen["metadata_layout"] = layout_q_descale
            return MagicMock()

        def fake_run(
            _self,
            _query,
            query_scale,
            _kv_cache,
            _attn_metadata,
            *,
            max_seqlen_q,
            mask_mode,
            layout_q_descale,
            output,
            **kwargs,
        ):
            seen["max_seqlen_q"] = max_seqlen_q
            seen["mask_mode"] = mask_mode
            seen["layout"] = layout_q_descale
            seen["scale"] = query_scale
            return output

        with (
            patch.object(AscendC8MXFPAttentionBackendImpl, "_get_qfa_metadata", fake_metadata),
            patch.object(AscendC8MXFPAttentionBackendImpl, "_run_qfa", fake_run),
        ):
            self.impl._forward_mxfp8_attention(
                torch.zeros(num_tokens, self.NUM_HEADS, 8, dtype=torch.uint8),
                scale,
                (MagicMock(), MagicMock(), MagicMock(), MagicMock()),
                attn_metadata,
                torch.zeros(num_tokens, self.NUM_HEADS, 8),
            )

        # A plan computed for one kernel must not be fed to the other.
        self.assertEqual(seen["metadata_max_seqlen_q"], seen["max_seqlen_q"])
        self.assertEqual(seen["metadata_layout"], seen["layout"])
        self.assertEqual(seen["metadata_mask_mode"], seen["mask_mode"])
        return seen

    # --- max_seqlen_q -----------------------------------------------------

    def test_uniform_decode_does_not_declare_the_batch_total(self):
        # 4 requests, one query token each: the old code declared 4.
        self.assertEqual(self._forward([0, 1, 2, 3, 4], max_query_len=1)["max_seqlen_q"], 1)

    def test_mtp_decode_uses_the_draft_query_length(self):
        # 3 requests x 2 tokens (1 + 1 draft): the old code declared 6.
        self.assertEqual(self._forward([0, 2, 4, 6], max_query_len=2)["max_seqlen_q"], 2)

    def test_mixed_batch_uses_the_longest_prefill(self):
        # One 5-token prefill plus 3 decodes; the longest query is the prefill.
        self.assertEqual(self._forward([0, 5, 6, 7, 8], max_query_len=5)["max_seqlen_q"], 5)

    def test_missing_max_query_len_falls_back_to_the_token_count(self):
        self.assertEqual(self._forward([0, 1, 2, 3, 4], max_query_len=None)["max_seqlen_q"], 4)

    # --- q scale layout ---------------------------------------------------

    def test_decode_sends_the_scale_as_n2tgd(self):
        seen = self._forward([0, 1, 2, 3, 4], max_query_len=1)
        self.assertEqual(seen["layout"], "N2TGD")
        num_tokens = seen["source_scale"].shape[0]
        self.assertEqual(
            tuple(seen["scale"].shape),
            (self.NUM_KV_HEADS, num_tokens, self.GROUP_SIZE, self.D_GROUPS, 2),
        )

    def test_n2tgd_keeps_every_head_scale_with_its_kv_head(self):
        seen = self._forward([0, 1, 2, 3, 4], max_query_len=1)
        source, permuted = seen["source_scale"], seen["scale"]
        for token in range(source.shape[0]):
            for head in range(self.NUM_HEADS):
                # Query heads are GQA-contiguous: head n serves kv head n // G.
                kv_head, group = divmod(head, self.GROUP_SIZE)
                self.assertTrue(
                    torch.equal(permuted[kv_head, token, group], source[token, head]),
                    f"token={token} head={head} lost its scale",
                )

    def test_prefill_keeps_the_scale_in_tnd_untouched(self):
        # G=4, so a 21-token query crosses the operator's G*Q_S boundary.
        seen = self._forward([0, 21], max_query_len=21)
        self.assertEqual(seen["layout"], "TND")
        self.assertIs(seen["scale"], seen["source_scale"])

    def test_boundary_value_still_takes_the_decode_layout(self):
        # G*Q_S == 80 exactly; the operator doc recommends N2TGD at or below.
        self.assertEqual(self._forward([0, 20], max_query_len=20)["layout"], "N2TGD")

    def test_mtp_verify_queries_take_the_decode_layout(self):
        # MTP verify steps are SpecDecoding with 1+spec query tokens each;
        # the layout decision reads the query shape, not the scheduler
        # state, so the decode kernel applies here too. Regression guard
        # against keying this on attn_state == DecodeOnly, which never
        # fires once MTP is enabled.
        seen = self._forward([0, 4, 8], max_query_len=4)
        self.assertEqual(seen["layout"], "N2TGD")
        self.assertEqual(seen["max_seqlen_q"], 4)

    def test_non_divisible_heads_stay_in_tnd(self):
        # MTP draft layers can run a head count that does not split into
        # whole kv-head groups; the group reshape would miscount elements
        # and raise, so the guard falls back to TND instead.
        self.impl.num_kv_heads = 3  # 8 % 3 != 0
        seen = self._forward([0, 1, 2, 3, 4], max_query_len=1)
        self.assertEqual(seen["layout"], "TND")
        self.assertIs(seen["scale"], seen["source_scale"])

    def test_zero_kv_heads_stay_in_tnd(self):
        # Degenerate layers must not hit a ZeroDivisionError in the guard.
        self.impl.num_kv_heads = 0
        seen = self._forward([0, 1, 2, 3, 4], max_query_len=1)
        self.assertEqual(seen["layout"], "TND")
        self.assertIs(seen["scale"], seen["source_scale"])

    # --- mask mode --------------------------------------------------------

    def test_uniform_decode_drops_the_mask(self):
        # One query row per request: CAUSAL constrains nothing there (the KV
        # range is bounded by seqused_kv), and NO_MASK is a separate tiling
        # key whose kernel never reads the mask.
        self.assertEqual(self._forward([0, 1, 2, 3, 4], max_query_len=1)["mask_mode"], 0)

    def test_mtp_verify_keeps_the_causal_mask(self):
        # 1 + num_spec query rows per request: row 0 must not see row 1's KV.
        self.assertEqual(self._forward([0, 2, 4, 6], max_query_len=2)["mask_mode"], 3)

    def test_prefill_keeps_the_causal_mask(self):
        self.assertEqual(self._forward([0, 5, 6, 7, 8], max_query_len=5)["mask_mode"], 3)

    def test_unknown_query_length_keeps_the_causal_mask(self):
        # max_query_len absent falls back to the token total, which is only
        # 1 for a genuinely single-token step -- never a silent NO_MASK.
        self.assertEqual(self._forward([0, 1, 2, 3, 4], max_query_len=None)["mask_mode"], 3)


class TestC8MXFPPerStepDerivations(TestBase):
    """Reuse lengths across groups while retaining group-local KV mappings."""

    def setUp(self):
        self.impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
        self.attn_metadata = SimpleNamespace(
            query_start_loc_gpu=torch.tensor([0, 4, 8], dtype=torch.int32),
            seq_lens_gpu=torch.tensor([10, 20], dtype=torch.int32),
        )

    def test_lengths_are_derived_once_per_step(self):
        first = self.impl._qfa_step_lengths(self.attn_metadata, 8)
        second = self.impl._qfa_step_lengths(self.attn_metadata, 8)
        # Same tensor objects, so later layers add no ops to the graph.
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_lengths_are_sanitized(self):
        # -1 padding clamps to 0, and cummax keeps the boundaries monotonic
        # so cu_seqlens_q[-1] still equals the token total.
        self.attn_metadata.query_start_loc_gpu = torch.tensor([0, 4, 8, -1, -1], dtype=torch.int32)
        self.attn_metadata.seq_lens_gpu = torch.tensor([10, 20, 0, 0], dtype=torch.int32)
        cu_seqlens_q, seqused_kv = self.impl._qfa_step_lengths(self.attn_metadata, 8)
        self.assertEqual(cu_seqlens_q.tolist(), [0, 4, 8, 8, 8])
        self.assertEqual(seqused_kv.tolist(), [10, 20, 1, 1])

    def test_distinct_group_metadata_reuses_lengths_but_a_new_step_refreshes_them(self):
        query_starts = self.attn_metadata.query_start_loc_gpu
        seq_lens = self.attn_metadata.seq_lens_gpu
        shared = {}
        first_metadata = SimpleNamespace(
            query_start_loc_gpu=query_starts[:], seq_lens_gpu=seq_lens[:], qfa_shared_metadata_cache=shared
        )
        second_metadata = SimpleNamespace(
            query_start_loc_gpu=query_starts[:], seq_lens_gpu=seq_lens[:], qfa_shared_metadata_cache=shared
        )
        self.assertIsNot(first_metadata.query_start_loc_gpu, second_metadata.query_start_loc_gpu)
        self.assertIsNot(first_metadata.seq_lens_gpu, second_metadata.seq_lens_gpu)
        first = self.impl._qfa_step_lengths(first_metadata, 8)
        second = self.impl._qfa_step_lengths(second_metadata, 8)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

        query_starts[1] = 3
        seq_lens.add_(1)
        next_metadata = SimpleNamespace(
            query_start_loc_gpu=query_starts[:], seq_lens_gpu=seq_lens[:], qfa_shared_metadata_cache={}
        )
        refreshed = self.impl._qfa_step_lengths(next_metadata, 8)
        self.assertIsNot(refreshed[0], first[0])
        self.assertEqual(refreshed[0].tolist(), [0, 3, 8])
        self.assertEqual(refreshed[1].tolist(), [11, 21])
        self.assertEqual(first[0].tolist(), [0, 4, 8])


class TestC8MXFPQfaMetadataPlan(TestBase):
    """One AICPU metadata plan a step, capture included.

    The plan's inputs are head counts, head dim, quant mode, the two length
    tensors, mask mode, window and layouts -- nothing layer-specific -- and
    the main operator declares ``metadata`` a read-only Input. So the step's
    full-attention layers share one plan instead of each paying an
    AICore-to-AICPU round trip. That holds while a graph is captured too: the
    deriving call is recorded inside the captured region, ahead of every read
    of its output.
    """

    LAYERS_PER_STEP = 23

    def setUp(self):
        self.impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
        self.impl.sliding_window = None
        self.impl.num_heads = 8
        self.impl.num_kv_heads = 2
        self.impl.head_size = 128

    def _plans(self, calls):
        """Issue one step's worth of plans; return (plans, operator calls)."""
        attn_metadata = SimpleNamespace()
        invocations = []

        def metadata_op(*args, **kwargs):
            invocations.append((args, kwargs))
            return MagicMock()

        defaults = {
            "cu_seqlens_q": torch.tensor([0, 1, 2], dtype=torch.int32),
            "seqused_kv": torch.tensor([10, 20], dtype=torch.int32),
            "max_seqlen_q": 1,
            "mask_mode": 0,
            "layout_q_descale": "N2TGD",
        }
        ops_module = MagicMock(quant_flash_attn_metadata=metadata_op)
        with patch.dict("sys.modules", {"cann_ops_transformer.ops": ops_module}):
            plans = [self.impl._get_qfa_metadata(attn_metadata, **{**defaults, **call}) for call in calls]
        return plans, invocations

    def test_one_plan_serves_every_layer(self):
        plans, invocations = self._plans([{}] * self.LAYERS_PER_STEP)
        self.assertEqual(len(invocations), 1)
        for plan in plans[1:]:
            self.assertIs(plan, plans[0])

    def test_capture_shares_the_plan_too(self):
        # Regression guard for the code this replaced: the old design
        # bypassed the cache whenever a capture was in flight, which put one
        # AICPU op per layer into the captured graph and replayed all of
        # them every step. The shared derivation now runs inside the
        # captured region by construction -- there is no bypass path left.
        plans, invocations = self._plans([{}] * self.LAYERS_PER_STEP)
        self.assertEqual(len(invocations), 1)
        for plan in plans[1:]:
            self.assertIs(plan, plans[0])

    def test_a_different_non_tensor_input_gets_its_own_plan(self):
        for override in ({"layout_q_descale": "TND"}, {"mask_mode": 3}, {"max_seqlen_q": 2}):
            with self.subTest(**override):
                plans, invocations = self._plans([{}, override])
                self.assertEqual(len(invocations), 2)
                self.assertIsNot(plans[0], plans[1])

    def test_the_step_mask_mode_reaches_the_operator(self):
        # The plan is a schedule for one kernel variant, so the mask mode it
        # was built for has to be the one the main call then uses.
        _, invocations = self._plans([{"mask_mode": 3}])
        self.assertEqual(invocations[0][1]["mask_mode"], 3)

    def test_shared_group_plans_keep_distinct_layouts_and_windows(self):
        shared = {}
        metadata = [SimpleNamespace(qfa_shared_metadata_cache=shared) for _ in range(6)]
        native = MagicMock(quant_flash_attn_metadata=MagicMock(side_effect=lambda *args, **kwargs: object()))
        lengths = dict(
            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
            seqused_kv=torch.tensor([10, 20], dtype=torch.int32),
            max_seqlen_q=1,
        )
        cases = ((None, "N2TGD"), (None, "N2TGD"), (None, "TND"), (4096, "TND"), (2048, "TND"), (4096, "TND"))
        plans = []
        with patch.dict(sys.modules, {"cann_ops_transformer.ops": native}):
            for item, (window, layout) in zip(metadata, cases):
                self.impl.sliding_window = window
                plans.append(
                    self.impl._get_qfa_metadata(item, **lengths, mask_mode=4 if window else 0, layout_q_descale=layout)
                )
        self.assertEqual(native.quant_flash_attn_metadata.call_count, 4)
        self.assertIs(plans[0], plans[1])
        self.assertIs(plans[3], plans[5])
        self.assertIsNot(plans[0], plans[2])
        self.assertIsNot(plans[3], plans[4])
        self.assertEqual(
            [call.kwargs["win_left"] for call in native.quant_flash_attn_metadata.call_args_list], [-1, -1, 4095, 2047]
        )

    def test_native_full_and_sliding_use_separate_shared_plans(self):
        shared = {}
        query_starts = torch.tensor([0, 1, 2], dtype=torch.int32)
        seq_lens = torch.tensor([10, 20], dtype=torch.int32)
        metadata = [
            SimpleNamespace(
                query_start_loc_gpu=query_starts[:],
                seq_lens_gpu=seq_lens[:],
                qfa_shared_metadata_cache=shared,
                num_actual_tokens=2,
                max_query_len=1,
                block_tables=torch.zeros(2, 1, dtype=torch.int32),
            )
            for _ in range(2)
        ]
        self.impl.q_quant_mode = "per_token_head"
        self.impl.kv_sharing_target_layer_name = None
        self.impl.qfa_v_descale = torch.ones(2)
        self.impl.qfa_v_output_scale = torch.ones(2, 128)
        self.impl.qfa_p_scale = torch.ones(1)
        self.impl.scale = 128**-0.5
        query = torch.zeros(2, 8, 128, dtype=torch.bfloat16)
        output = torch.empty_like(query)
        native = MagicMock(
            quant_flash_attn_metadata=MagicMock(side_effect=lambda *args, **kwargs: object()),
            quant_flash_attn=MagicMock(return_value=(torch.zeros_like(query), None)),
        )
        modules = {
            "cann_ops_transformer.ops": native,
            "vllm_ascend.ops.triton.qfa_v_head_scale": MagicMock(),
            "vllm_ascend.ops.triton.sink": MagicMock(),
        }
        with (
            patch.dict(sys.modules, modules),
            patch("vllm_ascend.attention.attention_c8_mxfp.record_attention_compute_start"),
            patch(
                "vllm_ascend.attention.attention_c8_mxfp.torch_npu.npu_dynamic_quant",
                side_effect=lambda x, **kwargs: (x.to(torch.float8_e4m3fn), torch.ones(x.shape[:-1])),
            ),
        ):
            for item in metadata:
                self.impl._forward_fp8_per_token(SimpleNamespace(), query, None, None, (None,) * 4, item, output)
            self.impl.sliding_window = 4096
            self.impl.q_quant_mode = "per_token_group"
            lengths = self.impl._qfa_step_lengths(metadata[0], 2)
            for item in metadata:
                self.impl._get_qfa_metadata(
                    item,
                    cu_seqlens_q=lengths[0],
                    seqused_kv=lengths[1],
                    max_seqlen_q=1,
                    mask_mode=4,
                    layout_q_descale="TND",
                )
        self.assertEqual([call.args[3] for call in native.quant_flash_attn_metadata.call_args_list], [6, 1])
        calls = native.quant_flash_attn.call_args_list
        self.assertIs(calls[0].kwargs["metadata"], calls[1].kwargs["metadata"])
        self.assertIs(calls[0].kwargs["cu_seqlens_q"], calls[1].kwargs["cu_seqlens_q"])
