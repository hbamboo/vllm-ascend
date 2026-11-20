# Dynamic K offsets for QFA

These experimental controls add online K offsets to full attention using QFA
mode 6 on Ascend 950. They require the matching native QFA package and Python
extension with `aclnnIQuestQuantFlashAttnKOffset` support. The old no-offset
entry point remains the default. Installing a custom native `.run` alone does
not update the Python extension: deploy its matching Python and C++ wrapper
sources together in a new image.

Use `additional_config` with the existing token/head Q/K and static V settings:

```json
{
  "c8_q_quant_mode": "per_token_head",
  "c8_k_quant_mode": "per_token_head",
  "c8_v_quant_mode": "per_channel",
  "c8_k_offset_mode": "per_token_head_scalar"
}
```

`c8_k_offset_mode` accepts:

- `none`: the existing symmetric FP8 K representation.
- `per_token_head_scalar`: an FP32 `(max(K) + min(K)) / 2` offset along the head
  dimension for every actual K token/KV head.
- `per_block_fp8_anchor`: the first actual post-RoPE key of each physical
  128-token block, encoded as FP8 with an FP32 scale per KV head.

Both modes subtract their decoded offset in FP32, then quantize the residual
to FP8 with an FP32 token/head scale. QFA adds `sum(Qhat) * offset` for the
scalar mode, or `Qhat dot decoded_anchor` for the anchor mode, after applying
Q/K descales and before masking and softmax. The sink stays in its original
coordinates. Q, static per-channel V and P448 retain their existing behavior;
SWA retains its existing QFA path. No offline calibration data is required.

Offsets occupy explicit sections of each cache packet. Scalar mode stores
4 bytes/token/KV head; anchor mode stores 132 bytes/physical block/KV head at
head dimension 128. Enabling either mode reserves 4 additional logical
bytes/token/KV head for all full layers, before hybrid page alignment. Prefix
and block copies include these sections. An anchor is initialized when the
physical page opens, retained during append, and replaced on page reuse.

An optional `c8_k_offset_layers` list selects full layers for tracing. Omit it
to enable the selected offset mode on every full layer. This option does not
switch attention layers to BF16. Acceptance runs should use all intended full
layers and keep hybrid cache management enabled.

Native offset support is restricted to Q layout `NTD`, Q descale `NT`, KV
`PA_BNBD`, output `TND`, and head/block dimensions 128. Other existing QFA
modes continue using the legacy interface. Mode 6 currently uses its regular
decode path rather than a FlashDecode branch. GE conversion is not supported
by this extension.

Relevant regression tests:

```bash
python tests/ut/attention/test_c8_qfa_k_offset.py
python tests/ut/attention/test_c8_qfa_fp8.py
pytest -q tests/ut/quantization/methods/test_mxfp_c8.py
ASCEND_RT_VISIBLE_DEVICES=0 python tests/e2e/nightly/single_node/ops/singlecard_ops/test_c8_qfa_k_offset.py
```

The NPU cache test covers eager execution, graph replay, append, prefix
metadata copying, padding and physical-page reassignment. Native attention
output/LSE checks and complete model evaluation are separate requirements;
cache tests alone do not demonstrate model accuracy or an EOS fix.
