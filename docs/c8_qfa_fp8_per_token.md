# QFA FP8 动态 per-token Q/K

Ascend 950 / CANN 9.2.1 路径使用原生 `QuantFlashAttn` 模式 6。
Q/K 均调用 `npu_dynamic_quant(..., dst_type=float8_e4m3fn, quant_mode="pertoken")`，
保留其 FP32 scale，不转为 E8M0。每个 token 的所有本地 head/channel 共用一个 scale，
随后将它广播到 QFA 的 token-head 接口。

V 保底方案沿用 checkpoint 的静态 per-channel scale 和原 `npu_quantize`，
不称为动态 V。模式 6 原生 V descale 仅支持 `[Nkv]`，因此传入全 1，
在 QFA 输出后乘 checkpoint channel scale：
`sum(P * (V8 * scale[channel])) = sum(P * V8) * scale[channel]`。
GQA 按 KV head 展开 scale；共享 KV 层使用 cache owner 的 scale。
learned-key sink 使用与 QK 相同的 `Q8 * q_scale`，根据原生 `[H,T]` LSE 修正分母。

```json
{
  "c8_q_quant_mode": "per_token",
  "c8_k_quant_mode": "per_token",
  "c8_v_quant_mode": "per_channel"
}
```

同时指定 `--kv-cache-dtype mxfp8 --dtype bfloat16`，滑窗层可继续通过
`--kv-cache-dtype-skip-layers sliding_window` 保留 C16。
当前要求 K/V head_dim=128，沿用基线 PCP/DCP 限制，不支持 PD KV transfer。
不设置 `c8_q_quant_mode` 时原 MXFP8/FIA 选择逻辑不变。

模式 6 的 kernel page 固定 128 token。512-token 调度页通过现有 BlockTables
机制拆成 4 个 kernel page；缓存每页为 `[K8 | V8 | K_scale_fp32 | padding]`，
三个视图保留物理 page pitch。V 固定 scale 保存在模型层，无需逐页填充或重标定。
Q 为 NTD，Q descale 为 NT，KV 为 PA_BNBD，输出为 TND；P scale 为 FP32 `[448]`。

真实 NPU 算子验证（2026-11-18，`Ascend950DT_9582`、torch 2.10.0、torch_npu 2.10.0.post7）：

```bash
python tests/ut/attention/test_c8_qfa_fp8.py
ASCEND_RT_VISIBLE_DEVICES=0 python \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_c8_qfa_fp8.py
```

3 项独立 CPU 页布局测试和 4 项真实 NPU 测试通过，包含 query 长度 1/6/129、
KV529、非顺序逻辑页、额外页 padding、cache 字节/FP32 scale 精确检查、sink/LSE、
生产 forward 的 NPUGraph 重放与 K 极值回归。合成 attention 相对 FP32 golden 的
最大输出绝对误差为 0.006951；不代表整模型 AIME 分数。

此前 `docs/c8_k_per_token.md` 的 97% AIME 和旧 6144/10 profiling 对应 BF16 Q 的 FIA
内部反量化方案，不是本模式 6。此路径必须独立完成整模型精度与 profiling 验证。
量化前 reshape 的性能优化继续留作后续工作。
