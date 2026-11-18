# QFA FP8 动态 per-token Q/K

Ascend 950 / CANN 9.2.1 路径使用原生 `QuantFlashAttn` 模式 6。
Q/K 均调用 `npu_dynamic_quant(..., dst_type=float8_e4m3fn, quant_mode="pertoken")`，
保留其 FP32 scale，不转为 E8M0。每个 token 的所有本地 head/channel 共用一个 scale，
随后将它广播到 QFA 的 token-head 接口。

V 同样使用 `npu_dynamic_quant`，将 `[T,Nkv,D]` 展成每个 head 一行，生成 FP32 `[Nkv]`
scale，由 QFA 原生消费。完整 Python 路径位于 `attention_c8_mxfp.py`，与原 MX 路径共用 Impl；
`AscendC8QFAFP8AttentionBackend` 仅选择 128-token kernel page。
K、V、FP32 K scale 通过现有 `scatter_pa_kv_cache_with_k_scale` 一次写入，
不再构造通用 ND scatter 索引，也不在输出端恢复 V channel scale。
learned-key sink 使用 `Q8 * q_scale` 和原生 `[H,T]` LSE，通过单个 Triton kernel 直接写输出。

```json
{
  "c8_q_quant_mode": "per_token",
  "c8_k_quant_mode": "per_token",
  "c8_v_quant_mode": "per_head"
}
```

同时指定 `--kv-cache-dtype mxfp8 --dtype bfloat16`，滑窗层可继续通过
`--kv-cache-dtype-skip-layers sliding_window` 保留 C16。
当前要求 K/V head_dim=128，沿用基线 PCP/DCP 限制，不支持 PD KV transfer。
不设置 `c8_q_quant_mode` 时原 MXFP8/FIA 选择逻辑不变。

模式 6 的 kernel page 固定 128 token。512-token 调度页通过现有 BlockTables
机制拆成 4 个 kernel page；缓存每页为 `[K8 | V8 | K_scale_fp32 | V_head_scale_fp32 | padding]`，
四个视图保留物理 page pitch。每页 V tag 占 `[Nkv]` 个 FP32，使用原有 padding，不增加 page 预算。
Q 为 NTD，Q descale 为 NT，KV 为 PA_BNBD，输出为 TND；P scale 为 FP32 `[448]`。

模式 6 的 V scale 对本次所有可见 token 共用，不能只换 scale 而不处理旧 payload。
每层 cache owner 保留 head bound，作为 32 元素对齐尾部附到 V 量化输入，
使 scale 随已见最大值单调增长。初始 bound 为 `float32.tiny * 448`，避免全零输入的中性 scale
污染后续尺度。普通 decode 没有新极值时历史 payload 不变。
每步从 GPU 长度和页表生成共享的引用页历史计数，按物理页 atomic-max 去重 prefix；
只有引用页的 scale 变化时才重标定有效历史 V，随后写新 V。
新页不读取原来的 tag/payload，零 query 行和未引用页不处理，避免破坏其他混合缓存状态。
head scale 与 bound 均使用稳定地址，在同一重标定 kernel 原地更新，支持图重放与 KV-sharing。
历史重标定会有额外 FP8 舍入误差，需独立进行整模型精度评测。

真实 NPU 算子验证（2026-11-18，`Ascend950DT_9582`、torch 2.10.0、torch_npu 2.10.0.post7）：

```bash
python tests/ut/attention/test_c8_qfa_fp8.py
ASCEND_RT_VISIBLE_DEVICES=0 python \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_c8_qfa_fp8.py
ASCEND_RT_VISIBLE_DEVICES=0 python \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_qfa_v_head_scale.py
```

10 项生产 attention/sink NPU 测试及 2 项 V 历史页 helper NPU 测试通过，包含 query 长度
1/6/129/6144、KV529、多个 KV head、非顺序页、padding、缓存字节/FP32 scale、LSE/sink、
V 从 1 增长到 100、共享 prefix 去重、新页脏 tag、未引用页保护，以及改变 Q/K/V/长度的图重放。
原生融合 scatter 另在 1/2 个 KV head、带 stride 页、负 slot/边界 slot 下通过字节检查。

此前 `docs/c8_k_per_token.md` 的 97% AIME 和旧 6144/10 profiling 对应 BF16 Q 的 FIA
内部反量化方案，不是本模式 6。
此前静态 V 保底版的单轮测试按用户要求停止于 29/30 已完成且全对，没有正式单轮分数，未启动十轮。
当前动态 per-head V 版本先做 profiling，再做单轮精度验证。
量化前 reshape 的性能优化继续留作后续工作。
