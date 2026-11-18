# QuantFlashAttnVllm

## 产品支持情况

| 产品 | 是否支持 |
|:--|:--:|
| Ascend 950PR/950DT 系列产品 | √ |
| Atlas A3 系列产品 | × |
| Atlas A2 系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列加速卡产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

QuantFlashAttnVllm（QFA）是面向量化注意力推理的 FlashAttention 算子，支持 MXFP8 / FP8 /
HiFloat8 输入与分页 KV 缓存，支持 BSND/BNSD/TND 等 layout 以及 NO_MASK / CAUSAL /
SLIDING_WINDOW（BAND）掩码模式。

- 算子来源：ops-transformer `feat/qfa-sliding-window-mxfp8` 分支
  （commit `3ff3060124af8aeeab9e0d1f71b3981d6597b3b4`，含 MXFP8 滑窗左边界掩码修复）。
- 相对上游的差异：仅做 vllm-ascend 接入所需的类型改名（`QuantFlashAttn` →
  `QuantFlashAttnVllm`，避免与已安装的 ops-transformer vendor 包注册冲突）、
  `opFile` / kernel 入口同名化，以及头文件依赖收敛；kernel / tiling 计算逻辑与上游逐字一致。

计算公式：

$$
Attention = \text{softmax}\left(\frac{Q @ \text{Dequant}(\tilde K)^T}{\sqrt{d_k}}\right) @ \text{Dequant}(\tilde V)
$$

其中 $\tilde K,\tilde V$ 为 PA_NZ 分页缓存中的量化 Key/Value，反量化参数以 E8M0
（MXFP8 模式）或 per-tensor scale 形式随缓存存放。

## 接口

PyTorch 侧注册为 `torch.ops._C_ascend.npu_quant_flash_attn_vllm` 和
`torch.ops._C_ascend.npu_quant_flash_attn_vllm_metadata`（schema 见
`csrc/torch_binding.cpp`），aclnn 侧为 `aclnnQuantFlashAttnVllm` /
`aclnnQuantFlashAttnVllmMetadata`。

### quant_flash_attn_vllm

| 参数 | 输入/输出/属性 | 说明 |
|:--|:--|:--|
| q/k/v | 输入 | MXFP8 模式为 FLOAT8_E4M3FN；HiFloat8 模式为 HIFLOAT8 |
| q_descale/k_descale/v_descale | 输入 | E8M0（MXFP8）或 FLOAT per-tensor scale |
| quant_mode | 属性 | 量化模式枚举，C8 滑窗路径使用 `A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32`（=1） |
| block_table | 可选输入 | PageAttention block 映射表 |
| cu_seqlens_q / cu_seqlens_kv / seqused_q / seqused_kv | 可选输入 | 变长序列描述 |
| sinks / attn_mask / metadata | 可选输入 | sink logits、int8 掩码、调度 metadata |
| softmax_scale | 属性 | QK 缩放系数 |
| mask_mode | 属性 | 0=NO_MASK，3=CAUSAL，4=SLIDING_WINDOW |
| win_left / win_right | 属性 | 滑窗左右边界 |
| layout_q / layout_q_descale / layout_kv / layout_out | 属性 | 支持 BSND/BNSD/TND/NTD 与 PA_NZ 等 |
| return_softmax_lse | 属性 | 为 True 时返回 `(N, T)`（TND）或 `(B, N, S)` LSE |
| attn_out | 输出 | BF16，shape 由 layout 推导 |
| softmax_lse | 输出 | FLOAT32；`return_softmax_lse=False` 时为空 tensor |

### quant_flash_attn_vllm_metadata

根据 `num_heads_q`、batch（由 `cu_seqlens_q` / `seqused_q` 推导）与当前设备
AIC/AIV 核数计算调度区大小，分配并填充 `(2, max_schedule_size)` int32 metadata。

## 约束说明

- 仅注册 `ascend950`（arch35）。
- 输入 K/V 的 stride 不支持含 0；K/V/scale 非连续由 aclnn 封装处理。
- 变长 + PA_NZ 场景使用 `layout_q="TND"`、`layout_kv="PA_NZ"`；
  decode 单 token 时 `layout_q_descale="N2TGD"`，prefill 为 `"TND"`。
- metadata 的分配大小依赖设备 AIC/AIV 核数（`aclrtGetResInCurrentThread`，
  失败时按 36/72 兜底），需与 AICPU 侧约定的 stride（16）和 4096 对齐一致。

## 目录说明

- `qfa_kernel_deps.h`：kernel 侧唯一依赖聚合头，只允许 `op_kernel/quant_flash_attn_vllm.cpp`
  在所有 QFA 头文件之前 include 一次。
- `qfa_host_deps.h`：host tiling 侧依赖聚合头。
- `common/`：上游 `attention/common` 的私有依赖快照（与 ops-transformer
  `feat/qfa-sliding-window-mxfp8` 逐字一致），避免与 vllm-ascend 已裁剪的共享
  `csrc/attention/common` 相互影响；`common/include/util.h` 为上游完整版，
  通过 `FLASH_ATTENTION_UTIL_H` guard 屏蔽仓内裁剪版。
- `op_host/`、`op_kernel/`、`op_api/`：算子实现（改名后）。

## 测试入口

```bash
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_qfa_c8_paged_layout.py
```
