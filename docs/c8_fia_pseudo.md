# IQuest-Q1：FP8 KV 存储、FIA BF16 计算

本分支面向 `/mnt/sfs_turbo/work/liuhaiyang/modelslim_full/out_full_c8`，复用该权重的线性层量化参数，不需要转换权重文件。K/V 均在运行时动态 per-token 量化，不需要预先校准 K scale。

在已有启动命令中保留以下配置，并把 `c8_mxfp_backend` 合并到已有的 `--additional-config` JSON 中：

```bash
--dtype bfloat16 \
--kv-cache-dtype mxfp8 \
--additional-config '{"c8_mxfp_backend":"fia_pseudo"}'
```

默认值是 `qfa`，仍走原 C8-MXFP/QFA 实现。`mxfp8` 这个 cache dtype 开关在新分支中用于复用现有的权重映射和 cache 预算，并不表示新分支仍执行 MXFP8 全量化。

## 新分支的计算方式

- Q：保持 BF16，不执行 `npu_dynamic_mx_quant`。
- K/V：各调用一次 `npu_dynamic_quant(..., dst_type=torch.float8_e4m3fn, quant_mode="pertoken")`，每个 token 的 K、V 分别产生一个 FP32 scale；该 token 在当前 TP rank 上的所有 KV head/channel 共享 scale。
- Cache：分页 ND 的 FP8 K/V 与 FP32 scale。包括 PrefillNoCache 在内的路径都先写量化 cache，再由 FIA 读取，避免 prefill 绕过 KV 量化。
- Attention：调用 `torch_npu.npu_fused_infer_attention_score_v2`，`query_quant_mode=0`、`key_quant_mode=4`、`value_quant_mode=4`，`dequant_scale_key/value` 指向分页 FP32 scale cache。由算子内部反量化 K/V 后计算，没有在 Python 侧反量化整个 cache。
- Sink：复用 C16 路径的 learned-key sink 融合后处理，使用原 BF16 Q 和 FIA 返回的 LSE。
- 图模式：使用 FIA v2 的 UpdatableGraph 任务，重放时更新实际 Q/KV 长度；scale cache 地址固定，scale 内容随新 token 更新。

## 量化 reshape 与 cache 写入

展平仅用于量化计算，写 cache 前恢复 TND：

```text
K/V: [T, Nkv, D]
    -> reshape [T, Nkv*D]
    -> dynamic_quant: FP8 [T, Nkv*D] + FP32 scale [T]
    -> FP8 恢复 [T, Nkv, D]
    -> FIA 专用 reshape_and_cache
```

新实现覆盖 `reshape_and_cache`，不调用 QFA 的 PA_NZ/E8M0 scatter。cache tuple 中：

| 分量 | 形状 | 数据类型 |
| --- | --- | --- |
| K cache | `[Bn, Bs, Nkv*D]` | FP8 E4M3FN |
| V cache | `[Bn, Bs, Nkv*D]` | FP8 E4M3FN |
| K scale cache | `[Bn, Bs]` | FP32 |
| V scale cache | `[Bn, Bs]` | FP32 |

用同一个 `slot_mapping` 计算 `page = slot // Bs`、`offset = slot % Bs`，将四份数据写入相同的 `[page, offset]`。每个物理 page 内保存 `[K | V | K_scale | V_scale | padding]`，四个视图的第 0 维 stride 保留真实 page pitch。不能将整个 cache 展平成连续的行：这些视图跨 page 有间隔，`reshape` 可能生成副本而丢失原始 cache 的写入关系。

负的 padding slot 写到 vLLM 保留的 null block 0。只更新本次 token 的数据和 scale；历史 token 的 scale 不变，decode 不需要重新量化历史缓存。V1/V2 zeroer 会同时清理两份动态 scale；原 QFA 的静态 E8M0 V scale 仍保留其原有初始化与恢复逻辑。

沿用 `mxfp8` 页预算时，每个 kernel page 需要额外两份 `Bs*4` 字节的 FP32 scale。目标模型每 rank 的 `Nkv=1, D=128` 正好满足预算；更宽的 KV 留有余量。页预算不足或 FP32 视图未按 4 字节对齐时初始化报错。

## 与原 checkpoint KV recipe 的区别

该权重的 attention recipe 为 `QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL`。其中 K 使用动态 per-token-group MXFP8，只有 V 提供静态 per-channel scale。

新分支改用 per-token FP8：K 的量化粒度从沿 D 每 32 个元素一组，改为每个 token 一组；V 从静态 per-channel 改为动态 per-token。`fa_v.scale` 仍按原映射加载，以兼容现有权重文件，但此分支不使用该 scale，也不创建单位 K scale。Q/K/V 线性层权重保持原有加载方式。

如果实验要求 K 也逐值保持原 MXFP8 分组量化结果，本分支不满足这一条件。不能把 `query` 改成 BF16 后继续传全量化的 6/6/8 参数组合。

## 范围与验证

- 面向 Ascend 950，BF16，full-attention 层，K/V head_dim 相同。
- Sliding-window 层继续由现有模型逻辑保留为 C16；PCP/DCP 不支持。
- 当前不支持 PD KV 传输：新 page 的数据/scale 排布不能与旧 QFA packet 格式混用；配置了 KV transfer 时初始化明确报错。
- 没有 NPU 实测。本次按用户要求仅修改本地代码；CPU 合约测试覆盖 page 边界与 stride、TND 恢复、两份动态 scale、跨页/多请求写入、decode 历史缓存、padding、V1/V2 zeroer 和图任务参数。同步后仍需验证服务器所安装 CANN 的 FIA mode 4 支持及整模型精度。

CPU 合约测试可独立执行，不依赖 vLLM/torch_npu 初始化：

```bash
python tests/ut/attention/test_c8_fia_pseudo.py -v
```

服务器上提供了真实算子验证入口，覆盖非连续分页数据与 scale、prefill、单 token decode 和 graph replay，并与“同一份 FP8 KV 反量化为 BF16”的 CPU attention 标杆比较。脚本只导入标准库、torch 和 torch_npu，从本地源码读取未修改的缓存辅助函数，不初始化 vLLM 平台插件。张量计算仍调用真实 NPU 算子；独立验证进程没有 PD group。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python tests/e2e/nightly/single_node/ops/singlecard_ops/test_c8_fia_pseudo.py -v
```

命令在同步后的仓库根目录执行，不依赖 `PYTHONPATH=.`。脚本在选择 NPU 后查询实际 runtime SOC，并打印设备名称和软件版本；不依赖 vllm-ascend 的构建期 `_build_info`。如果显示 `skipped`，表示没有执行 NPU 验证。

该验证入口未在本地 Windows/NPU 上执行。当前 TorchNPU Python 文档与底层 V5 文档的 decode 类型支持列表存在差异，因此不能省略服务器版本验证。保持原始 BF16 KV 的模型结果可作为另一组精度比较标杆，但不能用于要求此量化路径逐值相等的测试。

接口依据：[动态量化接口](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md)、[FIA v2 Python 接口](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md)、[底层 FIA V5 规格](https://github.com/hicann/ops-transformer/blob/master/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md)。底层 V5 允许 FP8 per-token/mode 4，分页 BnBsH 数据与 mode-4 scale 可带第 0 维 stride。
