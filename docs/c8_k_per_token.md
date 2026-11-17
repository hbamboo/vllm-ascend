# IQuest-Q1 C8：K/V 动态 per-token 量化

基于 `5125962e71149b6f1423bc47e4a9459dfb823e6b`，使用权重
`/mnt/sfs_turbo/work/liuhaiyang/modelslim_full/out_full_c8`，无需转换权重。

## 启用

在已安装、编译好本分支的环境中，将以下字段合并到原启动配置：

```bash
--dtype bfloat16 \
--kv-cache-dtype mxfp8 \
--kv-cache-dtype-skip-layers sliding_window \
--block-size 512 \
--additional-config '{"c8_k_quant_mode":"per_token","c8_v_quant_mode":"per_token"}'
```

完整 TP8 eager 示例为 `examples/serve_iquest_c8_k_per_token.sh`，默认端口 8003。
V 的默认配置仍为 `per_channel`；只设置 `c8_k_quant_mode=per_token` 时，
继续使用上一版 K-only QFA 路径。

| 配置 | Q | K | V | Attention |
| --- | --- | --- | --- | --- |
| 默认 | MXFP8 分组 | MXFP8 分组 | checkpoint per-channel | QFA |
| 仅 K per-token | MXFP8 分组 | 动态 per-token | checkpoint per-channel | QFA |
| K/V 均 per-token | BF16 | 动态 per-token | 动态 per-token | FIA v2 |

## K/V 动态量化与缓存

K 和 V 分别将 `[T,Nkv,D]` 展平为 `[T,Nkv*D]`，各调用一次
`npu_dynamic_mx_quant(block_size=Nkv*D, dst_type=float8_e4m3fn)`。
每个 token 在当前 TP rank 上的所有 KV head/channel 共享一个 E8M0 scale，
K 与 V 的 scale 独立。目标模型 TP8 时 `Nkv=1,D=128`。

保持上一版 K 的量化规则：`scale_alg=0`，指数为
`floor(log2(absmax)) - 8`。V 现在使用相同规则，随当前 token 动态生成 scale。
量化器返回的第二个 scale 字节为 padding，只取第一个有效字节。
FIA 使用其对应的 FP32 幂次值作为 descale，FP8 payload 无需重新量化。
这与使用任意 FP32 scale 的 `npu_dynamic_quant(quant_mode="pertoken")` 数值不同。

QFA MXFP8 的 V scale 沿 token 分组，无法表达每个 token 独立的 V scale；
因此 K/V 都 per-token 时选择 FIA v2，Q 使用 BF16，
`query_quant_mode=0,key_quant_mode=4,value_quant_mode=4`。
FIA 内部反量化 K/V；Python 不展开或反量化历史 KV cache。
该路径也不包含 QFA 的 softmax/P FP8 量化，不能将两条路径的差异仅归因于 V scale。

分页布局为 `[K | V | K_scale | V_scale | padding]`：

| 分量 | 形状 | 类型 |
| --- | --- | --- |
| K/V cache | `[Bn,Bs,Nkv*D]` | FP8 E4M3FN |
| K/V scale cache | `[Bn,Bs]` | FP32 |

每份视图保留真实 page stride，数据与 scale 使用同一 slot mapping。
只写本次 token；decode 不重新量化历史 token。
V1/V2 runner 不再填充 checkpoint V scale，zeroer 同时清理两份动态 scale。
`fa_v.scale` 仍加载以兼容权重文件，但动态路径不使用它。
目标 TP8/D128 下可复用现有 mxfp8 page 预算，其他形状页预算不足时初始化报错。

本路径面向 Ascend 950、BF16、full attention、相同 K/V head_dim。
本地 KV 宽度最大 1024，D 须被 64 整除；PCP/DCP 沿用基线限制，
当前不支持 PD KV transfer。滑窗层继续由 skip-layers 配置使用 C16。
图模式复用 FIA 的 UpdatableGraph task 和现有长度参数 provider。

接口依据：
[DynamicMxQuant](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_mx_quant.md)、
[FIA v2](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md)。

## 验证

```bash
python tests/ut/attention/test_c8_k_per_token.py -v
ASCEND_RT_VISIBLE_DEVICES=0 python \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_c8_k_per_token.py TestC8KVPerTokenNPU -v
```

2026-11-17 在 `10.222.47.74` 的 `hxd_debug` 容器、Ascend950DT_9582
（runtime SOC 260）上完成真实 NPU 验证：

- 6 组 eager：Q 长度 1/6/128，page padding 0/256 字节。
- 2 组 learned-key sink：使用生产 Triton sink kernel。
- 2 次 NPUGraph replay：同时改变 K/V 幅值，结果与同输入 eager 逐值一致。
- FP8 K/V 字节及两份 FP32 scale 精确比对通过。
- 对同一份量化 KV 反量化为 BF16 的 CPU attention 标杆，
  rtol/atol 均为 0.02，最大绝对误差为 0.0078125。

6 项独立 CPU 测试和 126 项 C8/配置/runner/zeroer 回归通过，覆盖动态 V
历史缓存保留、图任务长度 provider、跳过静态 scale 填充及动态 scale 清零。
Ruff check/format、Python 编译、启动脚本语法和 git diff 检查通过。

独立测试从源码加载生产函数、backend 方法及 Triton kernel，避免初始化 vLLM 分布式组；
量化、scatter、FIA 与 NPUGraph 均在 NPU 执行。
该算子测试使用合成 Q/K/V，并读取目标 checkpoint 的 V scale 验证权重兼容性；
动态路径的 V 数据和 scale 由运行时输入生成。

软件为 torch `2.10.0+cpu`、torch_npu `2.10.0.post7`、vLLM `0.30.0+empty`、
CANN `9.2.1`、cann-ops-transformer `1.0.0`。
整模型推理、生成精度与性能尚未验证；按用户要求不操作其他服务。
`bash format.sh ci` 因缺少 pre-commit 未执行，单独执行 Ruff 检查。
