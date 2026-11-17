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
`torch_npu.npu_dynamic_quant(dst_type=torch.float8_e4m3fn, quant_mode="pertoken")`。
每个 token 在当前 TP rank 上的所有 KV head/channel 共享一个 FP32 scale，
K 与 V 的 scale 独立，直接作为 FIA 的 descale；不经过 E8M0 编码或幂次转换。
目标模型 TP8 时 `Nkv=1,D=128`。

此前 K/V 路径使用 `npu_dynamic_mx_quant` 的 MX scale，现已改为上述 FP32 per-token 量化。
仅 K per-token 的 QFA 路径仍使用原有 MX 量化规则。

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

后续性能项：用 profiling 确认量化前 reshape 是否产生拷贝，以及能否省略。
当前精度验证保持该实现不变。

接口依据：
[DynamicQuant](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_quant.md)、
[FIA v2](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md)。

## 验证

```bash
python tests/ut/attention/test_c8_k_per_token.py -v
ASCEND_RT_VISIBLE_DEVICES=0 python \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_c8_k_per_token.py TestC8KVPerTokenNPU -v
```

2026-11-18，FP32 scale 版本在 `10.222.42.240` 的 `hxd_per_token` 容器、Ascend950DT_9582
（runtime SOC 260）上完成真实 NPU 验证：

- 6 组 eager：Q 长度 1/6/128，page padding 0/256 字节。
- 2 组 learned-key sink：使用生产 Triton sink kernel。
- 2 次 NPUGraph replay：同时改变 K/V 幅值，结果与同输入 eager 逐值一致。
- FP8 K/V 字节及两份 FP32 scale 精确比对通过。
- 对同一份量化 KV 反量化为 BF16 的 CPU attention 标杆，
  rtol/atol 均为 0.02，最大绝对误差为 0.015625。
- 从原始 BF16 输入验证 `±1.875` 和 `±3.75` 极值，避免旧 MX `scale_alg=0`
  分别截断至 `±1.75` 和 `±3.5`。FIA 测试通过生产 `forward(key,value)` 写缓存。

6 项独立 CPU 测试、4 项 NPU 测试和 126 项 C8/配置/runner/zeroer 回归通过，覆盖动态 V
历史缓存保留、图任务长度 provider、跳过静态 scale 填充及动态 scale 清零。
Ruff check/format、Python 编译、启动脚本语法和 git diff 检查通过。

独立测试从源码加载生产函数、backend 方法及 Triton kernel，避免初始化 vLLM 分布式组；
量化、scatter、FIA 与 NPUGraph 均在 NPU 执行。
该算子测试使用合成 Q/K/V，并读取目标 checkpoint 的 V scale 验证权重兼容性；
动态路径的 V 数据和 scale 由运行时输入生成。

软件为 torch `2.10.0+cpu`、torch_npu `2.10.0.post7`、vLLM `0.30.0+empty`、
CANN `9.2.1`。
此前 MX 版本在 `/mnt/sfs_turbo/hxd/eval/pertoken` 的 10 轮 AIME 得分为 88.33%，
低于用户提供的 C16 KV cache 基线约 97%。FP32 scale 版本按同一启动脚本和采样参数
先完成单轮 AIME：30/30（100%），30 个请求均成功且正常停止，无长度截断。
结果目录为 `/mnt/sfs_turbo/hxd/eval/pertoken_fp32_dynamic_quant_1round_20261118`。
服务使用 `/mnt/sfs_turbo/huawei/zuoqian/test/base/1116/1m-tp8dp1-pertoken/iquest-q1-start-mix.sh`。
新旧 10 轮评测均为 seed 42、batch 32、temperature 1、top_p 0.95、top_k 20、
max_tokens 512000、timeout 3600000，已逐项核对生成的评测配置一致。
单轮达标后完成 10 轮复测，正式得分为 **291/300（97.00%）**，
比旧 MX 版本的 265/300（88.33%）提升 **8.67 个百分点**，达到预期约 97%。
目录为 `/mnt/sfs_turbo/hxd/eval/pertoken_fp32_dynamic_quant_10round_20261118`，
正式报告为其下的 `reports/IQuest-Q1/aime26.json`，核对摘要为 `validation_summary.json`。

300 个请求均成功，errored=0；298 条正常停止，索引 258、295 两条达到 512000 token
上限，没有解析出最终答案，均按原评分规则计 0 分。其余 7 条为答案不匹配，未排除样本。
本轮总输出为 7006932 token。增量评分与正式报告、300 条 review 逐项核对一致。
服务健康检查通过，精度复测和增量评分进程均已结束。

长输出阶段观察到 KV cache 使用率约 98.5% 和容量等待；尚未进行固定输出长度的性能对照。
量化前 reshape 保持不变，按上文记录为后续性能项。
`bash format.sh ci` 因缺少 pre-commit 未执行，单独执行 Ruff 检查。

## 精度原因分析

旧 FIA 版本已对 K/V 动态逐 token 量化，scale cache 也已经是 FP32；
但 scale 先由 MX 量化器生成 E8M0 编码，再转为 FP32，数值仍只能是 2 的整数幂。
将 scale 转成 FP32 无法恢复量化 payload 已丢失的信息。

当前环境的 MX 默认 `scale_alg=0` 使用
`s = 2 ** (floor(log2(amax)) - 8)`。E4M3FN 最大有限值为 448，
因此重建上界为 `448*s = 1.75 * 2 ** floor(log2(amax))`。
当 token 的 `amax` 落在该二进制区间上部时，极值会饱和。
真实 NPU 合成输入验证如下，表中为反量化后的最大值：

| 原始 BF16 最大值 | 旧 MX alg0 | 新 npu_dynamic_quant |
| --- | --- | --- |
| 1.875 | 1.75 | 约 1.875 |
| 1.9921875 | 1.75 | 约 1.9921875 |

新版在当前 FP8 per-token 路径使用 `s = amax/448`，避免上述确定的极值截断，
并取消 scale 的幂次约束；非极值仍有 FP8 舍入误差，不保证每个元素的误差都更小。
K 的误差会影响 QK 与 softmax，V 的误差会影响加权结果；
它们可能改变后续自回归生成路径。

此前 attention golden 从已量化 KV 反量化后计算，能检查 FIA、分页、sink 和图重放，
却无法发现量化入口的损失。本次增加原始 BF16 极值对照，并通过生产 forward 写缓存。

此次相对 `b601394b2` 的推理行为改动仅在 K/V 量化 producer，
BF16 Q、FIA 参数、缓存布局、graph、sink 和评测配置均保持一致。
精度回升支持量化 producer 是本次关键差异；尚未统计真实模型 K/V 的饱和率，
也未做 K/V 分开或其他 scale 算法的整模型消融，不能将全部分差只归于截断，
也不能据此断言 FP32 scale 是唯一可行方式。
