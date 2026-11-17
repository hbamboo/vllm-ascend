# IQuest-Q1 C8：K per-token 量化

基于 `5125962e71149b6f1423bc47e4a9459dfb823e6b`，用于权重
`/mnt/sfs_turbo/work/liuhaiyang/modelslim_full/out_full_c8`。

## 启用

在当前已安装、编译好 vllm-ascend 的环境中，将本分支安装为运行版本。
保留原启动参数，将下面字段合并到已有的 `--additional-config`：

```bash
--kv-cache-dtype mxfp8 \
--kv-cache-dtype-skip-layers sliding_window \
--block-size 512 \
--additional-config '{"c8_k_quant_mode":"per_token"}'
```

完整 TP8 eager 启动示例为 `examples/serve_iquest_c8_k_per_token.sh`。
该脚本直接使用上述权重路径，默认端口为 8003；需要空闲的 8 张卡。
默认配置 `per_token_group` 保留原有 K 每 32 个 channel 一组的行为。
非法模式及不使用 mxfp8 cache 的 per-token 配置会报错。

## 量化与缓存语义

- K 输入 `[T, Nkv, D]` 展平为 `[T, Nkv*D]`，调用
  `npu_dynamic_mx_quant(block_size=Nkv*D, dst_type=float8_e4m3fn)`。
- 同一个 token 在当前 TP rank 上的所有 KV head/channel 共享一个 E8M0 scale。
  目标模型 TP8 时 `Nkv=1, D=128`，即每个 token 一组 128 个数。
- 保留原量化器的 `scale_alg=0` 规则：指数为
  `floor(log2(absmax)) - 8`。这是 E8M0 幂次 scale 的 per-token FP8，
  与采用任意 FP32 scale 的 `npu_dynamic_quant(quant_mode="pertoken")` 数值不同。
- 单组量化器返回两个 scale 字节，其中第二个字节为 padding。
  只将第一个有效字节复制到该 token 原有的所有 K scale 槽位；
  FP8 K 恢复 `[T,Nkv,D]` 后写入原来的 PA_NZ cache。
- Q 的 MXFP8 分组量化、V 的 checkpoint per-channel E8M0 scale、
  QFA、sink、分页布局和 cache 字节预算均沿用基线。
  无需转换权重，也不需要重新校准或补 K scale。
- 本地 KV 宽度 `Nkv*D` 最大为 1024，输入为 BF16/FP16，D 须被 64 整除。
  大于 1024 时需要增加 TP；量化算子不支持将更宽的 token 作为一个 block。

scale 被复制到原有槽位是为了兼容现有 QFA 接口，因此此改动不会缩减 scale cache。
跨 TP rank 的 K 不共享 scale。PCP/DCP 等能力沿用基线限制。

量化接口与 scale 规则依据
[Ascend 官方 DynamicMxQuant 文档](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dynamic_mx_quant.md)。

## 验证命令

```bash
python tests/ut/attention/test_c8_k_per_token.py -v
ASCEND_RT_VISIBLE_DEVICES=0 python \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_c8_k_per_token.py -v
```

独立测试入口从源码加载原始生产函数与 QFA backend 方法，避免单进程测试初始化
vLLM 分布式组；量化、Triton scatter、PA_NZ cache 写入、QFA 和 NPUGraph
均调用真实 NPU 实现。NPU 测试读取目标 checkpoint 的
`model.layers.1.self_attn.fa_v.scale`，使用合成 Q/K/V 验证接线；
对照标杆为同一份量化 Q/K/V 反量化后的 CPU attention。

2026-11-17 在 `10.222.47.74` 的 `hxd_debug` 容器、
Ascend950DT_9582（runtime SOC 260）上通过：

- 6 组 eager：Q 长度 1/6/128，物理 page padding 0/256 字节。
- 2 组 learned-key sink：Q 长度 1/6。
- 2 次 NPUGraph replay：改变 K 幅值及 KV 长度（513/529），
  每次输出与相同输入的 eager 结果逐值一致。
- K/V cache 字节、K scale 槽位和 checkpoint V scale 均精确比对通过。
- FP32 CPU attention 标杆容差为 rtol/atol 各 0.03；
  观察到的最大绝对误差为 0.021484375。
  QFA 还包含 softmax/P 量化，因此此标杆不要求逐值相等。

软件为 torch `2.10.0+cpu`、torch_npu `2.10.0.post7`、
vLLM `0.30.0+empty`、CANN `9.2.1`、cann-ops-transformer `1.0.0`。

另外，5 项独立 CPU 合约测试及 57 项 C8/配置回归测试通过，
Ruff check/format、Python 编译、启动脚本 bash 语法和 git diff 检查通过。
`bash format.sh ci` 因容器缺少 pre-commit 未能执行。

本次按用户要求保留 8002 端口现有服务，整模型推理、生成精度与性能尚未验证。
