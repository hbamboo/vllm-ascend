# SWA 动态 per-token FP8 KV

基于 `fix/c8-kv-fp32-pertoken` 的 `54d59ba212347214a0b177b162dac8f67e8f4f76`。
Full attention 和 SWA 均使用 BF16 Q、FP8 E4M3FN K/V，K/V 各自每 token 一个 FP32 scale。
FIA 内部反量化 KV；保留原 learned-key sink 修正。

## 启用

```bash
--dtype bfloat16 \
--kv-cache-dtype mxfp8 \
--block-size 512 \
--additional-config '{"c8_k_quant_mode":"per_token","c8_v_quant_mode":"per_token"}'
```

将 additional-config 字段合并到现有配置。移除原命令中的
`--kv-cache-dtype-skip-layers sliding_window`，否则 SWA 会按该显式选项继续使用 C16。
无需转换原 `out_full_c8` 权重，动态路径不使用校准的 V scale。

使用本次配套的独立 MTP draft 权重时，还需显式保留 draft 的 BF16 KV cache。
该 draft checkpoint 没有 `quantization_config`，其 `quant_config=None`，不会安装
ModelSlim 的动态 FP8 KV 后端。原来的 `sliding_window` skip 同时覆盖了 draft；
移除该 skip 后，需改为仅跳过 draft 对应的层索引：

```bash
--kv-cache-dtype-skip-layers 88
```

本权重有 88 个 target 层（25 个 full attention、63 个 SWA），索引 0～87 均使用 FP8 KV；
索引 88 对应独立 MTP draft，继续使用 BF16 KV。这里的 **88 依赖本权重的层数和 draft 层索引，
不是其他模型可直接沿用的通用常量**。

单 token SWA 通过仅补齐 query、丢弃 dummy 输出的方式绕过 CANN 9.2.1 的 sparse mode 限制。
实际 K/V 和 scale 仍仅按真实 token 写缓存；多 token MTP/prefill 使用原生 band 模式。
窗口采用现有 C16 的 `pre_tokens=W` 口径，即可见当前位置及之前 W 个位置。

## 独立 NPU 验证

在仓库根目录运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=. \
python tests/e2e/nightly/single_node/ops/singlecard_ops/test_c8_swa_per_token.py -v
```

该脚本使用真实生产 forward、DynamicQuant、scatter、FIA 和 sink kernel，不加载模型权重。
验证最短序列、Q=1/6/128、跨页窗口、混合 Q=1/Q=6 batch、缓存清零复用及输出/LSE 对照。
2026-11-18 在 `10.222.47.74` 的 `hxd_debug`、Ascend950DT_9582、CANN 9.2.1、
torch 2.10.0+cpu / torch_npu 2.10.0.post7 环境实测：

- 6 项 SWA NPU 测试全部通过，覆盖上述边界、sink、缓存写入和复用。
- raw NPUGraph 固定长度、修改 K/V 和 scale 的两次重放通过。
- 真实 UpdatableGraph 同图捕获 full attention 和 SWA，使用独立 workspace；
  KV 长度从 513 切换至 529，两层各自的 query 长度、输出和 LSE 对照通过。
- 原 full-attention 的 4 项动态 KV NPU 回归通过。
- 8 项独立 CPU 契约测试、482 项 pytest 回归通过。另有一项原有模型加载测试失败，
  已在未修改的 `54d59ba` 基线复现同样的 `npu_format_cast` CPU backend 错误，未修改无关代码。
- Ruff、格式、diff 和示例 shell 语法检查通过。

测试日志及代码 SHA256 记录在 `/mnt/sfs_turbo/hxd/debug/swa_fp8_20261118`。

尚未执行本分支整模型精度评测，也未测量端到端性能。
