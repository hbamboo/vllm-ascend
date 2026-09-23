# KV 传输时间线解析器

`parse_h2h_chain.py`：把 P/D 两侧的 H2H 逐层传输日志解析成一条可交互的 HTML 时间线
（泳道：受理→P / P 模型 / P D2H / H2H / D H2D / D 模型 / 首token 回传 / LAYER_DONE 往返）。

日志来源与打点约定见 `vllm_ascend/distributed/kv_transfer/utils/h2h_perf.py`：
- 新格式：`[h2h][perf] {"kind":...,"ts":<CLOCK_MONOTONIC>,"wall":"<ISO 毫秒>",...}`，由 `MC_TCP_PERF_LOG=1` 门控；
- 旧文本格式（`[mooncake][perf] P batch=...`）仍可解析，旧日志能原样重渲染。

用法（生成侧脚本在各 test_script 目录，例如 `test_script/layerwise_mamba/gen_122b_timeline.py`）：

```bash
python3 gen_122b_timeline.py <logs_dir> <out.html> --p-worker Worker_TP0_EP0 --d-worker Worker_DP
```

生成时会打印每个日志文件的 `行=/json/旧文本/未解析` 计数：**未解析必须为 0**，否则说明打点字段
与解析器不同步（时间线会静默少线）。`reqs=0` 时脚本会拒绝覆盖已有成品页并改写 `<out>.empty`。

注意：`P/D 不等分`(如 P tp4 → D tp2×dp2) 时同一层批会被对端多个 rank 各收一次，本解析器按
"同一 rank 参考行"给出行内数字（H2H read / H2D / LAYER_DONE 折线同源），多 rank 的聚合值
(max 耗时 / min 起点) 只在 tooltip 里作对照。
