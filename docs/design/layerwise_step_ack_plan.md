# 逐层 KV 传输「取消批间 ACK 等待」可行性判断与开发计划

- 状态：阶段 B 已实现（分支 `h2h_dev_p_then_d_ack_opt`，开关 `MC_TCP_STEP_ACK=1`）
- 范围：`MooncakeLayerwisePrefillThenDecodeConnector`（p_then_d，protocol=tcp）
- 目标读者：P/D 传输与调度同学

## 0. 结论（先读）

**可行，但要把“所有层”明确为“一个 forward step 的 layer sweep”，不能是“整个请求跨 chunk 的所有层”。** 建议分两步落地：

1. **第一阶段（推荐）**：保留每条 LAYER_DONE 的 ACK，但 P 侧**不再逐批等待**；同一 step 内连续发送，step 边界统一 drain。D 侧协议基本不变，风险最低。
2. **第二阶段（可选）**：改成“单条 step ACK”（D 侧等该 step 所有 job 完成后再回一个 ACK/NAK），需要 DEALER/ROUTER + `step_id/is_step_last` + D 侧按 step 累计完成。

同时纠正一个前提：**ACK 保护的不是 P 的 NPU KV 块，而是 staging slot 的复用**。D2H flush 是同步的，返回时数据已经在 staging；P 的 NPU 块理论上 flush 后就可以释放（当前延迟释放到 ACK 之后是保守做法）。真正不能提前覆盖的是 staging 中每个 layer 的固定 slot——下一个 step 的同层 flush 会覆盖它。

三个目标的可行性：

| 目标 | 是否满足 | 说明 |
|---|---|---|
| 1. staging/KV 在传输完成前不被覆盖 | **step 级可以**；请求级不行 | 同一 step 内各批层槽位不同，不需要互等；跨 step 的 layer 0 复用需要 step 边界 drain。跨 chunk 的请求会复用同一 step 的槽位，不能把 ACK 拖到请求结束。 |
| 2. 请求级完成/首 token/延迟释放 | 可以，但时机会变 | D 的 `done_list` 本来就是本地应用；P 的首 token/`finished_sending` 会从“批 ACK 后”推迟到“step 末 ACK 后”。多请求合批时，早结束的请求首 token 会晚一点，需要实测 TTFT。 |
| 3. D 侧重试、超时 NAK、重算 | 可以 | 需要在 `_run_recv_job` 加有界重试；mooncake 默认 `MC_TRANSFER_TIMEOUT=30s`（最小 5s），要配短超时或 watchdog；最终 NAK 走现有的 `update_failed_task` → recompute 路径。 |

## 1. 现状与瓶颈

当前 pull 模式每批（`MC_TCP_LAYER_BATCH=4`）的时序：

```text
P: flush N -> LD N -> 等 ACK N
D:                    pull N -> H2D N -> ACK N
P: flush N+1 -> LD N+1 -> ...
```

`_collect_layer_done_ack` 在批末阻塞（connector.py:1085-1099）；pipe 路径同样在 `_pipe_finalize` 里逐 job 收齐（1459-1472）。这个等待同时挡住了下一批的 flush 和 LD。

实测（`p0_no_mtp.log` / `d0_no_mtp.log`，P TP4→D TP4、4 层/批）：

| 指标 | 数值 |
|---|---|
| `pull` | p50 ~3.8ms，mean ~4.9ms |
| `h2d_only` | p50 ~17.5ms，mean ~16ms |
| P `flush` | ~1.4–1.5ms |
| 每批串行尾段 | flush + pull + h2d ≈ **22–24ms** |
| 每 step 批数 | ~12（48 层 + MTP）/ 4 |

D 侧 `h2d` 是瓶颈；当前 NIC 在 H2D 的 ~17ms 里基本空闲，pull 和下一次 flush 都被 ACK 串行化。

## 2. 目标与非目标

**目标**

- 同一 step 内 P 连续 flush/发送 LAYER_DONE，不再逐批等 ACK；
- step 边界统一 drain，保证下一 step 的 layer 0 不会覆盖未读完的 staging；
- D 侧对 pull 失败做有界重试，最终失败回 NAK，P 走现有请求级失败/重算；
- 默认关闭、可回滚，不影响旧的逐批 ACK 路径。

**非目标**

- 跨 step 的多代 staging（需要多缓冲整套 KV staging，内存代价大）；
- 把 H2D 改成多流（本文只做控制面流水，H2D 仍单流 FIFO）；
- 改变 D 侧重算逻辑；
- 支持“整个请求所有 chunk 一个 ACK”。

## 3. 方案设计

### 3.1 两阶段形态

**阶段 A（推荐先做）：per-message ACK 保留，P 不等待**

- P 侧控制面改成“多 outstanding”：每 peer 维护一个 REQ socket 池（深度 K，例如 4）或一个 DEALER；`_post_layer_done` 取一条空闲 socket 发送并记录到 `sent_ld`，不在批末阻塞。
- 同一 step 内持续发送，直到 `sent_ld` 数量达到 socket 池深度（背压）；收到 ACK 后归还 socket。
- **step 边界**（batch 含 `total_layers-1`，或调度侧给的 step 结束标记）统一 drain：收齐本 step 所有 ACK/NAK，再允许下一 step 的 layer 0 flush。
- D 侧协议不变（每条 LAYER_DONE 一个 ACK），只是 P 不再逐条等；改动集中在 P。

**阶段 B（可选，“单条 step ACK”）**

- P 用 DEALER（或 ROUTER/DEALER 组合），payload 增加 `step_id`、`is_step_last`、`seq`；中间 LAYER_DONE 只发不等回。
- D 侧记录每个 `(sender_path, step_id)` 的 job 完成计数；收到 `is_step_last` 的 job 且该 step 所有 job 都完成后，回**一个** `ACK`；任一 job 失败则回 `NAK`（带失败的 req_ids）。
- 收益与阶段 A 基本相同，但失败检测和首 token ACK 都被推迟到 step 末；协议和状态机复杂度明显更高。建议阶段 A 验证收益后再决定是否做 B。

### 3.2 数据流（阶段 A）

```text
P step S:
  batch 0: flush -> LD0 (socket #0)
  batch 1: flush -> LD1 (socket #1)
  ...
  batch B-1: flush -> LDB-1
  step 末: drain 所有 ACK/NAK，才允许 step S+1 的 layer 0 flush
D:
  收到 LD 即刻 pull + H2D；每批完成回 ACK
  pull/H2D(N+1) 与 H2D(N) 在 H2D 流上排队，pull 与 H2D 重叠
```

关键不变量：

- **同一 step 内，两条 outstanding LAYER_DONE 不得命中同一 layer 的 staging slot**。同一 step 的层是唯一的，天然满足；加断言防守。
- **跨 step 必须 drain**：layer 0 的 slot 在下一 step 复用；drain 的是“上一 step 所有已发 LD 的 ACK/NAK”，不是“最后一条”。
- **staging 不因流水而增加**：slot 是按 layer 固定的；同一 step 内不同层本来就是不同地址。

### 3.3 step 结束判定（关键设计点）

不能只用 `layer_idx == total_layers-1`：某个 rank 可能**没有最后一层的数据**，发送队列里根本不会出现该 task，`contains_last` 不成立，drain 就永远不触发。可选做法：

- 调度侧 connector 在每个 step 的 `metadata` 里带一个 `step_id` / `step_end` 标记，worker 把它塞进每个 SendTask（或单独发一个 sentinel task）；
- 或者用 `save_kv_layer` 的调用次数/层号回绕检测：`layer_idx` 回绕到 0 时，先 drain 上一 step；
- 最稳妥是调度侧显式给 step 边界，不依赖“本 rank 是否有末层数据”。

### 3.4 失败与超时语义

- **D 侧重试**：在 `_run_recv_job` 里对 `batch_transfer_sync_read` 做有界重试（建议 2 次，固定/指数退避），每次失败后整段重读（覆盖 D staging 中可能写了一半的内容）。P staging 在 step drain 前是稳定的，重读安全。
- **D 侧超时**：mooncake 的同步读超时是 `MC_TRANSFER_TIMEOUT`（默认 30s，最小 5s）+ `total_length`（1GiB/s 余量）；需要把 `MC_TRANSFER_TIMEOUT` 调到可接受值（例如 5–10s），或在 D 侧加 watchdog 线程；不能让一次 read 卡 30s 才 NAK。
- **最终失败回 NAK**：`_finish_recv_job` 回 `NAK`（现有路径），P 把该消息的 `req_ids` 加入 `failed_reqs`，停止为这些请求发送后续 LD，走现有 `_send_failed_signal` / `on_request_failed`；D 侧 `update_failed_task` → scheduler 重算。
- **NAK 与 in-flight 批**：阶段 A 下 P 可能已经发出了后续批；收到 NAK 后要能丢弃/忽略这些请求的后续段，D 侧也要把同一请求的后续 job 标失败/丢弃，避免“先失败后又应用 done_list”。
- **step 边界超时**：drain 必须有总超时（沿用 `LAYER_DONE_TIMEOUT_S`），超时即把本 step 未 ACK 的请求判失败并释放 staging，不能让 P 永远卡在 step 末。

### 3.5 P 侧请求级簿记

- **首 token**：当前在 group 末批 ACK 后发（connector.py:1174-1178）。阶段 A 下可以在 step 内**非阻塞地**处理到达的 ACK，group 末批 ACK 一到就发首 token，不必等 step 末；阶段 B（单 step ACK）会把它推迟到 step 末，需要实测 TTFT。
- **延迟释放**：当前 `finished_sending` 在 ACK 后上报。ACK 保护的是 staging，不是 NPU 块；如果确认 D 只读 staging，可以在 **D2H flush 完成后**就上报释放 NPU 块（可选优化，独立于本改造）。保守起见第一阶段保持现状。
- **`failed_reqs` / `on_send_done`**：需要支持多个 outstanding 批，按消息关联到请求；失败后停止该请求的后续发送。

## 4. 代码改动清单

| 模块 | 改动 |
|---|---|
| P `KVCacheSendingLayerThread` | `_peer_socks` 改为每 peer 的 REQ socket 池（或 DEALER）；`_post_layer_done` 取 socket、不阻塞；`_collect_layer_done_ack` 改为非阻塞/批量 drain；`sent_ld` 跨批保留；`_transfer_kv_cache_batch` / `_pipe_finalize` 去掉批末 collect，改为 step 边界 drain；新增 step-end 标记 |
| P 失败处理 | NAK 时停止后续发送；step drain 超时清理；`failed_reqs` 与 outstanding 批关联 |
| D `KVCacheRecvingLayerThread` | `_run_recv_job` 有界重试；最终失败 NAK；`_recv_jobs` 队列/池容量与背压；同一请求后续 job 的失败传播 |
| 协议 | 阶段 A 基本不改 payload；阶段 B 增加 `step_id/is_step_last/seq`，DEALER/ROUTER 空帧兼容 |
| envs | 新增 `MC_TCP_STEP_ACK`（默认 0）、`MC_TCP_LD_RETRY`、`MC_TCP_STEP_TIMEOUT_S`；复用 `MC_TRANSFER_TIMEOUT` |
| 测试 | UT：多 outstanding、乱序 ACK、NAK、step drain、staging slot 断言、混合版本；E2E：准确率、TTFT、故障注入 |

## 5. 开发计划

| 阶段 | 任务 | 工时 | 交付 |
|---|---|---|---|
| P0 | 设计评审、协议/状态机、step-end 标记方案、失败语义定稿 | 1d | 设计文档 + 评审结论 |
| P1 | P 侧多 outstanding（socket 池/DEALER）、非阻塞 ACK/drain、step 边界、staging 断言 + UT | 3d | 可开关的阶段 A 实现 + UT |
| P2 | D 侧有界重试、短超时/watchdog、NAK 传播、in-flight 失败处理 + UT | 2d | D 侧重试/失败路径 + UT |
| P3 | 故障注入（断连、超时、坏地址、NAK）、混合版本/回滚验证、打点 | 2d | 故障注入用例 + 指标 |
| P4 | 上机 A/B（pull/push、单/多请求、1P1D 与 DP2×TP2）、准确率、调参 | 3d | A/B 报告 + 结论 |
| 合计 | | **11d**（含缓冲 13d） | |

## 6. 收益预估

以实测每批 pull ~4.9ms、H2D ~16–17.5ms、flush ~1.5ms、每 step ~12 批为基准：

| 方案 | 每批尾段 | 每 step（12 批） | TTFT（p50 ~968ms） |
|---|---|---|---|
| 现状（逐批 ACK） | flush + pull + h2d ≈ 22–24ms | ~264–288ms | 基线 |
| 阶段 A/B 流水（保守） | max(h2d, pull) + 小量 ≈ 17–18ms | ~204–216ms | **-6% ~ -8%**（省 ~60–80ms） |
| 乐观（P 前向也被解锁） | 近似 H2D 主导 | ~180–200ms | **-10% ~ -15%**（省 ~100–150ms） |

分解：

- 直接收益：pull（~4.9ms）+ flush（~1.5ms）不再被 H2D 串行化，约 **6.4ms/批 × 12 ≈ 77ms/step**；这是最确定的部分。
- 额外收益：当前 send_queue 很小（1–2），P 的模型前向也被 ACK 反压；流水后 P 可在 step 内跑在前面，step 时间从 `Σ(模型+传输)` 趋向 `max(模型, H2D)`，乐观可多省 40–70ms。
- 网络利用率：当前 NIC 在 H2D 的 16–17.5ms 里基本空闲；流水后 pull 填进去，但瓶颈会转到 H2D（单流 FIFO），所以端到端收益上限是 H2D 之外的那段时间。
- D 侧重试：每次避免“整请求重算”，可省一次完整 prefill（该请求 TTFT 约 1s 量级）；收益取决于瞬时故障率，需实测，不计入上表。
- push 模式：pipe writer 已经把网络写与 H2D 重叠，收益小于 pull；需要分别 A/B。

**注意**：上表是单 step/单请求的理想化；TTFT 还包含调度、排队、模型前向等，实际收益可能更低。建议以 p50/p90 TTFT 和 `p_batch/d_batch` 的 `pull/h2d/flush` 窗口交叉验证。

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| step 结束判定漏触发（本 rank 无末层数据） | 调度侧显式 step_id/sentinel；layer 回绕兜底；UT 覆盖 |
| staging 槽位被跨 step 覆盖 | step 边界 drain + “同 step 同层 slot 不重复”断言；drain 超时则判失败 |
| NAK 到达时还有 in-flight 批 | 按 req_id 维护失败集合，停止后续发送；D 丢弃同请求后续 job；UT |
| 协议/版本不兼容 | `MC_TCP_STEP_ACK` 默认 0；两端同版本灰度；DEALER 空帧兼容；保留旧 REQ 路径 |
| D 侧 read 卡 30s | `MC_TRANSFER_TIMEOUT` 调小或 watchdog；重试有总预算 |
| H2D 单流仍是瓶颈 | 本改造只解决控制面串行；H2D 多流/event 另立项目 |
| P 侧内存在飞增加 | socket 池深度/背压；step 内批数有界；监控 `sent_ld` 长度 |
| 首 token 推迟（阶段 B） | 阶段 A 非阻塞处理 ACK；阶段 B 实测 TTFT 再决定 |

## 8. 验收标准

- 准确率：与直连/P 直出逐 token 一致，0 乱码/复读；多请求合批、chunked prefill、hybrid 模型全覆盖。
- 故障：断连/超时/坏地址/NACK 注入均能在有界时间内失败并触发重算，无挂死、无 staging 覆盖。
- 性能：pull 与 H2D 在时间线上重叠；TTFT p50/p90 相对基线 ≥5%（目标 8%）；TPOT/吞吐无回退；`p_batch` 的 `layerdone`/`misc` 与 `d_batch` 的 `pull`/`h2d` 可解释。
- 稳定性：长跑无内存增长、无队列无界增长、无 step drain 卡死；`MC_TCP_STEP_ACK=0` 可完全回退旧行为。

## 9. 开放问题

1. 阶段 A（per-message ACK 不等待）是否已经足够拿到全部收益？若是，阶段 B 的单 step ACK 可以不做。
2. 首 token 时机：step 内非阻塞 ACK 是否足以保持现有 TTFT？阶段 B 的延迟是否可接受？
3. 延迟释放是否要解耦到 D2H flush 之后（安全但改变现有语义）？
4. D 侧 H2D 多流/每 worker 独立流是否作为后续项目？
5. 多请求合批时，step 结束的判定以“模型最后一层”为准是否覆盖所有 rank 的发送情况？

**一句话**：方案可行，推荐先做“step 级屏障 + 批间不等待（ACK 仍逐条回、P 不逐条等）”，把 D 侧有界重试和 NAK 作为第二阶段；单条 step ACK 可作为后续优化。保守收益约 60–80ms/step（TTFT -6%~-8%），乐观 100–150ms（-10%~-15%），需上机 A/B 确认。

## 10. 实现记录（分支 `h2h_dev_p_then_d_ack_opt`）

已按阶段 B 落地，旧路径保留：

- 回退 P 侧并发改造（`62f66bf8d`、`e6bc5faf2`），保留 D 侧 recv pool
  （`10c9be7f7`）、`finished_sending` 过滤（`4a42d1526`）、H2D event（`acfa27ae0`）。
- 新增 `MC_TCP_STEP_ACK`（默认 0）：
  - P 每 peer 一条常驻 DEALER；同一步内连续发 LAYER_DONE，不等中间 ACK；
  - payload 追加 `seq/step_id/step_last`（11 元组）；
  - step 边界（`is_last`、层号断裂、或末批已满后下一 step 的第一个 task）统一 drain；
  - 本批没有数据的 peer 在 step 末补一条空 LAYER_DONE 作为 step_last 标记；
  - D 侧按 `(sender_path, step_id)` 累计完成，收到 step_last 且全部 job 完成后
    回唯一 `ACK`/`NAK`（带 step_id 与失败的 external req ids）；
  - P 的 `failed_reqs` 用 external→internal 映射回填；首 token/on_send_done/
    on_transfer_done 推迟到 step ACK 之后，避免失败重算复用旧首 token。
- 新增 `MC_TCP_LD_RETRY`（默认 2）：D 侧 `batch_transfer_sync_read` 有界重试，
  仍失败才回 NAK；`MC_TRANSFER_TIMEOUT` 需按部署调小（默认 30s，最小 5s）。
- 新增 UT `tests/ut/distributed/mooncake/test_layerwise_step_ack.py`：
  单 step 只回一个 ACK、乱序完成不提前 ACK、NAK 回填 internal id、空 step 标记、
  旧 8 元组协议不变。既有 recv pool / 上报过滤 / 延迟释放单测通过（29 passed）。
- 已知限制：P/D 必须同时升级（旧 D 不识别 DEALER 的 11 元组）；未上机验证，
  收益需按第 6 节口径做 A/B。
