# CPU Binding

## Overview

CPU binding is an **Ascend-native host-side optimization** for vLLM workers on
ARM servers. **Starting from vllm-ascend v0.18.0rc1, it is enabled by default
through `enable_cpu_binding=True`.**

The feature does not change model execution logic or numerical results. It only
controls CPU placement for the worker process, key runtime threads, memory
pages, and NPU IRQs when the host environment allows it. By keeping the main
worker, ACL, and release threads on dedicated CPU ranges, it **helps reduce
context-switch overhead from scheduler preemption on busy hosts.**

## Why CPU Binding?

On multi-socket ARM systems, the Linux scheduler may place worker threads on
CPUs far from the NPU that the worker drives. This can increase cross-NUMA
traffic, increase thread preemption, and introduce latency jitter. The Ascend
backend therefore owns a CPU allocation policy to **reduce cross-NUMA traffic,
reduce thread preemption, and improve latency stability** instead of relying on
upstream GPU NUMA binding flags.

This is also why upstream NUMA flags are adapted on Ascend:

- `--numa-bind` is converted to `additional_config={"enable_cpu_binding": true}`.
- `--numa-bind-nodes` and `--numa-bind-cpus` are ignored because Ascend computes CPU pools from NPU topology or global logical NPU IDs.

## How It Works?

The allocator derives its plan from runtime host state:

| Input | Source | Purpose |
| --- | --- | --- |
| Allowed CPUs | `/proc/self/status` `Cpus_allowed_list` | The only CPUs eligible for binding. Container cpusets are respected. |
| Logical NPU map | `npu-smi info -m` | Maps card/chip IDs to global logical NPU IDs and gives `total_logic_npus`. On Ascend 950, `Chip Logic ID` is not reported, so `NPU ID` is used as the logical ID. |
| Running NPUs | `npu-smi info` process table, filtered by `ASCEND_RT_VISIBLE_DEVICES` | Identifies the logical NPUs used by this worker process. A2/A3 process rows use `NPU Chip`; Ascend 950 process rows use `NPU ID`. |
| Topology affinity | `npu-smi info -t topo` | Provides NPU-to-CPU affinity for `topo_affinity` mode. |
| CPU NUMA map | `lscpu -e=CPU,NODE` | Used to extend single-NUMA affinity pools to the next NUMA node. |
| Thread topology | `lscpu` `Thread(s) per core` | Determines Ascend 950 cluster size: 8 CPUs for 1 thread per core, 16 CPUs for 2 threads per core. |
| UVB polling threads | `ps -Te` | Finds host `uvb_poll_window_thread` threads for Ascend 950 UVB CPU binding. Docker containers must use `--pid=host` to see these host threads. |

### Strategy Selection

The binding strategy is selected by Ascend device type:

| Device type | Strategy | Reason |
| --- | --- | --- |
| A3 | `global_slice` | A3 uses HCCS card-to-card interconnect. Each NPU is nearly equidistant from all NUMA nodes, so there is no strong NPU-to-NUMA affinity signal. Global logical NPU ID based slicing gives deterministic, non-overlapping CPU pools and CPU/NUMA isolation between workers. |
| Ascend 950 | `topo_affinity` | Ascend 950 uses NPU-to-CPU affinity from `npu-smi info -t topo` to choose an affinity NUMA node, then assigns one CPU cluster from that NUMA node to each worker. It also reports process rows by `NPU ID` instead of `NPU Chip`, skips IRQ binding, and binds host UVB polling threads. |
| A2 and Atlas 300 inference products | `topo_affinity` | A2 and Atlas 300 inference products provide NPU-to-CPU affinity information through `npu-smi info -t topo`, so they use this topology signal when available. |

If `topo_affinity` is selected but topo affinity is unavailable, the allocator falls back to `global_slice`.

### CPU Pool Construction

#### global_slice

`global_slice` is designed for devices without a useful NPU-to-CPU affinity
signal, including A3. Because A3's **HCCS interconnect makes the distance
from each NPU to each NUMA node nearly the same**, topology affinity is not a
useful placement signal. The allocator therefore partitions the sorted
`allowed_cpus` list by global logical NPU ID.

1. Determine `total_npus` in this order:
   - `total_logic_npus` from `npu-smi info -m`
   - number of topo affinity entries
   - number of running NPUs
2. Compute:
   - `base = len(allowed_cpus) // total_npus`
   - `extra = len(allowed_cpus) % total_npus`
3. Each logical NPU gets a deterministic slice:
   - NPU IDs `< extra` receive `base + 1` CPUs.
   - Remaining NPU IDs receive `base` CPUs.
4. Only running NPUs are materialized into `npu_cpu_pool`.

This is the key property: two independent worker processes with the same cpuset
but different visible NPU IDs still get **non-overlapping CPU pools** because
both processes slice against the same global NPU ID space. With a NUMA-aligned
cpuset, this also provides **CPU/NUMA isolation between workers**, so one worker
does not share the same CPU or NUMA slice with another worker.

`global_slice` requires enough CPUs for the selected device's role split:

- Devices with IRQ binding require `base >= 5`:
  2 CPUs for SQ/CQ IRQ binding, at least 1 CPU for the main worker, 1 CPU for
  ACL thread, and 1 CPU for release thread.

#### topo_affinity

`topo_affinity` is designed for A2, Atlas 300 inference products, Ascend 950,
and other non-A3 device types. A2 and Atlas 300 inference products expose
**meaningful NPU-to-CPU affinity information**, so the allocator starts from NPU
topology affinity when it is available and then avoids overlap for shared
affinity groups.

1. Build candidate NPUs from all logical NPUs:
   - always include running NPUs
   - include non-running NPUs only when their affinity overlaps this process's allowed cpuset
2. For each candidate NPU, intersect topo affinity with `allowed_cpus`.
3. If the intersection is empty for a candidate, binding fails for this rank.
4. If the affinity CPUs are all on one NUMA node, extend the pool with CPUs from the next NUMA node, constrained by `allowed_cpus`.
5. Group NPUs with identical extended pools and split each shared pool evenly across that group.
6. Keep only running NPUs in the final `npu_cpu_pool`.

The non-running candidate step is intentional. It prevents two independent
single-card workers from selecting the same CPU range when their visible NPUs
share the same topology affinity.

For Ascend 950, topology affinity is used differently:

1. Bind all visible host `uvb_poll_window_thread` threads to NUMA0 CPUs except CPU0, constrained by `allowed_cpus`. Docker containers must use `--pid=host` to make these host threads visible.
2. Use topo affinity to identify each NPU's single affinity NUMA node.
3. Parse `Thread(s) per core` from `lscpu` and set cluster size to 8 CPUs when it is 1, or 16 CPUs when it is 2.
4. Split each affinity NUMA's sorted allowed CPU list into contiguous clusters.
5. Assign clusters by sorted logical NPU ID, including hidden NPUs that share the same affinity NUMA.
6. Keep only running NPUs in the final `npu_cpu_pool`.

If Ascend 950 topo affinity is missing, spans multiple NUMA nodes, has too few
clusters, or reports an unsupported `Thread(s) per core`, worker CPU binding is
skipped without raising to the worker process.

### Role Split

After a CPU pool is built, the allocator splits it by role:

For devices with IRQ binding:

| Role | CPUs |
| --- | --- |
| SQ/CQ IRQ | `pool[0]`, `pool[1]` |
| Main worker process and subthreads | `pool[2:-2]` |
| ACL thread | `pool[-2]` |
| Release thread | `pool[-1]` |

For Ascend 950:

| Role | CPUs |
| --- | --- |
| Main worker process and subthreads | the whole assigned cluster |
| ACL thread | not separately pinned |
| Release thread | not separately pinned |

If a final pool has fewer CPUs than the selected role split requires, binding
fails for this rank and the worker logs a warning from the caller. The minimum
is 5 CPUs per NPU for devices with IRQ binding. Ascend 950 requires one full
cluster per worker.

When `BIND_LAYERWISE_CONNECTOR=1`, the last two CPUs of the main range are moved
to the layerwise KV connector transport threads, so the main range shrinks by
two CPUs:

| Role | CPUs |
| --- | --- |
| SQ/CQ IRQ | `pool[0]`, `pool[1]` |
| Main worker process and subthreads | `pool[2:-4]` |
| Layerwise KV connector transport threads | `pool[-4:-2]` |
| ACL thread | `pool[-2]` |
| Release thread | `pool[-1]` |

On Ascend 950 the same reservation happens at the end of the assigned cluster:
main keeps `cluster[:-2]` and the transport threads get the last two cluster
CPUs. If the main range would be left with no CPU, the reservation is skipped
with a warning and the previous split is kept, so enabling the switch never
fails worker startup.

#### Layerwise KV Connector Transport Threads

The layerwise KV connectors move KV cache in worker-process Python threads:
`KVCacheSendingLayerThread` (prefill side, D2H flush), `KVCacheRecvingLayerThread`
(decode side, TCP pull plus H2D), `KVCachePipeWriter` (pipelined writes) and
`KVTransferParamsRecvingThread` (prefill-then-decode parameter channel).
Without a reservation these threads share the main CPU range with forward,
HCCL and other runtime threads, so a busy host can preempt the send/receive
thread exactly when a layer is in flight.

Two properties of these threads shape the implementation:

- They are started during `initialize_from_config` (`register_kv_caches`), which
  runs before `compile_or_warm_up_model` where `bind_cpus` is called. A thread
  that pins itself at startup would be overwritten by the bulk
  `taskset -acp` of the main binding, so the transport threads must be pinned
  *after* that bulk bind, not by themselves.
- Their OS thread names are inherited from the creating thread, so all worker
  threads look alike in `/proc/<pid>/task/*/comm` and `ps -Te`. They are matched
  by Python thread name inside the worker process instead, and pinned by native
  thread ID.

The reservation is therefore applied in `bind_layerwise_connector_threads`,
which runs after the main `taskset -acp`. Every matched thread receives the
whole reserved set, and no other role includes those CPUs, so the send/receive
threads never share a CPU with the other worker threads or with the ACL/release
threads. If no matching thread exists (for example the flag was enabled on an
engine that does not run the layerwise connector), the CPUs stay idle for that
worker and a warning is logged.

## Conditional Host Tuning

After CPU affinity is applied, CPU binding can also apply two host-side tuning
steps when the environment supports them:

- Memory migration uses `migratepages` to move the worker process's existing
  pages to the selected NUMA node. This keeps the worker closer to the memory it
  reads and reduces remote-NUMA memory read latency.
- IRQ binding places NPU IRQ handling on the CPUs reserved for the corresponding
  NPU when `/proc/irq` is writable and IRQ files can be resolved.
  Ascend 950 skips this step.

These are conditional parts of CPU binding, not separate feature switches. If a
host prerequisite is missing, that step is skipped while CPU thread binding
still proceeds. Missing `migratepages` can still leave pages on remote NUMA
nodes, so **latency or throughput may regress compared with a full CPU binding
setup.**

## Examples

### A3 inference server with 640 CPUs and 16 NPUs

Inputs:

- `allowed_cpus = [0..639]`
- `total_logic_npus = 16`
- `running_npu_list = [0..15]`

Computation:

- `base = 640 // 16 = 40`
- `extra = 0`
- Worker `i` driving logical NPU `i` receives CPU slice
  `[i * 40 .. i * 40 + 39]`.

Global slice view:

```text
CPU range: 0                                                             639
           |-- worker0/NPU0 --|-- worker1/NPU1 --| ... |-- worker15/NPU15 --|
           |      0-39        |      40-79       | ... |      600-639       |
```

Role split inside each worker slice:

```text
40-CPU worker slice
| IRQ CPUs | main worker process and subthreads | ACL thread | release thread |
|  c0-c1   |              c2-c37                |    c38     |      c39       |
```

Concrete examples:

| Worker | Logical NPU | CPU pool | IRQ CPUs | Main CPUs | ACL CPU | Release CPU |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0-39 | 0-1 | 2-37 | 38 | 39 |
| 1 | 1 | 40-79 | 40-41 | 42-77 | 78 | 79 |
| ... | ... | ... | ... | ... | ... | ... |
| 15 | 15 | 600-639 | 600-601 | 602-637 | 638 | 639 |

This layout remains deterministic even when different worker processes share
the same cpuset, because slicing is based on the global logical NPU ID. With
`BIND_LAYERWISE_CONNECTOR=1`, worker 0 keeps `2-35` for main and hands `36-37`
to its layerwise KV connector transport threads; the other workers shift the
same way inside their own slice.

### A2 topo_affinity with hidden same-affinity NPUs

Inputs from an A2 topology:

- NPU0 affinity: 144-167
- NPU2 affinity: 144-167
- Process A sees only NPU0
- Process B sees only NPU2
- Both processes have `allowed_cpus = [144..191]`

The allocator includes the hidden same-affinity NPU as a candidate in each
process, splits the shared extended pool, and then keeps only the visible NPU in
the final pool.

Final pools:

| Process | Visible NPU | Final CPU pool |
| --- | --- | --- |
| A | 0 | 144-167 |
| B | 2 | 168-191 |

This avoids overlapping CPU pools even when the two workers are launched as independent single-card services.

## Logs

The allocator logs the selected mode and allocation plan:

```text
[cpu_bind_mode] mode=topo_affinity rank=0 visible_npus=[0]
The CPU allocation plan is as follows:
NPU0: main=[...] acl=[...] release=[...]
```

Ascend 950 uses a different role split, so its plan log does not include ACL or
release fields. UVB polling thread binding is reported separately when matching
threads are found:

```text
[cpu_bind_mode] mode=topo_affinity rank=0 visible_npus=[0]
The CPU allocation plan is as follows:
Ascend 950 NPU0: worker=[...]
[cpu_bind_ascend_950] uvb_poll_window_thread tids=[...] cpus=[...]
```

With `BIND_LAYERWISE_CONNECTOR=1`, the plan line reports the reserved CPUs and
the transport threads pinned to them:

```text
The CPU allocation plan is as follows:
NPU0: main=[...]  acl=[...]  release=[...]  layerwise=[36 37]
[cpu_bind_layerwise] NPU0: cpus=[36 37]  threads=[KVCacheRecvingLayerThread(tid=...)]
```

## Limitations

- CPU binding runs only on ARM. It is skipped on x86_64.
- Each final NPU pool must have enough CPUs for its role split: at least 5 CPUs
  for devices with IRQ binding. Ascend 950 requires one complete CPU cluster per worker.
- `global_slice` is deterministic and provides CPU/NUMA isolation when the
  cpuset is NUMA-aligned, but it cannot guarantee NUMA-local pools when CPU
  numbering or cpuset layout crosses NUMA boundaries.
- `topo_affinity` depends on usable output from `npu-smi info -t topo`.
- IRQ binding requires writable `/proc/irq` and resolvable PCI/IRQ information.
  Ascend 950 skips IRQ binding even when `/proc/irq` is writable.
- Ascend 950 UVB polling thread binding requires visibility into the host PID
  namespace. Docker containers must be created with `--pid=host`; otherwise
  `uvb_poll_window_thread` may not be found.
- Memory migration requires `migratepages`; otherwise only memory migration is
  skipped. CPU affinity still applies, but performance may degrade because
  existing pages are not moved to the target NUMA node and may be read through
  higher-latency remote NUMA access.
- Layerwise connector transport threads are matched by Python thread name in the
  worker process. Threads created after `bind_cpus` runs are not covered, and
  reserved CPUs stay idle on workers that run no layerwise connector. The two
  reserved CPUs are a fixed split; `LAYERWISE_CONNECTOR_CPUS` in
  `vllm_ascend/cpu_binding.py` is the single place to change it.
- If an exception escapes the binding flow, `NPUWorker` logs a warning and skips CPU binding for that rank.

## References

- Implementation: `vllm_ascend/cpu_binding.py`
- Worker integration: `vllm_ascend/worker/worker.py`
- Config: `vllm_ascend/ascend_config.py` and `docs/source/user_guide/configuration/additional_config.md`
- Tests: `tests/ut/device_allocator/test_cpu_binding.py`
