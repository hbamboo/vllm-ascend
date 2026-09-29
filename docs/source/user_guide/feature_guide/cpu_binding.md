# CPU Binding

**Starting from vllm-ascend v0.18.0rc1, CPU binding is enabled by default on
ARM-based Ascend servers.**

**You usually do not need to configure it manually.** Set `enable_cpu_binding`
only when you want to disable it or make the default explicit.

## Benefits of CPU Binding

CPU Binding improves **host-side scheduling** for multi-socket ARM servers with
Ascend NPUs. It is designed to solve three common host-side inference performance issues:

- **Lower cross-NUMA traffic.** Worker processes stay closer to the CPU and
  memory resources selected for their active NPU, reducing remote NUMA access.
- **Lower context-switch overhead from thread preemption.** Key runtime threads
  run on stable CPU ranges, reducing scheduler movement and CPU contention on
  busy hosts.
- **Better latency stability and multi-worker isolation.** Independent workers
  avoid sharing the same CPU/NUMA resources, which helps reduce tail-latency
  jitter and makes throughput more predictable during multi-NPU serving.

This feature is a host-side performance optimization. **It does not change model
execution logic or numerical outputs.** When memory migration support is
unavailable, CPU affinity still works, but memory locality may be worse and
latency or throughput may degrade.

## Usage

### Online Serving

Default behavior:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

Disable CPU binding:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --additional-config '{"enable_cpu_binding": false}'
```

### Offline Inference

Default behavior:

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
```

Disable CPU binding:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    additional_config={"enable_cpu_binding": False},
)
```

## Layerwise KV Connector Transfer Threads

**This switch is off by default.** Turn it on only on engines that use a
layerwise KV connector (the PD disaggregation path where KV cache is streamed
layer by layer), and only when CPU binding is enabled.

Layerwise KV transfer runs in worker threads (`KVCacheSendingLayerThread` on the
prefill side, `KVCacheRecvingLayerThread` on the decode side, plus the pipelined
write and parameter-channel threads). By default those threads share the same
CPUs as the model forward and other runtime threads, so a busy host can preempt
a send/receive thread while a layer is in flight.

With `BIND_LAYERWISE_CONNECTOR=1`, CPU binding takes the **last two CPUs of the
worker's main range** and gives them to those threads only. Every other thread
of the worker keeps a mask without these two CPUs, so the transfer threads and
the compute threads never run on the same CPU:

```bash
export BIND_LAYERWISE_CONNECTOR=1
vllm serve <your-model> --kv-transfer-config '...'
```

On a 320-CPU host with 16 NPUs (20 CPUs per worker, 16 of which form the main
range by default), one worker is split like this:

```text
20-CPU worker slice
| IRQ  |            main range (14)          | layerwise | ACL | release |
| c0-1 |              c2-c15                 |  c16-c17  | c18 |   c19   |
```

The allocation plan log shows the reserved CPUs, and a second line reports which
threads were pinned to them:

```text
NPU0: main=[2 3 ... 15]  acl=[18]  release=[19]  layerwise=[16 17]
[cpu_bind_layerwise] NPU0: cpus=[16 17]  threads=[KVCacheRecvingLayerThread(tid=...)]
```

Notes:

- The two reserved CPUs leave the main range, so the worker has two fewer CPUs
  for forward and other threads. If the remaining main range would be empty, the
  reservation is skipped and the previous split is kept.
- Reserved CPUs stay idle on workers that do not run a layerwise connector, so
  do not set this switch globally for engines that use other KV connectors.
- The decode side has one transfer thread that performs both receive and H2D
  copy; it receives both reserved CPUs.

## Requirements

Official vllm-ascend images have already included `util-linux` and `procps` /
`procps-ng` in v0.18.0rc1 and earlier releases. **Starting from v0.18.0rc1, the
official images also include `numactl`.**

If you are not using the official image, install the host tools manually:

```bash
# Ubuntu/Debian
sudo apt-get install -y util-linux numactl procps

# RHEL/CentOS/Alma/Rocky
sudo yum install -y util-linux numactl procps-ng

# openEuler
sudo dnf install -y util-linux numactl procps-ng
```

**Without `numactl` / `migratepages`, vLLM Ascend skips only memory migration.**
The worker process and runtime threads are still pinned, but pages already
placed on remote NUMA nodes are not migrated, which **can reduce locality and
degrade latency or throughput.**

For optimal locality, use a cpuset that is evenly distributed across NUMA
nodes. Unbalanced cpusets may reduce the locality benefit of CPU binding.

On Ascend 950, CPU binding uses NPU-to-CPU affinity from `npu-smi info -t topo`
to select the worker's affinity NUMA node. Each worker main process is pinned to
one CPU cluster from that NUMA node. The cluster size is derived from `lscpu`
`Thread(s) per core`: 8 CPUs when it is 1, and 16 CPUs when it is 2. Ascend 950
also pins host `uvb_poll_window_thread` threads to NUMA0 CPUs except CPU0,
constrained by the current cpuset. In Docker deployments, add `--pid=host` when
creating the container so vLLM Ascend can discover and bind these host threads.
Ascend 950 still can migrate memory pages when `migratepages` is available, but
it does not separately pin ACL/release threads and does not apply IRQ binding.

For IRQ binding, the process also needs permission to read `/proc/interrupts`
and write `/proc/irq/*/smp_affinity`. If `irqbalance` is running and the process
can use `systemctl`, vLLM Ascend stops it before applying IRQ affinity. In
containers where `systemctl` is unavailable, stop `irqbalance` on the host when
IRQ affinity matters.

Ascend 950 does not apply IRQ binding. When running on Ascend 950, the log contains
`[irq] IRQ binding skipped on Ascend 950.` and no `/proc/irq/*/smp_affinity` files are
written by this feature.

Ascend 950 allocation logs use `worker=[...]` instead of `acl=[...]` or
`release=[...]`, because ACL/release threads are not separately pinned on this
device type. When UVB polling threads are found and bound, the log also reports
their thread IDs and CPU pool:

```text
Ascend 950 NPU0: worker=[...]
[cpu_bind_ascend_950] uvb_poll_window_thread tids=[...] cpus=[...]
```

On the host, stop `irqbalance` before starting vLLM when you need stable IRQ
affinity:

```bash
sudo systemctl stop irqbalance
```

After the vLLM service exits, restart it if the host should return to the
default IRQ balancing policy:

```bash
sudo systemctl start irqbalance
```

## Troubleshooting

| Message | Meaning | Action |
| --- | --- | --- |
| `CPU binding skipped: non-ARM CPU detected.` | CPU binding only runs on ARM. | No action needed on x86_64. |
| `Can not get running npu info.` | No running NPU was found, or `ASCEND_RT_VISIBLE_DEVICES` filtered all NPUs. | Check visible NPU IDs and `npu-smi info`. |
| `Insufficient CPUs for binding...` | Fewer CPUs are available than the role split requires. Devices with IRQ binding need at least 5 CPUs per logical NPU. Ascend 950 needs one full cluster per worker. | Expand the cpuset or reduce visible NPUs. |
| `NPU topo affinity not found...` | Topology affinity is unavailable. | On Ascend 950, worker CPU binding is skipped. On other topo-affinity devices, vLLM Ascend falls back to `global_slice`. Check `npu-smi info -t topo` when topology affinity is expected. |
| `uvb_poll_window_thread not found... --pid=host` | Ascend 950 could not see host UVB polling threads. | Recreate the Docker container with `--pid=host`, then restart vLLM. |
| `failed to bind uvb_poll_window_thread... --pid=host` | Ascend 950 found a UVB polling thread but failed to bind it. | Check permissions and recreate the Docker container with `--pid=host` if running in Docker. |
| `The 'migratepages' command is not available...` | Memory migration is skipped, while CPU thread binding still proceeds. | Install `numactl` if NUMA locality or performance is affected. |
| `[irq] IRQ binding skipped on Ascend 950.` | Ascend 950 does not use the IRQ binding step. | No action needed. Worker main binding and memory migration still proceed. |
| `Bind cpus failed in rank...` | A binding step failed and CPU binding was skipped for that rank. | Check `taskset`, `lscpu`, `npu-smi`, cpuset size, and `/proc/irq` permissions. |
| `[cpu_bind_layerwise] insufficient CPUs to reserve layerwise connector cores.` | The main range is too small to give two CPUs to the transfer threads. | Expand the cpuset or reduce visible NPUs. CPU binding still applies without the reservation. |
| `[cpu_bind_layerwise] BIND_LAYERWISE_CONNECTOR=1 but no layerwise connector thread found.` | The worker runs no layerwise KV connector, or the variable did not reach the worker process. | Set the variable only on engines that use a layerwise KV connector. The two reserved CPUs stay idle for this worker. |
| `[cpu_bind_layerwise] failed to bind ...` | A matching transfer thread exited, or pinning was not permitted. | Check the worker log for the thread state; binding the remaining threads still proceeds. |
