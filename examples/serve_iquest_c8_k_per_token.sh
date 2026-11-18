#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m vllm.entrypoints.cli.main serve \
    /mnt/sfs_turbo/work/liuhaiyang/modelslim_full/out_full_c8 \
    --served-model-name IQuest-Q1-KPerToken \
    --tensor-parallel-size 8 \
    --distributed-executor-backend mp \
    --dtype bfloat16 \
    --quantization ascend \
    --kv-cache-dtype mxfp8 \
    --block-size 512 \
    --additional-config '{"c8_k_quant_mode":"per_token","c8_v_quant_mode":"per_token"}' \
    --enforce-eager \
    --max-model-len 4096 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8003 \
    "$@"
