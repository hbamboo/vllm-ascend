# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Calibrate fixed post-RoPE Q/K smoothing from independent real activations.

Run on Ascend 950:
  python calib_qk_smoothing.py CAPTURE_DIR OUTPUT_DIR --device npu:0
  python calib_qk_smoothing.py CAPTURE_DIR OUTPUT_DIR --tp-rank 0 --max-groups 1

Each .pt contains CPU tensors and metadata:
  layer_name, tp_rank, tp_size, sample_id, split (calibration/validation),
  phase (prefill/decode), query [Q,Nq,D], key/value [K,Nkv,D],
  query_positions [Q], key_positions [K], sink_key [Nkv,D] or None,
  sliding_window. Sources must be BF16 post-RoPE activations. Key positions
must be contiguous and include every KV position visible to every query.

Coefficients are selected ONLY on calibration samples. Validation is a
non-regression gate, never a second candidate search. Identity is retained
when no candidate passes. This tool uses an explicit FP32 attention oracle
with BF16 output/sink roundings and native dynamic FP8 quantization, NOT
the native FIA kernel. Native FIA and whole-model validation are separate.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def load_capture(path):
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = (
        "layer_name",
        "tp_rank",
        "tp_size",
        "sample_id",
        "split",
        "phase",
        "query",
        "key",
        "value",
        "query_positions",
        "key_positions",
        "sliding_window",
    )
    missing = set(required) - data.keys()
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    q, k, v = data["query"], data["key"], data["value"]
    qp, kp = data["query_positions"], data["key_positions"]
    if data["split"] not in ("calibration", "validation") or data["phase"] not in ("prefill", "decode"):
        raise ValueError(f"{path}: invalid split/phase")
    if (
        q.ndim != 3
        or k.ndim != 3
        or v.shape != k.shape
        or q.shape[0] == 0
        or k.shape[0] == 0
        or q.shape[2] != k.shape[2]
        or q.shape[1] % k.shape[1]
        or any(x.dtype != torch.bfloat16 for x in (q, k, v))
    ):
        raise ValueError(f"{path}: expected nonempty BF16 Q/K/V with compatible GQA heads/dim")
    if qp.shape != (q.shape[0],) or kp.shape != (k.shape[0],):
        raise ValueError(f"{path}: position lengths do not match Q/K")
    if qp.dtype != torch.int64 or kp.dtype != torch.int64:
        raise ValueError(f"{path}: positions must be int64")
    if not torch.equal(kp, torch.arange(int(kp[0]), int(kp[0]) + kp.numel())) or int(qp.min()) < 0:
        raise ValueError(f"{path}: key positions must be contiguous and query positions nonnegative")
    window = data["sliding_window"]
    if not isinstance(window, int) or window < 0:
        raise ValueError(f"{path}: expected a nonnegative integer sliding_window")
    softmax_scale = float(data.get("softmax_scale", q.shape[-1] ** -0.5))
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError(f"{path}: softmax_scale must be positive and finite")
    if int(kp[0]) > max(0, int(qp.min()) - window) or int(kp[-1]) < int(qp.max()):
        raise ValueError(f"{path}: incomplete visible KV interval")
    sink = data.get("sink_key")
    if sink is not None and sink.shape != k.shape[1:]:
        raise ValueError(f"{path}: sink_key shape does not match local KV heads/dim")
    if any(not bool(torch.isfinite(x).all()) for x in (q, k, v, *(() if sink is None else (sink,)))):
        raise ValueError(f"{path}: activation contains nonfinite values")
    rank, size = data["tp_rank"], data["tp_size"]
    if not isinstance(rank, int) or not isinstance(size, int) or not 0 <= rank < size:
        raise ValueError(f"{path}: invalid tensor-parallel metadata")
    return data


def build_candidates(data, alphas, limit):
    keys = [sample["key"].float() for sample in data if sample["split"] == "calibration"]
    count = sum(key.shape[0] for key in keys)
    rms = (sum(key.square().sum(0) for key in keys) / count).sqrt()
    absmax = torch.stack([key.abs().amax(0) for key in keys]).amax(0)
    candidates = {"identity": torch.ones_like(rms)}
    for statistic_name, statistic in (("rms", rms), ("absmax", absmax)):
        # Normalize each KV head's geometric mean to one; scalar rescaling
        # alone does not supply a finer per-channel quantization scheme.
        positive = statistic.clamp_min(1e-8)
        relative = positive / positive.log().mean(-1, keepdim=True).exp()
        for alpha in alphas:
            continuous = relative.pow(alpha).clamp(1 / limit, limit)
            powers = continuous.log2().round().clamp(math.ceil(-math.log2(limit)), math.floor(math.log2(limit))).exp2()
            for kind, coefficient in (("continuous", continuous), ("pow2", powers)):
                if not any(torch.equal(coefficient, existing) for existing in candidates.values()):
                    candidates[f"{statistic_name}_alpha{alpha:g}_{kind}"] = coefficient.contiguous()
    return candidates, {
        "rms_max_over_median": (rms.amax(-1) / rms.median(-1).values.clamp_min(1e-8)).tolist(),
        "absmax_max_over_median": (absmax.amax(-1) / absmax.median(-1).values.clamp_min(1e-8)).tolist(),
    }


def quantized_bf16(source, torch_npu):
    payload, scale = torch_npu.npu_dynamic_quant(source.flatten(1), dst_type=torch.float8_e4m3fn, quant_mode="pertoken")
    # The singleton native-FIA probe matches this effective BF16-scale
    # convention. Keep the original FP32 scale returned by the quantizer;
    # only this reference multiplication uses its BF16 effective value.
    decoded = (payload.to(torch.bfloat16).float() * scale.to(torch.bfloat16).float()[:, None]).bfloat16()
    return decoded.reshape_as(source)


def attention(query, key, value, visible, sink_logit, scale):
    group = query.shape[1] // key.shape[1]
    logits = (
        torch.matmul(query.transpose(0, 1).float(), key.repeat_interleave(group, 1).permute(1, 2, 0).float()) * scale
    )
    masked_logits = logits.masked_fill(~visible[None], float("-inf"))
    lse = masked_logits.logsumexp(-1).transpose(0, 1)
    raw = torch.matmul(masked_logits.softmax(-1), value.repeat_interleave(group, 1).transpose(0, 1).float())
    raw = raw.transpose(0, 1).contiguous().bfloat16()
    if sink_logit is None:
        return logits, raw
    factor = torch.sigmoid(lse - sink_logit).bfloat16()
    return logits, (raw.float() * factor[..., None].float()).bfloat16()


def prepare_sample(sample, device, torch_npu):
    q, k, v = (sample[name].to(device) for name in ("query", "key", "value"))
    qp, kp = (sample[name].to(device) for name in ("query_positions", "key_positions"))
    visible = (kp[None] <= qp[:, None]) & (kp[None] >= qp[:, None] - sample["sliding_window"])
    scale = float(sample.get("softmax_scale", q.shape[-1] ** -0.5))
    sink = sample.get("sink_key")
    sink_logit = None
    if sink is not None:
        sink_heads = sink.to(device).bfloat16().float().repeat_interleave(q.shape[1] // k.shape[1], 0)
        sink_logit = (q.float() * sink_heads).sum(-1).bfloat16().float() * scale
    reference_logits, reference_output = attention(q, k, v, visible, sink_logit, scale)
    return {
        "query": q,
        "key": k,
        "quantized_value": quantized_bf16(v, torch_npu),
        "visible": visible,
        "sink_logit": sink_logit,
        "scale": scale,
        "reference_logits": reference_logits,
        "reference_output": reference_output,
        "phase": sample["phase"],
        "split": sample["split"],
    }


def evaluate(samples, coefficient, device, torch_npu):
    totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    # Compute reciprocals on CPU, as the production coefficient loader does.
    query_scale = coefficient.repeat_interleave(samples[0]["query"].shape[1] // coefficient.shape[0], 0).to(device)
    key_scale = coefficient.reciprocal().to(device)
    for sample in samples:
        query = (sample["query"].float() * query_scale).bfloat16()
        key = (sample["key"].float() * key_scale).bfloat16()
        key = quantized_bf16(key, torch_npu)
        logits, output = attention(
            query, key, sample["quantized_value"], sample["visible"], sample["sink_logit"], sample["scale"]
        )
        logits_delta = (logits - sample["reference_logits"]).masked_fill(~sample["visible"][None], 0)
        reference_logits = sample["reference_logits"].masked_fill(~sample["visible"][None], 0)
        output_delta = output.float() - sample["reference_output"].float()
        values = (
            torch.stack(
                (
                    output_delta.square().sum(),
                    sample["reference_output"].float().square().sum(),
                    logits_delta.square().sum(),
                    reference_logits.square().sum(),
                )
            )
            .cpu()
            .tolist()
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite candidate output; inspect the captured inputs and coefficients")
        for phase in ("all", sample["phase"]):
            total = totals[phase]
            for index, value in enumerate(values):
                total[index] += value
            total[4] += 1
    return {
        phase: {
            "after_sink_relative_l2": math.sqrt(value[0] / max(value[1], 1e-30)),
            "qk_logits_relative_l2": math.sqrt(value[2] / max(value[3], 1e-30)),
            "samples": value[4],
        }
        for phase, value in totals.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tp-rank", type=int)
    parser.add_argument("--layer", action="append", help="Exact layer name; repeat to select multiple layers")
    parser.add_argument(
        "--max-groups", type=int, default=0, help="Limit layer/rank groups, retaining all files in each"
    )
    parser.add_argument("--max-files", type=int, default=0, help="Limit files; incomplete groups remain identity")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--coefficient-limit", type=float, default=4.0)
    parser.add_argument(
        "--min-calibration-improvement",
        type=float,
        default=0.01,
        help="Required fractional reduction in calibration attention relative-L2, default 1%%",
    )
    parser.add_argument(
        "--validation-max-regression",
        type=float,
        default=0.0,
        help="Allowed fractional relative-L2 regression overall and per phase, default 0",
    )
    args = parser.parse_args()
    if (
        args.coefficient_limit < 1
        or not 0 <= args.min_calibration_improvement < 1
        or args.validation_max_regression < 0
        or any(alpha < 0 for alpha in args.alphas)
    ):
        parser.error("Invalid coefficient range/improvement/regression/alpha")
    import torch_npu

    torch.npu.set_device(args.device)
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups, sample_splits = defaultdict(list), {}
    tp_size = None
    file_count = 0
    for path in sorted(args.capture_dir.rglob("*.pt")):
        # Read only mmap-backed metadata while indexing. Validate tensors
        # once when their selected layer/rank group is processed below.
        sample = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if args.tp_rank is not None and sample["tp_rank"] != args.tp_rank:
            continue
        if args.layer and sample["layer_name"] not in args.layer:
            continue
        key = (sample["tp_rank"], sample["layer_name"])
        if args.max_groups and key not in groups and len(groups) >= args.max_groups:
            continue
        if tp_size is None:
            tp_size = sample["tp_size"]
        if sample["tp_size"] != tp_size:
            raise ValueError("Mixed tensor-parallel sizes in captures")
        previous_split = sample_splits.setdefault(sample["sample_id"], sample["split"])
        if previous_split != sample["split"]:
            raise ValueError(f"sample_id {sample['sample_id']} overlaps calibration and validation")
        groups[key].append(path)
        file_count += 1
        if args.max_files and file_count >= args.max_files:
            break
    if not groups:
        raise ValueError("No captures matched the selected rank/layer filters")
    coefficients = {
        "format_version": 1,
        "tensor_parallel_size": tp_size,
        "ranks": {str(rank): {} for rank in range(tp_size)},
    }
    report = {
        "capture_dir": str(args.capture_dir.resolve()),
        "device": torch.npu.get_device_name(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "tp_size": tp_size,
        "selection": "lowest calibration after-sink relative-L2; validation non-regression gate only",
        "oracle": (
            "explicit FP32 QK/softmax/PV, BF16 output/sink rounding; "
            "FP8 native dynamic quant with effective BF16 scales"
        ),
        "native_fia_executed": False,
        "alphas": args.alphas,
        "coefficient_limit": args.coefficient_limit,
        "min_calibration_improvement": args.min_calibration_improvement,
        "validation_max_regression": args.validation_max_regression,
        "groups": [],
    }
    for (rank, layer_name), paths in sorted(groups.items()):
        captures = [load_capture(path) for path in paths]
        record = {
            "tp_rank": rank,
            "layer_name": layer_name,
            "files": [str(path) for path in paths],
            "sample_ids": sorted({sample["sample_id"] for sample in captures}),
            "exported": False,
        }
        calibration = [sample for sample in captures if sample["split"] == "calibration"]
        validation = [sample for sample in captures if sample["split"] == "validation"]
        if not calibration or not validation:
            record["reason"] = "incomplete_group_requires_calibration_and_validation"
            report["groups"].append(record)
            continue
        if any(
            sample["query"].shape[1:] != captures[0]["query"].shape[1:]
            or sample["key"].shape[1:] != captures[0]["key"].shape[1:]
            for sample in captures
        ):
            raise ValueError(f"{layer_name}/rank{rank}: inconsistent head layout")
        candidates, statistics = build_candidates(captures, args.alphas, args.coefficient_limit)
        record["channel_statistics"] = statistics
        prepared_calibration = [prepare_sample(sample, args.device, torch_npu) for sample in calibration]
        metrics = {
            name: evaluate(prepared_calibration, coefficient, args.device, torch_npu)
            for name, coefficient in candidates.items()
        }
        chosen = min(metrics, key=lambda name: metrics[name]["all"]["after_sink_relative_l2"])
        baseline_error = metrics["identity"]["all"]["after_sink_relative_l2"]
        improvement = 1 - metrics[chosen]["all"]["after_sink_relative_l2"] / max(baseline_error, 1e-30)
        record.update(
            calibration_candidates=metrics, calibration_selected=chosen, calibration_improvement_fraction=improvement
        )
        del prepared_calibration
        prepared_validation = [prepare_sample(sample, args.device, torch_npu) for sample in validation]
        validation_identity = evaluate(prepared_validation, candidates["identity"], args.device, torch_npu)
        validation_selected = (
            validation_identity
            if chosen == "identity"
            else evaluate(prepared_validation, candidates[chosen], args.device, torch_npu)
        )
        gates = {
            phase: validation_selected[phase]["after_sink_relative_l2"]
            <= baseline["after_sink_relative_l2"] * (1 + args.validation_max_regression)
            for phase, baseline in validation_identity.items()
        }
        chosen_is_better = chosen != "identity" and improvement >= args.min_calibration_improvement
        record.update(
            validation_identity=validation_identity,
            validation_selected=validation_selected,
            validation_phase_gates=gates,
        )
        if chosen_is_better and all(gates.values()):
            coefficients["ranks"][str(rank)][layer_name] = candidates[chosen].tolist()
            record.update(
                exported=True,
                reason="calibration_improved_and_validation_passed",
                coefficient_min=float(candidates[chosen].min()),
                coefficient_max=float(candidates[chosen].max()),
            )
        else:
            record["reason"] = (
                "calibration_improvement_below_threshold"
                if not chosen_is_better
                else "validation_regression_identity_retained"
            )
        report["groups"].append(record)
        (args.output_dir / "qk_smoothing.json").write_text(json.dumps(coefficients, indent=2) + "\n")
        (args.output_dir / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "tp_rank": rank,
                    "layer": layer_name,
                    "candidate": chosen,
                    "calibration_improvement": improvement,
                    "exported": record["exported"],
                    "reason": record["reason"],
                }
            ),
            flush=True,
        )
        del prepared_validation, captures
    report["processed_groups"] = len(report["groups"])
    report["exported_groups"] = sum(record["exported"] for record in report["groups"])
    (args.output_dir / "qk_smoothing.json").write_text(json.dumps(coefficients, indent=2) + "\n")
    (args.output_dir / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "calibration_complete",
                "groups": report["processed_groups"],
                "exported_groups": report["exported_groups"],
                "output": str(args.output_dir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
