# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for calibration data separation, visibility and selection policy."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "vllm_ascend").is_dir())
SPEC = importlib.util.spec_from_file_location("calibrate_c8_qk_smoothing", ROOT / "tools/calibrate_c8_qk_smoothing.py")
CALIBRATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATOR)


class TestQKSmoothingCalibration(unittest.TestCase):
    def make_capture(self, split="calibration"):
        key = torch.arange(1, 25).reshape(6, 1, 4).bfloat16()
        return {
            "layer_name": "model.layers.1.self_attn.attn",
            "tp_rank": 0,
            "tp_size": 1,
            "sample_id": split,
            "split": split,
            "phase": "prefill",
            "query": torch.ones(2, 2, 4, dtype=torch.bfloat16),
            "key": key,
            "value": key.clone(),
            "query_positions": torch.tensor([4, 5], dtype=torch.int64),
            "key_positions": torch.arange(6),
            "sink_key": torch.ones(1, 4),
            "sliding_window": 2,
        }

    def test_candidate_statistics_do_not_read_validation(self):
        calibration = self.make_capture()
        calibration["key"] = (torch.ones(6, 1, 4) * torch.tensor([1.0, 2.0, 5.0, 11.0])).bfloat16()
        validation = self.make_capture("validation")
        first, _ = CALIBRATOR.build_candidates([calibration, validation], [0.5, 1.0], 4.0)
        validation["key"][..., 0] *= 1000
        second, _ = CALIBRATOR.build_candidates([calibration, validation], [0.5, 1.0], 4.0)
        self.assertEqual(first.keys(), second.keys())
        self.assertIn("rms_alpha0.5_continuous", first)
        continuous = first["rms_alpha0.5_continuous"]
        self.assertFalse(torch.equal(continuous.log2(), continuous.log2().round()))
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_capture_requires_all_visible_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.pt"
            capture = self.make_capture()
            torch.save(capture, path)
            self.assertEqual(CALIBRATOR.load_capture(path)["query_positions"].tolist(), [4, 5])
            # Query at position4 can see position2, which this truncated K
            # interval drops even though it still contains both query positions.
            capture["key"] = capture["key"][3:]
            capture["value"] = capture["value"][3:]
            capture["key_positions"] = capture["key_positions"][3:]
            torch.save(capture, path)
            with self.assertRaisesRegex(ValueError, "incomplete visible KV interval"):
                CALIBRATOR.load_capture(path)

    def test_validation_rejects_calibration_winner_without_second_search(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            captures, output = directory / "captures", directory / "output"
            captures.mkdir()
            for split in ("calibration", "validation"):
                torch.save(self.make_capture(split), captures / f"{split}.pt")
            candidates = {
                "identity": torch.ones(1, 4),
                "calibration_best": torch.full((1, 4), 2.0),
                "validation_best": torch.full((1, 4), 4.0),
            }
            calls = []

            def evaluate(samples, coefficient, device, torch_npu):
                name = next(name for name, value in candidates.items() if torch.equal(value, coefficient))
                split = samples[0]["split"]
                calls.append((split, name))
                losses = {
                    "calibration": {"identity": 0.1, "calibration_best": 0.05, "validation_best": 0.07},
                    "validation": {"identity": 0.1, "calibration_best": 0.12, "validation_best": 0.01},
                }
                metric = {"after_sink_relative_l2": losses[split][name], "qk_logits_relative_l2": 0.02, "samples": 1}
                return {"all": metric, "prefill": metric}

            fake_npu = SimpleNamespace(set_device=lambda device: None, get_device_name=lambda: "CPU test")
            argv = ["calibrate", str(captures), str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch.dict(sys.modules, {"torch_npu": SimpleNamespace(__version__="CPU test")}),
                patch.object(torch, "npu", fake_npu, create=True),
                patch.object(CALIBRATOR, "build_candidates", return_value=(candidates, {})),
                patch.object(CALIBRATOR, "prepare_sample", side_effect=lambda sample, *_: sample),
                patch.object(CALIBRATOR, "evaluate", side_effect=evaluate),
            ):
                CALIBRATOR.main()
            report = json.loads((output / "calibration_report.json").read_text())
            coefficient_file = json.loads((output / "qk_smoothing.json").read_text())
            self.assertEqual(report["groups"][0]["calibration_selected"], "calibration_best")
            self.assertEqual(report["groups"][0]["reason"], "validation_regression_identity_retained")
            self.assertEqual(coefficient_file["ranks"]["0"], {})
            self.assertEqual([name for split, name in calls if split == "validation"], ["identity", "calibration_best"])


if __name__ == "__main__":
    unittest.main()
