# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the standalone NPU validator, using only stdlib."""

import builtins
import importlib.util
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


class TestFIAPseudoValidationImport(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[3]
        self.path = self.root / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_c8_fia_pseudo.py"
        self.torch = ModuleType("torch")
        self.torch.__version__ = "import-test"
        self.torch.npu = SimpleNamespace(
            is_available=Mock(return_value=True),
            set_device=Mock(),
            get_device_name=Mock(return_value="Ascend950"),
        )
        self.torch_npu = ModuleType("torch_npu")
        self.torch_npu.__version__ = "import-test"
        self.torch_npu.npu = SimpleNamespace(get_soc_version=Mock(return_value=260))

    def guarded_import(self, name, *args, **kwargs):
        if name.split(".")[0] in {"vllm", "vllm_ascend"}:
            raise AssertionError(f"Validator imported platform/plugin package: {name}")
        return self.original_import(name, *args, **kwargs)

    def load_validator(self):
        self.original_import = builtins.__import__
        spec = importlib.util.spec_from_file_location("standalone_fia_validator", self.path)
        module = importlib.util.module_from_spec(spec)
        with (
            patch.dict("sys.modules", {"torch": self.torch, "torch_npu": self.torch_npu}),
            patch("builtins.__import__", side_effect=self.guarded_import),
        ):
            spec.loader.exec_module(module)
        return module

    def test_import_avoids_vllm_and_device_initialization(self):
        self.load_validator()
        self.torch.npu.is_available.assert_not_called()
        self.torch.npu.set_device.assert_not_called()
        self.torch_npu.npu.get_soc_version.assert_not_called()

    def test_source_loader_preserves_production_helpers_and_tensor_backends(self):
        module = self.load_validator()
        with patch("builtins.__import__", side_effect=self.guarded_import):
            namespace = module.load_cache_helpers()
            self.assertEqual(namespace["mxfp_packet_size_bytes"](1, 512, 128, 128), 135168)
            self.assertIsNone(namespace["notify_kv_cache_written"]())
        self.assertIs(namespace["torch"], self.torch)
        self.assertIs(namespace["torch_npu"], self.torch_npu)
        for name, filename in (
            ("scatter_fia_fp8_cache", "attention_c8_fia.py"),
            ("mxfp_cache_views_for_spec", "attention_c8_mxfp.py"),
        ):
            self.assertEqual(Path(namespace[name].__code__.co_filename).name, filename)

    def test_setup_selects_device_before_runtime_soc_query(self):
        module = self.load_validator()
        order = []
        self.torch.npu.set_device.side_effect = lambda device: order.append(("device", device))
        self.torch_npu.npu.get_soc_version.side_effect = lambda: order.append(("soc", 260)) or 260
        with (
            patch("builtins.__import__", side_effect=self.guarded_import),
            patch("builtins.print"),
        ):
            module.TestFIAPerTokenNPU.setUpClass()
        self.assertEqual(order, [("device", 0), ("soc", 260)])
        self.assertTrue(callable(module.TestFIAPerTokenNPU.scatter_fia_fp8_cache))

    def test_unavailable_npu_skips_without_setting_device(self):
        module = self.load_validator()
        self.torch.npu.is_available.return_value = False
        with self.assertRaises(unittest.SkipTest):
            module.TestFIAPerTokenNPU.setUpClass()
        self.torch.npu.set_device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
