# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU-only algebra and production routing contracts; no Torch/NPU required."""

import ast
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock, patch


def dense_hadamard():
    """Independent parity definition, not the production butterfly algorithm."""
    return [[(-1.0 if bin(i & j).count("1") % 2 else 1.0) / math.sqrt(128) for j in range(128)] for i in range(128)]


def load_method(relative_file, class_name, method_name, namespace=None, *, first_if_only=False):
    root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
    path = root / relative_file
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    if first_if_only:
        method.body = [method.body[0]]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    scope = {} if namespace is None else dict(namespace)
    exec(compile(module, str(path), "exec"), scope)
    return scope[method_name]


class ShapeTensor:
    def __init__(self, shape):
        self.shape = shape

    def reshape(self, *shape):
        self.shape = shape
        return self

    def __getitem__(self, item):
        return self


class TestC8QKHadamard(unittest.TestCase):
    @unittest.skipIf(sys.version_info < (3, 10), "Production config_utils requires Python 3.10+")
    def test_existing_pydantic_entry_rejects_invalid_rotation(self):
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        spec = importlib.util.spec_from_file_location("hadamard_config_utils", root / "vllm_ascend/config_utils.py")
        config_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_utils)
        tree = ast.parse((root / "vllm_ascend/ascend_config.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendConfig")
        field = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id == "c8_qk_rotation")
        probe = ast.ClassDef(
            name="RotationConfig",
            bases=[],
            keywords=[],
            body=[field],
            decorator_list=[ast.Name(id="config", ctx=ast.Load())],
        )
        module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
        scope = {"config": config_utils.config, "Literal": Literal, "__name__": __name__}
        exec(compile(module, "actual_rotation_field", "exec"), scope)
        constructor = scope[probe.name]
        self.assertEqual(constructor().c8_qk_rotation, "none")
        self.assertEqual(constructor(c8_qk_rotation="hadamard").c8_qk_rotation, "hadamard")
        with self.assertRaises(ValueError):
            constructor(c8_qk_rotation="foo")

    def test_orthogonal_basis_and_sink_gqa_scores(self):
        h = dense_hadamard()
        for i in range(128):
            for j in range(128):
                self.assertAlmostEqual(sum(a * b for a, b in zip(h[i], h[j])), float(i == j), places=12)

        def rotate(x):
            return [sum(a * b for a, b in zip(row, x)) for row in h]

        keys = [[math.sin((d + 3 * head) / 9) for d in range(128)] for head in range(2)]
        for head in range(6):
            query = [math.cos((d + head) / 11) for d in range(128)]
            sink = keys[head // 3]
            qr, sr = rotate(query), rotate(sink)
            self.assertAlmostEqual(
                sum(a * b for a, b in zip(query, sink)), sum(a * b for a, b in zip(qr, sr)), places=11
            )
            for a, b in zip(rotate(qr), query):
                self.assertAlmostEqual(a, b, places=12)

    def test_configuration_is_opt_in_and_rejects_combined_offset(self):
        root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
        tree = ast.parse((root / "vllm_ascend/ascend_config.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendConfig")
        field = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and n.target.id == "c8_qk_rotation")
        self.assertEqual(ast.literal_eval(field.value), "none")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "derive_and_validate")
        # Execute the actual new entry validation, without unrelated vLLM setup.
        method.body = method.body[:2]
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        scope = {}
        exec(compile(module, "rotation_config", "exec"), scope)
        vc = SimpleNamespace(
            cache_config=SimpleNamespace(cache_dtype="mxfp8"), model_config=SimpleNamespace(dtype="bf16")
        )
        cfg = SimpleNamespace(c8_qk_rotation="hadamard", c8_k_offset_mode="none", c8_v_quant_mode="per_channel")
        with patch.dict(sys.modules, {"torch": SimpleNamespace(bfloat16="bf16")}):
            scope[method.name](cfg, vc)
            cfg.c8_k_offset_mode = "per_block_fp8_anchor"
            with self.assertRaisesRegex(ValueError, "offset_mode=none"):
                scope[method.name](cfg, vc)
            cfg.c8_k_offset_mode, cfg.c8_v_quant_mode = "none", "per_token"
            with self.assertRaisesRegex(ValueError, "actual C8"):
                scope[method.name](cfg, vc)
            cfg.c8_qk_rotation = "none"
            scope[method.name](cfg, vc)

    def test_forward_rotates_owner_qk_once_preserves_v_and_default(self):
        forward = load_method(
            "vllm_ascend/attention/attention_c8_mxfp.py", "AscendC8MXFPAttentionBackendImpl", "forward"
        )
        for mode, sharing, expected_calls in (("none", None, 0), ("hadamard", None, 2), ("hadamard", "owner", 1)):
            with self.subTest(mode=mode, sharing=sharing):
                query, key, value = ShapeTensor((7, 768)), ShapeTensor((7, 256)), object()
                q_rot, k_rot, output = ShapeTensor((7, 6, 128)), ShapeTensor((7, 2, 128)), object()
                rotation = MagicMock(side_effect=(q_rot, k_rot))
                downstream = MagicMock(return_value=output)
                impl = SimpleNamespace(
                    c8_qk_rotation=mode,
                    q_quant_mode="per_token_head",
                    num_heads=6,
                    num_kv_heads=2,
                    head_size=128,
                    kv_sharing_target_layer_name=sharing,
                    vllm_config=SimpleNamespace(kv_transfer_config=None),
                    _forward_fp8_per_token=downstream,
                )
                module = SimpleNamespace(hadamard128=rotation)
                with patch.dict(sys.modules, {"vllm_ascend.ops.triton.qk_hadamard": module}):
                    result = forward(impl, object(), query, key, value, (1, 2, 3, 4), object(), output)
                self.assertIs(result, output)
                self.assertEqual(rotation.call_count, expected_calls)
                self.assertIs(downstream.call_args.args[1], q_rot if mode == "hadamard" else query)
                self.assertIs(downstream.call_args.args[2], k_rot if expected_calls == 2 else key)
                self.assertIs(downstream.call_args.args[3], value)

    def test_sink_uses_fp32_derived_buffer_without_mutating_parameter(self):
        original, rotated = object(), object()
        layer = SimpleNamespace(sink_key=original, impl=SimpleNamespace(c8_qk_rotation="hadamard"))
        layer.register_buffer = MagicMock(side_effect=lambda name, tensor, **kwargs: setattr(layer, name, tensor))
        rotation = MagicMock(return_value=rotated)
        process = load_method(
            "vllm_ascend/quantization/methods/kv_cache/mxfp_c8.py",
            "AscendC8MXFPKVCacheAttentionMethod",
            "process_weights_after_loading",
            {"torch": SimpleNamespace(float32="fp32")},
            first_if_only=True,
        )
        with patch.dict(sys.modules, {"vllm_ascend.ops.triton.qk_hadamard": SimpleNamespace(hadamard128=rotation)}):
            process(object(), layer)
        rotation.assert_called_once_with(original, output_dtype="fp32")
        layer.register_buffer.assert_called_once_with("_c8_rotated_sink_key", rotated, persistent=False)
        self.assertIs(layer.sink_key, original)
        select = load_method(
            "vllm_ascend/attention/attention_c8_mxfp.py", "AscendC8MXFPAttentionBackendImpl", "_qfa_sink_key"
        )
        self.assertIs(select(SimpleNamespace(c8_qk_rotation="none"), layer), original)
        self.assertIs(select(SimpleNamespace(c8_qk_rotation="hadamard"), layer), rotated)


if __name__ == "__main__":
    unittest.main()
