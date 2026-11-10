"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
from ml_dtypes import bfloat16
import tensorflow as tf


_GOLDEN_PATH = Path(__file__).resolve().parents[1] / "assets" / "golden.py"
_MODULE_SPEC = importlib.util.spec_from_file_location("ftrl_golden", _GOLDEN_PATH)
golden = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(golden)


def _inputs(dtype, scalar_shape=()):
    rng = np.random.default_rng(42)
    states = [
        rng.uniform(-1.0, 1.0, (2, 3)),
        rng.uniform(0.1, 2.0, (2, 3)),
        rng.uniform(-1.0, 1.0, (2, 3)),
        rng.uniform(-0.5, 0.5, (2, 3)),
    ]
    scalars = [np.full(scalar_shape, value) for value in (0.1, 0.01, 0.001, -0.5)]
    return [np.asarray(value, dtype=dtype) for value in states + scalars]


class GoldenContractTest(unittest.TestCase):
    def _check_provider(self, dtype, scalar_shape=(), device="/CPU:0"):
        arrays = _inputs(dtype, scalar_shape)
        compute_arrays = (
            [value.astype(np.float32) for value in arrays]
            if dtype == bfloat16
            else arrays
        )
        expected = [value.astype(dtype) for value in golden._compute(compute_arrays)]
        with tf.device(device):
            tensors = [tf.convert_to_tensor(value) for value in arrays]
        if "GPU" in device:
            self.assertIn("GPU", tensors[0].device)
        before = [value.numpy().copy() for value in tensors]
        # A provider must execute independently of the CPU Golden and must not
        # leave the provider Tensor path, even when locally tested on CPU.
        with (
            mock.patch.object(
                golden, "_compute", side_effect=AssertionError("CPU Golden called")
            ),
            mock.patch.object(
                golden,
                "_prepare_inputs",
                side_effect=AssertionError("numpy adapter called"),
            ),
            mock.patch.object(
                type(tensors[0]),
                "numpy",
                side_effect=AssertionError("Tensor.numpy called"),
            ),
            mock.patch.object(
                tf.raw_ops, "ResourceApplyFtrl", wraps=tf.raw_ops.ResourceApplyFtrl
            ) as ftrl,
        ):
            outputs = golden.InplaceApplyFtrlTestSpec.third_party["tf"]()(*tensors)
        ftrl.assert_called_once()
        compute_dtype = tf.float32 if dtype == bfloat16 else tensors[0].dtype
        for name in ("grad", "lr", "l1", "l2", "lr_power"):
            self.assertEqual(ftrl.call_args.kwargs[name].dtype, compute_dtype)
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 3)
        for actual, reference in zip(outputs, expected):
            self.assertTrue(tf.is_tensor(actual))
            self.assertEqual(actual.dtype, tensors[0].dtype)
            self.assertEqual(actual.device, tensors[0].device)
            self.assertEqual(actual.shape, tensors[0].shape)
            # ResourceApplyFtrl and ApplyFtrl use the same native CPU update.
            # GPU kernels may round differently; their precision is gated by
            # the declared cross_check standard during remote GEIR execution.
            if "CPU" in tensors[0].device:
                np.testing.assert_array_equal(actual.numpy(), reference)
        for actual, original in zip(tensors, before):
            np.testing.assert_array_equal(actual.numpy(), original)

    def test_provider_preserves_dtype_device_and_inputs(self):
        for dtype in (np.float16, bfloat16, np.float32):
            for scalar_shape in ((), (1,)):
                with self.subTest(
                    dtype=np.dtype(dtype).name, scalar_shape=scalar_shape
                ):
                    self._check_provider(dtype, scalar_shape)

    def test_golden_enable_and_promote_preserve_reference_precision(self):
        for dtype in (np.float16, bfloat16, np.float32):
            arrays = _inputs(dtype)
            before = [value.copy() for value in arrays]
            promoted_dtype = np.float64 if dtype == np.float32 else np.float32
            for mode, input_dtype in (("Enable", dtype), ("Promote", promoted_dtype)):
                with self.subTest(dtype=np.dtype(dtype).name, mode=mode):
                    inputs = [value.astype(input_dtype) for value in arrays]
                    output_dtype = np.dtype(input_dtype)
                    calc_dtype = (
                        np.float32
                        if mode == "Enable" and dtype != np.float32
                        else input_dtype
                    )
                    expected = golden._compute(
                        [value.astype(calc_dtype) for value in inputs]
                    )
                    outputs = golden.InplaceApplyFtrlTestSpec.golden(
                        *inputs,
                        golden_mode=mode,
                        input_dtypes=[output_dtype.name] * 8,
                        output_dtypes=[output_dtype.name] * 3,
                    )
                    self.assertIsInstance(outputs, list)
                    self.assertEqual(len(outputs), 3)
                    for actual, reference in zip(outputs, expected):
                        self.assertEqual(actual.dtype, output_dtype)
                        self.assertEqual(actual.shape, inputs[0].shape)
                        np.testing.assert_array_equal(
                            actual, reference.astype(output_dtype)
                        )
                    for actual, original in zip(inputs, arrays):
                        np.testing.assert_array_equal(
                            actual, original.astype(input_dtype)
                        )
            for actual, original in zip(arrays, before):
                np.testing.assert_array_equal(actual, original)

    def test_provider_rejects_locking_in_constructor_and_call(self):
        provider = golden.InplaceApplyFtrlTestSpec.third_party["tf"]
        with self.assertRaisesRegex(ValueError, "use_locking=False"):
            provider(use_locking=True)
        tensors = [tf.convert_to_tensor(value) for value in _inputs(np.float32)]
        with self.assertRaisesRegex(ValueError, "use_locking=False"):
            provider()(*tensors, use_locking=True)

    @unittest.skipUnless(tf.config.list_physical_devices("GPU"), "TF GPU unavailable")
    def test_provider_keeps_gpu_device(self):
        for dtype in (np.float16, bfloat16, np.float32):
            with self.subTest(dtype=np.dtype(dtype).name):
                self._check_provider(dtype, device="/GPU:0")


if __name__ == "__main__":
    unittest.main()
