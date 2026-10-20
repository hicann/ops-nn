# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU contract checks for the three co-versioned RmsNormQuant golden files."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


def load_golden(operator):
    root = Path(__file__).resolve().parents[4]
    path = root / "norm" / operator / "tests" / "assets" / "golden.py"
    spec = importlib.util.spec_from_file_location(operator + "_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmptyGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = load_golden("rms_norm_quant")
        cls.v2 = load_golden("rms_norm_quant_v2")
        cls.v3 = load_golden("rms_norm_quant_v3")

    def test_aclnn_empty_shapes_and_packed_storage(self):
        for shape in ((0, 64), (3, 0), (0,), (0, 0), (2, 0, 64)):
            x = np.empty(shape, np.float16)
            gamma = np.ones((1, shape[-1]), np.float16)
            beta = np.zeros((shape[-1],), np.float16)
            scale = np.ones((1,), np.float32)
            for dst in (2, 29, 34, 35, 36):
                expected = shape[:-1] + (shape[-1] // 8,) if dst == 29 else shape
                with (
                    self.subTest(shape=shape, dst=dst),
                    patch.object(np, "mean", side_effect=AssertionError),
                ):
                    y1 = self.v1.aclnn_rms_norm_quant_golden(
                        x, gamma, beta, scale, scale, dst_type=dst
                    )
                    y3, rstd = self.v3.aclnn_rms_norm_quant_v3_golden(
                        x, gamma, scale, None, beta, dst_type=dst
                    )
                    self.assertEqual(tuple(y1.shape), expected)
                    self.assertEqual(tuple(y3.shape), expected)
                    self.assertEqual(y1.numel(), 0)
                    self.assertEqual(y3.numel(), 0)
                    self.assertIsNone(rstd)

    def test_aclnn_direct_int4_is_not_int32_packed(self):
        from ml_dtypes import int4

        for shape in ((0, 2), (2, 2), (0, 64), (2, 64)):
            x = np.ones(shape, np.float16)
            gamma = np.ones((shape[-1],), np.float16)
            beta = np.zeros((shape[-1],), np.float16)
            scale = np.array([0.5], np.float32)
            offset = np.array([0], np.float32)
            y = np.empty(shape, int4)
            outputs = (
                self.v1.aclnn_rms_norm_quant_golden(x, gamma, beta, scale, offset, y=y),
                self.v3.aclnn_rms_norm_quant_v3_golden(
                    x, gamma, scale, offset, beta, y=y
                )[0],
            )
            for output in outputs:
                with self.subTest(shape=shape):
                    self.assertEqual(output.shape, shape)
                    self.assertEqual(str(output.dtype), "int4")
                    np.testing.assert_array_equal(output, np.full(shape, 2, int4))

    def test_native_empty_dual_scalar_scales(self):
        x = np.empty((0, 2, 64), np.float16)
        gamma = np.ones((2, 64), np.float16)
        scale = np.array(0.5, np.float32)
        for dtype in ("int8", "int4", "hifloat8", "float8_e5m2", "float8_e4m3fn"):
            with (
                self.subTest(dtype=dtype),
                patch.object(np, "mean", side_effect=AssertionError),
            ):
                y1, y2 = self.v2.rms_norm_quant_v2(
                    x, gamma, scale, scale, output_dtypes=(dtype, dtype)
                )
                y3, y4, rstd = self.v3.rms_norm_quant_v3(
                    x,
                    gamma,
                    scale,
                    scale,
                    output_dtypes=(dtype, dtype, "float32"),
                    output_rstd=True,
                )
                for y in (y1, y2, y3, y4):
                    self.assertEqual(y.shape, x.shape)
                    self.assertEqual(y.size, 0)
                    self.assertEqual(str(y.dtype), dtype)
                self.assertEqual(rstd.shape, (0, 1, 1))
                self.assertEqual(rstd.dtype, np.float32)

    def test_empty_reduction_with_enabled_rstd_is_unsupported(self):
        for shape in ((3, 0), (0,)):
            x = np.empty(shape, np.float16)
            gamma = np.empty((0,), np.float16)
            scale = np.ones((1,), np.float32)
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "nonempty rstd"):
                    self.v3.rms_norm_quant_v3(x, gamma, scale, output_rstd=True)
                with self.assertRaisesRegex(ValueError, "nonempty rstd"):
                    self.v3.aclnn_rms_norm_quant_v3_golden(
                        x, gamma, scale, None, None, output_rstd=True
                    )

    def test_aclnn_empty_batch_with_enabled_rstd(self):
        x = np.empty((0, 64), np.float16)
        y, rstd = self.v3.aclnn_rms_norm_quant_v3_golden(
            x,
            np.ones((64,), np.float16),
            np.ones((1,), np.float32),
            None,
            None,
            output_rstd=True,
        )
        self.assertEqual(tuple(y.shape), (0, 64))
        self.assertEqual(tuple(rstd.shape), (0, 1))
        self.assertEqual(rstd.numel(), 0)

    def test_disabled_native_outputs_remain_suppressed(self):
        x = np.empty((2, 0), np.float16)
        gamma = np.empty((0,), np.float16)
        scale = np.ones((1,), np.float32)
        self.assertIsNone(self.v2.rms_norm_quant_v2(x, gamma, scale)[1])
        self.assertEqual(self.v3.rms_norm_quant_v3(x, gamma, scale)[1:], (None, None))

    def test_nonempty_computation_unchanged(self):
        x = np.ones((2, 8), np.float16)
        gamma = np.ones((8,), np.float16)
        beta = np.zeros((8,), np.float16)
        scale = np.array([0.5], np.float32)
        offset = np.array([0], np.float32)
        expected = np.full(x.shape, 2, np.int8)
        np.testing.assert_array_equal(
            self.v1.aclnn_rms_norm_quant_golden(x, gamma, beta, scale, offset), expected
        )
        np.testing.assert_array_equal(
            self.v2.rms_norm_quant_v2(x, gamma, scale)[0], expected
        )
        np.testing.assert_array_equal(
            self.v3.rms_norm_quant_v3(x, gamma, scale)[0], expected
        )
        y, rstd = self.v3.aclnn_rms_norm_quant_v3_golden(
            x, gamma, scale, offset, beta, output_rstd=True
        )
        np.testing.assert_array_equal(y, expected)
        np.testing.assert_allclose(
            rstd, np.full((2, 1), 1 / np.sqrt(1 + 1e-6), np.float32)
        )


if __name__ == "__main__":
    unittest.main()
