#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""CPU golden regressions: python3 -m unittest discover -s <this directory>."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from ml_dtypes import bfloat16


SPEC = importlib.util.spec_from_file_location(
    "dynamic_mx_quant_golden", Path(__file__).parents[1] / "assets" / "golden.py"
)
GOLDEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOLDEN)


class DynamicMxQuantGoldenTest(unittest.TestCase):
    def test_zero_blocks_keep_zero_scale_with_positive_lower_bound(self):
        for dtype in (np.float16, bfloat16, np.float32):
            for dst in ("float8_e4m3fn", "float8_e5m2"):
                for axis in (0, -1):
                    for block_size in (32, 64):
                        with self.subTest(
                            dtype=dtype, dst=dst, axis=axis, block=block_size
                        ):
                            # A full zero block, a clamped nonzero block, and a zero tail.
                            x = np.zeros((2, block_size * 2 + 1), dtype=dtype)
                            x[:, 1::2] = -0.0
                            x[:, block_size : 2 * block_size] = 1
                            if axis == 0:
                                x = x.T.copy()
                            scale, y = GOLDEN._mx_quantize(
                                x,
                                mx_ele_dtype=dst,
                                axis=axis,
                                block_size=block_size,
                                scale_alg=1,
                                max_low_bound=57344.0,
                            )
                            expected = np.zeros(scale.shape, np.uint8)
                            # E4M3 uses scale=128, E5M2 uses scale=1.
                            expected[:, 0, 1] = 134 if dst == "float8_e4m3fn" else 127
                            if axis == 0:
                                expected = np.zeros(scale.shape, np.uint8)
                                expected[0, :, 1] = (
                                    134 if dst == "float8_e4m3fn" else 127
                                )
                            np.testing.assert_array_equal(
                                scale.view(np.uint8), expected
                            )
                            zero = x == 0
                            np.testing.assert_array_equal(
                                np.asarray(y, np.float32)[zero], 0
                            )

    def test_zero_blocks_for_all_existing_algorithms(self):
        for dtype in (np.float16, bfloat16, np.float32):
            for alg, dst in (
                (0, "float4_e2m1"),
                (1, "float8_e4m3fn"),
                (2, "float4_e2m1"),
            ):
                with self.subTest(dtype=dtype, alg=alg):
                    x = np.zeros((3, 66), dtype=dtype)
                    x[:, 1::2] = -0.0
                    scale, y = GOLDEN._mx_quantize(x, mx_ele_dtype=dst, scale_alg=alg)
                    np.testing.assert_array_equal(scale.view(np.uint8), 0)
                    np.testing.assert_array_equal(np.asarray(y, np.float32), 0)

    def test_lower_bound_still_clamps_nonzero_blocks(self):
        x = np.ones((2, 32), np.float16)
        scale, _ = GOLDEN._mx_quantize(
            x, mx_ele_dtype="float8_e4m3fn", scale_alg=1, max_low_bound=448.0
        )
        np.testing.assert_array_equal(scale.view(np.uint8), [[[127, 0]], [[127, 0]]])

    def test_algorithm_two_uses_original_fp16_amax(self):
        x = np.zeros((2, 32), np.float16)
        x[:, :2] = 1
        x[:, 2] = [3.0, 3.001953125]
        scale, y = GOLDEN._mx_quantize(x, scale_alg=2)
        np.testing.assert_array_equal(scale.view(np.uint8), [[[126, 0]], [[127, 0]]])
        np.testing.assert_array_equal(
            np.asarray(y, np.float32)[:, :2], [[2, 2], [1, 1]]
        )

    def test_algorithm_two_custom_destination_maximum(self):
        x = np.full((1, 32), 3, np.float16)
        scale, _ = GOLDEN._mx_quantize(x, scale_alg=2, dst_type_max=3.0)
        np.testing.assert_array_equal(scale.view(np.uint8), [[[127, 0]]])

    def test_nvfp4_zero_and_padding_with_lower_bound(self):
        for dtype in (np.float16, bfloat16):
            for k in (2, 16, 18, 32, 48, 66, 65538):
                with self.subTest(dtype=dtype, k=k):
                    y, scale, _ = GOLDEN._nvfp4_reference(
                        np.zeros((2, k), dtype=dtype), max_low_bound=448.0
                    )
                    np.testing.assert_array_equal(y, 0)
                    np.testing.assert_array_equal(scale, 0)

    def test_nvfp4_nan_inf_and_signed_zero(self):
        for dtype in (np.float16, bfloat16):
            with self.subTest(dtype=dtype):
                x = np.zeros((4, 16), dtype=dtype)
                x[0] = np.nan
                x[1, :2] = [np.inf, -np.inf]
                x[2, :2] = [0.0, -0.0]
                x[3, :2] = [1.0, np.nan]
                y, scale, _ = GOLDEN._nvfp4_reference(x)
                np.testing.assert_array_equal(
                    scale.reshape(4, 2)[:, 0], [127, 126, 0, 35]
                )
                self.assertTrue(np.all(y.reshape(4, 8)[0] == 0x77))
                self.assertEqual(int(y.reshape(4, 8)[1, 0]), 0xF7)
                self.assertEqual(int(y.reshape(4, 8)[2, 0]), 0x80)


if __name__ == "__main__":
    unittest.main()
