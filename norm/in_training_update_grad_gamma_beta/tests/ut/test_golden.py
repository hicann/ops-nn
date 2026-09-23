#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU tests of the two reference implementations against exact rational sums."""

import importlib.util
import itertools
from fractions import Fraction
from pathlib import Path
import unittest

import numpy as np
import torch

_SPEC = importlib.util.spec_from_file_location(
    "intuggb_golden", Path(__file__).parents[1] / "assets/golden.py"
)
_GOLDEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GOLDEN)


def exact_sum(value):
    columns = int(np.prod(value.shape[1:]))
    rows = value.reshape(value.shape[0], columns)
    result = [
        float(sum((Fraction(float(x)) for x in rows[:, i]), Fraction()))
        for i in range(columns)
    ]
    with np.errstate(over="ignore"):
        return np.array(result, dtype=np.float32).reshape((1,) + value.shape[1:])


class GoldenTest(unittest.TestCase):
    def check_finite(self, gamma, beta):
        expected = [exact_sum(gamma), exact_sum(beta)]
        golden = _GOLDEN.InTrainingUpdateGradGammaBetaSpec.golden(gamma, beta)
        reference = _GOLDEN._TorchReference()(
            torch.from_numpy(gamma), torch.from_numpy(beta)
        )
        for i in range(2):
            np.testing.assert_array_equal(golden[i], expected[i])
            np.testing.assert_array_equal(reference[i].numpy(), expected[i])
            self.assertEqual(golden[i].dtype, np.float32)

    def test_all_cancellation_orders(self):
        maximum = np.finfo(np.float32).max
        gamma = np.array(
            list(itertools.permutations([maximum, 1.0, -maximum, 0.0])),
            dtype=np.float32,
        ).T.reshape(4, 24, 1, 1)
        self.check_finite(gamma, -gamma)

    def test_multiple_exponent_cancellation(self):
        gamma = np.array(
            [2.0**100, 2.0**60, 1.0, -(2.0**100), -(2.0**60), 2.0**-100, -0.5],
            dtype=np.float32,
        ).reshape(7, 1, 1, 1)
        self.check_finite(gamma, -gamma)

    def test_ordinary_and_subnormal_values(self):
        rng = np.random.default_rng(42)
        gamma = rng.uniform(-1.0, 1.0, size=(257, 13, 1, 1)).astype(np.float32)
        beta = np.zeros_like(gamma)
        beta[:3] = np.nextafter(np.float32(0), np.float32(1))
        self.check_finite(gamma, beta)

    def test_multiscale_regression_inputs(self):
        for suffix, shape in [
            ("permutations", (7, 5040, 1, 1)),
            ("full_range", (25, 13, 1, 1, 1)),
            ("four_levels", (7, 1, 1, 17)),
            ("row_tiles", (8193, 1, 1, 1)),
        ]:
            with self.subTest(suffix=suffix):
                template = np.zeros(shape, dtype=np.float32)
                inputs = _GOLDEN.InTrainingUpdateGradGammaBetaSpec.customize_inputs(
                    template,
                    template,
                    testcase_name="multiscale_test_multiscale_" + suffix,
                )
                self.check_finite(*inputs)

    def test_finite_threshold_and_true_overflow(self):
        maximum = np.finfo(np.float32).max
        threshold = np.nextafter(np.float32(float(maximum) / 10000), np.float32(0))
        gamma = np.full((10000, 1, 1, 1), threshold, dtype=np.float32)
        self.check_finite(gamma, -gamma)
        gamma = np.full((2, 1, 1, 1), maximum, dtype=np.float32)
        self.check_finite(gamma, -gamma)

    def test_empty_axes_and_single_row(self):
        for shape in [(0, 3, 1, 1), (4, 0, 1, 1), (1, 2, 1, 1, 3)]:
            with self.subTest(shape=shape):
                gamma = np.ones(shape, dtype=np.float32)
                self.check_finite(gamma, -gamma)

    def test_nonfinite_columns(self):
        gamma = np.array(
            [[np.nan, np.inf, -np.inf, np.inf], [1, 2, 3, -np.inf]], dtype=np.float32
        ).reshape(2, 4, 1, 1)
        expected = np.array(
            [np.nan, np.inf, -np.inf, np.nan], dtype=np.float32
        ).reshape(1, 4, 1, 1)
        outputs = _GOLDEN.InTrainingUpdateGradGammaBetaSpec.golden(gamma, gamma)
        references = _GOLDEN._TorchReference()(
            torch.from_numpy(gamma), torch.from_numpy(gamma)
        )
        for result in outputs + [value.numpy() for value in references]:
            np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
