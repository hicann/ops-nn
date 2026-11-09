#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "bn_training_update_golden", Path(__file__).with_name("golden.py")
)
golden_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden_module)


class GoldenTest(unittest.TestCase):
    def test_promote_preserves_nonfinite_values(self):
        x = np.array([[[[np.nan, np.inf, -np.inf]]]], dtype=np.float32)
        zero = np.zeros(1, dtype=np.float64)
        one = np.ones(1, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            y, *_ = golden_module.golden(
                x,
                zero,
                one,
                one,
                zero,
                zero,
                one,
                factor=0.1,
                epsilon=1e-5,
                input_dtypes=["float32"],
                original_input_dtypes=["float16"],
                golden_mode="Promote",
            )
        np.testing.assert_array_equal(y.ravel(), [np.nan, np.inf, -np.inf])

    def test_fp32_nonfinite_values_are_preserved(self):
        x = np.array([[[[np.nan, np.inf, -np.inf]]]], dtype=np.float32)
        zero = np.zeros(1, dtype=np.float32)
        one = np.ones(1, dtype=np.float32)
        y, *_ = golden_module.golden(
            x,
            zero,
            one,
            one,
            zero,
            zero,
            one,
            factor=0.1,
            epsilon=1e-5,
            input_dtypes=["float32"],
        )
        np.testing.assert_array_equal(y.ravel(), [np.nan, np.inf, -np.inf])

    def test_promote_retains_statistic_precision(self):
        x = np.ones((1, 1, 1, 3), dtype=np.float64)
        total = np.array([3.0 + 2**-30], dtype=np.float64)
        one = np.ones(1, dtype=np.float64)
        _, _, _, batch_mean, _ = golden_module.golden(
            x,
            total,
            total * 2,
            one,
            one,
            one,
            one,
            factor=0.1,
            epsilon=1e-5,
            input_dtypes=["float64"],
            original_input_dtypes=["float32"],
            golden_mode="Promote",
        )
        self.assertEqual(batch_mean.dtype, np.float64)
        self.assertGreater(batch_mean[0], 1.0)

    def test_third_party_layouts(self):
        import torch

        for fmt, shape in [
            ("NCHW", (2, 3, 2, 4)),
            ("NHWC", (2, 2, 4, 3)),
            ("NCDHW", (2, 3, 2, 2, 4)),
        ]:
            for dtype in (np.float16, np.float32):
                with self.subTest(format=fmt, shape=shape, dtype=dtype):
                    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape) / dtype(
                        16
                    )
                    stats = [
                        np.array(v, dtype=np.float32)
                        for v in (
                            [1, -2, 3],
                            [10, 20, 30],
                            [0.5, 1, 2],
                            [-1, 0, 1],
                            [-2, 0, 2],
                            [1, 2, 3],
                        )
                    ]
                    expected = golden_module.golden(
                        x,
                        *stats,
                        factor=0.25,
                        epsilon=0.125,
                        input_formats=[fmt],
                    )
                    reference = golden_module.BnTrainingUpdateThirdParty(
                        factor=0.25,
                        epsilon=0.125,
                        input_formats=[fmt],
                    )
                    actual = reference(*(torch.from_numpy(a) for a in [x, *stats]))
                    for a, e in zip(actual, expected):
                        np.testing.assert_allclose(a.numpy(), e, rtol=1e-6, atol=1e-6)

    def test_negative_variance_is_clamped_before_all_outputs(self):
        import torch

        x = np.array([0, 4, 0, 2, 0, 4], dtype=np.float32).reshape(1, 3, 1, 2)
        stats = [
            np.array(v, dtype=np.float32)
            for v in (
                [4, 2, 4],
                [4, np.nextafter(np.float32(2), np.float32(0)), 12],
                [2, 4, 0.5],
                [1, -1, 3],
                [0, 1, 2],
                [8, 16, 4],
            )
        ]
        actual = golden_module.golden(x, *stats, factor=0.5, epsilon=0.25)
        # The first two channels have negative raw variance, including cancellation near zero.
        expected = [
            np.array([-7, 9, -9, 7, 7 / 3, 11 / 3], dtype=np.float32).reshape(x.shape),
            np.array([1, 1, 2], dtype=np.float32),
            np.array([4, 8, 4], dtype=np.float32),
            np.array([2, 1, 2], dtype=np.float32),
            np.array([0, 0, 2], dtype=np.float32),
        ]
        reference = golden_module.BnTrainingUpdateThirdParty(factor=0.5, epsilon=0.25)
        torch_outputs = reference(*(torch.from_numpy(a) for a in [x, *stats]))
        for got, torch_got, want in zip(actual, torch_outputs, expected):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(torch_got.numpy(), want, rtol=1e-6, atol=1e-6)

    def test_statistics_use_rounded_reciprocal_multiplication(self):
        import torch

        x = np.zeros((1, 1, 1, 3), dtype=np.float32)
        stats = [np.array([v], dtype=np.float32) for v in (7, 30, 1, 0, 0, 1)]
        actual = golden_module.golden(x, *stats, factor=0.5, epsilon=0.25)
        reference = golden_module.BnTrainingUpdateThirdParty(factor=0.5, epsilon=0.25)
        torch_outputs = reference(*(torch.from_numpy(a) for a in [x, *stats]))
        # FP32(7) * FP32(1/3) differs from FP32(7) / FP32(3) by one ULP.
        expected_mean = np.array([2.3333335], dtype=np.float32)
        np.testing.assert_array_equal(actual[3], expected_mean)
        np.testing.assert_array_equal(torch_outputs[3].numpy(), expected_mean)

    def test_normalization_across_multiple_work_blocks(self):
        x = np.empty((1, 2, 1024, 1024), dtype=np.float32)
        x[:, 0] = 1
        x[:, 1] = -2
        stats = [
            np.array(v, dtype=np.float32)
            for v in (
                [0, 0],
                [0, 3.75 * 1024 * 1024],
                [1, 3],
                [1, -3],
                [0, 0],
                [1, 1],
            )
        ]
        y, *_ = golden_module.golden(x, *stats, factor=0.5, epsilon=0.25)
        np.testing.assert_array_equal(
            y[:, 0], np.full((1, 1024, 1024), 3, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            y[:, 1], np.full((1, 1024, 1024), -6, dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
