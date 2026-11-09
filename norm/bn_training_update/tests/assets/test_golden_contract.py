#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standalone CPU contract checks; expected values never come from another golden."""

import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["OMP_NUM_THREADS"] = "2"

from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "bn_contract_target", Path(__file__).with_name("golden.py")
)
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


def decimal_reference(x, stats, fmt, factor, epsilon, original_dtype):
    """Evaluate the contract from exact input floats, rounding only at output."""

    def dec(value):
        return Decimal.from_float(float(value))

    channels = len(stats[0])
    axis = 3 if fmt == "NHWC" else 1
    count = x.size // channels
    if not count:
        return [
            np.empty(x.shape),
            *[np.full(channels, np.nan) for _ in range(2)],
            *[np.zeros(channels) for _ in range(2)],
        ]
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.traps[InvalidOperation] = False
        ctx.traps[DivisionByZero] = False
        f, eps = dec(np.float32(factor)), dec(np.float32(epsilon))
        n = Decimal(count)
        bessel = n / (n - 1) if count > 1 else Decimal(0)
        y = np.empty(x.shape, dtype=np.float64)
        outputs = [np.empty(channels, dtype=np.float64) for _ in range(4)]
        multipliers, addends = [], []
        for c in range(channels):
            total, squares, scale, offset, old_mean, old_var = [
                dec(a[c]) for a in stats
            ]
            mean = total / n
            variance = squares / n - mean * mean
            if not variance.is_nan():
                variance = max(variance, Decimal(0))
            multiplier = scale / (variance + eps).sqrt()
            multipliers.append(multiplier)
            addends.append(offset - multiplier * mean)
            values = (
                f * mean + (1 - f) * old_mean,
                f * bessel * variance + (1 - f) * old_var,
                mean,
                variance,
            )
            for out, value in zip(outputs, values):
                out[c] = float(value)
        for index in np.ndindex(x.shape):
            c = index[axis]
            value = multipliers[c] * dec(x[index]) + addends[c]
            y[index] = float(value)
        return [y, *outputs]


class GoldenContractTest(unittest.TestCase):
    def check_outputs(self, actual, expected, x_dtype, stats_dtype, rtol=None):
        self.assertEqual(len(actual), 5)
        for i, (got, want) in enumerate(zip(actual, expected)):
            with self.subTest(output=i):
                dtype = np.dtype(x_dtype if i == 0 else stats_dtype)
                self.assertEqual(got.dtype, dtype)
                self.assertEqual(got.shape, want.shape)
                # Promote is double arithmetic; its FP32 carrier rounds y once.
                tolerance = (
                    rtol
                    if rtol is not None
                    else (2e-7 if dtype == np.float32 else 5e-14)
                )
                np.testing.assert_allclose(
                    got, want, rtol=tolerance, atol=tolerance, equal_nan=True
                )

    def check_promote(
        self, x, stats, fmt="NCHW", factor=0.3, epsilon=0.125, original="float32"
    ):
        expected = decimal_reference(x, stats, fmt, factor, epsilon, original)
        with np.errstate(all="ignore"):
            actual = target.golden(
                x,
                *stats,
                factor=factor,
                epsilon=epsilon,
                golden_mode="Promote",
                original_input_dtypes=[original] + ["float32"] * 6,
                input_dtypes=[x.dtype.name] + ["float64"] * 6,
                input_ori_formats=[fmt],
            )
        self.check_outputs(actual, expected, x.dtype, np.float64)
        return actual

    @staticmethod
    def stats(dtype=np.float64):
        # Deliberately unrelated to x; channel zero has negative raw variance.
        return [
            np.array(a, dtype=dtype)
            for a in ([8, -4], [1, 12], [0.75, -1.25], [0.2, -0.4], [3, -2], [5, 7])
        ]

    def test_promote_layouts_external_stats_and_factors(self):
        for fmt, shape in (
            ("NCHW", (2, 2, 2, 2)),
            ("NHWC", (2, 2, 2, 2)),
            ("NCDHW", (1, 2, 2, 2, 2)),
        ):
            for factor in (0, 1, 0.3):
                with self.subTest(format=fmt, factor=factor):
                    x = (np.arange(np.prod(shape), dtype=np.float64) - 19).reshape(
                        shape
                    )
                    result = self.check_promote(x, self.stats(), fmt, factor)
                    self.assertEqual(result[4][0], 0)
                    # Changing x alone cannot change any of the four statistics.
                    changed = self.check_promote(x + 103, self.stats(), fmt, factor)
                    for before, after in zip(result[1:], changed[1:]):
                        np.testing.assert_array_equal(before, after)

    def test_promote_retains_double_input_precision(self):
        x = np.array([0.1, 0.2, 0.3], dtype=np.float64).reshape(1, 1, 1, 3)
        stats = [
            np.array([v], dtype=np.float64)
            for v in (3 + 2**-30, 7 + 2**-29, 1 + 2**-28, 0.1, 0.2, 0.3)
        ]
        result = self.check_promote(x, stats)
        self.assertGreater(result[3][0], 1)

        import torch

        reference = target.BnTrainingUpdateThirdParty(
            factor=0.3, epsilon=0.125, input_ori_formats=["NCHW"]
        )
        actual = reference(*(torch.from_numpy(a) for a in [x, *stats]))
        expected = decimal_reference(x, stats, "NCHW", 0.3, 0.125, "float32")
        self.check_outputs(
            [a.numpy() for a in actual], expected, np.float64, np.float64
        )

    def test_promote_bessel_single_sample(self):
        x = np.array([17, -23], dtype=np.float64).reshape(1, 2, 1, 1)
        for factor in (0, 1, 0.3):
            with self.subTest(factor=factor):
                result = self.check_promote(x, self.stats(), factor=factor)
                if factor == 1:
                    np.testing.assert_array_equal(result[2], [0, 0])

    def test_promote_preserves_nonfinite_values_for_all_original_dtypes(self):
        x = np.array(
            [np.nan, np.inf, -np.inf, 1e8, -1e8, 1.00390625], dtype=np.float32
        ).reshape(1, 1, 1, 6)
        stats = [np.array([v], dtype=np.float64) for v in (0, 6, 1, 0, 2, 3)]
        for original in ("float16", "bfloat16", "float32"):
            with self.subTest(original_dtype=original):
                result = self.check_promote(x, stats, epsilon=0, original=original)
                np.testing.assert_array_equal(
                    result[0].ravel()[:5], [np.nan, np.inf, -np.inf, 1e8, -1e8]
                )
                # Both original half types use an FP32 carrier, without a BF16 cast.
                self.assertEqual(result[0].flat[-1], np.float32(1.00390625))
        for total in (np.inf, np.nan):
            with self.subTest(nonfinite_sum=total):
                nonfinite_stats = [a.copy() for a in stats]
                nonfinite_stats[0][0] = total
                self.check_promote(x, nonfinite_stats, original="float32")

    def test_arch35_empty_batch_extension(self):
        import torch

        for fmt, shape in (
            ("NCHW", (0, 2, 3, 4)),
            ("NHWC", (0, 3, 4, 2)),
            ("NCDHW", (0, 2, 3, 4, 5)),
        ):
            for original in ("float16", "bfloat16", "float32"):
                with self.subTest(format=fmt, original_dtype=original):
                    x = np.empty(shape, dtype=np.float32)
                    self.check_promote(x, self.stats(), fmt, original=original)
                    expected = decimal_reference(
                        x, self.stats(), fmt, 0.3, 0.125, original
                    )
                    actual = target.golden(
                        x,
                        *self.stats(np.float32),
                        factor=0.3,
                        epsilon=0.125,
                        input_ori_formats=[fmt],
                    )
                    self.check_outputs(actual, expected, np.float32, np.float32)
                    reference = target.BnTrainingUpdateThirdParty(
                        factor=0.3, epsilon=0.125, input_ori_formats=[fmt]
                    )
                    tensors = reference(
                        *(torch.from_numpy(a) for a in [x, *self.stats(np.float32)])
                    )
                    self.check_outputs(
                        [a.numpy() for a in tensors], expected, np.float32, np.float32
                    )

    def test_promote_accepts_missing_metadata_and_rejects_zero_channels(self):
        for dtype in (np.float32, np.float64):
            x = np.array([np.nan, np.inf, -np.inf, 1e8], dtype=dtype).reshape(
                1, 2, 1, 2
            )
            stats = self.stats()
            expected = decimal_reference(x, stats, "NCHW", 0.3, 0.125, "float32")
            for metadata in (
                {},
                {"input_dtypes": [np.dtype(dtype).name]},
                {"original_input_dtypes": []},
                {"original_input_dtypes": ["float16"]},
            ):
                with (
                    self.subTest(dtype=dtype, metadata=metadata),
                    np.errstate(invalid="ignore"),
                ):
                    result = target.golden(
                        x,
                        *stats,
                        factor=0.3,
                        epsilon=0.125,
                        golden_mode="Promote",
                        **metadata,
                    )
                    self.check_outputs(result, expected, dtype, np.float64)
        with self.assertRaisesRegex(ValueError, "C must be greater than zero"):
            target.golden(
                np.empty((1, 0, 2, 2)),
                *[np.empty(0) for _ in range(6)],
                factor=0.3,
                epsilon=0.125,
                golden_mode="Promote",
            )

    def test_torch_fp32_independently_and_consumes_both_stats(self):
        import torch

        torch.set_num_threads(2)
        for fmt, shape in (
            ("NCHW", (2, 2, 2, 2)),
            ("NHWC", (2, 2, 2, 2)),
            ("NCDHW", (1, 2, 2, 2, 2)),
        ):
            for factor in (0, 1, 0.3):
                with self.subTest(format=fmt, factor=factor):
                    x = (np.arange(np.prod(shape), dtype=np.float32) / 8 - 3).reshape(
                        shape
                    )
                    reference = target.BnTrainingUpdateThirdParty(
                        factor=factor, epsilon=0.125, input_ori_formats=[fmt]
                    )
                    results = []
                    for perturb in (None, 0, 1):
                        stats = self.stats(np.float32)
                        if perturb is not None:
                            stats[perturb] += np.float32(2)
                        expected = decimal_reference(
                            x, stats, fmt, factor, 0.125, "float32"
                        )
                        tensors = reference(*(torch.from_numpy(a) for a in [x, *stats]))
                        self.assertTrue(
                            all(
                                a.device.type == "cpu" and a.dtype == torch.float32
                                for a in tensors
                            )
                        )
                        actual = [a.numpy() for a in tensors]
                        self.check_outputs(
                            actual, expected, np.float32, np.float32, rtol=3e-6
                        )
                        results.append(actual)
                    for changed in results[1:]:
                        self.assertFalse(np.array_equal(results[0][0], changed[0]))
                        self.assertFalse(np.array_equal(results[0][4], changed[4]))
                    self.assertFalse(np.array_equal(results[0][3], results[1][3]))


if __name__ == "__main__":
    unittest.main()
