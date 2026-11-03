# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent mathematical checks for promoted affine Golden cancellation."""

import importlib.util
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
import torch


_SPEC = importlib.util.spec_from_file_location(
    "intu2_golden", Path(__file__).resolve().parents[1] / "assets/golden.py"
)
_GOLDEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GOLDEN)


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("beta", [-5.456753253936768, 0.125])
def test_exact_center_retains_beta_with_extreme_finite_gamma(layout, sign, beta):
    shape = (1, 1, 2, 5) if layout == "NCHW" else (1, 2, 5, 1)
    stat = (1, 1, 1, 1)
    gamma = sign * float(np.finfo(np.float32).max)
    with localcontext() as context:
        context.prec = 100
        centered = Decimal(1) - Decimal(10) / Decimal(10)
        expected = float(
            centered * Decimal.from_float(gamma) + Decimal.from_float(beta)
        )
    outputs = _GOLDEN._compute(
        torch.ones(shape, dtype=torch.float64),
        torch.full(stat, 10.0, dtype=torch.float64),
        torch.zeros(stat, dtype=torch.float64),
        torch.full(stat, gamma, dtype=torch.float64),
        torch.full(stat, beta, dtype=torch.float64),
        None,
        None,
        0.1,
        1e-5,
        layout,
    )
    np.testing.assert_array_equal(outputs[0].numpy(), np.full(shape, expected))


def test_nonfinite_scale_keeps_existing_ieee_expression():
    stat = (1, 1, 1, 1)
    outputs = _GOLDEN._compute(
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64).reshape(1, 1, 1, 3),
        torch.zeros(stat, dtype=torch.float64),
        torch.full(stat, 3.0, dtype=torch.float64),
        torch.full(stat, float("inf"), dtype=torch.float64),
        torch.zeros(stat, dtype=torch.float64),
        None,
        None,
        0.1,
        1e-5,
        "NCHW",
    )
    assert torch.isnan(outputs[0]).all()


@pytest.mark.parametrize(
    "dtype,r,value", [(np.float32, 3, 1.0 / 3.0), (np.float16, 8197, 1.9990234375)]
)
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_zero_std_uses_the_unrounded_mean(dtype, r, value, layout):
    value = float(dtype(value))
    sum_value = float(np.float32(value * r))
    with localcontext() as context:
        context.prec = 100
        centered = Decimal.from_float(value) - Decimal.from_float(sum_value) / Decimal(
            r
        )
        assert centered != 0
        expected = np.inf if centered > 0 else -np.inf
    shape = (1, 1, 1, r) if layout == "NCHW" else (1, 1, r, 1)
    stat = (1, 1, 1, 1)
    outputs = _GOLDEN._compute(
        torch.full(shape, value, dtype=torch.float64),
        torch.full(stat, sum_value, dtype=torch.float64),
        torch.zeros(stat, dtype=torch.float64),
        None,
        None,
        None,
        None,
        0.1,
        0.0,
        layout,
    )
    np.testing.assert_array_equal(outputs[0].numpy(), np.full(shape, expected))


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_finite_fp16_overflow_boundary_has_a_finite_nearest_output(sign):
    stat = (1, 1, 1, 1)
    outputs = _GOLDEN._compute(
        torch.full((1, 1, 1, 3), sign * 65504.0, dtype=torch.float64),
        torch.full(stat, -sign * 47.9970703125, dtype=torch.float64),
        torch.zeros(stat, dtype=torch.float64),
        None,
        None,
        None,
        None,
        0.1,
        1.0,
        "NCHW",
    )
    with localcontext() as context:
        context.prec = 100
        exact = (
            Decimal.from_float(sign * 65504.0)
            - Decimal.from_float(-sign * 47.9970703125) / 3
        )
    expected = float(exact)
    np.testing.assert_array_equal(outputs[0].numpy(), np.full((1, 1, 1, 3), expected))
    np.testing.assert_array_equal(
        outputs[0].numpy().astype(np.float16),
        np.full((1, 1, 1, 3), sign * 65504, dtype=np.float16),
    )


@pytest.mark.parametrize(
    "dtype,truth", [(np.float16, -3160.2751679916673), (np.float32, -536006.7774439028)]
)
def test_mixed_gate_accepts_nearest_representable_large_output(dtype, truth):
    golden = np.array([truth], dtype=np.float64)
    _GOLDEN._check_mixed_tolerance(golden.astype(dtype), golden, 0)


@pytest.mark.parametrize(
    "dtype,truth,step", [(np.float16, 1024.0, 1.0), (np.float32, 1048576.0, 0.125)]
)
def test_mixed_gate_rejects_outlier_above_32_ulp_even_with_99_percent_matches(
    dtype, truth, step
):
    golden = np.full(100, truth, dtype=np.float64)
    actual = golden.astype(dtype)
    actual[0] += 64 * step
    with pytest.raises(AssertionError, match="violates mixed tolerance"):
        _GOLDEN._check_mixed_tolerance(actual, golden, 0)


@pytest.mark.parametrize("dtype,mantissa_shift", [(np.float16, 5), (np.float32, 104)])
def test_maximum_finite_anchor_does_not_turn_the_limit_into_infinity(
    dtype, mantissa_shift
):
    golden = np.full(100, np.finfo(dtype).max, dtype=np.float64)
    actual = golden.astype(dtype)
    actual[0] = float(actual[0]) - np.ldexp(64.0, mantissa_shift)
    with pytest.raises(AssertionError, match="violates mixed tolerance"):
        _GOLDEN._check_mixed_tolerance(actual, golden, 0)


def test_mixed_gate_retains_small_value_absolute_cap():
    golden = np.zeros(100, dtype=np.float64)
    actual = golden.astype(np.float32)
    actual[0] = 0.02
    with pytest.raises(AssertionError, match="violates mixed tolerance"):
        _GOLDEN._check_mixed_tolerance(actual, golden, 0)


@pytest.mark.parametrize("dtype,error", [(np.float16, 0.05), (np.float32, 0.005)])
def test_mixed_gate_keeps_99_percent_ratio_threshold(dtype, error):
    golden = np.zeros(100, dtype=np.float64)
    actual = golden.astype(dtype)
    actual[0] = error
    _GOLDEN._check_mixed_tolerance(actual, golden, 0)
    actual[1] = error
    with pytest.raises(AssertionError, match="matched_ratio"):
        _GOLDEN._check_mixed_tolerance(actual, golden, 0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_mixed_gate_applies_output_narrowing_to_infinite_actual(sign):
    actual = np.array([sign * np.inf], dtype=np.float16)
    _GOLDEN._check_mixed_tolerance(
        actual, np.array([sign * 65520.0], dtype=np.float64), 0
    )
    with pytest.raises(AssertionError, match="NaN/Inf category mismatches"):
        _GOLDEN._check_mixed_tolerance(
            actual, np.array([sign * 65519.9990234375], dtype=np.float64), 0
        )


@pytest.mark.parametrize(
    "actual,golden", [(np.nan, 0.0), (np.inf, -np.inf), (np.inf, np.nan)]
)
def test_mixed_gate_rejects_nonfinite_category_and_sign_mismatches(actual, golden):
    with pytest.raises(AssertionError, match="NaN/Inf category mismatches"):
        _GOLDEN._check_mixed_tolerance(
            np.array([actual], dtype=np.float32), np.array([golden]), 0
        )


@pytest.mark.parametrize("extra_error", [0.0, 4.0])
def test_mixed_gate_finds_the_dynamic_anchor_after_a_chunk_boundary(extra_error):
    golden = np.zeros((1 << 20) + 1, dtype=np.float64)
    golden[-1] = -536006.7774439028
    actual = golden.astype(np.float32)
    actual[-1] += extra_error
    if extra_error:
        with pytest.raises(AssertionError, match="violates mixed tolerance"):
            _GOLDEN._check_mixed_tolerance(actual, golden, 0)
    else:
        _GOLDEN._check_mixed_tolerance(actual, golden, 0)
