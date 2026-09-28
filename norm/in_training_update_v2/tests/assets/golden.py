#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""INTrainingUpdateV2 golden and correlated-input generator.

The seven parameters follow ``in_training_update_v2_def.cpp`` exactly.  The
customizer derives ``sum`` and ``square_sum`` from ``x`` so normal cases model
the preceding INTrainingReduceV2 node.  Tagged ``negvar`` cases deliberately
provide a negative raw variance to cover the public numerical guard.
"""

import numpy as np
import torch


__spec__ = {"in_training_update_v2": "INTrainingUpdateV2TestSpec"}
__golden__ = {
    "kernel": {"in_training_update_v2": "in_training_update_v2_golden"},
}


_TOL = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}

_MIXED_TOLERANCE = {
    "float16": {"rtol": 2**-9, "atol": 2**-9, "max_abs": 1e-1},
    "float32": {"rtol": 2**-10, "atol": 2**-16, "max_abs": 1e-2},
}
_REQUIRED_MATCHED_RATIO = 0.99


def _attr(kwargs, name, default):
    value = kwargs.get(name)
    if value is None and isinstance(kwargs.get("attributes"), dict):
        value = kwargs["attributes"].get(name)
    if value is None:
        return default
    return float(value)


def _dtype_name(dtype):
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0] if dtype else None
    if dtype is None:
        return None
    name = str(dtype).lower().replace("torch.", "").replace("numpy.", "")
    return {
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "fp64": "float64",
        "double": "float64",
    }.get(name, name)


def _as_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    return torch.from_numpy(np.ascontiguousarray(np.asarray(value)))


def _empty_stat_like(value, device):
    tensor = _as_tensor(value)
    return torch.empty_like(tensor, dtype=_compute_dtype(tensor), device=device)


def _layout(kwargs):
    formats = kwargs.get("input_ori_formats") or kwargs.get("input_formats") or ()
    value = formats[0] if isinstance(formats, (list, tuple)) and formats else formats
    value = str(value).upper()
    if "NHWC" in value:
        return "NHWC"
    return "NCHW"


def _compute_dtype(*values):
    dtype = torch.float32
    for value in values:
        if value is not None and value.dtype.is_floating_point:
            dtype = torch.promote_types(dtype, value.dtype)
    return dtype


def _reshape_stat(value, n, c, name):
    if value.numel() != n * c:
        raise ValueError(f"{name} must contain N*C={n * c} elements")
    return value.reshape(n, c)


def _reshape_affine(value, n, c, name):
    if value.numel() not in (c, n * c):
        raise ValueError(f"{name} must contain C or N*C elements")
    rows = value.numel() // c
    matrix = value.reshape(rows, c)
    return matrix.expand(n, c) if rows == 1 else matrix


def _compute(
    x,
    sum_value,
    square_sum,
    gamma,
    beta,
    mean,
    variance,
    momentum,
    epsilon,
    layout,
):
    x_tensor = _as_tensor(x)
    if x_tensor.ndim != 4:
        raise ValueError(f"x must be rank 4, got rank {x_tensor.ndim}")
    n = x_tensor.shape[0]
    c = x_tensor.shape[-1] if layout == "NHWC" else x_tensor.shape[1]

    # Match the dedicated device empty template: inspect only x metadata and
    # return empty outputs before spatial math or optional tensor conversion.
    if n == 0 or c == 0:
        return [
            x_tensor.clone(),
            _empty_stat_like(sum_value, x_tensor.device),
            _empty_stat_like(square_sum, x_tensor.device),
        ]

    r = (
        x_tensor.shape[1] * x_tensor.shape[2]
        if layout == "NHWC"
        else x_tensor.shape[2] * x_tensor.shape[3]
    )
    if r <= 0:
        raise ValueError("H and W must be positive for a non-empty N/C tensor")

    sum_tensor = _as_tensor(sum_value)
    square_tensor = _as_tensor(square_sum)
    has_affine = gamma is not None and beta is not None
    has_running = mean is not None and variance is not None
    gamma_tensor = _as_tensor(gamma) if has_affine else None
    beta_tensor = _as_tensor(beta) if has_affine else None
    running_mean = _as_tensor(mean) if has_running else None
    running_variance = _as_tensor(variance) if has_running else None

    dtype = _compute_dtype(
        x_tensor,
        sum_tensor,
        square_tensor,
        gamma_tensor,
        beta_tensor,
        running_mean,
        running_variance,
    )
    x_compute = x_tensor.to(dtype=dtype)
    sum_matrix = _reshape_stat(sum_tensor.to(dtype=dtype), n, c, "sum")
    square_matrix = _reshape_stat(square_tensor.to(dtype=dtype), n, c, "square_sum")

    # Float attributes are quantized once to their public fp32 type.  Arithmetic
    # derived from shapes and attributes then stays in the promoted compute
    # dtype so Golden Promote remains an independent high-precision reference.
    scalar_device = x_compute.device

    def scalar(value):
        return torch.tensor(value, dtype=dtype, device=scalar_device)

    one = scalar(1.0)
    r_value = scalar(r)
    inv_r = one / r_value
    bessel = scalar(0.0) if r <= 1 else r_value / (r_value - one)
    momentum_value = scalar(float(np.float32(momentum)))
    one_minus_momentum = one - momentum_value
    epsilon_value = scalar(float(np.float32(epsilon)))

    current_mean = torch.mul(sum_matrix, inv_r)
    raw_variance = torch.sub(
        torch.mul(square_matrix, inv_r), torch.mul(current_mean, current_mean)
    )
    biased_variance = torch.where(
        raw_variance < 0, torch.zeros_like(raw_variance), raw_variance
    )
    current_variance = (
        torch.zeros_like(biased_variance)
        if r == 1
        else torch.mul(biased_variance, bessel)
    )
    std_value = torch.sqrt(torch.add(biased_variance, epsilon_value))

    broadcast_shape = (n, 1, 1, c) if layout == "NHWC" else (n, c, 1, 1)
    if has_affine:
        gamma_matrix = _reshape_affine(gamma_tensor.to(dtype=dtype), n, c, "gamma")
        beta_matrix = _reshape_affine(beta_tensor.to(dtype=dtype), n, c, "beta")
        scale = torch.div(gamma_matrix, std_value)
        bias = torch.sub(beta_matrix, torch.mul(scale, current_mean))
        y = torch.add(
            torch.mul(x_compute, scale.reshape(broadcast_shape)),
            bias.reshape(broadcast_shape),
        )
    else:
        y = torch.div(
            torch.sub(x_compute, current_mean.reshape(broadcast_shape)),
            std_value.reshape(broadcast_shape),
        )

    if has_running:
        running_mean_matrix = _reshape_stat(running_mean.to(dtype=dtype), n, c, "mean")
        running_var_matrix = _reshape_stat(
            running_variance.to(dtype=dtype), n, c, "variance"
        )
        batch_mean = torch.add(
            torch.mul(current_mean, momentum_value),
            torch.mul(running_mean_matrix, one_minus_momentum),
        )
        batch_variance = torch.add(
            torch.mul(current_variance, momentum_value),
            torch.mul(running_var_matrix, one_minus_momentum),
        )
    else:
        batch_mean = current_mean.clone()
        batch_variance = current_variance.clone()

    return [
        y,
        batch_mean.reshape(sum_tensor.shape),
        batch_variance.reshape(sum_tensor.shape),
    ]


def _kernel_golden(
    x,
    sum,
    square_sum,
    gamma=None,
    beta=None,
    mean=None,
    variance=None,
    momentum=0.1,
    epsilon=1e-5,
    **kwargs,
):
    momentum_value = _attr({**kwargs, "momentum": momentum}, "momentum", 0.1)
    epsilon_value = _attr({**kwargs, "epsilon": epsilon}, "epsilon", 1e-5)
    outputs = _compute(
        x,
        sum,
        square_sum,
        gamma,
        beta,
        mean,
        variance,
        momentum_value,
        epsilon_value,
        _layout(kwargs),
    )
    # Keep the computation precision even when an individual output's declared
    # dtype is narrower (e.g. promoted fp16 x with fp64 statistics). Device output
    # dtypes select comparison thresholds, not the precision of the CPU truth.
    return [
        np.ascontiguousarray(output.detach().cpu().contiguous().numpy())
        for output in outputs
    ]


def _customize_inputs(
    x,
    sum,
    square_sum,
    gamma=None,
    beta=None,
    mean=None,
    variance=None,
    testcase_name=None,
    **kwargs,
):
    del sum, square_sum
    layout = _layout(kwargs)
    testcase_name = testcase_name or ""
    if "exact_zero_var" in (testcase_name or ""):
        # Keep one complete spatial plane exactly constant.  This locks the
        # epsilon=0 IEEE category regression that otherwise depends on random
        # clipping happening to produce an all-boundary channel.
        x = np.array(x, copy=True, order="C")
        if x.size:
            constant = 176.0 if "large_r" in testcase_name else 1.0
            if layout == "NHWC":
                x[0, :, :, 0] = np.asarray(constant, dtype=x.dtype)
            else:
                x[0, 0, :, :] = np.asarray(constant, dtype=x.dtype)
    if "overflow_product_guard" in testcase_name:
        # Both square_sum * R and sum * sum overflow FP32 for this plane, but
        # their exact values differ. The exact-zero detector must not treat
        # the common (+Inf, -Inf) product expansion as equality.
        x = np.array(x, copy=True, order="C")
        values = np.asarray([8.0e18, 1.2e19], dtype=x.dtype)
        if layout == "NHWC":
            x[0, 0, :, 0] = values
        else:
            x[0, 0, 0, :] = values
    # Precision coverage requires NaN, +Inf, -Inf and mixed infinities for
    # every supported x dtype.  Keep these values out of the generic random
    # range syntax and inject them only into explicitly tagged supplemental
    # cases so all existing rows remain byte-for-byte and semantically stable.
    if any(
        tag in testcase_name for tag in ("x_nan", "x_posinf", "x_neginf", "x_both_inf")
    ):
        x = np.array(x, copy=True, order="C")
        flat_x = x.reshape(-1)
        if "x_nan" in testcase_name:
            flat_x[0] = np.nan
        elif "x_posinf" in testcase_name:
            flat_x[-1] = np.inf
        elif "x_neginf" in testcase_name:
            flat_x[flat_x.size // 2] = -np.inf
        else:
            flat_x[0] = -np.inf
            flat_x[-1] = np.inf

    x_f32 = np.ascontiguousarray(np.asarray(x).astype(np.float32, copy=False))
    x_tensor = torch.from_numpy(x_f32)
    reduce_axes = (1, 2) if layout == "NHWC" else (2, 3)
    sum_value = (
        torch.sum(x_tensor, dim=reduce_axes, keepdim=True, dtype=torch.float32)
        .cpu()
        .numpy()
    )
    square_value = (
        torch.sum(
            torch.mul(x_tensor, x_tensor),
            dim=reduce_axes,
            keepdim=True,
            dtype=torch.float32,
        )
        .cpu()
        .numpy()
    )
    if "negvar" in testcase_name:
        sum_value = np.ones_like(sum_value, dtype=np.float32)
        square_value = np.zeros_like(square_value, dtype=np.float32)
    if "r1_nanvar" in testcase_name:
        sum_value = np.full_like(sum_value, np.nan, dtype=np.float32)
        square_value = np.full_like(square_value, np.nan, dtype=np.float32)
    if "r1_infvar" in testcase_name:
        sum_value = np.zeros_like(sum_value, dtype=np.float32)
        square_value = np.full_like(square_value, np.inf, dtype=np.float32)
    if variance is not None:
        variance = np.maximum(np.asarray(variance), np.float32(0.125)).astype(
            np.float32, copy=False
        )
    return (
        x,
        np.ascontiguousarray(sum_value),
        np.ascontiguousarray(square_value),
        gamma,
        beta,
        mean,
        variance,
    )


def _check_mixed_tolerance(actual, golden, output_index):
    """Enforce the single-Golden precision gate before three-party comparison."""
    actual_array = np.asarray(actual)
    golden_array = np.asarray(golden)
    if actual_array.shape != golden_array.shape:
        raise AssertionError(
            f"output {output_index} shape mismatch: "
            f"actual={actual_array.shape}, golden={golden_array.shape}"
        )
    if actual_array.size == 0:
        return

    dtype_name = _dtype_name(actual_array.dtype)
    limits = _MIXED_TOLERANCE.get(dtype_name)
    if limits is None:
        raise AssertionError(
            f"output {output_index} has unsupported precision dtype {actual_array.dtype}"
        )

    actual_flat = actual_array.reshape(-1)
    golden_flat = golden_array.reshape(-1)
    golden_nan = np.isnan(golden_flat)
    golden_posinf = np.isposinf(golden_flat)
    golden_neginf = np.isneginf(golden_flat)
    golden_finite = np.isfinite(golden_flat)
    special_matches = (
        (golden_nan & np.isnan(actual_flat))
        | (golden_posinf & np.isposinf(actual_flat))
        | (golden_neginf & np.isneginf(actual_flat))
    )
    finite_pairs = golden_finite & np.isfinite(actual_flat)
    category_matches = special_matches | finite_pairs
    if not np.all(category_matches):
        mismatch_count = int(np.count_nonzero(~category_matches))
        raise AssertionError(
            f"output {output_index} has {mismatch_count} NaN/Inf category mismatches"
        )

    matched = special_matches.copy()
    max_abs_error = 0.0
    if np.any(finite_pairs):
        actual_f64 = actual_flat[finite_pairs].astype(np.float64)
        golden_f64 = golden_flat[finite_pairs].astype(np.float64)
        abs_error = np.abs(actual_f64 - golden_f64)
        finite_matched = abs_error <= (
            limits["atol"] + limits["rtol"] * np.abs(golden_f64)
        )
        matched[finite_pairs] = finite_matched
        max_abs_error = float(np.max(abs_error))

    matched_ratio = float(np.count_nonzero(matched)) / float(matched.size)
    ulp_at_one = float(np.spacing(np.array(1.0, dtype=actual_array.dtype)))
    max_abs_limit = max(limits["max_abs"], 32.0 * ulp_at_one)
    if matched_ratio < _REQUIRED_MATCHED_RATIO or max_abs_error > max_abs_limit:
        raise AssertionError(
            f"output {output_index} violates mixed tolerance: "
            f"matched_ratio={matched_ratio:.8f} "
            f"(required>={_REQUIRED_MATCHED_RATIO}), "
            f"max_abs_error={max_abs_error:.8e} (limit<={max_abs_limit:.8e})"
        )


def _precision_pre_compare(
    y,
    batch_mean,
    batch_variance,
    golden_y,
    golden_batch_mean,
    golden_batch_variance,
):
    """Validate NPU outputs against the strict Golden without replacing cross-check."""
    outputs = (y, batch_mean, batch_variance)
    goldens = (golden_y, golden_batch_mean, golden_batch_variance)
    for output_index, (actual, golden) in enumerate(zip(outputs, goldens)):
        _check_mixed_tolerance(actual, golden, output_index)


class _INTrainingUpdateV2Compose:
    """Independent Torch composition matching the arch35 fp32 arithmetic."""

    def __init__(self, momentum=0.1, epsilon=1e-5, **kwargs):
        self.momentum = _attr({**kwargs, "momentum": momentum}, "momentum", 0.1)
        self.epsilon = _attr({**kwargs, "epsilon": epsilon}, "epsilon", 1e-5)
        # Materialize attribute rounding before torch.compile traces _impl.
        # Converting NumPy scalars to Python floats inside a full graph is an
        # unsupported Tensor.item-style operation on the A100 runner.
        self._momentum_f32 = float(np.float32(self.momentum))
        self._old_weight_f32 = float(np.float32(1.0) - np.float32(self.momentum))
        self._epsilon_f32 = float(np.float32(self.epsilon))
        self.layout = _layout(kwargs)
        self._compiled = None

    def _impl(
        self,
        x,
        sum,
        square_sum,
        gamma=None,
        beta=None,
        mean=None,
        variance=None,
    ):
        """Independent fp32 Torch composition for cross-check and timing."""
        if x.ndim != 4:
            raise ValueError(f"x must be rank 4, got rank {x.ndim}")
        n = x.shape[0]
        c = x.shape[-1] if self.layout == "NHWC" else x.shape[1]
        r = (
            x.shape[1] * x.shape[2]
            if self.layout == "NHWC"
            else x.shape[2] * x.shape[3]
        )
        if n == 0 or c == 0:
            return [
                x.clone(),
                torch.empty_like(sum, dtype=torch.float32),
                torch.empty_like(square_sum, dtype=torch.float32),
            ]
        if r <= 0:
            raise ValueError("H and W must be positive for a non-empty N/C tensor")

        x_f32 = x.to(dtype=torch.float32)
        sum_f32 = sum.to(dtype=torch.float32).reshape(n, c)
        square_f32 = square_sum.to(dtype=torch.float32).reshape(n, c)
        scalar_kwargs = {"dtype": torch.float32, "device": x.device}
        one = torch.ones((), **scalar_kwargs)
        zero = torch.zeros((), **scalar_kwargs)
        r_f32 = one * r
        r_minus_one_f32 = one * (r - 1)
        r_is_one = r_f32 == one
        inv_r = one / r_f32
        bessel_denominator = torch.where(r_is_one, one, r_minus_one_f32)
        bessel = torch.where(r_is_one, zero, r_f32 / bessel_denominator)
        momentum = torch.tensor(self._momentum_f32, **scalar_kwargs)
        old_weight = torch.tensor(self._old_weight_f32, **scalar_kwargs)
        epsilon = torch.tensor(self._epsilon_f32, **scalar_kwargs)

        current_mean = sum_f32 * inv_r
        biased_variance = torch.clamp(
            square_f32 * inv_r - current_mean * current_mean, min=0.0
        )
        current_variance = torch.where(
            r_is_one, torch.zeros_like(biased_variance), biased_variance * bessel
        )
        denominator = torch.sqrt(biased_variance + epsilon)
        broadcast_shape = (n, 1, 1, c) if self.layout == "NHWC" else (n, c, 1, 1)

        if gamma is not None and beta is not None:
            gamma_f32 = gamma.to(dtype=torch.float32).reshape(-1, c)
            beta_f32 = beta.to(dtype=torch.float32).reshape(-1, c)
            if gamma_f32.shape[0] == 1:
                gamma_f32 = gamma_f32.expand(n, c)
            if beta_f32.shape[0] == 1:
                beta_f32 = beta_f32.expand(n, c)
            multiplier = gamma_f32 / denominator
            addend = beta_f32 - multiplier * current_mean
            y_f32 = x_f32 * multiplier.reshape(broadcast_shape) + addend.reshape(
                broadcast_shape
            )
        else:
            y_f32 = (
                x_f32 - current_mean.reshape(broadcast_shape)
            ) / denominator.reshape(broadcast_shape)

        if mean is not None and variance is not None:
            old_mean = mean.to(dtype=torch.float32).reshape(n, c)
            old_variance = variance.to(dtype=torch.float32).reshape(n, c)
            batch_mean = current_mean * momentum + old_mean * old_weight
            batch_variance = current_variance * momentum + old_variance * old_weight
        else:
            batch_mean = current_mean.clone()
            batch_variance = current_variance.clone()

        return [
            y_f32.to(dtype=x.dtype),
            batch_mean.reshape(sum.shape),
            batch_variance.reshape(sum.shape),
        ]

    def __call__(
        self,
        x,
        sum,
        square_sum,
        gamma=None,
        beta=None,
        mean=None,
        variance=None,
        **kwargs,
    ):
        if kwargs.get("input_ori_formats"):
            self.layout = _layout(kwargs)
        inputs = (x, sum, square_sum, gamma, beta, mean, variance)
        if self._compiled is None:
            # Performance evidence labels this path as torch.compile.  Keep it
            # strict: a graph break or compilation/execution failure must fail
            # the remote leg instead of silently timing eager execution.
            self._compiled = torch.compile(self._impl, dynamic=True, fullgraph=True)
        return self._compiled(*inputs)


class INTrainingUpdateV2TestSpec:
    golden = _kernel_golden
    customize_inputs = _customize_inputs
    pre_compare = _precision_pre_compare
    third_party = {"torch": _INTrainingUpdateV2Compose}
    tolerance = _TOL


def in_training_update_v2_golden(
    x,
    sum,
    square_sum,
    gamma=None,
    beta=None,
    mean=None,
    variance=None,
    momentum=0.1,
    epsilon=1e-5,
    *args,
    **kwargs,
):
    del args
    return tuple(
        _kernel_golden(
            x,
            sum,
            square_sum,
            gamma,
            beta,
            mean,
            variance,
            momentum=momentum,
            epsilon=epsilon,
            **kwargs,
        )
    )


# 【不存在】ACLNN 通路：CMakeLists.txt 明确使用 aclnn_exclude，仓内不新增
# REG_OP 公共契约之外的 API。
# 【不存在】E2E 通路：该融合算子通过 GE 图模式使用，仓内无框架绑定。
