#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare kernel output with an independent NumPy FP32 reference.

Usage:
    python3 compare_data.py 'float32' <mode> [epsilon] [has_zero_points] [check_x] [bias_broadcast]
"""

import math
import sys
from pathlib import Path

import numpy as np


def load_tensor(path, dtype):
    raw = Path(path).read_bytes()
    element_size = np.dtype(dtype).itemsize
    if not raw or len(raw) % element_size != 0:
        raise ValueError(f"invalid {path} size: {len(raw)}")
    return np.frombuffer(raw, dtype=dtype).copy()


def first_mismatch(actual, expected):
    indices = np.flatnonzero(actual != expected)
    return int(indices[0]) if indices.size else None


def main(
    dtype, mode, epsilon, has_zero_points=True, check_x=True, bias_broadcast=False
):
    if dtype != "float32":
        raise ValueError("kernel UT comparison supports float32 only")
    if mode not in {"mul", "div", "per_tensor"}:
        raise ValueError(f"unknown quantization mode: {mode}")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a number")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if any(
        type(flag) is not bool for flag in (has_zero_points, check_x, bias_broadcast)
    ):
        raise TypeError("has_zero_points, check_x, and bias_broadcast must be bool")

    prefix = f"{dtype}_"
    x1 = load_tensor(prefix + "input_x1.bin", np.float32)
    x2 = load_tensor(prefix + "input_x2.bin", np.float32)
    bias = load_tensor(prefix + "input_bias.bin", np.float32)
    gamma = load_tensor(prefix + "input_gamma.bin", np.float32)
    beta = load_tensor(prefix + "input_beta.bin", np.float32)
    scales = load_tensor(prefix + "input_scales.bin", np.float32)
    zero_points = load_tensor(prefix + "input_zero_points.bin", np.float32)

    cols = gamma.size
    if cols == 0 or x1.size % cols != 0:
        raise ValueError("input size is incompatible with gamma")
    shape = (x1.size // cols, cols)
    if bias_broadcast:
        bias = bias[:cols]
    for name, tensor, expected_size in (
        ("x2", x2, x1.size),
        ("bias", bias, cols if bias_broadcast else x1.size),
        ("beta", beta, cols),
        ("scales", scales, cols),
        ("zero_points", zero_points, cols),
    ):
        if tensor.size != expected_size:
            raise ValueError(
                f"{name} has {tensor.size} elements, expected {expected_size}"
            )

    bias_shape = (1, cols) if bias_broadcast else shape
    x = (
        x1.reshape(shape)
        + x2.reshape(shape)
        + bias[: np.prod(bias_shape)].reshape(bias_shape)
    ).astype(np.float32)
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = (x - mean).astype(np.float32)
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    reciprocal_std = np.reciprocal(np.sqrt(variance + np.float32(epsilon))).astype(
        np.float32
    )
    normalized = ((centered * reciprocal_std) * gamma + beta).astype(np.float32)
    if mode == "per_tensor":
        quantized_fp = normalized * scales[0]
    elif mode == "div":
        if np.any(scales == 0):
            raise ValueError("division scales must be nonzero")
        quantized_fp = normalized / scales
    else:
        quantized_fp = normalized * scales
    if has_zero_points:
        quantized_fp = quantized_fp + (
            zero_points[0] if mode == "per_tensor" else zero_points
        )

    if not np.isfinite(quantized_fp).all():
        raise ValueError("reference quantization produced non-finite values")
    rounded = np.rint(quantized_fp)
    if np.any((rounded < -128) | (rounded > 127)):
        raise ValueError("reference quantization exceeds the confirmed int8 domain")

    expected_y = rounded.astype(np.int8).ravel()
    actual_y = load_tensor(prefix + "output_y.bin", np.int8)
    if actual_y.size != expected_y.size:
        raise AssertionError(f"y size mismatch: {actual_y.size} != {expected_y.size}")
    mismatch = first_mismatch(actual_y, expected_y)
    if mismatch is not None:
        raise AssertionError(
            f"y mismatch at {mismatch}: actual={int(actual_y[mismatch])}, expected={int(expected_y[mismatch])}"
        )

    if check_x:
        expected_x = x.ravel()
        actual_x = load_tensor(prefix + "output_x.bin", np.float32)
        if actual_x.size != expected_x.size:
            raise AssertionError(
                f"x size mismatch: {actual_x.size} != {expected_x.size}"
            )
        close = np.isclose(actual_x, expected_x, rtol=0.0, atol=1e-6, equal_nan=False)
        if not close.all():
            mismatch = int(np.flatnonzero(~close)[0])
            raise AssertionError(
                f"x mismatch at {mismatch}: actual={float(actual_x[mismatch])}, expected={float(expected_x[mismatch])}"
            )

    print(f"[PASS] mode={mode}, elements={expected_y.size}")


def parse_flag(value, name):
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def parse_epsilon(value):
    epsilon = float(value)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    return epsilon


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4, 5, 6, 7):
        print(__doc__)
        sys.exit(1)
    main(
        sys.argv[1],
        sys.argv[2],
        parse_epsilon(sys.argv[3]) if len(sys.argv) >= 4 else 1e-5,
        parse_flag(sys.argv[4], "has_zero_points") if len(sys.argv) >= 5 else True,
        parse_flag(sys.argv[5], "check_x") if len(sys.argv) >= 6 else True,
        parse_flag(sys.argv[6], "bias_broadcast") if len(sys.argv) >= 7 else False,
    )
