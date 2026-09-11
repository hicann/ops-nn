#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden generator for QuantizeAddLayerNorm (ascend950 kernel UT, fp32).

Formula:
    x = x1 + x2 + bias
    norm = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    mul mode        : y = int8(round(norm * scales + zero_points))      (axis = -65535)
    per_channel div : y = int8(round(norm / scales + zero_points))      (axis other)
    per_tensor      : y = int8(round(norm * scales[0] + zero_points[0])) (axis = 65535)

Usage:
    python3 gen_data.py '(N, D)' '(D)' 'float32' [eps]
"""

import sys
import numpy as np


def parse_shape(shape_str):
    shape_str = shape_str.strip().strip("(").strip(")")
    return tuple(int(x) for x in shape_str.split(",") if x.strip() != "")


def quantize_add_layer_norm(x1, x2, gamma, beta, bias, scales, zero_points, eps, mode):
    x = x1.astype(np.float32) + x2.astype(np.float32) + bias.astype(np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(np.power(x - mean, 2), axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    norm = (x - mean) * rstd * gamma.astype(np.float32) + beta.astype(np.float32)

    s = scales.astype(np.float32)
    zp = zero_points.astype(np.float32) if zero_points is not None else None
    if mode == "per_tensor":
        s = s.reshape(-1)[0:1]
        if zp is not None:
            zp = zp.reshape(-1)[0:1]

    if mode == "div":
        q = norm / s
    else:  # mul / per_tensor
        q = norm * s
    if zp is not None:
        q = q + zp

    # kernel: Truncate(CAST_RINT) -> fp16 -> int8(CAST_TRUNC); fp32 rint + clip matches here
    q = np.rint(q)
    q = np.clip(q, -128, 127)
    return q.astype(np.int8), x.astype(np.float32)


def to_bin(arr, path):
    np.asarray(arr).tofile(path)


def gen_data_and_golden(x_shape_str, gamma_shape_str, d_type="float32", eps=1e-5):
    assert d_type == "float32", "kernel UT golden currently supports float32 only"
    x_shape = parse_shape(x_shape_str)
    gamma_shape = parse_shape(gamma_shape_str)
    d = x_shape[-1]
    assert tuple(gamma_shape) == (d,), "gamma shape must be (D,)"

    rng = np.random.default_rng(20260804)
    x1 = rng.uniform(-0.5, 0.5, x_shape).astype(np.float32)
    x2 = rng.uniform(-0.5, 0.5, x_shape).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, x_shape).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, gamma_shape).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, gamma_shape).astype(np.float32)
    scales = rng.uniform(0.5, 2.0, (d,)).astype(np.float32)
    zero_points = rng.integers(-3, 4, (d,)).astype(np.float32)

    prefix = "{}_".format(d_type)
    to_bin(x1, prefix + "input_x1.bin")
    to_bin(x2, prefix + "input_x2.bin")
    to_bin(bias, prefix + "input_bias.bin")
    to_bin(gamma, prefix + "input_gamma.bin")
    to_bin(beta, prefix + "input_beta.bin")
    to_bin(scales, prefix + "input_scales.bin")
    to_bin(zero_points, prefix + "input_zero_points.bin")

    for mode in ("mul", "div", "per_tensor"):
        y, x = quantize_add_layer_norm(
            x1, x2, gamma, beta, bias, scales, zero_points, eps, mode
        )
        to_bin(y, prefix + "golden_y_{}.bin".format(mode))
        if mode == "mul":
            to_bin(x, prefix + "golden_x.bin")
    print(
        "gen_data ok: x_shape={}, gamma_shape={}, eps={}".format(
            x_shape, gamma_shape, eps
        )
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    eps_val = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], eps_val)
