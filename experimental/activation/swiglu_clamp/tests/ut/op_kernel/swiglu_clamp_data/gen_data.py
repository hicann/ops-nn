#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# SwigluClamp kernel UT golden generator.
# x [..., 2N] -> golden y [..., N]:
#   gate = x[..., :N], up = x[..., N:]
#   golden = silu(gate).clip(max=limit) * up.clip(-limit, limit)
#   silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
#
# NOTE: SwigluClamp does NOT support +/-inf or nan (see README constraints), so the
# input pool uses finite values that cover the clamp boundary (limit) and the silu
# linear/saturating regions.
#
# usage: gen_data.py '<shape>' <float16|float32> <limit>
#   shape is x's shape, last dim must be even (2N).

import sys
import os
import numpy as np


def parse_shape(shape_str):
    shape_str = shape_str.strip("(").strip(")")
    return tuple(int(x) for x in shape_str.split(","))


def silu(x):
    # silu(g) = g * sigmoid(g), computed in fp32 for accuracy
    return x / (1.0 + np.exp(-x))


def gen(shape_str, dtype_str, limit_str):
    dtype_map = {"float32": np.float32, "float16": np.float16}
    if dtype_str not in dtype_map:
        raise ValueError("dtype must be float16 or float32")
    np_dtype = dtype_map[dtype_str]

    limit = float(limit_str)
    shape = parse_shape(shape_str)
    size = int(np.prod(shape))

    # finite boundary values: cover clamp edges (±limit), silu sat region, small/zero.
    pool = np.array(
        [
            0.0,
            0.25,
            -0.25,
            0.5,
            -0.5,
            1.0,
            -1.0,
            2.0,
            -2.0,
            3.0,
            -3.0,
            5.0,
            -5.0,
            limit,
            -limit,
            limit + 1.0,
            -(limit + 1.0),
        ]
    )
    x = np.random.choice(pool, size=size).reshape(shape).astype(np_dtype)

    N = shape[-1] // 2
    gate = x[..., :N].astype(np.float32)
    up = x[..., N:].astype(np.float32)

    s = silu(gate)
    s = np.minimum(s, limit)  # silu-then-clamp upper bound
    up_c = np.clip(up, -limit, limit)  # up bidirectional clamp
    golden = (s * up_c).astype(np_dtype)

    x.astype(np_dtype).tofile("{0}_input_t_swiglu_clamp.bin".format(dtype_str))
    golden.astype(np_dtype).tofile("{0}_golden_t_swiglu_clamp.bin".format(dtype_str))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: gen_data.py '<shape>' <float16|float32> <limit>")
        exit(1)
    os.system("rm -rf *.bin")
    gen(sys.argv[1], sys.argv[2], sys.argv[3])
