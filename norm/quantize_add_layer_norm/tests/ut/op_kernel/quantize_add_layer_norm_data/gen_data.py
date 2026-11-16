#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate deterministic FP32 inputs for the QuantizeAddLayerNorm kernel UT.

Usage:
    python3 gen_data.py '(N, D)' '(D)' 'float32'
"""

import sys
from pathlib import Path

import numpy as np


def parse_shape(value):
    return tuple(
        int(item)
        for item in value.strip().strip("(").strip(")").split(",")
        if item.strip()
    )


def uniform(shape, low, high, generator):
    return generator.uniform(low, high, size=shape).astype(np.float32)


def write_tensor(path, tensor):
    Path(path).write_bytes(np.ascontiguousarray(tensor).tobytes())


def generate(x_shape_value, gamma_shape_value, dtype):
    if dtype != "float32":
        raise ValueError("kernel UT input generation supports float32 only")

    x_shape = parse_shape(x_shape_value)
    gamma_shape = parse_shape(gamma_shape_value)
    if (
        not x_shape
        or any(dim <= 0 for dim in x_shape + gamma_shape)
        or gamma_shape != (x_shape[-1],)
    ):
        raise ValueError("gamma shape must equal the last dimension of x")

    generator = np.random.default_rng(20260804)
    tensors = {
        "x1": uniform(x_shape, -0.5, 0.5, generator),
        "x2": uniform(x_shape, -0.5, 0.5, generator),
        "bias": uniform(x_shape, -0.2, 0.2, generator),
        "gamma": uniform(gamma_shape, 0.5, 1.5, generator),
        "beta": uniform(gamma_shape, -0.5, 0.5, generator),
        "scales": uniform(gamma_shape, 0.5, 2.0, generator),
        "zero_points": generator.integers(
            -3, 4, size=gamma_shape, dtype=np.int32
        ).astype(np.float32),
    }
    for name, tensor in tensors.items():
        write_tensor(f"{dtype}_input_{name}.bin", tensor)

    print(f"generated inputs: x_shape={x_shape}, gamma_shape={gamma_shape}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    generate(sys.argv[1], sys.argv[2], sys.argv[3])
