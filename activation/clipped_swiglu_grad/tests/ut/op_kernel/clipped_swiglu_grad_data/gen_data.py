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

import numpy as np
import sys


def do_clippedSwigluGrad(
    x, grad_y, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True
):
    ndim = x.ndim
    dim = dim % ndim
    pre = 1
    for i in range(dim):
        pre *= x.shape[i]
    cut = 1
    for i in range(dim, ndim):
        cut *= x.shape[i]

    xf = x.astype(np.float32).reshape(pre, cut)
    dyf = grad_y.astype(np.float32).reshape(pre, cut // 2)

    if interleaved:
        a = xf[:, 0::2]
        b = xf[:, 1::2]
    else:
        h = cut // 2
        a = xf[:, :h]
        b = xf[:, h:]

    A = np.clip(a, None, limit)
    B = np.clip(b, -limit, limit)
    s = 1.0 / (1.0 + np.exp(-alpha * A))

    maskA = (a <= limit).astype(np.float32)
    maskB = ((b >= -limit) & (b <= limit)).astype(np.float32)

    da = dyf * (B + bias) * s * (1.0 + alpha * A * (1.0 - s)) * maskA
    db = dyf * A * s * maskB

    dx = np.zeros((pre, cut), dtype=np.float32)
    if interleaved:
        dx[:, 0::2] = da
        dx[:, 1::2] = db
    else:
        h = cut // 2
        dx[:, :h] = da
        dx[:, h:] = db

    return dx.reshape(x.shape)


params_info = {
    "test_case_bf16_shortH": {"x_shape": [3200, 5760], "dtype": np.float16},
    "test_case_fp16_shortH": {"x_shape": [3200, 5760], "dtype": np.float16},
    "test_case_fp32_shortH": {"x_shape": [3200, 5760], "dtype": np.float32},
    "test_case_bf16_longH": {"x_shape": [3200, 23040], "dtype": np.float16},
    "test_case_fp16_longH": {"x_shape": [3200, 23040], "dtype": np.float16},
    "test_case_fp32_longH": {"x_shape": [3200, 23040], "dtype": np.float32},
}


def main():
    case = sys.argv[1]
    info = params_info[case]
    dtype = info["dtype"]
    if "bf16" in case:
        try:
            from ml_dtypes import bfloat16

            dtype = bfloat16
        except ImportError:
            dtype = np.float16

    np.random.seed(42)
    x_shape = info["x_shape"]
    x = np.random.randn(*x_shape).astype(dtype)
    grad_y_shape = list(x_shape)
    grad_y_shape[-1] = grad_y_shape[-1] // 2
    grad_y = np.random.randn(*grad_y_shape).astype(dtype)

    group_index = np.random.randint(1, 11, size=(10,), dtype=np.int64)

    x.tofile("input_x.bin")
    grad_y.tofile("input_grad_y.bin")
    group_index.tofile("input_group_index.bin")


if __name__ == "__main__":
    main()
