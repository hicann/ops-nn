#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
TTK custom golden for clipped_swiglu_grad (backward of ClippedSwiglu).

Inputs (positional, in op-def order):
    grad_y      : numpy array (fp16/fp32/bf16), same shape as forward output y
    x           : numpy array (fp16/fp32/bf16), forward input
    group_index : numpy int64 array, OPTIONAL
Attributes (via **kwargs):
    dim         : int   (default -1)
    alpha       : float (default 1.702)
    limit       : float (default 7.0)
    bias        : float (default 1.0)
    interleaved : bool  (default True)
Output:
    grad_x : same dtype as x, same shape as x. NON-inplace.

Backward formula:
    A = clamp(a, max=limit);  B = clamp(b, -limit, limit);  s = sigmoid(alpha * A)
    maskA = (a <= limit);     maskB = (-limit <= b <= limit)
    da = dy * (B + bias) * s * (1 + alpha * A * (1 - s)) * maskA
    db = dy * A * s * maskB
    dx scatter: interleaved -> dx[::2]=da, dx[1::2]=db;  front/back -> dx[:h]=da, dx[h:]=db
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def _prod(seq):
    p = 1
    for v in seq:
        p *= int(v)
    return p


def __golden_clipped_swiglu_grad(*input_arrays, **kwargs):
    grad_y = np.asarray(input_arrays[0])
    x = np.asarray(input_arrays[1])
    group_index = None
    if len(input_arrays) > 2 and input_arrays[2] is not None:
        group_index = np.asarray(input_arrays[2])

    dim = int(kwargs.get("dim", -1))
    alpha = float(kwargs.get("alpha", 1.702))
    limit = float(kwargs.get("limit", 7.0))
    bias = float(kwargs.get("bias", 1.0))
    interleaved = bool(kwargs.get("interleaved", True))

    output_dtypes = kwargs.get("output_dtypes")
    if output_dtypes is not None and len(output_dtypes) > 0:
        target = str(output_dtypes[0])
    else:
        target = str(x.dtype)

    orig_shape = list(x.shape)
    ndim = len(orig_shape)
    dim_pos = dim % ndim

    pre = _prod(orig_shape[:dim_pos]) if dim_pos > 0 else 1
    cut = _prod(orig_shape[dim_pos:])

    xf = x.astype(np.float32).reshape(pre, cut)
    dyf = grad_y.astype(np.float32).reshape(pre, cut // 2)

    group = pre
    if group_index is not None:
        group = min(int(group_index.sum()), pre)

    xt = xf[:group]
    dyt = dyf[:group]

    if interleaved:
        a = xt[:, 0::2]
        b = xt[:, 1::2]
    else:
        h = cut // 2
        a = xt[:, :h]
        b = xt[:, h:]

    # 重算正向中间量
    A = np.clip(a, None, limit)
    B = np.clip(b, -limit, limit)
    with np.errstate(over="ignore", invalid="ignore"):
        s = 1.0 / (1.0 + np.exp(-alpha * A))

    # clamp 的 mask（基于原始值 a, b）
    maskA = (a <= limit).astype(np.float32)
    maskB = ((b >= -limit) & (b <= limit)).astype(np.float32)

    # 反向梯度公式
    da = dyt * (B + bias) * s * (1.0 + alpha * A * (1.0 - s)) * maskA
    db = dyt * A * s * maskB

    # 散回 dx
    dx = np.zeros((pre, cut), dtype=np.float32)
    dxt = dx[:group]
    if interleaved:
        dxt[:, 0::2] = da
        dxt[:, 1::2] = db
    else:
        h = cut // 2
        dxt[:, :h] = da
        dxt[:, h:] = db

    dx = dx.reshape(orig_shape)

    if target == "bfloat16":
        dx = dx.astype(_bf16) if _bf16 is not None else dx
    else:
        dx = dx.astype(target)
    return [dx]


__golden__ = {"kernel": {"clipped_swiglu_grad": "__golden_clipped_swiglu_grad"}}
