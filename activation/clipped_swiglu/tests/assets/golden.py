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
TTK custom golden for clipped_swiglu (ClippedSwiglu / variant SwiGlu with clamp + grouping).

Inputs (positional, in op-def order):
    x           : numpy array (fp16/fp32/bf16)
    group_index : numpy int64 array, OPTIONAL (absent -> input_arrays[1] is None or missing)
Attributes (via **kwargs, from CSV `attributes`):
    dim         : int   (default -1)  合轴切分轴
    alpha       : float (default 1.702)
    limit       : float (default 7.0)
    bias        : float (default 1.0)
    interleaved : bool  (default True)  True=奇偶切分, False=前后切分
Output:
    y : same dtype as x, shape = x with the cut axis halved. NON-inplace.

Reference (mirrors tests/ut/op_kernel/clipped_swiglu_data/gen_data.py::do_clippedSwiglu
and docs/aclnnClippedSwiglu.md):
    1. 按 dim 合轴 -> [pre, cut]
    2. group = min(sum(group_index), pre); 仅前 group 行参与计算, 其余行输出 0
    3. interleaved: A=x[:,::2], B=x[:,1::2]; else 前后: A=x[:,:h], B=x[:,h:]  (h=cut//2)
    4. A = clamp(A, max=limit); B = clamp(B, -limit, limit)
       y = A * sigmoid(alpha*A) * (B + bias)
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


def __golden_clipped_swiglu(*input_arrays, **kwargs):
    x = np.asarray(input_arrays[0])
    group_index = None
    if len(input_arrays) > 1 and input_arrays[1] is not None:
        group_index = np.asarray(input_arrays[1])

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
    # 归一化 dim 到正索引, 用于合轴与输出 shape 计算
    ndim = len(orig_shape)
    dim_pos = dim % ndim

    pre = _prod(orig_shape[:dim_pos]) if dim_pos > 0 else 1
    cut = _prod(orig_shape[dim_pos:])

    xf = x.astype(np.float32).reshape(pre, cut)

    group = pre
    if group_index is not None:
        group = min(int(group_index.sum()), pre)

    xt = xf[:group]
    if interleaved:
        a = xt[:, 0::2]
        b = xt[:, 1::2]
    else:
        h = cut // 2
        a = xt[:, :h]
        b = xt[:, h:]

    a = np.clip(a, None, limit)
    b = np.clip(b, -limit, limit)
    with np.errstate(over="ignore", invalid="ignore"):
        sig = 1.0 / (1.0 + np.exp(-alpha * a))
        res = a * sig * (b + bias)

    y = np.zeros((pre, cut // 2), dtype=np.float32)
    y[:group] = res.astype(np.float32)

    out_shape = list(orig_shape)
    out_shape[dim_pos] = out_shape[dim_pos] // 2
    y = y.reshape(out_shape)

    if target == "bfloat16":
        y = y.astype(_bf16) if _bf16 is not None else y
    else:
        y = y.astype(target)
    return [y]


__golden__ = {"kernel": {"clipped_swiglu": "__golden_clipped_swiglu"}}
