#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
from ml_dtypes import bfloat16, float8_e5m2, float8_e4m3fn

try:
    from en_dtypes import hifloat8
    _HAS_HIFLOAT8 = True
except ImportError:
    _HAS_HIFLOAT8 = False

__golden__ = {
    "kernel": {
        "quant_batch_matmul_inplace_add": "quant_batch_matmul_inplace_add_golden"
    }
}


def customize_inputs(x1, x2, x2_scale, y, x1_scale=None, *, transpose_x1=False,
                     transpose_x2=False, group_size=0, **kwargs):
    return x1, x2, x2_scale, y, x1_scale


def pre_compare(*outputs, **kwargs):
    return list(outputs)


def quant_batch_matmul_inplace_add_golden(x1, x2, x2_scale, y, x1_scale=None, *,
                                           transpose_x1=False, transpose_x2=False,
                                           group_size=0, **kwargs):
    x1, x2, x2_scale, y, x1_scale = customize_inputs(
        x1, x2, x2_scale, y, x1_scale,
        transpose_x1=transpose_x1, transpose_x2=transpose_x2,
        group_size=group_size, **kwargs)

    x1_f32 = _to_float32(x1)
    x2_f32 = _to_float32(x2)

    if transpose_x1:
        x1_f32 = np.swapaxes(x1_f32, -2, -1)
    if transpose_x2:
        x2_f32 = np.swapaxes(x2_f32, -2, -1)

    is_mx = _is_mx_scale(x2_scale)

    if is_mx:
        acc = _mxfp8_matmul(x1_f32, x2_f32, x1_scale, x2_scale)
    else:
        acc = _per_tensor_matmul(x1_f32, x2_f32, x1_scale, x2_scale)

    y_out = acc + y.astype(np.float32)

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        y_out = _cast_output_dtype(y_out, output_dtypes[0])
    else:
        y_out = y_out.astype(np.float32)

    return [y_out]


def _to_float32(arr):
    dtype_str = str(arr.dtype)
    if "e4m3" in dtype_str or "e5m2" in dtype_str:
        return arr.astype(np.float32)
    if "hifloat" in dtype_str or "hif8" in dtype_str:
        return arr.astype(np.float32)
    return arr.astype(np.float32)


def _is_mx_scale(scale):
    if scale is None:
        return False
    return "e8m0" in str(scale.dtype)


def _e8m0_to_float(arr):
    raw = arr.view(np.uint8).astype(np.float32)
    result = np.power(2.0, raw - 127.0)
    result[raw == 255.0] = np.nan
    return result


def _per_tensor_matmul(x1, x2, x1_scale, x2_scale):
    acc = np.matmul(x1, x2)
    s1 = x1_scale.astype(np.float32).reshape(-1)[0] if x1_scale is not None else 1.0
    s2 = x2_scale.astype(np.float32).reshape(-1)[0] if x2_scale is not None else 1.0
    acc = acc * (s1 * s2)
    return acc


def _mxfp8_matmul(x1, x2, x1_scale, x2_scale):
    M, K = x1.shape
    N = x2.shape[1]

    x1s_f = _e8m0_to_float(x1_scale)
    x2s_f = _e8m0_to_float(x2_scale)

    x1s_flat = x1s_f.reshape(M, -1)
    x2s_flat = x2s_f.reshape(-1, N)

    num_groups = x1s_flat.shape[1]
    group_size_k = K // num_groups

    out = np.zeros((M, N), dtype=np.float32)
    for g in range(num_groups):
        k_start = g * group_size_k
        k_end = (g + 1) * group_size_k
        partial = np.matmul(x1[:, k_start:k_end], x2[k_start:k_end, :])
        s1 = x1s_flat[:, g][:, np.newaxis]
        s2 = x2s_flat[g, :][np.newaxis, :]
        out += partial * s1 * s2

    return out


class QuantBatchMatmulInplaceAddAssets:

    golden = quant_batch_matmul_inplace_add_golden
    customize_inputs = customize_inputs
    pre_compare = pre_compare

    tolerance = {
        "float32": {"standard": "IsClose", "rtol": 1e-3, "atol": 1e-3},
    }


def _cast_output_dtype(arr, dtype_name):
    dtype_map = {"float16": np.float16, "float32": np.float32, "bfloat16": bfloat16}
    target = dtype_map.get(dtype_name)
    if target is not None:
        return arr.astype(target)
    return arr.astype(dtype_name)
