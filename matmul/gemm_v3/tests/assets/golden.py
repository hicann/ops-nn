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
from ml_dtypes import bfloat16

__golden__ = {
    "kernel": {
        "gemm_v3": "gemm_v3_golden"
    }
}

_FRACTAL_DIMS = {
    "float16":  (16, 16),
    "bfloat16": (16, 16),
    "float32":  (16, 8),
}


def customize_inputs(a, b, c=None, *, transpose_a=False, transpose_b=False,
                     alpha=1.0, beta=1.0, enable_hf32=False, **kwargs):
    input_formats = kwargs.get('input_formats', ())
    input_ori_shapes = kwargs.get('input_ori_shapes', ())
    if len(input_formats) > 1 and input_formats[1] == 'FRACTAL_NZ':
        b_dtype_str = _dtype_to_str(b.dtype)
        ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
        b = _nz_to_nd(b, b_dtype_str, ori_shape)
    return a, b, c


def pre_compare(*outputs, **kwargs):
    return list(outputs)


def gemm_v3_golden(a, b, c=None, *, transpose_a=False, transpose_b=False,
                   alpha=1.0, beta=1.0, enable_hf32=False, **kwargs):
    a, b, c = customize_inputs(
        a, b, c,
        transpose_a=transpose_a, transpose_b=transpose_b,
        alpha=alpha, beta=beta, enable_hf32=enable_hf32, **kwargs)

    a_dtype = a.dtype
    b_dtype = b.dtype

    is_hf32 = enable_hf32

    if is_hf32 and a_dtype == np.float32:
        a = _hf32_truncate(a)
        b = _hf32_truncate(b)

    if a_dtype in (np.float16, bfloat16):
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        comp_dtype = np.float32
    elif a_dtype == np.float32:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        comp_dtype = np.float64
    else:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        comp_dtype = np.float64

    if transpose_a:
        a = np.swapaxes(a, -2, -1)
    if transpose_b:
        b = np.swapaxes(b, -2, -1)

    mm_out = np.matmul(a, b)

    mm_out = mm_out * alpha

    if c is not None:
        c = c.astype(comp_dtype)
        mm_out = mm_out + beta * c

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        mm_out = _cast_output_dtype(mm_out, output_dtypes[0])

    return [mm_out]


class GemmV3Assets:

    golden = gemm_v3_golden
    customize_inputs = customize_inputs
    pre_compare = pre_compare

    tolerance = {
        "float32": {"standard": "IsClose", "rtol": 1e-4, "atol": 1e-4},
        "float16": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0,
            "max_re_rtol": 10.0,
            "rmse_rtol": 2.0,
            "small_value": 0.001,
            "small_value_atol": 1e-5,
        },
        "bfloat16": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0,
            "max_re_rtol": 10.0,
            "rmse_rtol": 2.0,
            "small_value": 0.001,
            "small_value_atol": 1e-5,
        },
    }


def _hf32_truncate(arr):
    int_view = arr.view(np.uint32)
    truncated = (((int_view >> 11) + 1) >> 1) << 12
    return truncated.view(np.float32)


def _nz_to_nd(data, dtype_str, ori_shape=None):
    m0, n0 = _FRACTAL_DIMS.get(dtype_str, (16, 16))
    shape = data.shape
    batch_dims = len(shape) - 4
    perm = list(range(batch_dims)) + [batch_dims + 1, batch_dims + 2, batch_dims + 0, batch_dims + 3]
    data = np.transpose(data, perm)
    m1 = shape[batch_dims + 1]
    m0_actual = shape[batch_dims + 2]
    n1 = shape[batch_dims + 0]
    n0_actual = shape[batch_dims + 3]
    data = data.reshape(*shape[:batch_dims], m1 * m0_actual, n1 * n0_actual)
    if ori_shape is not None:
        target_M = ori_shape[-2]
        target_N = ori_shape[-1]
        data = data[..., :target_M, :target_N]
    return data


def _nd_to_fractal_nz(data, dtype_str):
    m0, n0 = _FRACTAL_DIMS.get(dtype_str, (16, 16))
    M, N = data.shape[-2], data.shape[-1]
    m1 = (M + m0 - 1) // m0
    n1 = (N + n0 - 1) // n0
    padded_M = m1 * m0
    padded_N = n1 * n0
    pad_width = [(0, 0)] * (len(data.shape) - 2) + [(0, padded_M - M), (0, padded_N - N)]
    data = np.pad(data, pad_width, mode="constant")
    data = data.reshape(*data.shape[:-2], m1, m0, n1, n0)
    batch_dims = len(data.shape) - 4
    perm = list(range(batch_dims)) + [batch_dims + 2, batch_dims + 0, batch_dims + 1, batch_dims + 3]
    data = np.transpose(data, perm)
    return data


def _dtype_to_str(dtype):
    dtype_map = {
        np.float16: "float16",
        np.float32: "float32",
        np.float64: "float64",
        bfloat16: "bfloat16",
    }
    return dtype_map.get(dtype, str(dtype))


def _cast_output_dtype(arr, dtype_name):
    dtype_map = {"float16": np.float16, "float32": np.float32, "bfloat16": bfloat16}
    target = dtype_map.get(dtype_name)
    if target is not None:
        return arr.astype(target)
    return arr.astype(dtype_name)
