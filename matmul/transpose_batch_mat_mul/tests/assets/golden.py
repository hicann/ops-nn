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
        "transpose_batch_mat_mul": "transpose_batch_mat_mul_golden"
    }
}

_FRACTAL_DIMS = {
    "float16":  (16, 16),
    "bfloat16": (16, 16),
    "float32":  (16, 8),
}


def customize_inputs(x1, x2, bias=None, scale=None, *, perm_x1=(0, 1, 2),
                     perm_x2=(0, 1, 2), perm_y=(1, 0, 2), enable_hf32=False,
                     batch_split_factor=1, **kwargs):
    input_formats = kwargs.get('input_formats', ())
    input_ori_shapes = kwargs.get('input_ori_shapes', ())
    if len(input_formats) > 1 and input_formats[1] == 'FRACTAL_NZ':
        x2_dtype_str = _dtype_to_str(x2.dtype)
        ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
        x2 = _nz_to_nd(x2, x2_dtype_str, ori_shape)
    return x1, x2, bias, scale


def pre_compare(*outputs, **kwargs):
    return list(outputs)


def transpose_batch_mat_mul_golden(x1, x2, bias=None, scale=None, *,
                                    perm_x1=(0, 1, 2), perm_x2=(0, 1, 2),
                                    perm_y=(1, 0, 2), enable_hf32=False,
                                    batch_split_factor=1, **kwargs):
    x1, x2, bias, scale = customize_inputs(
        x1, x2, bias, scale,
        perm_x1=perm_x1, perm_x2=perm_x2, perm_y=perm_y,
        enable_hf32=enable_hf32, batch_split_factor=batch_split_factor, **kwargs)

    x1_dtype = x1.dtype

    is_hf32 = enable_hf32

    if is_hf32 and x1_dtype == np.float32:
        x1 = _hf32_truncate(x1)
        x2 = _hf32_truncate(x2)

    if x1_dtype in (np.float16, bfloat16):
        x1 = x1.astype(np.float32)
        x2 = x2.astype(np.float32)
        comp_dtype = np.float32
    elif x1_dtype == np.float32:
        x1 = x1.astype(np.float64)
        x2 = x2.astype(np.float64)
        comp_dtype = np.float64
    else:
        x1 = x1.astype(np.float64)
        x2 = x2.astype(np.float64)
        comp_dtype = np.float64

    x1_t = np.transpose(x1, axes=list(perm_x1))

    if tuple(perm_x2) == (0, 2, 1):
        x2_t = np.swapaxes(x2, -2, -1)
    else:
        x2_t = x2

    mm_out = np.matmul(x1_t, x2_t)

    if scale is not None:
        B, M, N = mm_out.shape
        mm_out = mm_out.transpose(1, 0, 2).reshape(M, B * N)
        scale_f = scale.astype(comp_dtype).reshape(1, B * N)
        mm_out = mm_out * scale_f
        mm_out = np.round(mm_out).clip(-128, 127).astype(np.int8)
        mm_out = mm_out.reshape(M, 1, B * N)
    elif batch_split_factor > 1:
        B, M, N = mm_out.shape
        inner_batch = B // batch_split_factor
        mm_out = mm_out.reshape(batch_split_factor, inner_batch, M, N)
        mm_out = mm_out.transpose(0, 2, 1, 3)
        mm_out = mm_out.reshape(batch_split_factor, M, inner_batch * N)
    else:
        mm_out = np.transpose(mm_out, axes=list(perm_y))

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        mm_out = _cast_output_dtype(mm_out, output_dtypes[0])

    return [mm_out]


class TransposeBatchMatMulAssets:

    golden = transpose_batch_mat_mul_golden
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
    dtype_map = {"float16": np.float16, "float32": np.float32, "bfloat16": bfloat16, "int8": np.int8}
    target = dtype_map.get(dtype_name)
    if target is not None:
        return arr.astype(target)
    return arr.astype(dtype_name)
