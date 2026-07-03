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

try:
    from en_dtypes import float4_e2m1
    _HAS_FLOAT4 = True
except ImportError:
    _HAS_FLOAT4 = False

__golden__ = {
    "kernel": {
        "dual_level_quant_batch_matmul": "dual_level_quant_batch_matmul_golden"
    }
}

_DTYPE_ATTR_MAP = {1: np.float16, 27: bfloat16}


def customize_inputs(x1, x2, x1_level0_scale, x1_level1_scale, x2_level0_scale,
                     x2_level1_scale, bias=None, *, dtype=1, transpose_x1=False,
                     transpose_x2=True, level0_group_size=512, level1_group_size=32,
                     **kwargs):
    input_formats = kwargs.get('input_formats', ())
    input_ori_shapes = kwargs.get('input_ori_shapes', ())
    if len(input_formats) > 1 and input_formats[1] == 'FRACTAL_NZ':
        ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
        x2 = _nz_to_nd(x2, ori_shape)
    return x1, x2, x1_level0_scale, x1_level1_scale, x2_level0_scale, x2_level1_scale, bias


def pre_compare(*outputs, **kwargs):
    return list(outputs)


def dual_level_quant_batch_matmul_golden(x1, x2, x1_level0_scale, x1_level1_scale,
                                          x2_level0_scale, x2_level1_scale, bias=None, *,
                                          dtype=1, transpose_x1=False, transpose_x2=True,
                                          level0_group_size=512, level1_group_size=32,
                                          **kwargs):
    x1, x2, x1_level0_scale, x1_level1_scale, x2_level0_scale, x2_level1_scale, bias = customize_inputs(
        x1, x2, x1_level0_scale, x1_level1_scale, x2_level0_scale, x2_level1_scale, bias,
        dtype=dtype, transpose_x1=transpose_x1, transpose_x2=transpose_x2,
        level0_group_size=level0_group_size, level1_group_size=level1_group_size, **kwargs)

    x1_f32 = x1.astype(np.float32)
    x2_f32 = x2.astype(np.float32)

    if transpose_x2:
        x2_f32 = np.swapaxes(x2_f32, -2, -1)

    x1_l0 = x1_level0_scale.astype(np.float32)
    x2_l0 = x2_level0_scale.astype(np.float32)
    x1_l1 = _e8m0_to_float(x1_level1_scale)
    x2_l1 = _e8m0_to_float(x2_level1_scale)

    M, K = x1_f32.shape
    N = x2_f32.shape[1]

    num_l1_groups = x1_l1.shape[-2] * x1_l1.shape[-1] if x1_l1.ndim >= 2 else x1_l1.shape[0]
    x1_l1_flat = x1_l1.reshape(M, -1)
    x2_l1_flat = x2_l1.reshape(N, -1)

    num_l0_groups = x1_l0.shape[1]
    l1_per_l0 = level0_group_size // level1_group_size

    acc = np.zeros((M, N), dtype=np.float32)
    for g0 in range(num_l0_groups):
        acc_l0 = np.zeros((M, N), dtype=np.float32)
        for g1 in range(l1_per_l0):
            g1_global = g0 * l1_per_l0 + g1
            if g1_global >= num_l1_groups:
                break
            k_start = g1_global * level1_group_size
            k_end = min(k_start + level1_group_size, K)
            if k_end <= k_start:
                break
            partial = np.matmul(x1_f32[:, k_start:k_end], x2_f32[k_start:k_end, :])
            s1 = x1_l1_flat[:, g1_global][:, np.newaxis]
            s2 = x2_l1_flat[:, g1_global][np.newaxis, :]
            acc_l0 += partial * s1 * s2
        acc += x1_l0[:, g0][:, np.newaxis] * x2_l0[g0, :][np.newaxis, :] * acc_l0

    if bias is not None:
        acc = acc + bias.astype(np.float32).reshape(1, -1)

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        acc = _cast_output_dtype(acc, output_dtypes[0])
    else:
        target_dtype = _DTYPE_ATTR_MAP.get(dtype, np.float16)
        acc = acc.astype(target_dtype)

    return [acc]


class DualLevelQuantBatchMatmulAssets:

    golden = dual_level_quant_batch_matmul_golden
    customize_inputs = customize_inputs
    pre_compare = pre_compare

    tolerance = {
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


def _e8m0_to_float(arr):
    raw = arr.view(np.uint8).astype(np.float32)
    result = np.power(2.0, raw - 127.0)
    result[raw == 255.0] = np.nan
    return result


def _cast_output_dtype(arr, dtype_name):
    dtype_map = {"float16": np.float16, "float32": np.float32, "bfloat16": bfloat16}
    target = dtype_map.get(dtype_name)
    if target is not None:
        return arr.astype(target)
    return arr.astype(dtype_name)


def _nz_to_nd(data, ori_shape=None):
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
