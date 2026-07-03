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
import ml_dtypes
import numpy as np
from ml_dtypes import bfloat16, float8_e5m2, float8_e4m3fn

__golden__ = {
    "kernel": {
        "transpose_quant_batch_mat_mul": "transpose_quant_batch_mat_mul_golden"
    },
    "e2e": {
        "torch_npu.npu_transpose_quant_batchmatmul": "torch_npu_npu_transpose_quant_batchmatmul_golden"
    }
}

_DTYPE_ATTR_MAP = {1: np.float16, 27: bfloat16}

_FRACTAL_DIMS = {
    "float16":  (16, 16),
    "bfloat16": (16, 16),
    "float32":  (16, 8),
}


def customize_inputs(x1, x2, bias=None, x1_scale=None, x2_scale=None, *,
                     dtype=1, group_size=0, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2),
                     perm_y=(1, 0, 2), batch_split_factor=1, **kwargs):
    input_formats = kwargs.get('input_formats', ())
    input_ori_shapes = kwargs.get('input_ori_shapes', ())
    if len(input_formats) > 1 and input_formats[1] == 'FRACTAL_NZ':
        x2_dtype_str = _dtype_to_str(x2.dtype)
        ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
        x2 = _nz_to_nd(x2, x2_dtype_str, ori_shape)
    return x1, x2, bias, x1_scale, x2_scale


def pre_compare(*outputs, **kwargs):
    return list(outputs)


def transpose_quant_batch_mat_mul_golden(x1, x2, bias=None, x1_scale=None,
                                          x2_scale=None, *, dtype=1, group_size=0,
                                          perm_x1=(1, 0, 2), perm_x2=(0, 1, 2),
                                          perm_y=(1, 0, 2), batch_split_factor=1,
                                          **kwargs):
    x1, x2, bias, x1_scale, x2_scale = customize_inputs(
        x1, x2, bias, x1_scale, x2_scale,
        dtype=dtype, group_size=group_size, perm_x1=perm_x1, perm_x2=perm_x2,
        perm_y=perm_y, batch_split_factor=batch_split_factor, **kwargs)

    x1_f32 = x1.astype(np.float32)
    x2_f32 = x2.astype(np.float32)

    x1_t = np.transpose(x1_f32, axes=list(perm_x1))

    if tuple(perm_x2) == (0, 2, 1):
        x2_t = np.swapaxes(x2_f32, -2, -1)
        if x2_scale is not None and x2_scale.ndim == 4:
            x2_scale = np.transpose(x2_scale, (0, 2, 1, 3))
    else:
        x2_t = x2_f32

    is_mx = _is_mx_scale(x1_scale)

    if is_mx:
        acc = _mxfp8_matmul(x1_t, x2_t, x1_scale, x2_scale)
    else:
        acc = _kc_matmul(x1_t, x2_t, x1_scale, x2_scale)

    out = np.transpose(acc, axes=list(perm_y))

    output_dtypes = kwargs.get("output_dtypes", None)
    if output_dtypes is not None:
        out = _cast_output_dtype(out, output_dtypes[0])
    else:
        target_dtype = _DTYPE_ATTR_MAP.get(dtype, np.float16)
        out = out.astype(target_dtype)

    return [out]


def _is_mx_scale(scale):
    if scale is None:
        return False
    dtype_str = str(scale.dtype)
    return "e8m0" in dtype_str


def _e8m0_to_float(arr):
    raw = arr.view(np.uint8).astype(np.float32)
    result = np.power(2.0, raw - 127.0)
    result[raw == 255.0] = np.nan
    return result


def _kc_matmul(x1, x2, x1_scale, x2_scale):
    acc = np.matmul(x1, x2)

    if x2_scale is not None:
        acc = acc * x2_scale.astype(np.float32).reshape(1, 1, -1)

    if x1_scale is not None:
        acc = acc * x1_scale.astype(np.float32).reshape(1, -1, 1)

    return acc


def _mxfp8_matmul(x1, x2, x1_scale, x2_scale):
    B, M, K = x1.shape
    N = x2.shape[2]
    group_size_k = 32
    num_groups = K // group_size_k

    x1_scale_f = _e8m0_to_float(x1_scale)
    x2_scale_f = _e8m0_to_float(x2_scale)

    x1_scale_flat = x1_scale_f.reshape(M, B, num_groups)
    x1_scale_flat = np.transpose(x1_scale_flat, (1, 0, 2))

    x2_scale_flat = x2_scale_f.reshape(B, num_groups, N)

    out = np.zeros((B, M, N), dtype=np.float32)
    for g in range(num_groups):
        k_start = g * group_size_k
        k_end = (g + 1) * group_size_k
        partial = np.matmul(x1[:, :, k_start:k_end], x2[:, k_start:k_end, :])
        s1 = x1_scale_flat[:, :, g][:, :, np.newaxis]
        s2 = x2_scale_flat[:, g, :][:, np.newaxis, :]
        out += partial * s1 * s2

    return out


class TransposeQuantBatchMatMulAssets:

    golden = transpose_quant_batch_mat_mul_golden
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


# ===== E2E golden (torch-based, retained for e2e mode) =====

def torch_npu_npu_transpose_quant_batchmatmul_golden(x1, x2, dtype, *, bias=None, x1_scale=None,
                                                     x2_scale=None, group_sizes=None, perm_x1=None,
                                                     perm_x2=None, perm_y=None, batch_split_factor=None, **kwargs):
    import torch
    x1 = _torch_to_numpy(x1)
    x2 = _torch_to_numpy(x2)
    bias = _torch_to_numpy(bias)
    x1_scale = _torch_to_numpy(x1_scale)
    x2_scale = _torch_to_numpy(x2_scale)
    x1_dtype = str(x1.dtype)
    x2_dtype = str(x2.dtype)
    x1_scale_dtype = ""
    if isinstance(x1_scale, torch.Tensor):
        x1_scale_dtype = str(x1_scale.dtype)
    pertoken_flag = (x1_scale is not None
                     and x1_dtype in ("int8", "float8_e4m3fn", "float8_e5m2")
                     and x1_scale_dtype in ("float32",))

    x1 = _transpose_(x1, perm_x1)
    x2 = _transpose_(x2, perm_x2)

    x1 = _x_dtype_cope(x1, x1_dtype)
    x2 = _x_dtype_cope(x2, x2_dtype)
    if pertoken_flag:
        out = _pertoken_calculation(x1, x2, x1_scale, x2_scale, bias, is_bias_epilogue=False, output_dtype=dtype)
    else:
        raise TypeError("Please check whether this quantitative method is supported")
    out = _transpose_(out, perm_y)
    out = _out_dtype_cope(out, dtype)
    return out


def _pertoken_calculation(x1, x2, x1_scale, x2_scale, bias, is_bias_epilogue=False, output_dtype=2):
    import torch
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    out = torch.matmul(x1, x2)
    if bias is not None:
        bias = torch.from_numpy(bias)
    if bias is not None and (bias.dtype == "int32" or (bias.dtype == "float32" and not is_bias_epilogue)):
        out = torch.add(out, bias)
    out = (out * x2_scale.astype(np.float32))

    x1_scale = torch.from_numpy(x1_scale)
    pertoken_scale_slice = torch.unsqueeze(x1_scale, dim=1).to(torch.float32)
    out = out * pertoken_scale_slice
    if bias is not None and is_bias_epilogue:
        out = torch.tensor(out, dtype=torch.float)
        bias = torch.tensor(bias, dtype=torch.float)
        out = torch.add(out, bias)
    return out.numpy()


def _x_dtype_cope(x, x_dtype):
    if x_dtype in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2", "hifloat8"):
        x = x.astype(np.float32)
    else:
        x = x.astype(np.int32)
    return x


def _transpose_(x, array_trans_x):
    import torch
    if isinstance(x, torch.Tensor):
        x = torch.permute(x, array_trans_x)
    elif isinstance(x, np.ndarray):
        x = np.transpose(x, array_trans_x)
    else:
        raise TypeError("This is not a tensor or an array can be transposed")
    return x


def _out_dtype_cope(out, output_dtype):
    import torch
    if output_dtype == torch.float32:
        out = out.astype(np.float32)
    elif output_dtype == torch.float16:
        out = out.astype(np.float16)
    elif output_dtype == torch.int8:
        out = np.clip(out, -128, 127)
        out = out.astype(np.int8)
    elif output_dtype == torch.bfloat16:
        out = out.astype(bfloat16)
    elif output_dtype == torch.float8_e4m3fn:
        out = out.astype(float8_e4m3fn)
    elif output_dtype == torch.int32:
        out = out.astype(np.int32)
    else:
        raise TypeError("invalid output dtype")
    return out


def _torch_to_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float8_e5m2:
            x = x.to(torch.float32).numpy().astype(float8_e5m2)
        elif x.dtype == torch.float8_e4m3fn:
            x = x.to(torch.float32).numpy().astype(float8_e4m3fn)
        else:
            x = x.numpy()
    return x
