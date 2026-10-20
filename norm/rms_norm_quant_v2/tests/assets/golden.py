#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


_DST_SAT_MAX = {
    "int8": 127.0,
    "int4": 7.0,
    "float8_e5m2": 57344.0,
    "float8_e4m3fn": 448.0,
    "hifloat8": 32768.0,
}
_DST_SAT_MIN = {
    "int8": -128.0,
    "int4": -8.0,
    "float8_e5m2": -57344.0,
    "float8_e4m3fn": -448.0,
    "hifloat8": -32768.0,
}


__golden__ = {"kernel": {"rms_norm_quant_v2": "rms_norm_quant_v2"}}
__spec__ = {"rms_norm_quant_v2": "RmsNormQuantV2TestSpec"}


def _empty_quant_output(shape, dtype):
    """Construct the requested dtype without evaluating an empty reduction."""
    if dtype == "int4":
        from ttk.utilities.dtypes import numpy_int4

        dtype = numpy_int4()
    elif dtype in ("float8_e5m2", "float8_e4m3fn"):
        import ml_dtypes

        dtype = getattr(ml_dtypes, dtype)
    elif dtype == "hifloat8":
        from en_dtypes import hifloat8

        dtype = hifloat8
    return np.empty(shape, dtype=dtype)


def rms_norm_quant_v2(
    input_x,
    input_gamma,
    input_scales1,
    input_scales2=None,
    input_zero_points1=None,
    input_zero_points2=None,
    input_beta=None,
    epsilon=1e-6,
    div_mode=True,
    output_dtypes=None,
    dst_type=2,
    **kwargs,
):
    DST_TYPE_MAP = {
        2: "int8",
        29: "int4",
        34: "hifloat8",
        35: "float8_e5m2",
        36: "float8_e4m3fn",
    }
    if output_dtypes is None:
        quant_dtype = DST_TYPE_MAP.get(dst_type, "int8")
        output_dtypes = (quant_dtype, quant_dtype)
    y1_dtype = str(output_dtypes[0]) if output_dtypes is not None else "int8"
    y2_dtype = (
        str(output_dtypes[1])
        if (output_dtypes is not None and len(output_dtypes) > 1)
        else None
    )

    if input_x.size == 0:
        return (
            _empty_quant_output(input_x.shape, y1_dtype),
            _empty_quant_output(input_x.shape, y2_dtype)
            if input_scales2 is not None
            else None,
        )

    # Flatten only the normalization axes. Quantization parameters describe
    # their own suffix of x, which need not have the same rank as gamma.
    original_x_shape = input_x.shape
    norm_dim_num = max(1, input_gamma.ndim)
    n = input_x.ndim - norm_dim_num
    input_gamma = input_gamma.reshape(-1)
    x_shape = input_x.shape[0:n] + input_gamma.shape
    input_x = input_x.reshape(x_shape)

    len_shape_x = len(input_x.shape)
    len_shape_gamma = len(input_gamma.shape)
    axis = len_shape_x - len_shape_gamma
    input_x_dtype = input_x.dtype
    input_scales_dtype = input_scales1.dtype
    input_zp_dtype = None
    if input_zero_points1 is not None:
        input_zp_dtype = input_zero_points1.dtype
    if input_zero_points2 is not None:
        input_zp_dtype = input_zero_points2.dtype

    if input_scales_dtype is not np.float32:
        input_scales1 = input_scales1.astype(np.float32)
    if input_scales2 is not None:
        if input_scales_dtype is not np.float32:
            input_scales2 = input_scales2.astype(np.float32)
    if input_zero_points1 is not None:
        if input_zp_dtype == np.int32:
            input_zero_points1 = input_zero_points1.astype(np.float32)
        elif input_scales_dtype is not np.float32:
            input_zero_points1 = input_zero_points1.astype(np.float32)
    if input_zero_points2 is not None:
        if input_zp_dtype == np.int32:
            input_zero_points2 = input_zero_points2.astype(np.float32)
        elif input_scales_dtype is not np.float32:
            input_zero_points2 = input_zero_points2.astype(np.float32)

    x_fp32 = input_x.astype(np.float32)
    variance = np.mean(np.power(x_fp32, 2), axis=axis, keepdims=True)
    std = np.sqrt(variance + epsilon)
    rstd = 1 / std
    result_mid = x_fp32 * rstd

    if input_x_dtype == np.float32:
        y_array = result_mid * input_gamma
    elif input_x_dtype == np.float16:
        input_gamma_fp32 = input_gamma.astype(np.float32)
        y_array = result_mid * input_gamma_fp32
    else:
        input_gamma_fp32 = input_gamma.astype(np.float32)
        y_array = result_mid * input_gamma_fp32

    if input_beta is not None:
        beta_reshaped = input_beta.reshape(input_gamma.shape)
        y_array = y_array + beta_reshaped.astype(np.float32)

    # Restore x's axes before broadcasting scalar or multidimensional scales
    # and zero points; never index shape[0] on a scalar quantization parameter.
    y_array = y_array.reshape(original_x_shape)

    if div_mode:
        input_scales1 = 1.0 / input_scales1
        if input_scales2 is not None:
            input_scales2 = 1.0 / input_scales2

    y1_quant = y_array * input_scales1
    y2_quant = None
    if input_zero_points1 is not None:
        y1_quant = y1_quant + input_zero_points1

    if input_scales2 is not None:
        y2_quant = y_array * input_scales2
    if input_zero_points2 is not None:
        y2_quant = y2_quant + input_zero_points2

    def quant_process(y_quant, q_dtype):
        y_quant = np.nan_to_num(
            y_quant,
            nan=0.0,
            posinf=_DST_SAT_MAX.get(q_dtype, 0.0),
            neginf=_DST_SAT_MIN.get(q_dtype, 0.0),
        )
        if q_dtype in _DST_SAT_MAX:
            y_quant = np.clip(y_quant, _DST_SAT_MIN[q_dtype], _DST_SAT_MAX[q_dtype])
        if q_dtype in ("int8", "int4"):
            y_quant = y_quant.astype(np.float16).astype(np.float32)
            y_quant = np.round(y_quant, 0)
        elif q_dtype not in ("hifloat8", "float8_e5m2", "float8_e4m3fn"):
            y_quant = np.round(y_quant, 0)
        if q_dtype == "int8":
            y_quant = y_quant.clip(-128, 127)
            y_quant = y_quant.astype("int8", copy=False)
        elif q_dtype == "int4":
            from ttk.utilities.dtypes import numpy_int4

            y_quant = np.clip(y_quant, -8, 7)
            y_quant = y_quant.astype(numpy_int4(), copy=False)
        elif q_dtype == "int32":
            y_quant = np.nan_to_num(y_quant, nan=0)
            y_quant = np.clip(y_quant, -2147483648, 2147483647)
            y_quant = y_quant.astype("int32", copy=False)
        elif q_dtype == "float8_e5m2":
            from ml_dtypes import float8_e5m2 as numpy_float8_e5m2

            y_quant = y_quant.astype(numpy_float8_e5m2, copy=False)
        elif q_dtype == "float8_e4m3fn":
            from ml_dtypes import float8_e4m3fn as numpy_float8_e4m3fn

            y_quant = y_quant.astype(numpy_float8_e4m3fn, copy=False)
        elif q_dtype == "hifloat8":
            from en_dtypes import hifloat8 as numpy_hifloat8

            y_quant = y_quant.astype(numpy_hifloat8, copy=False)
        return y_quant

    return quant_process(y1_quant, y1_dtype), quant_process(
        y2_quant, y2_dtype
    ) if y2_quant is not None else None


class RmsNormQuantV2TestSpec:
    """整数量化输出按 1 LSB 判；FP8 输出按相邻量化码判。

    y = round(x / scale) 的 round 是阶跃函数：商落在 .5 附近时，浮点末位 1 ulp 的差异
    就足以让结果跳到相邻整数格。而末位差异来自归约次序、FMA 融合、硬件舍入等实现自由度，
    不同实现之间必然存在。要求逐位相等等于要求复现某一个实现的浮点运算次序，
    故整数按绝对误差 <= 1 判定，FP8 按 requant 的码距判定。
    """

    golden = staticmethod(rms_norm_quant_v2)
    tolerance = {
        "int8": {"standard": "quant"},
        "int4": {"standard": "quant"},
        "float8_e5m2": {"standard": "requant"},
        "float8_e4m3fn": {"standard": "requant"},
    }
