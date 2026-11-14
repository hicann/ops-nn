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


DATA_TYPE_INT_TO_STR = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    11: "double",
    12: "bool",
    27: "bfloat16",
    29: "int4",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    37: "float8_e8m0",
    40: "float4_e2m1",
    41: "float4_e1m2",
}


def _get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in str(dt):
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


def _set_float_overflow_mode(dst_type, round_data):
    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    max_norm = _get_dtype_range(dst_type_str)[1]
    np.clip(round_data, a_min=-max_norm, a_max=max_norm, out=round_data)
    round_data = np.nan_to_num(round_data, nan=0.0, copy=False)
    return round_data


__golden__ = {"kernel": {"quant_max": "quant_max_golden"}}


def quant_max_golden(x, scale, *, round_mode="rint", dst_type=35, **kwargs):
    """
    Golden function for quant_max.
    All the parameters (names and order) follow @quant_max_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    if x.dtype == np.float16:
        x_fp32 = x.astype(np.float32)
        mul_scale = x_fp32 * scale
        amax = np.max(np.abs(x_fp32)).astype(x.dtype, copy=False)
    elif x.dtype == np.dtype("bfloat16"):
        x_fp32 = x.astype(np.float32)
        mul_scale = x_fp32 * scale
        amax = np.max(np.abs(x_fp32)).astype(x.dtype, copy=False)
    else:
        mul_scale = x * scale
        amax = np.max(np.abs(x))

    mul_scale = _set_float_overflow_mode(dst_type, mul_scale)
    if dst_type == 34:
        from en_dtypes import hifloat8

        mul_scale = mul_scale.astype(hifloat8, copy=False)
    elif dst_type == 35:
        from ml_dtypes import float8_e5m2

        mul_scale = mul_scale.astype(float8_e5m2, copy=False)
    elif dst_type == 36:
        from ml_dtypes import float8_e4m3fn

        mul_scale = mul_scale.astype(float8_e4m3fn, copy=False)
    output = _set_float_overflow_mode(dst_type, mul_scale)
    return output, amax
