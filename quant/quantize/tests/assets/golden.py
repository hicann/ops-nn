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
import torch

__golden__ = {
    "aclnn": {
        "aclnnQuantize": "aclnn_quantize_golden",
    },
    "kernel": {"quantize": "quantize_golden"},
}


def quantize_golden(x, scales, zero_points=None, *, dtype, axis=1, **kwargs):
    """
    Golden function for quantize.
    All the parameters (names and order) follow @quantize_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    scale = scales
    offset = zero_points
    dst_type_str = str(kwargs["output_dtypes"][0])
    x_shape = x.shape
    scale_shape = scale.shape
    if len(x_shape) != len(scale_shape):
        # 支持per tensor，per channel，per head
        tmp_scale_shape = [1] * len(x_shape)
        tmp_scale_shape[axis] = scale_shape[0]
        scale = np.reshape(scale, tmp_scale_shape)

    cast_dtype = "float32"
    x = x.astype(cast_dtype)
    scale = scale.astype(cast_dtype)

    scale_rst = x / scale
    if offset is not None:
        offset = offset.astype(cast_dtype)
        offset = np.reshape(offset, scale.shape)
        scale_rst = scale_rst + offset

    round_data = (
        scale_rst
        if dst_type_str in ("hifloat8", "float8_e5m2", "float8_e4m3fn")
        else np.round(scale_rst, 0)
    )
    if dst_type_str == "int8":
        round_data = round_data.clip(-128, 127)
        round_data = round_data.astype("int8", copy=False)
    elif dst_type_str == "uint8":
        round_data = round_data.clip(0, 255)
        round_data = round_data.astype("uint8", copy=False)
    elif dst_type_str == "int32":
        round_data = np.nan_to_num(
            round_data, nan=0
        )  # CastToInt32(fp32(nan)) cpu是-2147483648，gpu是0，标杆和gpu一致
        round_data = np.clip(
            round_data, -2147483648, 2147483647
        )  # CastToInt32(fp32(inf)) cpu上是-2147483648，gpu是2147483647，标杆和gpu一致
        round_data = round_data.astype("int32", copy=False)
    elif dst_type_str == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        np.clip(round_data, a_min=-57344.0, a_max=57344.0, out=round_data)
        round_data = np.nan_to_num(round_data, nan=0.0, copy=False)
        round_data = round_data.astype(float8_e5m2, copy=False)
    elif dst_type_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        round_data = round_data.astype(float8_e4m3fn, copy=False)
    elif dst_type_str == "hifloat8":
        from en_dtypes import hifloat8

        round_data = round_data.astype(hifloat8, copy=False)

    return round_data


def aclnn_quantize_golden(x, scales, zeroPoints, dtype, axis, out, **kwargs):
    if hasattr(dtype, "item"):
        dtype = dtype.item()
    if hasattr(axis, "item"):
        axis = axis.item()
    x_f = x.to(torch.float32)
    scale = scales.to(torch.float32)
    offset = zeroPoints.to(torch.float32) if zeroPoints is not None else None
    x_shape = x_f.shape
    scale_shape = scale.shape
    if len(x_shape) != len(scale_shape):
        tmp_scale_shape = [1] * len(x_shape)
        tmp_scale_shape[axis] = scale_shape[0]
        scale = scale.reshape(tmp_scale_shape)
    scale_rst = x_f / scale
    if offset is not None:
        offset = offset.reshape(scale.shape)
        scale_rst = scale_rst + offset
    round_data = torch.round(scale_rst).clamp(-128, 127).to(torch.int8)
    return [round_data]
