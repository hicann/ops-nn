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
        "aclnnDynamicQuantV3": "aclnn_dynamic_quant_v3_golden",
        "aclnnDynamicQuant": "aclnn_dynamic_quant_golden",
    },
    "kernel": {"dynamic_quant": "dynamic_quant_golden"},
}


def _dynamic_quant_common(
    x,
    smooth_scales=None,
    group_index=None,
    dst_type=2,
    quant_mode="pertoken",
    is_symmetrical=True,
    output_dtype_str="",
):
    # get scale max const value
    scale_max = np.float32(0.0)
    scale_max_no_sym = np.float32(0.0)
    if dst_type == 2:
        scale_max = np.float32(127.0)
        scale_max_no_sym = np.float32(255.0)
    elif dst_type == 29:
        scale_max = np.float32(7.0)
        scale_max_no_sym = np.float32(15.0)
    elif dst_type == 34:
        scale_max = np.float32(32768.0)
        scale_max_no_sym = np.float32(65536.0)
    elif dst_type == 35:
        scale_max = np.float32(57344.0)
        scale_max_no_sym = np.float32(114688.0)
    elif dst_type == 36:
        scale_max = np.float32(448.0)
        scale_max_no_sym = np.float32(896.0)

    # handle smooth_scales
    if smooth_scales is not None:
        smooth_scales = smooth_scales.astype("float32")
    else:
        smooth_scales = 1

    # compute
    if group_index is not None:
        x = x.reshape(-1, x.shape[-1])
        S, H = x.shape
        E = group_index.shape[0]
        input_mul = np.empty(shape=(0, H), dtype="float32")
        for row_scale in range(E):
            x_start_row = 0 if row_scale == 0 else group_index[row_scale - 1]
            x_end_row = group_index[row_scale]
            if x_start_row < x_end_row:
                mul_rows = x[x_start_row:x_end_row] * smooth_scales[row_scale]
                input_mul = np.concatenate([input_mul, mul_rows], axis=0)
    else:
        x = x.astype("float32")
        input_mul = x * smooth_scales

    offset = None
    if is_symmetrical is False:
        input_abs = input_mul
        input_max = (
            np.max(input_abs)
            if quant_mode == "pertensor"
            else np.max(input_abs, axis=-1, keepdims=True)
        )
        input_min = (
            np.min(input_abs)
            if quant_mode == "pertensor"
            else np.min(input_abs, axis=-1, keepdims=True)
        )
        scale = (input_max - input_min) * (np.float32(1.0) / scale_max_no_sym)
        offset = scale_max - (input_max / scale)
        input_scaled = input_mul / scale + offset
    else:
        input_abs = np.abs(input_mul)
        input_max = (
            np.max(input_abs)
            if quant_mode == "pertensor"
            else np.max(input_abs, axis=-1, keepdims=True)
        )
        scale = input_max * (np.float32(1.0) / scale_max)
        input_scaled = input_mul / scale

    # cast to dst_type
    round_data = (
        input_scaled
        if output_dtype_str in ("hifloat8", "float8_e5m2", "float8_e4m3fn")
        else np.round(input_scaled, 0)
    )

    if dst_type == 2:
        round_data = round_data.astype("int8", copy=False)
    elif dst_type == 29:
        from ml_dtypes import int4

        round_data = round_data.astype(int4, copy=False)
    elif dst_type == 35:
        from ml_dtypes import float8_e5m2

        round_data = round_data.astype(float8_e5m2, copy=False)
    elif dst_type == 36:
        from ml_dtypes import float8_e4m3fn

        round_data = round_data.astype(float8_e4m3fn, copy=False)
    elif dst_type == 34:
        from en_dtypes import hifloat8

        round_data = round_data.astype(hifloat8, copy=False)

    if is_symmetrical is False:
        output_data = [round_data, scale.squeeze(-1), offset.squeeze(-1)]
    else:
        output_data = [round_data, scale.squeeze(-1)]
    return output_data


def _dynamic_quant_perchannel(
    x,
    smooth_scales=None,
    group_index=None,
    dst_type=2,
    quant_mode="perchannel",
    is_symmetrical=True,
    output_dtype_str="",
):
    # get scale max const value
    scale_max = np.float32(0.0)
    scale_max_no_sym = np.float32(0.0)
    if dst_type == 2:
        scale_max = np.float32(127.0)
        scale_max_no_sym = np.float32(255.0)
    elif dst_type == 29:
        scale_max = np.float32(7.0)
        scale_max_no_sym = np.float32(15.0)
    elif dst_type == 34:
        scale_max = np.float32(32768.0)
        scale_max_no_sym = np.float32(65536.0)
    elif dst_type == 35:
        scale_max = np.float32(57344.0)
        scale_max_no_sym = np.float32(114688.0)
    elif dst_type == 36:
        scale_max = np.float32(448.0)
        scale_max_no_sym = np.float32(896.0)

    # handle smooth_scales
    if smooth_scales is not None:
        smooth_scales = smooth_scales.astype("float32")[:, None]
    else:
        smooth_scales = 1

    # compute
    x = x.astype("float32")
    input_mul = x * smooth_scales

    offset = None
    if is_symmetrical is False:
        input_abs = input_mul
        input_max = np.max(input_abs, axis=-2, keepdims=True)
        input_min = np.min(input_abs, axis=-2, keepdims=True)
        scale = (input_max - input_min) * (np.float32(1.0) / scale_max_no_sym)
        offset = scale_max - (input_max / scale)
        input_scaled = input_mul / scale + offset
    else:
        input_abs = np.abs(input_mul)
        input_max = np.max(input_abs, axis=-2, keepdims=True)
        scale = input_max * (np.float32(1.0) / scale_max)
        input_scaled = input_mul / scale

    # cast to dst_type
    round_data = (
        input_scaled
        if output_dtype_str in ("hifloat8", "float8_e5m2", "float8_e4m3fn")
        else np.round(input_scaled, 0)
    )

    if dst_type == 2:
        round_data = round_data.astype("int8", copy=False)
    elif dst_type == 29:
        from ml_dtypes import int4

        round_data = round_data.astype(int4, copy=False)
    elif dst_type == 35:
        from ml_dtypes import float8_e5m2

        round_data = round_data.astype(float8_e5m2, copy=False)
    elif dst_type == 36:
        from ml_dtypes import float8_e4m3fn

        round_data = round_data.astype(float8_e4m3fn, copy=False)
    elif dst_type == 34:
        from en_dtypes import hifloat8

        round_data = round_data.astype(hifloat8, copy=False)

    if is_symmetrical is False:
        output_data = [round_data, scale.squeeze(-2), offset.squeeze(-2)]
    else:
        output_data = [round_data, scale.squeeze(-2)]
    return output_data


def dynamic_quant_golden(
    x, smooth_scales=None, group_index=None, *, dst_type=2, **kwargs
):
    """
    Golden function for dynamic_quant.
    All the parameters (names and order) follow @dynamic_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    # get params
    quant_mode = kwargs.get("quant_mode", "perToken")
    is_symmetrical = kwargs.get("is_symmetrical", True)
    output_dtype_str = str(kwargs["output_dtypes"][0])
    if quant_mode == "perchannel":
        return _dynamic_quant_perchannel(
            x,
            smooth_scales,
            group_index,
            dst_type,
            quant_mode,
            is_symmetrical,
            output_dtype_str,
        )
    return _dynamic_quant_common(
        x,
        smooth_scales,
        group_index,
        dst_type,
        quant_mode,
        is_symmetrical,
        output_dtype_str,
    )


def aclnn_dynamic_quant_golden(x, smoothScalesOptional, yOut, scaleOut, **kwargs):
    """
    Aclnn golden for aclnnDynamicQuant.
    """
    x_f = x.to(torch.float32) if x.dtype != torch.float32 else x
    smooth_scales = (
        smoothScalesOptional.to(torch.float32)
        if smoothScalesOptional is not None
        else None
    )
    x_scaled = x_f * smooth_scales if smooth_scales is not None else x_f
    amax = torch.amax(
        torch.abs(x_scaled).view(-1, x_scaled.shape[-1]), dim=-1, keepdim=True
    )
    scale = amax / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(x_scaled / scale)
    quantized = quantized.clamp(-128, 127).to(torch.int8)
    return [quantized, scale]


def aclnn_dynamic_quant_v3_golden(
    x,
    smoothScalesOptional,
    groupIndexOptional,
    dstType,
    isSymmetrical,
    quantMode,
    yOut,
    scaleOut,
    offsetOut,
    **kwargs,
):
    """
    Aclnn golden for aclnnDynamicQuantV3.
    """
    if hasattr(dstType, "item"):
        dstType = dstType.item()
    if hasattr(isSymmetrical, "item"):
        isSymmetrical = bool(isSymmetrical.item())
    if hasattr(quantMode, "item"):
        quantMode = quantMode.item()
    x_f = x.to(torch.float32) if x.dtype != torch.float32 else x
    smooth_scales = (
        smoothScalesOptional.to(torch.float32)
        if smoothScalesOptional is not None
        else None
    )
    x_scaled = x_f * smooth_scales if smooth_scales is not None else x_f
    scale_max = 127.0
    scale_max_no_sym = 255.0
    offset = None
    if not isSymmetrical:
        input_max = torch.max(x_scaled, dim=-1, keepdim=True).values
        input_min = torch.min(x_scaled, dim=-1, keepdim=True).values
        scale = (input_max - input_min) / scale_max_no_sym
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        offset = scale_max - (input_max / scale)
        input_scaled = x_scaled / scale + offset
    else:
        input_abs = torch.abs(x_scaled)
        input_max = torch.max(input_abs, dim=-1, keepdim=True).values
        scale = input_max / scale_max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        input_scaled = x_scaled / scale
    round_data = torch.round(input_scaled)
    if dstType == 2:
        if isSymmetrical:
            round_data = round_data.clamp(-128, 127).to(torch.int8)
        else:
            round_data = round_data.clamp(0, 255).to(torch.uint8)
    scale_out = scale.squeeze(-1)
    if offset is not None:
        offset_out = offset.squeeze(-1)
    else:
        offset_out = torch.zeros_like(scale_out)
    return [round_data, scale_out, offset_out]
