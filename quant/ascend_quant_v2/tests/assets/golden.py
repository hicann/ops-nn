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
        "aclnnAscendQuantV3": "aclnn_ascend_quant_v3_golden",
    },
    "kernel": {"ascend_quant_v2": "ascend_quant_v2_golden"},
}


def ascend_quant_v2_golden(
    x,
    scale,
    offset=None,
    *,
    sqrt_mode=False,
    round_mode="round",
    dst_type=2,
    axis=-1,
    **kwargs,
):
    """
    Golden function for ascend_quant_v2.
    All the parameters (names and order) follow @ascend_quant_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    cast_dtype = "float32"
    x = x.astype(cast_dtype)
    scale = scale.astype(cast_dtype)
    offset = offset.astype(cast_dtype) if offset is not None else None

    scale_rst = x * (scale**2) if sqrt_mode else x * scale
    if offset is not None:
        scale_rst = scale_rst + offset

    if round_mode == "round":
        round_data = (
            np.round(scale_rst, 8)
            if str(kwargs["output_dtypes"][0])
            in ("hifloat8", "float8_e5m2", "float8_e4m3fn")
            else np.round(scale_rst, 0)
        )
    elif round_mode == "floor":
        round_data = np.floor(scale_rst)
    elif round_mode == "ceil":
        round_data = np.ceil(scale_rst)
    else:
        round_data = np.trunc(scale_rst)

    if dst_type == 2:
        round_data = np.clip(round_data, -128, 127)
        round_data = round_data.astype("int8", copy=False)
    elif dst_type == 29:
        from ml_dtypes import int4

        round_data = np.clip(round_data, -8, 7)
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

    return round_data


def aclnn_ascend_quant_v3_golden(
    x, scale, offset, sqrtMode, roundMode, dstType, axis, y, **kwargs
):
    if hasattr(sqrtMode, "item"):
        sqrtMode = bool(sqrtMode.item())
    if hasattr(roundMode, "decode"):
        roundMode = roundMode.decode()
    x_f = x.to(torch.float32)
    scale_f = scale.to(torch.float32)
    offset_f = offset.to(torch.float32) if offset is not None else None
    scale_rst = x_f * (scale_f**2) if sqrtMode else x_f * scale_f
    if offset_f is not None:
        scale_rst = scale_rst + offset_f
    if roundMode == "round":
        round_data = torch.round(scale_rst)
    elif roundMode == "floor":
        round_data = torch.floor(scale_rst)
    elif roundMode == "ceil":
        round_data = torch.ceil(scale_rst)
    else:
        round_data = torch.round(scale_rst)
    round_data = round_data.clamp(-128, 127).to(torch.int8)
    return [round_data]
