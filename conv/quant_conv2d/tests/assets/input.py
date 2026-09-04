#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__input__ = {"kernel": {"quant_conv2d": "quant_conv2d_input"}}

NCHW_FORMAT = "NCHW"


def quant_conv2d_input(
    x,
    filter,
    scale,
    bias=None,
    offset=None,
    *,
    dtype: int,
    strides: list,
    pads: list = [0, 0, 0, 0],
    dilations: list = [1, 1, 1, 1],
    groups: int = 1,
    data_format: str = NCHW_FORMAT,
    offset_x: int = 0,
    round_mode: str = "rint",
    **kwargs,
):
    """
    Input function for quant_conv2d.
    All the parameters (names and order) follow @quant_conv2d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x: Input tensor
        filter: Filter tensor
        scale: Scale tensor
        bias: Bias tensor (optional)
        offset: Offset tensor (optional)
        **kwargs: Extended context including:
            - input_dtypes: List of input data types
            - full_soc_version: Full SoC version (e.g., 'Ascend950')
            - short_soc_version: Short SoC version (e.g., 'Ascend950')
    Returns:
        List of processed inputs for dynamic case
    """

    x_input = x
    filter_input = filter
    scale_input = scale
    bias_input = bias
    offset_input = offset

    if scale_input is not None:
        scale_np = (
            scale_input
            if isinstance(scale_input, np.ndarray)
            else np.array(scale_input)
        )
        scale_input = np.bitwise_and(
            scale_np.astype(np.float32).view(np.uint32), 0xFFFFE000
        ).view(np.float32)
        scale_input = scale_input.view(np.uint32).astype(scale_np.dtype)

    return [x_input, filter_input, scale_input, bias_input, offset_input]
