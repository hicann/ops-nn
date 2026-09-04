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

__input__ = {"kernel": {"extend_conv_transpose": "extend_conv_transpose_input"}}


def encode_deq_scale_u64(fp32_vals):
    """
    Encode float32 dequantization scales into the uint64 fixpipe format:
    low 32 bits = fp32 bit pattern masked by 0xFFFFE000 (19 significant bits),
    matching the hardware fixpipe multiplication field (trans_quant_param_v2).
    """
    vals = np.ascontiguousarray(np.asarray(fp32_vals, dtype=np.float32))
    u32 = vals.view(np.uint32).copy()
    u32 &= np.uint32(0xFFFFE000)
    return u32.astype(np.uint64)


def extend_conv_transpose_input(
    input_size,
    x,
    filter,
    bias=None,
    scale=None,
    *,
    strides: list = None,
    pads: list = None,
    dilations: list = None,
    groups: int = 1,
    data_format: str = "NDHWC",
    output_padding: list = None,
    offset_x: int = 0,
    fusion_mode: int = 0,
    y_quant_mode: int = 0,
    **kwargs,
):
    """
    Input function for extend_conv_transpose.
    All the parameters (names and order) follow @extend_conv_transpose_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Randomly generated uint64 scale data would decode to arbitrary fp32 bit patterns
    (possibly NaN/Inf/huge values), so scale is re-encoded into valid fixpipe format
    with well-behaved values.

    Args:
        input_size: Output shape (const input, int32 tensor)
        x: Input tensor (int8)
        filter: Filter tensor (int8)
        bias: Bias tensor (optional, int32)
        scale: Dequant scale tensor (optional, uint64)
        **kwargs: Extended context including:
            - input_dtypes: List of input data types
            - full_soc_version: Full SoC version (e.g., 'Ascend950PR')
            - short_soc_version: Short SoC version (e.g., 'Ascend950')

    Returns:
        List of processed inputs
    """
    if scale is not None:
        scale_np = scale if isinstance(scale, np.ndarray) else np.array(scale)
        n = scale_np.size
        if n == 1:
            # SCALAR_QUANT: single deq scale
            fp32_vals = np.array([0.05], dtype=np.float32)
        else:
            # VECTOR_QUANT: per-output-channel deq scales in [0.05, 0.5]
            fp32_vals = np.linspace(0.05, 0.5, n).astype(np.float32)
        scale = encode_deq_scale_u64(fp32_vals)

    return [input_size, x, filter, bias, scale]
