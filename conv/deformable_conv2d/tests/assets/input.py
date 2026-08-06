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

__input__ = {
    "aclnn": {"aclnnDeformableConv2d": "deformable_conv2d_input"},
    "e2e": {"torch_npu.npu_deformable_conv2d": "deformable_conv2d_input"},
}


def _safe_uniform(shape, dtype, low=-1, high=1):
    if low is None:
        low = -1
    if high is None:
        high = 1
    import torch

    if isinstance(dtype, torch.dtype):
        dtype_map = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.bfloat16: np.float32,
            torch.int64: np.int64,
            torch.int32: np.int32,
            torch.int8: np.int8,
        }
        dtype = dtype_map.get(dtype, np.float32)
    return np.random.uniform(low, high, shape).astype(dtype)


def deformable_conv2d_input(
    x,
    weight,
    offset,
    bias=None,
    kernel_size=None,
    stride=None,
    padding=None,
    dilation=None,
    groups=1,
    deformable_groups=1,
    modulated=True,
    out=None,
    deform_out=None,
    **kwargs,
):
    """
    Input function for deformable_conv2d operator.
    All parameters follow @deformable_conv2d_def.cpp without outputs.
    All input Tensors are numpy.ndarray.


    Args:
        x: Input feature map tensor, shape (N, C, H, W) in NCHW format
        weight: Convolution weight tensor, shape (outC, inC/groups, kH, kW)
        offset: Offset tensor, shape (N, 3*deformable_groups*kH*kW, outH, outW)
        bias: Optional bias tensor, shape (outC,)
        kernel_size: Kernel size [kH, kW], REQUIRED
        stride: Stride values [n, c, h, w], REQUIRED
        padding: Padding values [top, bottom, left, right], REQUIRED
        dilation: Dilation values [n, c, h, w], default [1,1,1,1]
        groups: Number of groups for grouped convolution, default 1
        deformable_groups: Number of deformable groups, default 1
        modulated: Whether to use modulated deformable convolution, default True
        **kwargs: Extended context including:
            - input_dtypes: List[dtype] - input data types
            - input_ori_shapes: List[tuple] - original input shapes
            - input_formats: List[str] - input formats
            - input_ori_formats: List[str] - original input formats
            - input_ranges: List[tuple] - input data ranges


    Returns:
        List of input tensors: [x, weight, offset, bias]
    """
    # Get input ranges from kwargs
    input_ranges = kwargs.get("input_ranges", [])

    if x is not None:
        r = (
            input_ranges[0]
            if input_ranges and len(input_ranges) > 0 and input_ranges[0]
            else (-10, 10)
        )
        x = _safe_uniform(x.shape, x.dtype, r[0], r[1])

    if weight is not None:
        r = (
            input_ranges[1]
            if input_ranges and len(input_ranges) > 1 and input_ranges[1]
            else (-1, 1)
        )
        weight = _safe_uniform(weight.shape, weight.dtype, r[0], r[1])

    if offset is not None:
        r = (
            input_ranges[2]
            if input_ranges and len(input_ranges) > 2 and input_ranges[2]
            else (-1, 1)
        )
        offset = _safe_uniform(offset.shape, offset.dtype, r[0], r[1])

    if bias is not None:
        r = (
            input_ranges[3]
            if input_ranges and len(input_ranges) > 3 and input_ranges[3]
            else (-1, 1)
        )
        bias = _safe_uniform(bias.shape, bias.dtype, r[0], r[1])

    return [
        x,
        weight,
        offset,
        bias,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        deformable_groups,
        modulated,
        out,
        deform_out,
    ]
