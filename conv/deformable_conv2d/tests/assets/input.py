#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FIT FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__input__ = {
    "aclnn": {
        "aclnnDeformableConv2d": "deformable_conv2d_input"
    }
}


def deformable_conv2d_input(x, weight, offset, bias=None,
                            *, kernel_size, stride, padding, dilation,
                            groups=1, deformable_groups=1, modulated=True,
                            **kwargs):
    '''
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
    '''
    # Get input ranges from kwargs
    input_ranges = kwargs.get('input_ranges', [])
    
    # Apply input ranges if available
    if input_ranges and len(input_ranges) >= 3:
        x_range = input_ranges[0] if input_ranges[0] else (-10, 10)
        weight_range = input_ranges[1] if input_ranges[1] else (-1, 1)
        offset_range = input_ranges[2] if input_ranges[2] else (-1, 1)
        
        low, high = x_range
        x = np.random.uniform(low, high, x.shape).astype(x.dtype)
        
        low, high = weight_range
        weight = np.random.uniform(low, high, weight.shape).astype(weight.dtype)
        
        low, high = offset_range
        offset = np.random.uniform(low, high, offset.shape).astype(offset.dtype)
        
        if bias is not None and len(input_ranges) > 3:
            bias_range = input_ranges[3] if input_ranges[3] else (-1, 1)
            low, high = bias_range
            bias = np.random.uniform(low, high, bias.shape).astype(bias.dtype)
    
    return [x, weight, offset, bias]
