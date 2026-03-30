#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in免 root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__input__ = {
    "kernel": {
        "extend_conv2d": "extend_conv2d_input"
    }
}


def extend_conv2d_input(x, filter, bias=None, offset_w=None,
                        scale0=None, relu_weight0=None, clip_value0=None,
                        scale1=None, relu_weight1=None, clip_value1=None,
                        *,
                        strides, 
                        pads :list=[0, 0, 0, 0], 
                        dilations: list=[1, 1, 1, 1], 
                        groups: int=1, 
                        data_format: str=NCHW_FORMAT,  # attributes
                        offset_x: int = 0, 
                        round_mode: str = "rint", 
                        pad_mode: str = "SPECIFIC",   
                        enable_hf32: bool = False,  
                        enable_relu0: bool = False, 
                        enable_relu1: bool = False,  
                        dual_output: bool = False, 
                        dtype0: int = -1, 
                        dtype1: int = -1, 
                        **kwargs):
    """
    Input function for extend_conv2d.
    All the parameters (names and order) follow @extend_conv2d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x: Input tensor
        filter: Filter tensor
        bias: Bias tensor (optional)
        offset_w: Offset w tensor (optional)
        scale0: Scale0 tensor (optional)
        relu_weight0: Relu weight0 tensor (optional)
        clip_value0: Clip value0 tensor (optional)
        scale1: Scale1 tensor (optional)
        relu_weight1: Relu weight1 tensor (optional)
        clip_value1: Clip value1 tensor (optional)
        **kwargs: Extended context including:
            - input_dtypes: List of input data types
            - full_soc_version: Full SoC version (e.g., 'Ascend910B2')
            - short_soc_version: Short SoC version (e.g., 'Ascend910B')
    Returns:
        List of processed inputs
    """
    x_input = x
    filter_input = filter
    bias_input = bias
    offset_w_input = offset_w
    scale0_input = scale0
    relu_weight0_input = relu_weight0
    clip_value0_input = clip_value0
    scale1_input = scale1
    relu_weight1_input = relu_weight1
    clip_value1_input = clip_value1

    if scale0 is not None:
        scale0_np = scale0 if isinstance(scale0, np.ndarray) else np.array(scale0)
        scale0_input = np.bitwise_and(scale0_np.astype(np.float32).view(np.uint32), 0xffffe000).view(np.float32)
        scale0_input = scale0_input.view(np.uint32).astype(np.uint64)

    return [x_input, filter_input, bias_input, offset_w_input, scale0_input,
            relu_weight0_input, clip_value0_input, scale1_input,
            relu_weight1_input, clip_value1_input]
