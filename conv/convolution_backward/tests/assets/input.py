#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from typing import List, Optional, Union
import numpy as np
import torch

__input__ = {
    "aclnn": {
        "aclnnConvolutionBackward": "aclnn_convolution_backward_input",
        "aclnnConvTbcBackward": "aclnn_conv_tbc_backward_input",
    },
    "torch_api": {
        "torch.ops.aten.convolution_backward": "aten_convolution_backward_input",
    }
}


def aclnn_convolution_backward_input(
    gradOutput,
    input,
    weight,
    biasSizes: Optional[List[int]] = None,
    stride: Union[int, List[int]] = 1,
    padding: Union[int, List[int]] = 0,
    dilation: Union[int, List[int]] = 1,
    transposed: bool = False,
    outputPadding: Union[int, List[int]] = 0,
    groups: int = 1,
    outputMask: List[bool] = [True, True, False],
    cubeMathType: int = 0,
    gradInput=None,
    gradWeight=None,
    gradBias=None,
    **kwargs
):
    """
    Input function for aclnnConvolutionBackward.
    Parameter names and order follow aclnn_convolution_backward.h:
    aclnnConvolutionBackwardGetWorkspaceSize(gradOutput, input, weight, biasSizes,
                                              stride, padding, dilation, transposed,
                                              outputPadding, groups, outputMask, cubeMathType,
                                              gradInput, gradWeight, gradBias)
    
    Args:
        gradOutput: Gradient of output (NCL/NCHW/NCDHW)
        input: Original input tensor
        weight: Weight tensor
        biasSizes: Bias shape (aclIntArray)
        stride: Convolution stride (aclIntArray)
        padding: Padding (aclIntArray)
        dilation: Dilation (aclIntArray)
        transposed: Whether transposed convolution
        outputPadding: Output padding for transposed conv (aclIntArray)
        groups: Number of groups
        outputMask: Output mask (aclBoolArray) [gradInput, gradWeight, gradBias]
        cubeMathType: Cube math type (int8_t)
        gradInput: Pre-allocated gradInput output
        gradWeight: Pre-allocated gradWeight output
        gradBias: Pre-allocated gradBias output
        **kwargs: Additional context
    
    Returns:
        List of inputs
    """
    return [gradOutput, input, weight]


def aclnn_conv_tbc_backward_input(
    self,
    input,
    weight,
    bias=None,
    pad: int = 0,
    cubeMathType: int = 0,
    gradInput=None,
    gradWeight=None,
    gradBias=None,
    **kwargs
):
    """
    Input function for aclnnConvTbcBackward.
    Parameter names and order follow aclnn_convolution_backward.h:
    aclnnConvTbcBackwardGetWorkspaceSize(self, input, weight, bias, pad, cubeMathType,
                                          gradInput, gradWeight, gradBias)
    
    Args:
        self: Convolution output gradient (ND/NCL)
        input: Convolution input (ND/NCL)
        weight: Convolution weight (ND/NCL)
        bias: Convolution bias
        pad: Padding (int64_t)
        cubeMathType: Cube math type (int8_t)
        gradInput: Pre-allocated gradInput output
        gradWeight: Pre-allocated gradWeight output
        gradBias: Pre-allocated gradBias output
        **kwargs: Additional context
    
    Returns:
        List of inputs
    """
    return [self, input, weight, bias]


def aten_convolution_backward_input(
    grad_output,
    input,
    weight,
    bias_sizes: Optional[List[int]] = None,
    stride: Union[int, List[int]] = 1,
    padding: Union[int, List[int]] = 0,
    dilation: Union[int, List[int]] = 1,
    transposed: bool = False,
    output_padding: Union[int, List[int]] = 0,
    groups: int = 1,
    output_mask: List[bool] = [True, True, False],
    **kwargs
):
    """
    Input function for torch.ops.aten.convolution_backward.
    Supports 1D, 2D, 3D convolution backward.
    """
    return [grad_output, input, weight]