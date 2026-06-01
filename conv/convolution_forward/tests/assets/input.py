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

from typing import List, Optional, Tuple, Union
import numpy as np
import torch

__input__ = {
    "aclnn": {
        "aclnnConvolution": "aclnn_convolution_input",
        "aclnnConvTbc": "aclnn_conv_tbc_input",
        "aclnnConvDepthwise2d": "aclnn_conv_depthwise2d_input",
        "aclnnQuantConvolution": "aclnn_quant_conv2d_input",
    },
    "e2e": {
        "torch.conv1d": "torch_conv1d_input",
        "torch.conv2d": "torch_conv2d_input",
        "torch.conv3d": "torch_conv3d_input",
        "torch.nn.functional.conv1d": "torch_conv1d_input",
        "torch.nn.functional.conv2d": "torch_conv2d_input",
        "torch.nn.functional.conv3d": "torch_conv3d_input",
        "torch_npu.npu_conv2d": "torch_conv2d_input",
        "torch_npu.npu_conv3d": "torch_conv3d_input",
        "torch_npu.npu_quant_conv2d": "torch_npu_quant_conv2d_input",
        "torch.ops.aten.convolution": "aten_convolution_input",
    }
}


def aclnn_convolution_input(
    input,
    weight,
    bias=None,
    stride: Union[int, List[int]] = 1,
    padding: Union[int, List[int]] = 0,
    dilation: Union[int, List[int]] = 1,
    transposed: bool = False,
    outputPadding: Union[int, List[int]] = 0,
    groups: int = 1,
    output=None,
    cubeMathType: int = 0,
    **kwargs
):
    """
    Input function for aclnnConvolution.
    Parameter names and order follow aclnn_convolution.h:
    aclnnConvolutionGetWorkspaceSize(input, weight, bias, stride, padding, dilation,
                                      transposed, outputPadding, groups, output, cubeMathType)
    
    Supports 1D, 2D, 3D convolutions.
    """
    return [input, weight, bias]


def aclnn_conv_tbc_input(
    self,
    weight,
    bias=None,
    pad: int = 0,
    output=None,
    cubeMathType: int = 0,
    **kwargs
):
    """
    Input function for aclnnConvTbc.
    Parameter names and order follow aclnn_convolution.h:
    aclnnConvTbcGetWorkspaceSize(self, weight, bias, pad, output, cubeMathType)
    
    TBC format: (T, B, C) where T is time/sequence, B is batch, C is channels.
    """
    return [self, weight, bias]


def aclnn_conv_depthwise2d_input(
    self,
    weight,
    kernelSize: Union[int, List[int]] = None,
    bias=None,
    stride: Union[int, List[int]] = 1,
    padding: Union[int, List[int]] = 0,
    dilation: Union[int, List[int]] = 1,
    out=None,
    cubeMathType: int = 0,
    **kwargs
):
    """
    Input function for aclnnConvDepthwise2d.
    Parameter names and order follow aclnn_convolution.h:
    aclnnConvDepthwise2dGetWorkspaceSize(self, weight, kernelSize, bias, stride,
                                          padding, dilation, out, cubeMathType)
    
    Depthwise convolution where groups = input_channels.
    """
    return [self, weight, bias]


def aclnn_quant_conv2d_input(
    x,
    weight,
    scale,
    bias=None,
    offset=None,
    strides: Union[int, List[int]] = 1,
    pads: Union[int, List[int]] = 0,
    dilations: Union[int, List[int]] = 1,
    groups: int = 1,
    offset_x: int = 0,
    round_mode: str = "rint",
    output_dtype: Union[str, int, torch.dtype] = "int8",
    input_dtype: Union[str, int, torch.dtype] = None,
    weight_dtype: Union[str, int, torch.dtype] = None,
    **kwargs
):
    """
    Input function for aclnnQuantConvolution (quantized convolution).
    
    Args:
        x: Input feature map (N, C, H, W) - int8, float8_e4m3fn, hifloat8
        weight: Weight tensor (OutC, InC/groups, kH, kW)
        scale: Per-channel scale (OutC) - stored as int64 (float32 binary masked)
        bias: Bias tensor (OutC) or None
        offset: Offset tensor or None (reserved)
        strides: Convolution stride
        pads: Padding
        dilations: Dilation
        groups: Number of groups
        offset_x: Padding offset value
        round_mode: Rounding mode
        output_dtype: Output dtype (str/int/torch.dtype)
        input_dtype: Input dtype
        weight_dtype: Weight dtype
        **kwargs: Additional context
    
    Returns:
        List of inputs
    """
    scale_input = None
    if scale is not None:
        scale_np = scale if isinstance(scale, np.ndarray) else np.array(scale)
        scale_float = scale_np.astype(np.float32)
        scale_input = np.bitwise_and(scale_np.astype(np.float32).view(np.uint32), 0xffffe000).view(np.float32)
        scale_input = scale_input.view(np.uint32).astype(np.uint64)
        
    return [x, weight, scale_input, bias, offset]


def torch_npu_quant_conv2d_input(
    x,
    weight,
    scale,
    bias=None,
    offset=None,
    strides: Union[int, List[int]] = 1,
    pads: Union[int, List[int]] = 0,
    dilations: Union[int, List[int]] = 1,
    groups: int = 1,
    offset_x: int = 0,
    round_mode: str = "rint",
    output_dtype: Union[str, int, torch.dtype] = None,
    input_dtype: Union[str, int, torch.dtype] = None,
    weight_dtype: Union[str, int, torch.dtype] = None,
    **kwargs
):
    """
    Input function for torch_npu.npu_quant_conv2d.
    
    Args:
        x: Input tensor
        weight: Weight tensor
        scale: Scale tensor (int64 storing float32 binary)
        bias: Bias tensor (optional)
        offset: Offset tensor (optional)
        strides: Stride
        pads: Padding
        dilations: Dilation
        groups: Groups count
        offset_x: Padding offset
        round_mode: Rounding mode
        output_dtype: Output dtype
        input_dtype: Input dtype
        weight_dtype: Weight dtype
        **kwargs: Additional context
    
    Returns:
        List of inputs
    """
    scale_input = None
    if scale is not None:
        scale_np = scale if isinstance(scale, np.ndarray) else np.array(scale)
        scale_input = np.bitwise_and(scale_np.astype(np.float32).view(np.uint32), 0xffffe000).view(np.float32)
        scale_input = scale_input.view(np.uint32).astype(np.uint64)

    return [x, weight, scale_input, bias, offset]


def torch_conv1d_input(
    input,
    weight,
    bias=None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    **kwargs
):
    """
    Input function for torch.conv1d / torch.nn.functional.conv1d.
    """
    return [input, weight, bias]


def torch_conv2d_input(
    input,
    weight,
    bias=None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    **kwargs
):
    """
    Input function for torch.conv2d / torch.nn.functional.conv2d.
    """
    return [input, weight, bias]


def torch_conv3d_input(
    input,
    weight,
    bias=None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int], str] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
    **kwargs
):
    """
    Input function for torch.conv3d / torch.nn.functional.conv3d.
    """
    return [input, weight, bias]


def aten_convolution_input(
    input,
    weight,
    bias=None,
    stride: Union[int, List[int]] = 1,
    padding: Union[int, List[int]] = 0,
    dilation: Union[int, List[int]] = 1,
    transposed: bool = False,
    output_padding: Union[int, List[int]] = 0,
    groups: int = 1,
    benchmark: bool = False,
    deterministic: bool = False,
    cudnn_allow_tf32: bool = True,
    **kwargs
):
    """
    Input function for torch.ops.aten.convolution.
    Supports 1D, 2D, 3D convolutions based on input tensor dimensions.
    """
    return [input, weight, bias]