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
    bias=None,
    scale=None,
    offset=None,
    output=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    outputPadding=0,
    groups=1,
    offsetx=0,
    roundMode='',
    **kwargs
):
    """
    Input function for aclnnQuantConvolution.
    Parameter names and order follow aclnn_quant_convolution.h:
    aclnnQuantConvolutionGetWorkspaceSize(input, weight, bias, scale, offset, output,
                                          stride, padding, dilation, transposed,
                                          outputPadding, groups, offsetx, roundMode)
    """
    if scale is not None and not isinstance(scale, (int, float)):
        n_channels = scale.shape[0] if hasattr(scale, 'shape') else len(scale)
        scale_float = np.random.uniform(0.01, 1.0, size=n_channels).astype(np.float32)
        scale_float = np.bitwise_and(scale_float.view(np.uint32), 0xffffe000).view(np.float32)
        scale_encoded = torch.from_numpy(scale_float.view(np.uint32).astype(np.int64))
        if isinstance(scale, torch.Tensor):
            scale.copy_(scale_encoded)
        elif isinstance(scale, np.ndarray):
            scale[:] = scale_encoded.numpy()
        else:
            scale = scale_encoded

    return [x, weight, bias, scale, offset]


def torch_npu_quant_conv2d_input(
    x,
    weight,
    scale,
    strides=1,
    pads=0,
    dilations=1,
    groups=1,
    offset_x=0,
    round_mode='rint',
    output_dtype=None,
    bias=None,
    offset=None,
    input_dtype=None,
    weight_dtype=None,
    **kwargs
):
    """
    Input function for torch_npu.npu_quant_conv2d.
    Parameter order matches the API signature.

    For int8 quantized conv, scale must store float32 binary representation
    in int64 format. TTK generates random int64 which produces garbage float32
    values. Generate proper float32 scale values and encode as int64.
    For float8/hifloat8 types, scale encoding is left as-is.
    """
    if scale is not None and not isinstance(scale, (int, float)):
        x_dtype = str(x.dtype).split('.')[-1] if hasattr(x, 'dtype') else ''
        if x_dtype == 'int8':
            n_channels = scale.shape[0] if hasattr(scale, 'shape') else len(scale)
            scale_float = np.random.uniform(0.01, 1.0, size=n_channels).astype(np.float32)
            scale_float = np.bitwise_and(scale_float.view(np.uint32), 0xffffe000).view(np.float32)
            scale_encoded = torch.from_numpy(scale_float.view(np.uint32).astype(np.int64))
            if isinstance(scale, torch.Tensor):
                scale.copy_(scale_encoded)
            elif isinstance(scale, np.ndarray):
                scale[:] = scale_encoded.numpy()
            else:
                scale = scale_encoded

    return [x, weight, bias, scale, offset]


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