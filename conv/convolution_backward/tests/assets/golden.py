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

__golden__ = {
    "aclnn": {
        "aclnnConvolutionBackward": "aclnn_convolution_backward_golden",
        "aclnnConvTbcBackward": "aclnn_conv_tbc_backward_golden",
    },
    "torch_api": {
        "torch.ops.aten.convolution_backward": "aten_convolution_backward_golden",
    }
}


def get_conv_dim(input_shape, weight_shape):
    """
    Determine convolution dimension from tensor shapes.
    Returns: 1, 2, or 3
    """
    input_spatial_dims = len(input_shape) - 2
    weight_spatial_dims = len(weight_shape) - 2
    return max(input_spatial_dims, weight_spatial_dims)


def to_float32(t):
    """Convert torch tensor or numpy array to float32."""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        dtype_str = str(t.dtype)
        if any(s in dtype_str for s in ['hifloat8', 'float8', 'float4', 'int4', 'bfloat16']):
            return t.float()
        return t.to(torch.float32)
    return t.astype(np.float32)


def ensure_list(param, num_dims):
    """Ensure parameter is a list with correct number of dimensions."""
    if isinstance(param, (list, tuple)):
        if len(param) == num_dims:
            return list(param)
        elif len(param) == 1:
            return [int(param[0])] * num_dims
        return list(param)
    return [int(param)] * num_dims


def simulate_hf32_precision(data, short_soc_version=None):
    """
    Simulate HF32 (Half Float 32) precision.
    Ascend910B (default): truncates lower 12 bits of float32 mantissa, keeping 20 bits with rounding.
    Ascend910_95: truncates lower 13 bits of float32 mantissa, keeping 19 bits with rounding.
    """
    input_hf32 = data.view(np.int32)
    if short_soc_version in ("Ascend910B",):
        input_hf32 = np.right_shift(np.right_shift(input_hf32, 11) + 1, 1)
        input_hf32 = np.left_shift(input_hf32, 12)
    else:
        input_hf32 = np.right_shift(np.right_shift(input_hf32, 12) + 1, 1)
        input_hf32 = np.left_shift(input_hf32, 13)
    return input_hf32.view(np.float32)


def _compute_conv_backward(gradOutput, input, weight, stride, padding,
                           dilation, groups, conv_dim, transposed=False, outputPadding=0,
                           outputMask=[True, True, False], biasSizes=None,
                           cubeMathType=0, short_soc_version=None):
    stride = ensure_list(stride, conv_dim)
    padding = ensure_list(padding, conv_dim)
    dilation = ensure_list(dilation, conv_dim)
    outputPadding = ensure_list(outputPadding, conv_dim)

    if isinstance(gradOutput, np.ndarray):
        gradOutput = torch.from_numpy(gradOutput)
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(weight, np.ndarray):
        weight = torch.from_numpy(weight)

    if gradOutput is not None:
        gradOutput = gradOutput.float()
    if input is not None:
        input = input.float()
    if weight is not None:
        weight = weight.float()

    input_dtype_str = str(input.dtype).split('.')[-1] if input is not None else 'float32'
    if input_dtype_str == 'float32' and input is not None and weight is not None:
        if cubeMathType in [1, 3]:
            input_np = simulate_hf32_precision(input.numpy().astype(np.float32), short_soc_version)
            weight_np = simulate_hf32_precision(weight.numpy().astype(np.float32), short_soc_version)
            gradOutput_np = simulate_hf32_precision(gradOutput.numpy().astype(np.float32), short_soc_version)
            input = torch.from_numpy(input_np)
            weight = torch.from_numpy(weight_np)
            gradOutput = torch.from_numpy(gradOutput_np)
        elif cubeMathType == 2:
            gradOutput = gradOutput.to(torch.float16).to(torch.float32)
            input = input.to(torch.float16).to(torch.float32)
            weight = weight.to(torch.float16).to(torch.float32)

    if not outputMask[2]:
        biasSizes = None
    elif biasSizes is None or (isinstance(biasSizes, (list, torch.Size)) and len(biasSizes) == 0):
        biasSizes = list(weight.shape[:1])

    return torch.ops.aten.convolution_backward(
        gradOutput, input, weight, biasSizes,
        stride, padding, dilation, transposed, outputPadding, groups, outputMask
    )


def aclnn_convolution_backward_golden(
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
    ACLNN API golden for aclnnConvolutionBackward.
    Parameter names and order follow aclnn_convolution_backward.h:
    aclnnConvolutionBackwardGetWorkspaceSize(gradOutput, input, weight, biasSizes,
                                              stride, padding, dilation, transposed,
                                              outputPadding, groups, outputMask, cubeMathType,
                                              gradInput, gradWeight, gradBias)
    
    Supports 1D, 2D, 3D convolution backward.
    """
    input_shape = input.shape if isinstance(input, torch.Tensor) or hasattr(input, 'shape') else None
    weight_shape = weight.shape if isinstance(weight, torch.Tensor) or hasattr(weight, 'shape') else None
    conv_dim = get_conv_dim(input_shape, weight_shape)
    short_soc_version = kwargs.get("short_soc_version", None)

    grad_input, grad_weight, grad_bias = _compute_conv_backward(
        gradOutput, input, weight, stride, padding,
        dilation, groups, conv_dim, transposed, outputPadding, outputMask, biasSizes,
        cubeMathType=cubeMathType, short_soc_version=short_soc_version
    )
    
    return (grad_input, grad_weight, grad_bias)


def aclnn_conv_tbc_backward_golden(
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
    ACLNN API golden for aclnnConvTbcBackward.
    Parameter names and order follow aclnn_convolution_backward.h:
    aclnnConvTbcBackwardGetWorkspaceSize(self, input, weight, bias, pad, cubeMathType,
                                          gradInput, gradWeight, gradBias)

    TBC format: (T, B, C) where T is time/sequence, B is batch, C is channels.
    Equivalent to conv1d with input shape (B, C, T).
    """
    short_soc_version = kwargs.get("short_soc_version", None)

    if isinstance(self, np.ndarray):
        self = torch.from_numpy(self)
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(weight, np.ndarray):
        weight = torch.from_numpy(weight)

    self = self.float()
    input = input.float()
    weight = weight.float()

    input_dtype_str = str(input.dtype).split('.')[-1]
    if input_dtype_str == 'float32':
        if cubeMathType in [1, 3]:
            self_np = simulate_hf32_precision(self.numpy().astype(np.float32), short_soc_version)
            input_np = simulate_hf32_precision(input.numpy().astype(np.float32), short_soc_version)
            weight_np = simulate_hf32_precision(weight.numpy().astype(np.float32), short_soc_version)
            self = torch.from_numpy(self_np)
            input = torch.from_numpy(input_np)
            weight = torch.from_numpy(weight_np)
        elif cubeMathType == 2:
            self = self.to(torch.float16).to(torch.float32)
            input = input.to(torch.float16).to(torch.float32)
            weight = weight.to(torch.float16).to(torch.float32)

    output_mask = [True, True, bias is not None]
    bias_sizes = list(weight.shape[:1]) if bias is not None else None

    grad_input_ncl, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
        self.permute(1, 2, 0),
        input.permute(1, 2, 0),
        weight,
        bias_sizes,
        [1],
        [pad],
        [1],
        False,
        [0],
        1,
        output_mask
    )

    if grad_input_ncl is not None:
        grad_input_ncl = grad_input_ncl.permute(2, 0, 1)

    return (grad_input_ncl, grad_weight, grad_bias)


def aten_convolution_backward_golden(
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
    Golden for torch.ops.aten.convolution_backward.
    Supports 1D, 2D, 3D convolution backward.
    """

    input_shape = input.shape if isinstance(input, torch.Tensor) or hasattr(input, 'shape') else None
    weight_shape = weight.shape if isinstance(weight, torch.Tensor) or hasattr(weight, 'shape') else None
    conv_dim = get_conv_dim(input_shape, weight_shape)
    short_soc_version = kwargs.get("short_soc_version", None)

    grad_input, grad_weight, grad_bias = _compute_conv_backward(
        grad_output, input, weight, stride, padding,
        dilation, groups, conv_dim, transposed, output_padding, output_mask, bias_sizes,
        short_soc_version=short_soc_version
    )
    
    return (grad_input, grad_weight, grad_bias)