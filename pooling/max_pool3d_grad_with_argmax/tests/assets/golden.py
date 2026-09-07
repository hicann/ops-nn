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
from copy import deepcopy

__golden__ = {
    "aclnn": {
        "aclnnMaxPool2dWithIndicesBackward": "aclnn_max_pool2d_with_indices_backward_golden",
    },
    "kernel": {"max_pool3d_grad_with_argmax": "max_pool3d_grad_with_argmax_golden"},
}


def max_pool3d_grad_with_argmax_golden(
    x,
    grad,
    argmax,
    *,
    ksize,
    strides,
    pads,
    dilation=[1, 1, 1],
    ceil_mode=False,
    data_format="NCDHW",
    **kwargs,
):
    """
    Golden function for max_pool3d_grad_with_argmax.
    All the parameters (names and order) follow @max_pool3d_grad_with_argmax_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    input_x = deepcopy(x)
    inpu_x_dtype = input_x.dtype

    attr_re_ksize = ksize
    attr_re_strides = strides
    attr_re_pads = pads
    attr_op_dilations = dilation
    attr_op_ceil_mode = ceil_mode
    attr_op_format = data_format

    max_out = deepcopy(grad)
    max_indices = deepcopy(argmax)

    # NDHWC -> NCDHW
    if attr_op_format == "NDHWC":
        input_x = input_x.transpose(0, 4, 1, 2, 3)
        max_out = max_out.transpose(0, 4, 1, 2, 3)
        max_indices = max_indices.transpose(0, 4, 1, 2, 3)

    max_indices = max_indices.astype(np.int64)
    if inpu_x_dtype == "bfloat16" or inpu_x_dtype == "float16":
        input_x = input_x.astype(np.float32)
        max_out = max_out.astype(np.float32)
    input_x = torch.from_numpy(input_x)
    max_out = torch.from_numpy(max_out)
    max_indices = torch.from_numpy(max_indices)

    backward_out = torch.ops.aten.max_pool3d_with_indices_backward(
        max_out,
        input_x,
        kernel_size=attr_re_ksize,
        stride=attr_re_strides,
        padding=attr_re_pads,
        dilation=attr_op_dilations,
        ceil_mode=attr_op_ceil_mode,
        indices=max_indices,
    )
    if inpu_x_dtype == "bfloat16" or inpu_x_dtype == "float16":
        backward_out = backward_out.numpy().astype(inpu_x_dtype)
    else:
        backward_out = backward_out.numpy()

    # NCDHW -> NDHWC
    if attr_op_format == "NDHWC":
        backward_out = backward_out.transpose(0, 2, 3, 4, 1)

    return backward_out


def aclnn_max_pool2d_with_indices_backward_golden(
    gradOutput,
    self,
    indices,
    kernelSize=0,
    stride=0,
    padding=0,
    dilation=0,
    ceilMode=0,
    gradInput=None,
    **kwargs,
):
    """
    Aclnn golden for aclnnMaxPool2dWithIndicesBackward.
    Parameters follow @aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    grad = gradOutput
    grad_dtype = grad.dtype
    indices_dtype = indices.dtype

    input_grad_format = kwargs.get("tensor_formats", ["NCHW"])[0]
    if grad_dtype == torch.float16 or grad_dtype == torch.bfloat16:
        grad = grad.to(torch.float32)
        self = self.to(torch.float32)
    if indices_dtype == torch.int32:
        indices = indices.to(torch.int64)
    if input_grad_format == "NHWC":
        grad = grad.permute(0, 3, 1, 2)
        self = self.permute(0, 3, 1, 2)
        indices = indices.permute(0, 3, 1, 2)

    ceilMode = bool(ceilMode) if ceilMode else False

    backward_out = torch.ops.aten.max_pool2d_with_indices_backward(
        grad,
        self,
        kernel_size=kernelSize,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceilMode,
        indices=indices,
    )
    if grad_dtype == torch.float16:
        backward_out = backward_out.to(torch.float16)
    elif grad_dtype == torch.bfloat16:
        backward_out = backward_out.to(torch.bfloat16)
    if input_grad_format == "NHWC":
        backward_out = backward_out.permute(0, 2, 3, 1)
    return backward_out
