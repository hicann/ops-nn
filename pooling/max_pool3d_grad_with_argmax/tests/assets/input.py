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
from functools import reduce
import operator

__input__ = {
    "aclnn": {"aclnnMaxPool2dWithIndicesBackward": "aclnn_max_pool_backward_input"},
}


def aclnn_max_pool_backward_input(*args, **kwargs):
    """
    Input function for aclnnMaxPool2dWithIndicesBackward.
    Generate valid self, gradOutput and indices via forward maxpool.
    Tensors: gradOutput(0), self(1), indices(2), gradInput(3)

    Following opstest approach: generate random self, run forward maxpool,
    use forward output as gradOutput, forward indices as indices.
    """
    import torch
    import torch.nn.functional as F

    grad_output = args[0]
    self_input = args[1]
    indices = args[2]

    attrs = kwargs.get("attributes", {})
    kernel_size = attrs.get("kernelSize", [2, 2])
    stride = attrs.get("stride", [2, 2])
    padding = attrs.get("padding", [1, 1])
    dilation = attrs.get("dilation", [1, 1])
    ceil_mode = attrs.get("ceilMode", False)

    input_grad_format = kwargs.get("tensor_formats", ["NCHW"])[0]

    # Generate random self data (like opstest: random int8 cast to float)
    self_shape = self_input.shape
    ele_num = reduce(operator.mul, self_shape)
    random_array = np.random.randint(
        low=np.iinfo(np.int8).min,
        high=np.iinfo(np.int8).max + 1,
        size=(ele_num,),
        dtype=np.int8,
    )
    input_x = random_array.reshape(self_shape)

    if self_input.dtype == torch.float16 or self_input.dtype == torch.bfloat16:
        input_x = input_x.astype(np.float32)
        x_torch = torch.from_numpy(input_x)
    else:
        input_x = input_x.astype(np.float32)
        x_torch = torch.from_numpy(input_x)

    # NHWC -> NCHW for forward maxpool
    if input_grad_format == "NHWC":
        x_torch = x_torch.permute(0, 3, 1, 2)

    max_out, max_indices = F.max_pool2d_with_indices(
        x_torch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=bool(ceil_mode),
    )

    # Convert back to original dtype
    orig_dtype = self_input.dtype
    if orig_dtype == torch.float16 or orig_dtype == torch.bfloat16:
        max_out = max_out.to(orig_dtype)
        x_torch = x_torch.to(orig_dtype)

    # Convert indices to expected dtype
    indices_dtype = indices.dtype
    max_indices = max_indices.to(indices_dtype)

    # NHWC -> back
    if input_grad_format == "NHWC":
        x_torch = x_torch.permute(0, 2, 3, 1)
        max_out = max_out.permute(0, 2, 3, 1)
        max_indices = max_indices.permute(0, 2, 3, 1)

    # Copy results into existing tensors in-place
    self_input.copy_(x_torch)
    grad_output.copy_(max_out)
    indices.copy_(max_indices)
