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
import torch

__golden__ = {
    "aclnn": {
        "aclnnRepeatInterleaveWithDim": "aclnn_repeat_interleave_with_dim_golden",
        "aclnnRepeatInterleaveTensor": "aclnn_repeat_interleave_tensor_golden",
        "aclnnRepeatInterleaveIntWithDim": "aclnn_repeat_interleave_int_with_dim_golden",
        "aclnnRepeatInterleaveInt": "aclnn_repeat_interleave_int_golden",
        "aclnnRepeatInterleave": "aclnn_repeat_interleave_golden",
    },
    "kernel": {"repeat_interleave": "repeat_interleave_golden"},
}


def repeat_interleave_golden(x, repeats, *, axis=1000, **kwargs):
    """
    Golden function for repeat_interleave.
    All the parameters (names and order) follow @repeat_interleave_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    output_shapes = kwargs.get("output_shapes", [[]])
    output_shape = output_shapes[0] if output_shapes else []

    repeats_val = repeats.item() if isinstance(repeats, np.ndarray) else repeats
    axis_val = axis % len(x.shape)

    if output_shape and repeats_val != output_shape[axis_val]:
        repeats_val = output_shape[axis_val]

    dtypes = {"uint64": "int64", "uint16": "int16", "uint32": "int32"}
    if x.dtype.name in dtypes.keys():
        input_x = x.view(dtypes[x.dtype.name])
    else:
        input_x = x

    input_dtype = input_x.dtype
    if input_dtype.name == "bfloat16":
        x_torch = torch.from_numpy(input_x.view(np.int16)).view(torch.bfloat16)
    else:
        x_torch = torch.from_numpy(input_x)

    res_torch = torch.repeat_interleave(x_torch, repeats_val, axis_val)

    if input_dtype.name == "bfloat16":
        return res_torch.view(torch.int16).numpy().view(x.dtype)
    return res_torch.numpy().view(x.dtype)


def aclnn_repeat_interleave_golden(self, repeats, outputSize=0, out=None, **kwargs):
    """
    Aclnn golden for aclnnRepeatInterleave.
    Parameters follow @aclnnRepeatInterleaveGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    input = self
    repeats = repeats
    return torch.repeat_interleave(input, repeats)


def aclnn_repeat_interleave_int_golden(
    self, repeats=0, outputSize=0, out=None, **kwargs
):
    """
    Aclnn golden for aclnnRepeatInterleaveInt.
    Parameters follow @aclnnRepeatInterleaveIntGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    input = self
    if hasattr(repeats, "item"):
        repeats = repeats.item()
    return torch.repeat_interleave(input, repeats)


def aclnn_repeat_interleave_int_with_dim_golden(
    self, repeats=0, dim=0, outputSize=0, out=None, **kwargs
):
    """
    Aclnn golden for aclnnRepeatInterleaveIntWithDim.
    Parameters follow @aclnnRepeatInterleaveIntWithDimGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    input = self
    if hasattr(repeats, "item"):
        repeats = repeats.item()
    if hasattr(dim, "item"):
        dim = dim.item()
    return torch.repeat_interleave(input, repeats, dim)


def aclnn_repeat_interleave_tensor_golden(repeats, outputSize=0, out=None, **kwargs):
    """
    Aclnn golden for aclnnRepeatInterleaveTensor.
    Parameters follow @aclnnRepeatInterleaveTensorGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    repeats = repeats
    return torch.repeat_interleave(repeats)


def aclnn_repeat_interleave_with_dim_golden(
    self, repeats, dim=0, outputSize=0, out=None, **kwargs
):
    """
    Aclnn golden for aclnnRepeatInterleaveWithDim.
    Parameters follow @aclnnRepeatInterleaveWithDimGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    input = self
    repeats = repeats
    if hasattr(dim, "item"):
        dim = dim.item()
    return torch.repeat_interleave(input, repeats, dim)
