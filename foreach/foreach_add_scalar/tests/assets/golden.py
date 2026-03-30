#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import numpy as np
import torch


__golden__ = {
    "kernel": {
        "foreach_add_scalar": "foreach_add_scalar_golden"
    },
    "aclnn": {
        "aclnnForeachAddScalar": "aclnn_foreach_add_scalar_golden",
        "aclnnForeachAddScalarV2": "aclnn_foreach_add_scalar_golden"
    }
}


def _foreach_add_core(input_list, scalar_val, dtype):
    '''
    Core function for foreach_add operation.
    
    Args:
        input_list: List of tensors (torch.Tensor)
        scalar_val: Scalar value to add
        dtype: Target dtype for output
    
    Returns:
        Tuple of output tensors
    '''
    import torch
    
    result_pts = torch._foreach_add(input_list, scalar_val)
    if getattr(dtype, '__module__', None) == 'torch':
        return tuple(result_pt.to(dtype) for result_pt in result_pts)
    else: # numpy
        return tuple(result_pt.numpy().astype(dtype) for result_pt in result_pts)


def _convert_tensor_list_to_float32_if_needed(tensor_list, dtype_str):
    '''
    Convert tensor list to float32 if dtype is float16 or bfloat16.
    
    Args:
        tensor_list: List of tensors (torch.Tensor)
        dtype_str: String representation of dtype
    
    Returns:
        Converted tensor list
    '''
    if 'bfloat16' in dtype_str or 'float16' in dtype_str:
        return [inp.to(torch.float32) for inp in tensor_list]
    return tensor_list


def foreach_add_scalar_golden(x, scalar, **kwargs):
    '''
    Golden function for foreach_add_scalar.
    All the parameters (names and order) follow @foreach_add_scalar_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of output tensors
    '''
    import torch

    dtype = x[0].dtype
    dtype_str = str(dtype)
    scalar_val = scalar[0] if isinstance(scalar, np.ndarray) else scalar

    input_pts = []
    for inp in x:
        if 'float16' in dtype_str:
            inp = inp.astype(np.float32)
        inp_pt = torch.from_numpy(inp)
        input_pts.append(inp_pt)

    return _foreach_add_core(input_pts, scalar_val, dtype)


def aclnn_foreach_add_scalar_golden(x, scalar, out, **kwargs):
    '''
    Aclnn golden for aclnnForeachAddScalar & aclnnForeachAddScalarV2.
    All the parameters (name & order) follow \
        function `aclnnForeachAddScalarGetWorkspaceSize` in @aclnn_foreach_add_scalar.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    import torch

    dtype = x[0].dtype
    dtype_str = str(dtype)

    input_pts = _convert_tensor_list_to_float32_if_needed(x, dtype_str)
    
    if 'float16' in dtype_str:
        scalar = scalar.to(torch.float32)

    return _foreach_add_core(input_pts, scalar.item(), dtype)
