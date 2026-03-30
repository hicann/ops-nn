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


__golden__ = {
    "kernel": {
        "foreach_addcmul_scalar": "foreach_addcmul_scalar_golden"
    },
    "aclnn": {
        "aclnnForeachAddcmulScalar": "aclnn_foreach_addcmul_scalar_golden",
        "aclnnForeachAddcmulScalarV2": "aclnn_foreach_addcmul_scalar_golden"
    }
}


def _convert_to_float32_if_needed(tensor, dtype_str):
    if 'float16' in dtype_str:
        if isinstance(tensor, np.ndarray):
            return tensor.astype(np.float32)
        else:
            import torch
            return tensor.to(torch.float32)
    return tensor


def _convert_tensor_list_to_float32_if_needed(tensor_list, dtype_str):
    import torch
    result = []
    for tensor in tensor_list:
        tensor = _convert_to_float32_if_needed(tensor, dtype_str)
        if isinstance(tensor, np.ndarray):
            result.append(torch.from_numpy(tensor))
        else:
            result.append(tensor)
    return result


def _foreach_addcmul_core(result_pts, dtype):
    import torch
    if isinstance(dtype, torch.dtype):
        return tuple(result_pt.to(dtype) for result_pt in result_pts)
    else:
        return tuple(result_pt.numpy().astype(dtype) for result_pt in result_pts)


def _process_scalar(scalar):
    import torch

    scalar_dtype = scalar.dtype
    if getattr(scalar_dtype, '__module__', None) != 'torch':  # 'numpy'
        if isinstance(scalar, np.ndarray):
            if scalar_dtype.name == 'bfloat16':
                scalar = torch.from_numpy(scalar.view(np.int16)).view(torch.bfloat16)
            else:
                scalar = torch.from_numpy(scalar)
        else:
            scalar = torch.tensor(scalar)
    
    scalar = _convert_to_float32_if_needed(scalar, str(scalar_dtype))
    return scalar.item()


def _foreach_addcmul_impl(x1, x2, x3, scalar, dtype_str):
    import torch
    
    input_pts = _convert_tensor_list_to_float32_if_needed(x1, dtype_str)
    input_pts1 = _convert_tensor_list_to_float32_if_needed(x2, dtype_str)
    input_pts2 = _convert_tensor_list_to_float32_if_needed(x3, dtype_str)
    
    scalar_val = _process_scalar(scalar)
    
    result_pts = torch._foreach_addcmul(input_pts, input_pts1, input_pts2, scalar_val)
    
    dtype = x1[0].dtype
    return _foreach_addcmul_core(result_pts, dtype)


def foreach_addcmul_scalar_golden(x1, x2, x3, scalar, **kwargs):
    '''
    Golden function for foreach_addcmul_scalar.
    All the parameters (names and order) follow @foreach_addcmul_scalar_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x1, x2, x3: Input tensor lists (numpy.ndarray)
        scalar: Scalar value (numpy.ndarray)
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of output tensors (numpy.ndarray)
    '''
    dtype_str = str(x1[0].dtype)
    scalar_val = scalar[0] if isinstance(scalar, np.ndarray) else scalar
    
    if 'float16' in dtype_str:
        scalar_val = scalar_val.astype(np.float32)
    
    return _foreach_addcmul_impl(x1, x2, x3, scalar_val, dtype_str)


def aclnn_foreach_addcmul_scalar_golden(x1, x2, x3, scalar, out, **kwargs):
    '''
    Aclnn golden for aclnnForeachAddcmulScalar and aclnnForeachAddcmulScalarV2.
    All the parameters (name & order) follow \
        function `aclnnForeachAddcmulScalarV2GetWorkspaceSize` in @aclnn_foreach_addcmul_scalar_v2.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        x1: First input tensor list
        x2: Second input tensor list
        x3: Third input tensor list
        scalar: Scalar value (for V1: torch.Tensor with shape [1]; for V2: scalar attribute)
        out: Output tensor list (placeholder, not used)
        **kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    dtype_str = str(x1[0].dtype)
    
    return _foreach_addcmul_impl(x1, x2, x3, scalar, dtype_str)
