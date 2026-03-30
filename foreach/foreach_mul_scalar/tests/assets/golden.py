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
        "foreach_mul_scalar": "foreach_mul_scalar_golden"
    },
    "aclnn": {
        "aclnnForeachMulScalar": "aclnn_foreach_mul_scalar_golden",
        "aclnnForeachMulScalarV2": "aclnn_foreach_mul_scalar_golden"
    }
}


def foreach_mul_scalar_golden(x, scalar, **kwargs):
    '''
    Golden function for foreach_mul_scalar.
    All the parameters (names and order) follow @foreach_mul_scalar_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of output tensors
    '''
    import torch

    scalar_val = scalar[0] if isinstance(scalar, np.ndarray) else scalar

    dtype = x[0].dtype
    dtype_str = str(dtype)

    input_pts = []
    for inp in x:
        if 'float16' in dtype_str:  # bfloat16 & float16
            inp = inp.astype(np.float32)
        inp_pt = torch.from_numpy(inp)
        input_pts.append(inp_pt)

    input_tuple = tuple(input_pt for input_pt in input_pts)
    result_pts = torch._foreach_mul(input_tuple, scalar_val)

    return tuple(result_pt.numpy().astype(dtype) for result_pt in result_pts)


def aclnn_foreach_mul_scalar_golden(x, scalar, out, **kwargs):
    '''
    Aclnn golden for aclnnForeachMulScalar and aclnnForeachMulScalarV2.
    All the parameters (name & order) follow
        function `aclnnForeachMulScalarGetWorkspaceSize` in @aclnn_foreach_mul_scalar.h
        and `aclnnForeachMulScalarV2GetWorkspaceSize` in @aclnn_foreach_mul_scalar_v2.h
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch,
        the Tensors in the parameters are all torch.Tensor.
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        x: input tensor list
        scalar: scalar value (tensor with shape [1] for V1, scalar for V2)
        out: output tensor list (not used in calculation)
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    import torch

    dtype = x[0].dtype
    dtype_str = str(dtype)

    input_pts = []
    for inp in x:
        if 'float16' in dtype_str:  # bfloat16 & float16
            inp_pt = inp.to(torch.float32)
        else:
            inp_pt = inp
        input_pts.append(inp_pt)

    scalar_val = scalar.item() if hasattr(scalar, 'item') else scalar
    if 'float16' in dtype_str:  # bfloat16 & float16
        if isinstance(scalar_val, (int, float)):
            pass
        elif hasattr(scalar_val, 'to'):
            scalar_val = scalar_val.to(torch.float32).item() if hasattr(scalar_val, 'item') else scalar_val

    result_pts = torch._foreach_mul(input_pts, scalar_val)

    return tuple(result_pt.to(dtype) for result_pt in result_pts)
