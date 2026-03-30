#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import math
import numpy as np


__golden__ = {
    "kernel": {
        "foreach_norm": "foreach_norm_golden"
    },
    "aclnn": {
        "aclnnForeachNorm": "aclnn_foreach_norm_golden"
    }
}


def foreach_norm_golden(x, scalar, **kwargs):
    '''
    Golden function for foreach_norm.
    All the parameters (names and order) follow @foreach_norm_def.cpp without outputs.
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

    if math.isinf(scalar_val):
        mode = float('inf')
    elif int(scalar_val) == 1:
        mode = 1
    else:
        mode = 2

    input_pts = []
    for inp in x:
        if 'float16' in dtype_str:  # bfloat16 & float16
            inp = inp.astype(np.float64)
        else:
            inp = inp.astype(np.float64)
        inp_pt = torch.from_numpy(inp)
        input_pts.append(inp_pt)

    input_tuple = tuple(input_pt for input_pt in input_pts)
    result_pts = torch._foreach_norm(input_tuple, mode)

    return tuple(result_pt.numpy().astype(dtype) for result_pt in result_pts)


def aclnn_foreach_norm_golden(x, scalar, out, **kwargs):
    '''
    Aclnn golden for aclnnForeachNorm.
    All the parameters (name & order) follow
        function `aclnnForeachNormGetWorkspaceSize` in aclnn_foreach_norm.h
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch,
        the Tensors in the parameters are all torch.Tensor.
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    import torch

    if math.isinf(scalar):
        mode = float('inf')
    elif int(scalar) == 1:
        mode = 1
    else:
        mode = 2

    dtype = x[0].dtype
    input_pts: list = []
    for inp in x:
        in_dtype = inp.dtype
        if 'float16' in str(in_dtype):  # bfloat16 & float16
            inp = inp.to(torch.float64)
        else:
            inp = inp.to(torch.float64)
        input_pts.append(inp)

    result_pts = torch._foreach_norm(input_pts, mode)

    return tuple(result_pt.to(dtype) for result_pt in result_pts)
