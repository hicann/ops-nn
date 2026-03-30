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
        "foreach_sqrt": "foreach_sqrt_golden"
    },
    "aclnn": {
        "aclnnForeachSqrt": "aclnn_foreach_sqrt_golden"
    }
}


def foreach_sqrt_golden(x, **kwargs):
    '''
    Golden function for foreach_sqrt.
    All the parameters (names and order) follow @foreach_sqrt_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of output tensors
    '''
    input_dtype = x[0].dtype
    input_dtype_str = str(input_dtype)
    
    output_pts = []
    for inp in x:
        if 'float16' in input_dtype_str:  # bfloat16 & float16
            inp = inp.astype(np.float32)
        sqrt_res = np.sqrt(inp)
        output_pts.append(sqrt_res.astype(input_dtype))

    return tuple(output_pts)


def aclnn_foreach_sqrt_golden(x, out, **kwargs):
    '''
    Aclnn golden for aclnnForeachSqrt.
    All the parameters (name & order) follow
        function `aclnnForeachSqrtGetWorkspaceSize` in aclnn_foreach_sqrt.h
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
    
    output_pts = []
    for inp in x:
        in_dtype = inp.dtype
        if 'float16' in str(in_dtype):  # bfloat16 & float16
            inp = inp.to(torch.float32)
        sqrt_res = torch.sqrt(inp).to(in_dtype)
        output_pts.append(sqrt_res)
    return tuple(output_pts)
