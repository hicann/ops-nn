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

__golden__ = {"kernel": {"swi_glu_grad": "swi_glu_grad_golden"}}


def swi_glu_grad_golden(dy, x, *, dim, **kwargs):
    '''
    Golden function for swi_glu_grad.
    All the parameters (names and order) follow @swi_glu_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch
    
    x_dtype = x.dtype
    
    if str(x_dtype) in ("bfloat16", "float16"):
        x = x.astype(np.float32)
        dy = dy.astype(np.float32)
    
    x_torch = torch.from_numpy(x)
    dy_torch = torch.from_numpy(dy)
    
    inputA, inputB = torch.split(x_torch, x_torch.size(dim) // 2, dim)
    sigmoid = torch.sigmoid(inputA)
    outA = dy_torch * sigmoid * (1 + inputA * (1 - sigmoid)) * inputB
    outB = dy_torch * sigmoid * inputA
    
    result = torch.concat((outA, outB), dim=dim)
    
    if "bfloat16" in str(x_dtype):
        result = result.to(torch.float32).numpy().astype(x_dtype)
    elif "float16" in str(x_dtype):
        result = result.to(torch.float16).numpy()
    else:
        result = result.numpy()
    
    return result
