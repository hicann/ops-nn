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
        "fast_gelu_grad": "fast_gelu_grad_golden"
    }
}


def fast_gelu_grad_golden(dy, x, **kwargs):
    '''
    Golden function for fast_gelu_grad.
    All the parameters (names and order) follow @fast_gelu_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    x_dtype = x.dtype
    if x_dtype.name in ["bfloat16", "float16"]:
        x = x.astype(np.float32)
        dy = dy.astype(np.float32)
    
    value1MulsX = np.multiply(-1.702, x)
    temp1Reg = np.exp(value1MulsX)
    temp1Reg = np.add(temp1Reg, 1.0)
    div_down_rec = np.reciprocal(temp1Reg)
    temp2Reg = np.add(div_down_rec, -1.0)
    temp2Reg = np.multiply(temp2Reg, value1MulsX)
    temp2Reg = np.add(temp2Reg, 1.0)
    div_down_rec = np.multiply(div_down_rec, temp2Reg)
    div_down_rec = np.multiply(div_down_rec, dy)
    result = div_down_rec
    
    if x_dtype.name in ["bfloat16", "float16"]:
        return result.astype(x_dtype, copy=False)
    else:
        return result
