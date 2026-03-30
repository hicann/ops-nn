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
import functools


__golden__ = {
    "kernel": {
        "fast_gelu": "fast_gelu_golden"
    }
}


def fast_gelu_golden(x, **kwargs):
    '''
    Golden function for fast_gelu.
    All the parameters (names and order) follow @fast_gelu_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    input_dtype = x.dtype
    if input_dtype.name in ["bfloat16", "float16"]:
        x = x.astype(np.float32)
    
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda a, b: a * b, x.shape)
    x = np.reshape(x, fuseshape)
    
    const_1_value = 1
    attr = 1.702
    calc_dtype = x.dtype
    attr_opp = 0 - attr
    attr_half = attr / 2

    const_0 = np.array(attr_opp, calc_dtype)
    const_1 = np.array(const_1_value, calc_dtype)
    denominator = np.multiply(x, const_0)
    denominator = np.exp(denominator)
    denominator = np.add(denominator, const_1)

    denominator = np.reciprocal(denominator)
    result = np.multiply(x, denominator)
    return result.astype(input_dtype, copy=False)
