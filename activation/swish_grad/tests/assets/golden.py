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

__golden__ = {"kernel": {"swish_grad": "swish_grad_golden"}}


def swish_grad_golden(grad, x, y, *, scale, **kwargs):
    '''
    Golden function for swish_grad.
    All the parameters (names and order) follow @swish_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    dtype = x.dtype
    
    if x.size == 0:
        return np.zeros(grad.shape, dtype=grad.dtype)
    
    if dtype.name in ('float16', 'bfloat16'):
        input_gradients = grad.astype("float32")
        fwd_input = x.astype("float32")
        fwd_output = y.astype("float32")
    else:
        input_gradients = grad
        fwd_input = x
        fwd_output = y
    
    beta_scalar = np.array([scale], dtype=np.float32)
    beta_output = fwd_output
    beta_input = np.multiply(fwd_input, beta_scalar)
    
    sigmoid_res = 1 / (1 + np.exp(-beta_input))
    
    grad_x = sigmoid_res * (1 + beta_input * (1 - sigmoid_res))
    res = np.multiply(input_gradients, grad_x)
    
    if dtype.name in ('float16', 'bfloat16'):
        return res.astype(dtype)
    
    return res
