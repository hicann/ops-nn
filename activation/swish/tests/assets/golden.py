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

__golden__ = {"kernel": {"swish": "swish_golden"}}


def swish_golden(x, *, scale=1.0, **kwargs):
    '''
    Golden function for swish.
    All the parameters (names and order) follow @swish_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch
    
    dtype = x.dtype
    if dtype.name in ('float16', 'bfloat16'):
        x = x.astype(np.float32)
    
    if scale == 1.0:
        x_torch = torch.from_numpy(x)
        m = torch.nn.SiLU()
        res = m(x_torch)
        return res.numpy().astype(dtype, copy=False)
    else:
        return _swish_overflow(x, scale, dtype, **kwargs)


def _swish_overflow(data_input, scale, dtype, **kwargs):
    short_soc_version = kwargs.get('short_soc_version', '')
    
    if dtype.name in ('float16', 'bfloat16'):
        data_input = data_input.astype(np.float32)
    
    if short_soc_version in ("Ascend950",):
        scale_arr = np.array([scale], dtype=data_input.dtype)
        multi = data_input * scale_arr * -1.0
        exp = np.exp(multi)
        add = exp + 1.0
        res = np.divide(data_input, add)
    else:
        scale_arr = np.array([scale], dtype=data_input.dtype)
        scale_input = np.multiply(data_input, scale_arr)
        abs_scale_input = np.abs(scale_input)
        minus_abs = np.multiply(abs_scale_input, np.array([-1.0], dtype=data_input.dtype))
        sign_diff = np.add(scale_input, minus_abs)
        half_sign_diff = np.multiply(sign_diff, np.array([0.5], dtype=data_input.dtype))
        
        exp_top = np.exp(half_sign_diff)
        exp_bottom = np.exp(minus_abs)
        one_plus_exp = np.add(exp_bottom, np.array([1.0], dtype=data_input.dtype))
        
        input_mul_exp = np.multiply(data_input, exp_top)
        res = np.divide(input_mul_exp, one_plus_exp)
    
    return res.astype(dtype, copy=False)
