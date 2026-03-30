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

__golden__ = {"kernel": {"rms_norm_grad": "rms_norm_grad_golden"}}


def _prod_func(src_list):
    '''
    Golden function for rms_norm_grad.
    All the parameters (names and order) follow @rms_norm_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    res = 1
    for i in src_list:
        res *= i
    return res


def rms_norm_grad_golden(dy, x, rstd, gamma, **kwargs):
    import torch
    
    def _to_torch(arr):
        if arr.dtype.name == 'bfloat16':
            return torch.from_numpy(arr.view(np.float16)).view(torch.bfloat16)
        return torch.from_numpy(arr)
    
    ori_dtype = x.dtype
    _np_to_torch = {np.float16: torch.float16, np.float32: torch.float32,
                    np.int8: torch.int8, np.int16: torch.int16, np.int32: torch.int32}
    
    def _to_torch(arr):
        if arr.dtype.name == 'bfloat16':
            return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
        return torch.from_numpy(arr)
    
    def _torch_to_np(t):
        if t.dtype == torch.bfloat16:
            return t.view(torch.int16).numpy().view(np.dtype('bfloat16'))
        return t.numpy()
    
    dy_torch = _to_torch(dy).to(torch.float32)
    x_torch = _to_torch(x).to(torch.float32)
    rstd_torch = torch.tensor(rstd)
    gamma_torch = _to_torch(gamma).to(torch.float32)
    
    n = _prod_func(rstd.shape)
    d = _prod_func(gamma.shape)
    
    if n == 0 and d != 0:
        dx = torch.empty(x_torch.shape)
        dgamma = torch.full((d,), torch.nan)
        if ori_dtype.name == 'bfloat16':
            dx = _torch_to_np(dx)
        else:
            dx = dx.to(ori_dtype).numpy()
        dgamma = dgamma.to(torch.float32)
        return dx, dgamma.numpy()
    
    x_reshaped = x_torch.reshape((n, d))
    gamma_reshaped = gamma_torch.reshape(1, d)
    grad_y_reshaped = dy_torch.reshape((n, d))
    rstd_reshaped = rstd_torch.reshape(n, 1)
    
    dgamma = (grad_y_reshaped * (x_reshaped * rstd_reshaped)).sum(0, keepdims=False)
    dx = (grad_y_reshaped * gamma_reshaped - x_reshaped * rstd_reshaped * (grad_y_reshaped * gamma_reshaped * (x_reshaped * rstd_reshaped)).mean(-1, keepdims=True)) * rstd_reshaped
    
    if ori_dtype.name == 'bfloat16':
        dx = _torch_to_np(dx)
    else:
        dx = dx.to(ori_dtype).numpy()
    dgamma = dgamma.to(torch.float32)
    
    return dx, dgamma.numpy()
