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


__golden__ = {
    "kernel": {
        "add_rms_norm": "add_rms_norm_golden"
    }
}


def add_rms_norm_golden(x1, x2, gamma, # inputs
                        epsilon: float=1e-6, # attributes
                        **kwargs):
    '''
    Golden function for add_rms_norm.
    All the parameters (names and order) follow @add_rms_norm_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    x1_dtype = x1.dtype
    if x1_dtype.name == 'bfloat16':
        x1_tensor = torch.from_numpy(x1.view(np.float16)).view(torch.bfloat16)
        x2_tensor = torch.from_numpy(x2.view(np.float16)).view(torch.bfloat16)
        gamma_tensor = torch.from_numpy(gamma.view(np.float16)).view(torch.bfloat16)
    else:
        x1_tensor = torch.from_numpy(x1)
        x2_tensor = torch.from_numpy(x2)
        gamma_tensor = torch.from_numpy(gamma)
    
    post_action = kwargs.get('_post_action')
    short_soc_version = kwargs.get('short_soc_version')
    if short_soc_version in ["Ascend910B", "Ascend910_93"]:
        y_tensor, var_tensor, x_tensor, y_fp32_tensor = \
            add_rms_golden_v1(x1_tensor, x2_tensor, gamma_tensor, epsilon, post_action)
    else:
        y_tensor, var_tensor, x_tensor, y_fp32_tensor = \
            add_rms_golden_v2(x1_tensor, x2_tensor, gamma_tensor, epsilon, post_action)

    if x1_dtype.name == 'bfloat16':
        y = y_tensor.view(torch.float16).numpy().view(x1_dtype)
        x = x_tensor.view(torch.float16).numpy().view(x1_dtype)
    else:
        y = y_tensor.numpy()
        x = x_tensor.numpy()
    rstd = var_tensor.numpy()
    if post_action == 'cast':  # add_rms_norm_cast
        y_fp32 = y_fp32_tensor.numpy()
        return y_fp32, y, rstd, x
    else:
        return y, rstd, x


def add_rms_golden_v1(x1, x2, gamma, eps, post_action):
    import torch

    # 不同分支走的cast方案不同
    if x1.dtype == torch.bfloat16:
        x = (x1.type(torch.float32) + x2.type(torch.float32)).type(x1.dtype)
    else:
        x = x1 + x2

    xFp32 = x.type(torch.float32)
    if 0 in xFp32.shape[:len(xFp32.shape) - len(gamma.shape)]:  # A 轴存在 0
        xFp32_1 = xFp32.reshape((*xFp32.shape[:len(xFp32.shape) - len(gamma.shape)], 0))
    else:
        xFp32_1 = xFp32.reshape((*xFp32.shape[:len(xFp32.shape) - len(gamma.shape)], -1))
    rstd = torch.rsqrt(xFp32_1.pow(2).mean(-1, keepdim=True)\
        .reshape((*xFp32.shape[:len(xFp32.shape) - len(gamma.shape)], *([1] * len(gamma.shape)))) + eps)
    tmpX = xFp32 * rstd

    if x1.dtype == torch.bfloat16:
        tmpX = tmpX.type(torch.bfloat16).type(torch.float32)
        y = (tmpX * gamma.type(torch.float32)).type(x1.dtype)
    elif x.dtype == torch.float16:
        tmpX = tmpX.type(torch.float16)
        y = tmpX * gamma
    else:
        y = tmpX * gamma
    y_fp32 = y.type(torch.float32) if post_action == 'cast' else None
    return y, rstd, x, y_fp32


def add_rms_golden_v2(x1, x2, gamma, eps, post_action):
    import torch

    x = x1.type(torch.float32) + x2.type(torch.float32)
    if 0 in x.shape[:len(x.shape) - len(gamma.shape)]:  # A 轴存在 0
        x_1 = x.reshape((*x.shape[:len(x.shape) - len(gamma.shape)], 0))
    else:
        x_1 = x.reshape((*x.shape[:len(x.shape) - len(gamma.shape)], -1))
    rstd = torch.rsqrt(x_1.pow(2).mean(-1, keepdim=True)\
        .reshape((*x.shape[:len(x.shape) - len(gamma.shape)], *([1] * len(gamma.shape)))) + eps)
    tmpX = x * rstd

    y_fp32 = (tmpX * gamma.type(torch.float32))
    y = y_fp32.type(x1.dtype)
    x = x.type(x1.dtype)
    return y, rstd, x, y_fp32
