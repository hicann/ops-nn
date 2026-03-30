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

__input__ = {"kernel": {"rms_norm_grad": "rms_norm_grad_input"}}


def _prod_func(src_list):
    res = 1
    for i in src_list:
        res *= i
    return res


def _to_torch(arr):
    import torch
    if arr.dtype.name == 'bfloat16':
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def rms_norm_grad_input(dy, x, rstd, gamma, **kwargs):
    '''
    Input function for rms_norm_grad.
    All the parameters (names and order) follow rms_norm_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: input_dtypes, full_soc_version, short_soc_version, testcase_name

    Returns:
        Input tensors
    '''
    import torch

    n = _prod_func(rstd.shape)
    d = _prod_func(gamma.shape)

    epsilon = 1e-6
    input_x = x.reshape((n, d))

    tensor_x = _to_torch(input_x)
    rstd = torch.rsqrt(tensor_x.float().pow(2).mean(-1, keepdim=True) + epsilon)
    rstd = rstd.float().detach().numpy()

    return [dy, x, rstd, gamma]
