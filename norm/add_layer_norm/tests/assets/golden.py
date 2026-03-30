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
        "add_layer_norm": "add_layer_norm_golden"
    }
}


def add_layer_norm_golden(x1, x2, gamma, beta, bias = None, # inputs
                          epsilon: float=1e-5, additional_output: bool=False, # attributes
                          **kwargs):
    '''
    Golden function for add_layer_norm.
    All the parameters (names and order) follow @add_layer_norm_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    x1_dtype = x1.dtype
    x2_dtype = x2.dtype
    x1_x2_any_fp32 = x1.dtype.name == 'float32' or x2.dtype.name == 'float32'
    calc_dtype = np.float64 if x1_dtype.name == np.float32 else np.float32
    calc_dtype_torch = torch.float64 if x1_dtype.name == np.float32 else torch.float32

    x1 = x1.astype(calc_dtype)
    x2 = x2.astype(calc_dtype)
    gamma = gamma.astype(calc_dtype)
    beta = beta.astype(calc_dtype)

    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    gamma = torch.from_numpy(gamma)
    beta = torch.from_numpy(beta)

    epsilon = np.float64(epsilon) if x1_dtype.name == np.float32 else np.float32(epsilon)

    if bias is not None:
        bias = bias.astype(calc_dtype)
        bias = torch.from_numpy(bias)
        x = x1 + x2 + bias
    else:
        x = x1 + x2

    row_size = x1.shape[-1]
    x = x.reshape(-1, row_size).to(calc_dtype_torch)

    y, x_mean, x_rstd = torch.ops.aten.native_layer_norm(x, [row_size], weight=gamma, bias=beta, eps=epsilon)

    y_out = y.numpy().astype('float32' if x1_x2_any_fp32 else x1_dtype)
    mean_out = x_mean.numpy().astype("float32")
    rstd_out = x_rstd.numpy().astype("float32")
    x_out = x.numpy().astype('float32' if x1_x2_any_fp32 else x1_dtype)

    return y_out, mean_out, rstd_out, x_out
