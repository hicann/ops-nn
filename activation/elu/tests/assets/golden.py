#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        "elu": "elu_golden"
    },
    "aclnn": {
        "aclnnElu": "aclnn_elu_golden",
        "aclnnInplaceElu": "aclnn_inplace_elu_golden"
    }
}


def elu_golden(x, *, alpha: float = 1.0, scale: float = 1.0, input_scale: float = 1.0, **kwargs):
    '''
    Golden function for elu.
    All the parameters (names and order) follow @elu_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    x_dtype = x.dtype
    if x_dtype.name in ("bfloat16", "float16"):
        x = x.astype(np.float32)
    
    x_torch = torch.from_numpy(x)
    output_y = torch.ops.aten.elu(x_torch, alpha=alpha, scale=scale, input_scale=input_scale)
    result = output_y.numpy()
    
    if x_dtype.name in ("bfloat16", "float16"):
        result = result.astype(x_dtype, copy=False)
    
    return result


def _aclnn_elu_impl(input_tensor, alpha, scale, inputScale):
    import torch
    alpha_val = alpha.item()
    scale_val = scale.item()
    input_scale_val = inputScale.item()
    return torch.ops.aten.elu(input_tensor, alpha=alpha_val, scale=scale_val, input_scale=input_scale_val)


def aclnn_elu_golden(selfT, alpha, scale, inputScale, out, **kwargs):
    '''
    Aclnn golden for aclnnElu.
    All the parameters (name & order) follow \
        function `aclnnEluGetWorkspaceSize` in @aclnn_elu.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    return _aclnn_elu_impl(selfT, alpha, scale, inputScale)


def aclnn_inplace_elu_golden(selfRef, alpha, scale, inputScale, **kwargs):
    '''
    Aclnn golden for aclnnInplaceElu.
    All the parameters (name & order) follow \
        function `aclnnInplaceEluGetWorkspaceSize` in @aclnn_elu.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    return _aclnn_elu_impl(selfRef, alpha, scale, inputScale)
