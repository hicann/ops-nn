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
    "kernel": {"rms_norm": "rms_norm_golden"},
    "aclnn": {"aclnnRmsNorm": "aclnn_rms_norm_golden"},
}


def rms_norm_golden(x, gamma, *, epsilon=0.000001, **kwargs):
    """
    Golden function for rms_norm.
    All the parameters (names and order) follow @rms_norm_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    input_dtype = x.dtype
    x_fp32 = x.astype(np.float32)
    gamma_fp32 = gamma.astype(np.float32)

    len_shape_x = len(x.shape)
    len_shape_gamma = len(gamma.shape)

    if len_shape_gamma == 1:
        variance = np.mean(np.power(x_fp32, 2), axis=-1, keepdims=True)
    else:
        original_list = [0, 1, 2, 3, 4, 5, 6, 7]
        axis = [
            original_list[i] for i in range(len_shape_x - len_shape_gamma, len_shape_x)
        ]
        variance = np.mean(np.power(x_fp32, 2), axis=tuple(axis), keepdims=True)

    std = np.sqrt(variance + epsilon)
    rstd = 1 / std
    result_mid = x_fp32 * rstd

    golden = result_mid * gamma_fp32
    return golden.astype(input_dtype, copy=False), rstd


def aclnn_rms_norm_golden(x, gamma, epsilon, y_out, rstd_out, **kwargs):
    """
    Aclnn golden for aclnnRmsNorm.
    All the parameters (name & order) follow
        function `aclnnRmsNormGetWorkspaceSize` in aclnn_rms_norm.h
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch,
        the Tensors in the parameters are all torch.Tensor.
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    """
    import torch

    input_dtype = x.dtype
    if "float16" in str(input_dtype):
        x = x.to(torch.float32)
        gamma = gamma.to(torch.float32)

    len_shape_x = len(x.shape)
    len_shape_gamma = len(gamma.shape)

    if len_shape_gamma == 1:
        variance = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
    else:
        dims = list(range(len_shape_x - len_shape_gamma, len_shape_x))
        variance = torch.mean(torch.pow(x, 2), dim=dims, keepdim=True)

    std = torch.sqrt(variance + epsilon)
    rstd = 1 / std
    result_mid = x * rstd

    golden = result_mid * gamma
    return golden.to(input_dtype), rstd.to(torch.float32)
