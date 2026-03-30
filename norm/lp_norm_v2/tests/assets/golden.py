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
import math

__golden__ = {"kernel": {"lp_norm_v2": "lp_norm_v2_golden"}}


class Constant:
    CONST_EPSILON_FP16 = 1e-7


def get_unique_axis(input_tensor, axis):
    if len(axis) == 0:
        return list(range(len(input_tensor.shape)))
    unique_list = []
    tensorLen = len(input_tensor.shape)
    tensorLen = 1 if tensorLen == 0 else tensorLen
    for item in axis:
        if item < 0:
            item = item + tensorLen
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


def is_empty_tensor_for_inf(input_tensor, p, axis):
    if not (math.isinf(p)):
        return False
    if input_tensor.numel() != 0:
        return False
    if len(axis) == 0:
        return True
    for i in range(len(input_tensor.shape)):
        if i in axis and input_tensor.shape[i] == 0:
            return True
    return False


def lp_norm_v2_golden(x, *, axes=[], keepdim=False, p=2.0, epsilon=1e-12, **kwargs):
    '''
    Golden function for lp_norm_v2.
    All the parameters (names and order) follow @lp_norm_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    x_type = x.dtype
    axis = axes

    x_tensor = torch.from_numpy(x) if not isinstance(x, torch.Tensor) else x
    unique_axis = get_unique_axis(x_tensor, axis)
    empty_tensor_for_inf = is_empty_tensor_for_inf(x_tensor, p, unique_axis)

    if empty_tensor_for_inf:
        if not keepdim:
            res_shape = [x_tensor.shape[i] for i in range(len(x_tensor.shape)) if i not in unique_axis]
        else:
            res_shape = [x_tensor.shape[i] if i not in unique_axis else 1 for i in range(len(x_tensor.shape))]
        res_shape = tuple(res_shape)
        fill_val = -float("inf") if p > 0 else float("inf")
        res = torch.full(res_shape, torch.tensor(fill_val, dtype=x_tensor.dtype))
    else:
        res = torch.linalg.vector_norm(x_tensor, ord=p, dim=unique_axis, keepdim=keepdim)

    res_np = res.numpy() if isinstance(res, torch.Tensor) else res
    if x_type == "float16" and epsilon < Constant.CONST_EPSILON_FP16:
        if math.isclose(epsilon, 0.0):
            std_no = 0
        else:
            std_no = Constant.CONST_EPSILON_FP16
    else:
        std_no = float(epsilon)
    res = np.maximum(res_np, std_no)

    return res.astype(x.dtype, copy=False)
