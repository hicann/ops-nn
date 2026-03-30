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

__golden__ = {"kernel": {"log_softmax_grad": "log_softmax_grad_golden"}}


def __eliminate_duplicate_axes(axis, input_tensor):
    if axis is None:
        axis = [-1]
    if isinstance(axis, int):
        axis = [axis]
    axis = tuple(set([_ax if _ax >= 0 else len(input_tensor.shape) + _ax for _ax in axis]))
    return axis


def log_softmax_grad_golden(input_dy, input_x, *, axes=None, axis=None, **kwargs):
    '''
    Golden function for log_softmax_grad.
    All the parameters (names and order) follow @log_softmax_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    dtype = input_dy.dtype
    if axis is None:
        axis = axes
    axis = __eliminate_duplicate_axes(axis, input_dy)

    if 'float16' in str(dtype):
        input_x = input_x.astype("float32")
        input_dy = input_dy.astype("float32")
    exp_x = np.exp(input_x)
    sum_dy = np.sum(input_dy, axis=axis, keepdims=True)
    data_sum_broadcast = np.broadcast_to(sum_dy, input_dy.shape)
    data_mul = np.multiply(exp_x, data_sum_broadcast)
    res = np.subtract(input_dy, data_mul)
    return res.astype(dtype, copy=False)
