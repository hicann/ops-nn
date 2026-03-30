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

import functools
import numpy as np

__golden__ = {"kernel": {"mse_loss": "mse_loss_golden", "mse_loss_grad": "mse_loss_grad_golden"}}


def mse_loss_golden(input0, input1, *, reduction="mean", **kwargs):
    '''
    Golden function for mse_loss.
    All the parameters (names and order) follow @mse_loss_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    ori_dtype = input0.dtype
    reshape_input0 = [functools.reduce(lambda x, y: x * y, input0.shape)]
    reshape_input1 = [functools.reduce(lambda x, y: x * y, input1.shape)]
    predict = np.reshape(input0, reshape_input0)
    label = np.reshape(input1, reshape_input1)
    
    if "float16" in ori_dtype.name:
        predict = predict.astype("float32")
        label = label.astype("float32")
    
    reduce_elts = 1.0
    for i in reshape_input0:
        reduce_elts *= i
    cof = reduce_elts ** (-1)
    
    axis_d = tuple(range(len(reshape_input0)))
    res = np.subtract(predict, label)
    res_sqr = np.multiply(res, res)
    
    y = 0.0
    if reduction == 'mean':
        y = np.sum(res_sqr, axis=axis_d, keepdims=False)
        y = np.multiply(y, cof)
    elif reduction == 'sum':
        y = np.sum(res_sqr, axis=axis_d, keepdims=False)
    elif reduction == 'none':
        y = res_sqr
    
    if "float16" in ori_dtype.name:
        y = y.astype(ori_dtype, copy=False)
    
    return y


def mse_loss_grad_golden(predict, label, dout, *, reduction="mean", **kwargs):
    ori_dtype = predict.dtype
    p_shape = predict.shape
    l_shape = label.shape
    d_shape = dout.shape
    
    shape_list = [p_shape, l_shape, d_shape]
    max_shape = shape_list[0]
    for s in shape_list[1:]:
        max_shape = np.broadcast(max_shape, s).shape if hasattr(np, 'broadcast') else \
                    [max(s1, s2) for s1, s2 in zip(max_shape, s)]
    
    predict = np.broadcast_to(predict, max_shape)
    label = np.broadcast_to(label, max_shape)
    dout = np.broadcast_to(dout, max_shape)
    
    if ori_dtype.name in ("float16", "bfloat16"):
        predict = predict.astype("float32")
        label = label.astype("float32")
        dout = dout.astype("float32")
    
    sub_res = np.subtract(predict, label)
    if 'mean' == reduction:
        reduce_elts = 1.0
        for i in p_shape:
            reduce_elts *= i
        cof = (reduce_elts**(-1)) * 2.0
    else:
        cof = 2.0
    
    norm_grad = sub_res * cof
    grad_res = norm_grad * dout
    return grad_res.astype(ori_dtype, copy=False)
