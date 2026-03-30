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

import math
import numpy as np

__golden__ = {"kernel": {"binary_cross_entropy_grad": "binary_cross_entropy_grad_golden"}}


def _broadcast_to_maxshape(shapes: list):
    def _max(_shape):
        no_one_shape = [s for s in _shape if s != 1]
        if len(no_one_shape) == 0:
            max_value = 1
        else:
            max_value = no_one_shape[0]
        return max_value
    max_dim_length = max(len(list(shape)) for shape in shapes)
    input_shapes = []
    for shape in shapes:
        input_shapes.append([1 for _ in range(max_dim_length - len(shape))] + list(shape))
    input_shapes = list(map(list, zip(*input_shapes)))
    max_shape = [_max(shape) for shape in input_shapes]
    input_shapes = list(map(list, zip(*input_shapes)))
    return (*input_shapes, max_shape)


class Constant:
    SCALAR_ONE = 1
    SCALAR_NEGATIVE_ONE = -1
    SCALAR_EPS = 1e-12


def binary_cross_entropy_grad_golden(x, y, grad_output, weight=None, *, reduction='mean', **kwargs):
    '''
    Golden function for binary_cross_entropy_grad.
    All the parameters (names and order) follow @binary_cross_entropy_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    x_shape = x.shape
    y_shape = y.shape
    grad_shape = grad_output.shape
    shape_list = _broadcast_to_maxshape([x_shape, y_shape, grad_shape])
    x = np.broadcast_to(x, shape_list[-1])
    y = np.broadcast_to(y, shape_list[-1])
    grad_output = np.broadcast_to(grad_output, shape_list[-1])
    if weight is not None:
        weight = np.broadcast_to(weight, shape_list[-1])
    ori_dtype = x.dtype
    if ori_dtype.name == "float16" or ori_dtype.name == "bfloat16":
        x = x.astype("float32")
        y = y.astype("float32")
        grad_output = grad_output.astype("float32")
        if weight is not None:
            weight = weight.astype("float32")

    val1 = np.subtract(x, y)
    minus_predict = np.multiply(x, Constant.SCALAR_NEGATIVE_ONE)
    val2_tmp = np.add(minus_predict, Constant.SCALAR_ONE)
    val2 = np.multiply(x, val2_tmp)
    val2 = np.maximum(val2, Constant.SCALAR_EPS)
    grad_val = np.multiply(grad_output, val1)
    result = np.divide(grad_val, val2)
    if weight is not None:
        result = np.multiply(weight, result)

    if reduction == "mean":
        reduce_elts = 1.0
        for i in shape_list[-1]:
            reduce_elts *= i
        cof = reduce_elts if math.isclose(reduce_elts, 0.0) else reduce_elts ** (-1)
        result = np.multiply(result, cof)

    return result.astype(ori_dtype, copy=False)
