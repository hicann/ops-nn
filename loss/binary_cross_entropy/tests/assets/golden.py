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

__golden__ = {"kernel": {"binary_cross_entropy": "binary_cross_entropy_golden"}}


class Constant:
    CONST_MIN = -100
    CONST_ONE = 1
    CONST_NEG_ONE = -1


def binary_cross_entropy_golden(input_data_x1, input_data_x2, input_data_weight=None, *, reduction='mean', **kwargs):
    '''
    Golden function for binary_cross_entropy.
    All the parameters (names and order) follow @binary_cross_entropy_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    ori_dtype = input_data_x1.dtype
    shape = input_data_x1.shape
    if ori_dtype == "float16" or ori_dtype == "bfloat16":
        input_data_x1 = input_data_x1.astype("float32")
        input_data_x2 = input_data_x2.astype("float32")
        if input_data_weight is not None:
            input_data_weight = input_data_weight.astype("float32")

    x_log_tmp = np.log(input_data_x1)
    x_log_tmp = np.maximum(x_log_tmp, Constant.CONST_MIN)
    data_mul1 = np.multiply(x_log_tmp, input_data_x2)
    x_neg_tmp = np.multiply(input_data_x1, Constant.CONST_NEG_ONE)
    x1_tmp = np.add(x_neg_tmp, Constant.CONST_ONE)
    y_neg_tmp = np.multiply(input_data_x2, Constant.CONST_NEG_ONE)
    y1_tmp = np.add(y_neg_tmp, Constant.CONST_ONE)
    x1_log_tmp = np.log(x1_tmp)
    x1_log_tmp = np.maximum(x1_log_tmp, Constant.CONST_MIN)
    data_mul2 = np.multiply(x1_log_tmp, y1_tmp)
    data_sum = np.add(data_mul1, data_mul2)
    result = np.multiply(data_sum, Constant.CONST_NEG_ONE)

    if input_data_weight is not None:
        result = np.multiply(result, input_data_weight)

    reduce_elts = 1.0
    for i in shape:
        reduce_elts *= i
    cof = reduce_elts ** (-1)

    axis_d = []
    for i, _ in enumerate(shape):
        axis_d.append(i)
    axis_d = tuple(axis_d)

    if reduction == "mean":
        result = np.multiply(result, cof)
        result = np.sum(result, axis=axis_d, keepdims=False)
    elif reduction == "sum":
        result = np.sum(result, axis=axis_d, keepdims=False)
    elif reduction == "none":
        pass

    return result.astype(ori_dtype)
