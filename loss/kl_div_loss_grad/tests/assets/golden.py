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

__golden__ = {"kernel": {"kl_div_loss_grad": "kl_div_loss_grad_golden"}}


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


def kl_div_loss_grad_golden(grad, input_, target, *, reduction='mean', log_target=False, **kwargs):
    '''
    Golden function for kl_div_loss_grad.
    All the parameters (names and order) follow @kl_div_loss_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    grad_shape = grad.shape
    input_shape = input_.shape
    target_shape = target.shape
    dtype = input_.dtype

    shape_list = _broadcast_to_maxshape([grad_shape, input_shape, target_shape])

    grad = np.broadcast_to(grad, shape_list[-1])
    target = np.broadcast_to(target, shape_list[-1])

    if dtype.name == "float16" or dtype.name == "bfloat16":
        grad = grad.astype("float32")
        input_ = input_.astype("float32")
        target = target.astype("float32")

    if log_target:
        target = np.exp(target)
    res = grad * target
    res = -1 * res
    Element_total = input_.size
    batch_size = input_.shape[0]

    if reduction == "batchmean":
        res = res / batch_size
    elif reduction == "mean":
        res = res / Element_total
    return res.astype(dtype, copy=False)
