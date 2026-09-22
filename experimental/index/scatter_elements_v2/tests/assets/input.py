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

__input__ = {"kernel": {"scatter_elements_v2": "scatter_elements_v2_input"}}


def scatter_elements_v2_input(
    var, indices, updates, *, axis=0, reduction="none", include_self=True, **kwargs
):
    """
    Input function for scatter_elements_v2.
    All the parameters (names and order) follow @scatter_elements_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    shape, dtype = indices.shape, indices.dtype
    if axis < 0:
        axis = axis + len(shape)
    value_max = var.shape[axis]
    if reduction in ["none", "mul", "add"]:
        batch_shape = shape[:axis] + shape[axis + 1 :]
        tmp_shape = batch_shape + (shape[axis],)
        batch_num = 1
        for num in batch_shape:
            batch_num *= num
        shape_order = list(range(len(shape) - 1))
        shape_order.insert(axis, len(shape) - 1)
        x = np.random.choice(range(0, value_max), size=shape[axis], replace=False)
        x = np.expand_dims(x, axis=0)

        indices = np.repeat(x, batch_num, axis=0)
        np.apply_along_axis(np.random.shuffle, axis=1, arr=indices)
        indices = indices.reshape(tmp_shape).transpose(tuple(shape_order)).astype(dtype)
    else:
        indices = np.random.uniform(0, value_max, shape).astype(dtype)
    return [var, indices, updates]
