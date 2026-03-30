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

__golden__ = {"kernel": {"scatter": "scatter_golden"}}


def scatter_golden(var, indices, update_value, *, axis=0, **kwargs):
    '''
    Golden function for scatter.
    All the parameters (names and order) follow @scatter_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import copy
    
    dtype_dict = {"float32": 4, "int8": 1, "int32": 4, "float16": 2, "int64": 8, "bfloat16": 2}
    all_shape = len(var.shape)
    abs_axis = axis
    if axis < 0:
        abs_axis = all_shape + axis
    
    if not (all_shape == 4 and abs_axis == 3 and var.shape[2] % dtype_dict.get(str(var.dtype), 1) == 0
            and var.shape[3] % dtype_dict.get(str(var.dtype), 1) == 0):
        trans_shape_0 = var.shape[0]
        update_shape_0 = update_value.shape[0]
        
        seceond_dim = 1
        update_second_dim = 1
        
        for i in range(1, abs_axis):
            seceond_dim *= var.shape[i]
            update_second_dim *= update_value.shape[i]
        trans_shape_1 = seceond_dim
        update_shape_1 = update_second_dim
        
        trans_shape_2 = var.shape[abs_axis]
        update_shape_2 = update_value.shape[abs_axis]
        
        fourth_dim = 1
        update_fourth_dim = 1
        for i in range(abs_axis + 1, all_shape):
            fourth_dim *= var.shape[i]
            update_fourth_dim *= update_value.shape[i]
        trans_shape_3 = fourth_dim
        update_shape_3 = update_fourth_dim
        
        var = var.reshape(trans_shape_0, trans_shape_1, trans_shape_2, trans_shape_3)
        update_value = update_value.reshape(update_shape_0, update_shape_1, update_shape_2, update_shape_3)
        
        axis = 2
    
    shape_0 = update_value.shape[0]
    shape_2 = update_value.shape[2]
    shape_3 = update_value.shape[3]
    output = copy.deepcopy(var)
    indices_value = indices.astype(np.int64)
    
    if len(indices.shape) == 2:
        if axis == -2 or axis == 2:
            for i in range(indices.shape[0]):
                for k in range(shape_2):
                    output[indices_value[i][0], :, indices_value[i][1] + k, :] = update_value[i, :, k, :]
        elif axis == -1 or axis == 3:
            for i in range(indices.shape[0]):
                for l in range(shape_3):
                    output[indices_value[i][0], :, :, indices_value[i][1] + l] = update_value[i, :, :, l]
    else:
        if axis == -2 or axis == 2:
            for i in range(shape_0):
                indices_key = indices_value[i]
                for k in range(shape_2):
                    output[i, :, indices_key + k, :] = update_value[i, :, k, :]
        elif axis == -1 or axis == 3:
            for i in range(shape_0):
                indices_key = indices_value[i]
                for l in range(shape_3):
                    output[i, :, :, indices_key + l] = update_value[i, :, :, l]
    
    return output
