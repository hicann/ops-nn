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

__golden__ = {"kernel": {"relu_grad_v2": "relu_grad_v2_golden"}}


def _vsel_bit(condition, input1, input2):
    '''
    Golden function for relu_grad_v2.
    All the parameters (names and order) follow @relu_grad_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    input1_list = input1.reshape(-1)
    input2_list = input2.reshape(-1)
    condition = condition.reshape(-1)
    condition_length = len(condition)
    output = np.zeros((condition_length * 8,), input1_list.dtype)
    
    for i in range(condition_length):
        mask = condition[i]
        for j in range(8):
            flag = mask % 2
            mask //= 2
            real_index = i * 8 + j
            output[real_index] = input1_list[real_index] if flag == 1 else input2_list[real_index]
    return output


def relu_grad_v2_golden(gradients, mask, **kwargs):
    gradients_shape = gradients.shape
    ipt0_dtype = gradients.dtype
    zeros = np.zeros(gradients_shape, dtype=ipt0_dtype)
    res = _vsel_bit(mask, gradients, zeros)
    return res.astype(ipt0_dtype, copy=False)
