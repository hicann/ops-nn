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

__golden__ = {"kernel": {"nll_loss_grad": "nll_loss_grad_golden"}}


def nll_loss_grad_golden(input_x, y_grad_value, target_value, weight_value, total_weight_value, *, ignore_index=-100, reduction="mean", **kwargs):
    '''
    Golden function for nll_loss_grad.
    All the parameters (names and order) follow @nll_loss_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    x_shape = input_x.shape
    if len(x_shape) == 1:
        x_shape = [1, x_shape[0]]
    
    x_grad_shape = x_shape
    n_dim = x_shape[0]
    loss = np.zeros(x_shape).astype(np.float32)
    
    for i in range(0, n_dim):
        if (target_value[i] == ignore_index and ignore_index >= 0 and ignore_index < x_shape[-1]) or \
            target_value[i] < 0 or target_value[i] >= x_shape[-1]:
            continue
        valid_weight = weight_value[target_value[i]]
        
        if reduction == 'none':
            loss[i][target_value[i]] = -1 * y_grad_value[i] * valid_weight
        elif reduction == 'sum':
            loss[i][target_value[i]] = -1 * y_grad_value[0] * valid_weight
        elif reduction == 'mean':
            loss[i][target_value[i]] = -1 * valid_weight * (y_grad_value[0] / total_weight_value[0])
    
    loss = loss.reshape(x_grad_shape)
    return loss
