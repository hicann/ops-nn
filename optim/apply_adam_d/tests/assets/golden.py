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


__golden__ = {
    "kernel": {
        "apply_adam_d": "apply_adam_d_golden"
    }
}


def apply_adam_d_golden(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, # inputs
                        use_locking: bool=False, use_nesterov: bool=False, # attributes
                        **kwargs):
    '''
    Golden function for apply_adam_d.
    All the parameters (names and order) follow @apply_adam_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''

    input_dtype = var.dtype
    if input_dtype.name in ("bfloat16", "float16"):
        var = var.astype("float32")
        m = m.astype("float32")
        v = v.astype("float32")
        grad = grad.astype("float32")
        beta1_power = beta1_power.astype("float32")
        beta2_power = beta2_power.astype("float32")
        lr = lr.astype("float32")
        beta1 = beta1.astype("float32")
        beta2 = beta2.astype("float32")
        epsilon = epsilon.astype("float32")
    
    # scalars
    beta1_power = beta1_power[0]
    beta2_power = beta2_power[0]
    lr = lr[0]
    beta1 = beta1[0]
    beta2 = beta2[0]
    epsilon = epsilon[0]
    
    # update m/v
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    alpha = lr * np.sqrt(1 - beta2_power) / (1 - beta1_power)
    
    # calculate var
    if use_nesterov:
        # Nesterov version
        m_hat = m_new * beta1 + (1 - beta1) * grad
        var_update = alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        var_new = var - var_update
    else:
        # nomalize version
        var_update = alpha * m_new / (np.sqrt(v_new) + epsilon)
        var_new = var - var_update

    var = var_new.astype(input_dtype, copy=False)
    m = m_new.astype(input_dtype, copy=False)
    v = v_new.astype(input_dtype, copy=False)

    return var, m, v
