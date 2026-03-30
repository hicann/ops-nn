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
        "apply_adam_w": "apply_adam_w_golden"
    }
}


def apply_adam_w_golden(var, m, v, beta1_power, beta2_power, lr, weight_decay, 
                        beta1, beta2, epsilon, grad, max_grad_norm = None, # inputs
                        amsgrad: bool=False, maximize: bool=False, # attributes
                        **kwargs):
    '''
    Golden function for apply_adam_w.
    All the parameters (names and order) follow @apply_adam_w_def.cpp without outputs.
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
        if max_grad_norm is not None:
            max_grad_norm = max_grad_norm.astype("float32")
        beta1_power = beta1_power.astype("float32")
        beta2_power = beta2_power.astype("float32")
        lr = lr.astype("float32")
        weight_decay = weight_decay.astype("float32")
        beta1 = beta1.astype("float32")
        beta2 = beta2.astype("float32")
        epsilon = epsilon.astype("float32")

    gt = -grad if maximize else grad
    m_out = m * beta1 - (beta1 + (-1)) * gt
    v_out = v * beta2 - (beta2 + (-1)) * gt * gt

    var_t = var * (1 + (-lr * weight_decay))

    beta1_power_out = beta1_power * beta1
    beta2_power_out = beta2_power * beta2

    if amsgrad and max_grad_norm is not None:
        max_grad_norm_out = np.maximum(max_grad_norm, v_out)
        sqrt_v_t = np.sqrt(max_grad_norm_out / (1 - beta2_power_out))
        denom = sqrt_v_t + epsilon
    else:
        v_t = v_out / (1 - beta2_power_out)
        sqrt_v_t = np.sqrt(v_t)
        denom = sqrt_v_t + epsilon
    m_t = m_out / (beta1_power_out - 1)
    m_t_mul_lr = lr * m_t

    m_t_mul_lr_div_denom = m_t_mul_lr / denom
    var_out = var_t + m_t_mul_lr_div_denom

    var_out = var_out.astype(input_dtype, copy=False)
    m_out = m_out.astype(input_dtype, copy=False) 
    v_out = v_out.astype(input_dtype, copy=False)
    return var_out, m_out, v_out
