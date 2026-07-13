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


__golden__ = {"kernel": {"inplace_apply_power_sign": "inplace_apply_power_sign_golden"}}


def inplace_apply_power_sign_golden(
    var, m, lr, logbase, sign_decay, beta, grad, use_locking: bool = False, **kwargs
):
    input_dtype = var.dtype

    # Compute in float64 for maximum precision, regardless of input dtype,
    # then cast the result back to the original dtype.
    var = var.astype("float64")
    m = m.astype("float64")
    lr = lr.astype("float64")
    logbase = logbase.astype("float64")
    sign_decay = sign_decay.astype("float64")
    beta = beta.astype("float64")
    grad = grad.astype("float64")

    lr_val = lr.flat[0]
    logbase_val = logbase.flat[0]
    sign_decay_val = sign_decay.flat[0]
    beta_val = beta.flat[0]

    m_new = beta_val * m + (1.0 - beta_val) * grad
    sign_gm = np.sign(m_new) * np.sign(grad)
    var_new = var - lr_val * np.exp(logbase_val * sign_decay_val * sign_gm) * grad

    var_out = var_new.astype(input_dtype, copy=False)
    m_out = m_new.astype(input_dtype, copy=False)

    return var_out, m_out
