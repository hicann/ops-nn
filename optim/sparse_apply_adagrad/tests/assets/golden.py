#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__golden__ = {"kernel": {"sparse_apply_adagrad": "sparse_apply_adagrad_golden"}}

# 与 kernel 中硬编码的 EPSILON (float32 1e-8) 保持一致：
#   accum[idx] = accum[idx] + grad[i] * grad[i]        (update_slots=True)
#   var[idx]   = var[idx] - lr * grad[i] / (sqrt(accum[idx]) + EPSILON)
_EPSILON = np.float32(1e-8)


def sparse_apply_adagrad_golden(
    var, accum, lr, grad, indices, *, update_slots=True, use_locking=False, **kwargs
):
    lr_val = float(lr.item()) if hasattr(lr, "item") else float(lr)
    var_out = np.array(var, copy=True)
    accum_out = np.array(accum, copy=True)
    grad_np = np.asarray(grad)
    indices_np = np.asarray(indices).flatten().astype(np.int32)

    for grad_row, idx in zip(grad_np, indices_np):
        if idx < 0 or idx >= var_out.shape[0]:
            continue
        if update_slots:
            accum_out[idx] += grad_row * grad_row
        var_out[idx] -= lr_val * grad_row / (np.sqrt(accum_out[idx]) + _EPSILON)

    return var_out, accum_out
