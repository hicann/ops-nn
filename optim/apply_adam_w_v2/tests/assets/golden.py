#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np


__golden__ = {"kernel": {"apply_adam_w_v2": "apply_adam_w_v2_golden"}}


def _is_low_precision(dtype):
    return dtype.name in ("bfloat16", "float16")


def apply_adam_w_v2_golden(
    var,
    m,
    v,
    grad,
    step,
    max_grad_norm=None,  # inputs (follow def.cpp Input order)
    lr: float = 0.1,
    beta1: float = 0.1,
    beta2: float = 0.1,
    weight_decay: float = 0.1,
    eps: float = 1e-8,
    amsgrad: bool = False,
    maximize: bool = False,  # attributes
    **kwargs,
):
    """
    Golden function for apply_adam_w_v2.

    Parameter names/order follow @apply_adam_w_v2_def.cpp Input declarations
    (var, m, v, grad, step, max_grad_norm) followed by the float/bool attributes.
    All input Tensors are numpy.ndarray; lr/beta1/beta2/weight_decay/eps are float
    scalar attributes; step is a tensor (INT64 or FLOAT32) with a single element.

    Compute contract is verified against the AICore kernel
    (op_kernel/apply_adam_w_v2_fp.h and op_kernel/arch35/apply_adam_w_v2_dag.h):
      - step is incremented by 1 internally before bias correction
      - maximize negates grad
      - mixed precision: promote to float32 for the whole computation, cast back at the end

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        (var_out, m_out, v_out, max_grad_norm_out) aligned with output_shapes and
        output_inplace_indexes=(0, 1, 2, 5).
    """
    var_dtype = var.dtype
    grad_dtype = grad.dtype

    var_low = _is_low_precision(var_dtype)
    grad_low = _is_low_precision(grad_dtype)

    # Mixed-precision promotion, mirroring executor_apply_adam_w_v2.py:
    #   - var low precision -> promote every tensor to float32
    #   - else grad low precision -> promote only grad and max_grad_norm to float32
    if var_low:
        var = var.astype("float32")
        m = m.astype("float32")
        v = v.astype("float32")
        grad = grad.astype("float32")
        if max_grad_norm is not None:
            max_grad_norm = max_grad_norm.astype("float32")
    elif grad_low:
        grad = grad.astype("float32")
        if max_grad_norm is not None:
            max_grad_norm = max_grad_norm.astype("float32")

    # scalar attributes computed in float32
    lr = np.float32(lr)
    beta1 = np.float32(beta1)
    beta2 = np.float32(beta2)
    weight_decay = np.float32(weight_decay)
    eps = np.float32(eps)

    # step tensor -> scalar; interface adds 1 internally (see kernel step_ += 1)
    step_val = float(np.asarray(step).reshape(-1)[0]) + 1.0

    if maximize:
        grad = -grad

    # weight decay: param.mul_(1 - lr * weight_decay)
    var_t = var * (1.0 - lr * weight_decay)

    # exp_avg.lerp_(grad, 1 - beta1): m_out = beta1 * m + (1 - beta1) * grad
    m_out = beta1 * m + (1.0 - beta1) * grad
    # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    v_out = beta2 * v + (1.0 - beta2) * grad * grad

    bias_correction1 = 1.0 - np.float32(beta1) ** step_val
    bias_correction2 = 1.0 - np.float32(beta2) ** step_val
    bias_correction2_sqrt = np.sqrt(bias_correction2)

    if amsgrad and max_grad_norm is not None:
        max_grad_norm_out = np.maximum(max_grad_norm, v_out)
        denom = np.sqrt(max_grad_norm_out) / bias_correction2_sqrt + eps
    else:
        max_grad_norm_out = max_grad_norm
        denom = np.sqrt(v_out) / bias_correction2_sqrt + eps

    step_size = lr / bias_correction1
    # param.addcdiv_(exp_avg, denom, value=-step_size)
    var_out = var_t - step_size * (m_out / denom)

    # Cast back to target precision, mirroring executor_apply_adam_w_v2.py
    if var_low:
        var_out = var_out.astype(var_dtype, copy=False)
        m_out = m_out.astype(var_dtype, copy=False)
        v_out = v_out.astype(var_dtype, copy=False)
        if max_grad_norm_out is not None:
            max_grad_norm_out = max_grad_norm_out.astype(var_dtype, copy=False)
    elif grad_low:
        if max_grad_norm_out is not None:
            max_grad_norm_out = max_grad_norm_out.astype(grad_dtype, copy=False)

    return var_out, m_out, v_out, max_grad_norm_out
