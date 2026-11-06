#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch CPU golden for ApplyCamePart4.

NumPy is used only for TTK array conversion and input sanitization. The golden
math itself is expressed with Torch tensor operations.

Semantics (aligned with op_kernel/arch35/apply_came_part4.h; n = len(r_in),
m = len(c_in); N/M from global_shape when given, else n/m):

    sum_r     = sum(r_in)                                  (in-kernel when absent)
    r_out     = beta3 * r_in + (1 - beta3) / M * sum_u_r
    c_out     = beta3 * c_in + (1 - beta3) / N * sum_u_c
    denom     = beta3 * sum_r / N + (1 - beta3) * sum_u_rc / (M * N)
    param_out = (1 - lr * weight_decay) * param_in - lr * m / sqrt(r_out x c_out / denom)

fp16/bf16 path: inputs cast to fp32, computed in fp32, rounded (RNE, torch cast)
back to the low precision dtype; the param update consumes the ROUNDED r_out/c_out.

The kernel has no epsilon/clamp. References reject non-finite/non-positive denom
or normalized r*c (including zeros introduced by output rounding), rather than
changing the formula or passing NaN/Inf to the precision comparator.
"""

import numpy as np
import torch
import ml_dtypes  # noqa: F401 - registers bfloat16 with NumPy

__spec__ = {
    "apply_came_part4": "ApplyCamePart4KernelSpec",
}
__golden__ = {
    "kernel": {"apply_came_part4": "apply_came_part4_golden"},
}

_KERNEL_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _to_compute(x):
    """Convert to torch tensor: retain fp64 (TTK Promote), lift fp16/bf16 to fp32."""
    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.is_floating_point() and t.dtype not in (torch.float16, torch.bfloat16):
            return t
        return t.to(torch.float32)
    arr = np.asarray(x)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
    return torch.as_tensor(arr)


def _dtype_of(x):
    if torch.is_tensor(x):
        return x.dtype
    np_dtype = np.asarray(x).dtype
    if np_dtype == np.float16:
        return torch.float16
    if np_dtype == ml_dtypes.bfloat16:
        return torch.bfloat16
    if np_dtype == np.float64:
        return torch.float64
    return torch.float32


def _checked_param_update(param, m, r_out, c_out, denom, lr, weight_decay, out_dtype):
    """Follow the kernel's reciprocal/multiply path on its finite, positive domain."""
    if not torch.isfinite(denom).all() or not (denom > 0).all():
        raise ValueError("ApplyCamePart4 reference requires finite denom > 0")
    rc = torch.outer(r_out.to(param.dtype), c_out.to(param.dtype))
    # ProcessParam computes the reciprocal once, then ComputeParam multiplies rc.
    normalized_rc = rc * denom.reciprocal()
    if not torch.isfinite(normalized_rc).all() or not (normalized_rc > 0).all():
        raise ValueError(
            "ApplyCamePart4 reference requires finite rc / denom > 0 after r/c rounding"
        )
    param_out = (
        (1.0 - lr * weight_decay) * param - (lr * m) / torch.sqrt(normalized_rc)
    ).to(out_dtype)
    if not torch.isfinite(param_out).all():
        raise ValueError("ApplyCamePart4 reference produced non-finite param output")
    return param_out


def apply_came_part4_golden(
    param_in,
    m,
    r_in,
    c_in,
    weight_decay,
    lr,
    beta3,
    sum_u_r,
    sum_u_c,
    sum_u_rc,
    sum_r=None,
    global_shape=None,
    *attr_values,
    **kwargs,
):
    out_dtype = _dtype_of(param_in)

    param_f = _to_compute(param_in)
    m_f = _to_compute(m)
    r_f = _to_compute(r_in).reshape(-1)
    c_f = _to_compute(c_in).reshape(-1)
    wd = _to_compute(weight_decay).reshape(-1)[0]
    lr_v = _to_compute(lr).reshape(-1)[0]
    beta3_v = _to_compute(beta3).reshape(-1)[0]
    sum_u_r_f = _to_compute(sum_u_r).reshape(-1)
    sum_u_c_f = _to_compute(sum_u_c).reshape(-1)
    sum_u_rc_v = _to_compute(sum_u_rc).reshape(-1)[0]
    compute_dtype = param_f.dtype

    if sum_r is None:
        sum_r_v = r_f.sum()
    else:
        sum_r_v = _to_compute(sum_r).reshape(-1)[0]

    n, mm = r_f.shape[0], c_f.shape[0]
    if global_shape is None:
        n_g, m_g = float(n), float(mm)
    else:
        gs = np.asarray(global_shape).reshape(-1)
        n_g, m_g = float(gs[0]), float(gs[1])

    one = torch.tensor(1.0, dtype=compute_dtype)
    # r/c 更新:fp32 计算后 round 回输出 dtype(torch cast 为 RNE,对齐 CAST_RINT)
    r_out = (beta3_v * r_f + (one - beta3_v) / m_g * sum_u_r_f).to(out_dtype)
    c_out = (beta3_v * c_f + (one - beta3_v) / n_g * sum_u_c_f).to(out_dtype)
    # param 更新:以 round 后的 r_out/c_out(cast 回 compute_dtype)为输入
    denom = beta3_v * sum_r_v / n_g + (one - beta3_v) * sum_u_rc_v / (m_g * n_g)
    param_out = _checked_param_update(
        param_f, m_f, r_out, c_out, denom, lr_v, wd, out_dtype
    )
    return [param_out, r_out, c_out]


def customize_inputs(
    param_in,
    m,
    r_in,
    c_in,
    weight_decay,
    lr,
    beta3,
    sum_u_r,
    sum_u_c,
    sum_u_rc,
    sum_r=None,
    global_shape=None,
    **kwargs,
):
    """约束常规随机用例取值域；边界用例由 golden 显式检查，拒绝非有限参考结果。"""
    for tensor in (r_in, c_in, sum_u_r, sum_u_c):
        tensor[...] = np.abs(np.asarray(tensor)) + 0.1
    if sum_r is not None:
        sum_r[...] = np.abs(np.asarray(sum_r)) + 0.1
    sum_u_rc[...] = np.abs(np.asarray(sum_u_rc)) + 0.1
    weight_decay[...] = np.abs(np.asarray(weight_decay)) % 0.1
    lr[...] = np.abs(np.asarray(lr)) % 0.01 + 1e-5
    beta3[...] = np.abs(np.asarray(beta3)) % 0.999
    if global_shape is not None:
        n = np.asarray(r_in).size
        mm = np.asarray(c_in).size
        global_shape[...] = np.maximum(np.asarray(global_shape), [n, mm])
    return (
        param_in,
        m,
        r_in,
        c_in,
        weight_decay,
        lr,
        beta3,
        sum_u_r,
        sum_u_c,
        sum_u_rc,
        sum_r,
        global_shape,
    )


class _ApplyCamePart4Compose:
    """Third-party reference executed on the remote GPU server."""

    def __init__(self, **kwargs):
        # TTK may pass the input pool to both construction and invocation.
        # Bind tensors only in __call__, using the registered GE input names.
        pass

    def __call__(
        self,
        param,
        m,
        r,
        c,
        weight_decay,
        lr,
        beta3,
        sum_u_r,
        sum_u_c,
        sum_u_rc,
        sum_r=None,
        global_shape=None,
        **kwargs,
    ):
        out_dtype = param.dtype
        param_f = param.to(torch.float32)
        m_f = m.to(torch.float32)
        r_f = r.to(torch.float32).reshape(-1)
        c_f = c.to(torch.float32).reshape(-1)
        wd = weight_decay.to(torch.float32).reshape(-1)[0]
        lr_v = lr.to(torch.float32).reshape(-1)[0]
        beta3_v = beta3.to(torch.float32).reshape(-1)[0]
        sum_u_rc_v = sum_u_rc.to(torch.float32).reshape(-1)[0]
        sum_r_v = r_f.sum() if sum_r is None else sum_r.to(torch.float32).reshape(-1)[0]

        if global_shape is None:
            n_g, m_g = float(r_f.numel()), float(c_f.numel())
        else:
            n_g, m_g = global_shape.to(torch.float32).reshape(-1).unbind()

        r_out = (
            beta3_v * r_f
            + (1.0 - beta3_v) / m_g * sum_u_r.to(torch.float32).reshape(-1)
        ).to(out_dtype)
        c_out = (
            beta3_v * c_f
            + (1.0 - beta3_v) / n_g * sum_u_c.to(torch.float32).reshape(-1)
        ).to(out_dtype)
        denom = beta3_v * sum_r_v / n_g + (1.0 - beta3_v) * sum_u_rc_v / (m_g * n_g)
        # The parameter update consumes the rounded low-precision r/c outputs.
        param_out = _checked_param_update(
            param_f, m_f, r_out, c_out, denom, lr_v, wd, out_dtype
        )
        return [param_out, r_out, c_out]


class ApplyCamePart4KernelSpec:
    golden = apply_came_part4_golden
    customize_inputs = customize_inputs
    third_party = {"torch": _ApplyCamePart4Compose}
    tolerance = _KERNEL_TOLERANCE


ApplyCamePart4TestSpec = ApplyCamePart4KernelSpec


# 【不存在】aclnn 通路: 本算子为 GE 图内算子(GE-only),canndev 亦无 aclnnApplyCamePart4。
