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
__input__ = {
    "kernel": {"apply_came_part4": "customize_inputs"},
}

_KERNEL_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _to_f32(x):
    if torch.is_tensor(x):
        return x.detach().cpu().to(torch.float32)
    return torch.as_tensor(np.asarray(x).astype(np.float32), dtype=torch.float32)


def _to_numpy_for_golden(x):
    if torch.is_tensor(x):
        if x.dtype == torch.bfloat16:
            return x.detach().cpu().to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
        return x.detach().cpu().numpy()
    return x


def _dtype_of(x):
    if torch.is_tensor(x):
        return x.dtype
    np_dtype = np.asarray(x).dtype
    if np_dtype == np.float16:
        return torch.float16
    if np_dtype == ml_dtypes.bfloat16:
        return torch.bfloat16
    return torch.float32


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
    sum_r,
    global_shape,
    *attr_values,
    **kwargs,
):
    out_dtype = _dtype_of(param_in)

    param_f = _to_f32(param_in)
    m_f = _to_f32(m)
    r_f = _to_f32(r_in).reshape(-1)
    c_f = _to_f32(c_in).reshape(-1)
    wd = _to_f32(weight_decay).reshape(-1)[0]
    lr_v = _to_f32(lr).reshape(-1)[0]
    beta3_v = _to_f32(beta3).reshape(-1)[0]
    sum_u_r_f = _to_f32(sum_u_r).reshape(-1)
    sum_u_c_f = _to_f32(sum_u_c).reshape(-1)
    sum_u_rc_v = _to_f32(sum_u_rc).reshape(-1)[0]

    if sum_r is None:
        sum_r_v = r_f.sum()
    else:
        sum_r_v = _to_f32(sum_r).reshape(-1)[0]

    n, mm = r_f.shape[0], c_f.shape[0]
    if global_shape is None:
        n_g, m_g = float(n), float(mm)
    else:
        gs = np.asarray(global_shape).reshape(-1)
        n_g, m_g = float(gs[0]), float(gs[1])

    one = torch.tensor(1.0, dtype=torch.float32)
    # r/c 更新:fp32 计算后 round 回输出 dtype(torch cast 为 RNE,对齐 CAST_RINT)
    r_out = (beta3_v * r_f + (one - beta3_v) / m_g * sum_u_r_f).to(out_dtype)
    c_out = (beta3_v * c_f + (one - beta3_v) / n_g * sum_u_c_f).to(out_dtype)
    # param 更新:以 round 后的 r_out/c_out(cast 回 fp32)为输入
    denom = beta3_v * sum_r_v / n_g + (one - beta3_v) * sum_u_rc_v / (m_g * n_g)
    s = torch.outer(r_out.to(torch.float32), c_out.to(torch.float32)) / denom
    param_out = ((one - lr_v * wd) * param_f - lr_v * m_f / torch.sqrt(s)).to(out_dtype)
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
    sum_r,
    global_shape,
):
    """约束取值域,保证金色可算(sqrt 内 r_out*c_out/denom 非负、denom != 0)。"""
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
        sum_r,
        sum_u_r,
        sum_u_c,
        sum_u_rc,
        global_shape,
    )


class _ApplyCamePart4Compose:
    """Third-party reference executed on the remote GPU server."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *inputs, **kwargs):
        merged = dict(self.kwargs)
        merged.update(kwargs)
        outputs = apply_came_part4_golden(
            *[_to_numpy_for_golden(value) for value in inputs],
            **merged,
        )
        device = inputs[0].device if inputs and torch.is_tensor(inputs[0]) else "cpu"
        return [torch.as_tensor(np.asarray(out), device=device) for out in outputs]


class ApplyCamePart4KernelSpec:
    golden = apply_came_part4_golden
    customize_inputs = customize_inputs
    third_party = {"torch": _ApplyCamePart4Compose}
    tolerance = _KERNEL_TOLERANCE


ApplyCamePart4TestSpec = ApplyCamePart4KernelSpec


# 【不存在】aclnn 通路: 本算子为 GE 图内算子(GE-only),canndev 亦无 aclnnApplyCamePart4。
