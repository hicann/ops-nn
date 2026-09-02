#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import torch


__golden__ = {"kernel": {"lamb_update_with_lr_v2": "lamb_update_with_lr_v2_golden"}}


def _scalars(*xs):
    """取标量并落到 float32 torch 标量张量上。不能返回 Python float——那是 fp64，标量
    运算会被抬到双精度，而算子在 fp32 上算（A2 的 TBE compute 里 dtype='float32'，
    arch35 DAG 的计算类型 U = float）。numpy 只用于取值与 dtype 转换。"""
    return tuple(
        torch.from_numpy(np.asarray(x, "float32").reshape(-1)[:1])[0] for x in xs
    )


def _t(x):
    """numpy.ndarray -> float32 torch tensor; numpy is kept for I/O and dtype conversion only."""
    return torch.from_numpy(np.asarray(x).astype("float32"))


def _fp32_div(a, b):
    """IEEE-754 fp32 scalar division through torch.div (x/0 -> +-inf, 0/0 -> nan, never raises)."""
    return torch.div(
        torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    ).item()


def lamb_update_with_lr_v2_golden(x1, x2, x3, x4, x5, greater_y, select_e, **kwargs):
    """Golden for LambUpdateWithLrV2. Params follow lamb_update_with_lr_v2_def.cpp (without outputs). All inputs are numpy.ndarray.

    Computed by composing torch ops (torch.div for the ratio, torch tensor arithmetic for the
    update) instead of a hand-written numpy formula: red line R3 requires the golden to be a
    competitor-operator composition, and a naive numpy expression tends to make exactly the
    same rounding mistakes as the kernel under test, which would disguise a precision
    shortfall as a pass.
    """
    dt = x4.dtype
    a, b, lr, gy, se = _scalars(x1, x2, x3, greater_y, select_e)
    upd, param = _t(x4), _t(x5)
    # Match kernel Vec::Div<float> (arch35 DivAlgo::INTRINSIC): IEEE-754 fp32 division.
    # b<=gy stays on select_e (kernel Select); when b>gy and b==0 the kernel outputs inf
    # (v2 has NO clip), so mirror that instead of raising ZeroDivisionError.
    inner = _fp32_div(a, b) if b > gy else se
    ratio = inner if a > gy else se
    return [(param - lr * ratio * upd).numpy().astype(dt)]


# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 在保留原 golden 的基础上补三方标杆能力。
# golden     = CPU 真值，如实转写算子定义（两步拼接，不用融合算子）
# third_party= 三方标杆，用 torch 的自然形式（含融合算子）在设备侧跑，供 cross_check 比对
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
}


def _tp_t(x):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。"""
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    return t.to(torch.float32)


def _tp_s(x):
    return _tp_t(x).reshape(-1)[0]


class _LambUpdateWithLrV2Compose:
    def __call__(self, x1, x2, x3, x4, x5, greater_y, select_e, **kwargs):
        a, b, lr, gy, se = (_tp_s(v_) for v_ in (x1, x2, x3, greater_y, select_e))
        upd, param = _tp_t(x4), _tp_t(x5)
        inner = torch.div(a, b) if bool(b > gy) else se
        ratio = inner if bool(a > gy) else se
        return [param - lr * ratio * upd]


class LambUpdateWithLrV2KernelSpec:
    golden = lamb_update_with_lr_v2_golden
    third_party = {"torch": _LambUpdateWithLrV2Compose}
    tolerance = _TOL_KERNEL


__spec__ = {"lamb_update_with_lr_v2": "LambUpdateWithLrV2KernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# e2e / ONNX / 融合 pass: 均未交付。
