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


__golden__ = {"kernel": {"lamb_update_with_lr": "lamb_update_with_lr_golden"}}


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


def _s(x):
    """python float -> float32 标量张量，供 NaN 传播语义的 minimum/maximum 使用。"""
    return torch.tensor(x, dtype=torch.float32)


def _fp32_div(a, b):
    """IEEE-754 fp32 scalar division through torch.div (x/0 -> +-inf, 0/0 -> nan, never raises)."""
    return torch.div(
        torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    ).item()


def lamb_update_with_lr_golden(
    input_greater1,
    input_greater_realdiv,
    input_realdiv,
    input_mul0,
    input_mul1,
    input_sub,
    greater_y,
    select_e,
    minimum_y,
    **kwargs,
):
    """Golden for LambUpdateWithLr. Params follow lamb_update_with_lr_def.cpp (without outputs). All inputs are numpy.ndarray.

    Computed by composing torch ops (torch.div for the ratio, torch tensor arithmetic for the
    update) instead of a hand-written numpy formula: red line R3 requires the golden to be a
    competitor-operator composition, and a naive numpy expression tends to make exactly the
    same rounding mistakes as the kernel under test, which would disguise a precision
    shortfall as a pass.
    """
    dt = input_mul1.dtype
    g1, grd, rd, lr, gy, se, miny = _scalars(
        input_greater1,
        input_greater_realdiv,
        input_realdiv,
        input_mul0,
        greater_y,
        select_e,
        minimum_y,
    )
    upd, param = _t(input_mul1), _t(input_sub)
    # Match kernel Vec::Div<float> (arch35 DivAlgo::INTRINSIC): IEEE-754 fp32 division.
    # x/0 -> +-inf, 0/0 -> nan (never raises); clipped downstream by min/max like the kernel.
    realdiv0 = _fp32_div(grd, rd)
    select0 = realdiv0 if g1 > gy else se
    select1 = select0 if grd > gy else se
    # 内核是 Vec::Min<U>/Vec::Max<U> 硬件矢量指令，与 torch.minimum/maximum 一样传播 NaN；
    # Python 内置 min()/max() 是比较语义、会静默丢掉 NaN（max(x, nan) -> x），不是同一个函数。
    # greater_y / minimum_y 为 NaN 时内核整片输出 NaN，用内置版本会给出有限值而全盘对不上。
    clip = torch.maximum(torch.minimum(_s(select1), _s(miny)), _s(gy)).item()
    return [(param - clip * lr * upd).numpy().astype(dt)]


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


class _LambUpdateWithLrCompose:
    def __call__(
        self,
        input_greater1,
        input_greater_realdiv,
        input_realdiv,
        input_mul0,
        input_mul1,
        input_sub,
        greater_y,
        select_e,
        minimum_y,
        **kwargs,
    ):
        g1, grd, rd, lr, gy, se, miny = (
            _tp_s(x)
            for x in (
                input_greater1,
                input_greater_realdiv,
                input_realdiv,
                input_mul0,
                greater_y,
                select_e,
                minimum_y,
            )
        )
        upd, param = _tp_t(input_mul1), _tp_t(input_sub)
        realdiv0 = torch.div(grd, rd)
        select0 = realdiv0 if bool(g1 > gy) else se
        select1 = select0 if bool(grd > gy) else se
        clip = torch.maximum(torch.minimum(select1, miny), gy)
        return [param - (clip * lr) * upd]


class LambUpdateWithLrKernelSpec:
    golden = lamb_update_with_lr_golden
    third_party = {"torch": _LambUpdateWithLrCompose}
    tolerance = _TOL_KERNEL


__spec__ = {"lamb_update_with_lr": "LambUpdateWithLrKernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# e2e / ONNX / 融合 pass: 均未交付。
