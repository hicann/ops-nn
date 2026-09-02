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
import torch


__golden__ = {"kernel": {"lamb_next_mv_with_decay": "lamb_next_mv_with_decay_golden"}}


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


def lamb_next_mv_with_decay_golden(
    input_mul3,
    input_mul2,
    input_realdiv1,
    input_mul1,
    input_mul0,
    input_realdiv0,
    input_mul4,
    mul0_x,
    mul1_sub,
    mul2_x,
    mul3_sub1,
    mul4_x,
    add2_y,
    **kwargs,
):
    """Golden for LambNextMVWithDecay. Params follow lamb_next_m_v_with_decay_def.cpp (without outputs). All inputs are numpy.ndarray.

    Computed by composing torch tensor ops (torch.add/torch.sqrt) instead of a hand-written
    numpy formula: red line R3 requires the golden to be a competitor-operator composition,
    and a naive numpy expression tends to make exactly the same rounding mistakes as the
    kernel under test, which would disguise a precision shortfall as a pass.
    """
    dt = input_mul3.dtype
    g2, v, g, m, param = [
        _t(x) for x in (input_mul3, input_mul2, input_mul1, input_mul0, input_mul4)
    ]
    rd1, rd0, b1, omb1, b2, omb2, wd, eps = _scalars(
        input_realdiv1,
        input_realdiv0,
        mul0_x,
        mul1_sub,
        mul2_x,
        mul3_sub1,
        mul4_x,
        add2_y,
    )
    # 两步(先乘后加)拼接,不用 torch.add(alpha=)/addcmul 的融合形式:后者可能走 FMA 单次舍入,
    # 与算子定义的「Muls 再 Add」两步舍入不是同一个运算序列。golden 要如实转写定义。
    next_v = v * b2 + g2 * omb2
    next_m = m * b1 + g * omb1
    v_unb, m_unb = next_v / rd1, next_m / rd0
    pw = param * wd
    y1 = pw + m_unb / torch.sqrt(v_unb + eps)
    y4 = pw + m_unb / (torch.sqrt(v_unb) + eps)
    return [
        y1.numpy().astype(dt),
        next_m.numpy().astype(dt),
        next_v.numpy().astype(dt),
        y4.numpy().astype(dt),
    ]


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


class _LambNextMVWithDecayCompose:
    def __call__(
        self,
        input_mul3,
        input_mul2,
        input_realdiv1,
        input_mul1,
        input_mul0,
        input_realdiv0,
        input_mul4,
        mul0_x,
        mul1_sub,
        mul2_x,
        mul3_sub1,
        mul4_x,
        add2_y,
        **kwargs,
    ):
        g2, v, g, m, param = (
            _tp_t(t)
            for t in (input_mul3, input_mul2, input_mul1, input_mul0, input_mul4)
        )
        rd1, rd0, b1, omb1, b2, omb2, wd, eps = (
            _tp_s(t)
            for t in (
                input_realdiv1,
                input_realdiv0,
                mul0_x,
                mul1_sub,
                mul2_x,
                mul3_sub1,
                mul4_x,
                add2_y,
            )
        )
        # 三方用 torch 的融合形式 (addcmul) 表达同一定义
        next_v = torch.addcmul(v * b2, g2, torch.ones_like(g2), value=omb2)
        next_m = torch.addcmul(m * b1, g, torch.ones_like(g), value=omb1)
        v_unb, m_unb = next_v / rd1, next_m / rd0
        pw = param * wd
        y1 = pw + m_unb / torch.sqrt(v_unb + eps)
        y4 = pw + m_unb / (torch.sqrt(v_unb) + eps)
        return [y1, next_m, next_v, y4]


class LambNextMVWithDecayKernelSpec:
    golden = lamb_next_mv_with_decay_golden
    third_party = {"torch": _LambNextMVWithDecayCompose}
    tolerance = _TOL_KERNEL


__spec__ = {"lamb_next_mv_with_decay": "LambNextMVWithDecayKernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# e2e / ONNX / 融合 pass: 均未交付。
