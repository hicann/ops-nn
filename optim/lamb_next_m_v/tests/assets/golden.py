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


__golden__ = {"kernel": {"lamb_next_mv": "lamb_next_mv_golden"}}


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


def lamb_next_mv_golden(
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
    """Golden for LambNextMV. Params follow lamb_next_m_v_def.cpp (without outputs). All inputs are numpy.ndarray.

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
    y1 = param * wd + m_unb / torch.sqrt(v_unb + eps)
    y4 = m_unb / (torch.sqrt(v_unb) + eps)
    return [
        y1.numpy().astype(dt),
        next_m.numpy().astype(dt),
        next_v.numpy().astype(dt),
        y4.numpy().astype(dt),
    ]
