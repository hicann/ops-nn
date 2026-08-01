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


__golden__ = {"kernel": {"lamb_apply_weight_assign": "lamb_apply_weight_assign_golden"}}


def _scalars(*xs):
    """取标量并落到 float32 torch 标量张量上。不能返回 Python float——那是 fp64，标量运算
    会被抬到双精度，而算子两代都在 fp32 上算（A2 compute_ratio 里 dtype='float32'；
    arch35 LambApplyWeightAssignCompute<T, U> 的 U = float）。numpy 只用于取值与 dtype
    转换，计算一律交给 torch。"""
    return tuple(
        torch.from_numpy(np.asarray(x, "float32").reshape(-1)[:1])[0] for x in xs
    )


_F32_TINY = torch.tensor(float(np.finfo(np.float32).tiny), dtype=torch.float32)


def _div_ftz(a, b):
    """torch.div 的 fp32 除法 + FTZ，对齐算子两代共同的行为。

    A2(910B) 的 Div 没有 config 参数（asc-devkit Div.md 里带 config 的原型对 Atlas A2
    标注"不支持"），只有单指令一条路，Subnormal 必然 FTZ；arch35 的 Vec::Div 用默认
    DivConfig{DivAlgo::INTRINSIC}，文档写明该档"Subnormal 均被 FTZ"。CPU 默认不 FTZ，
    所以要显式补这一步，否则商落入 [1e-45, 1.1754944e-38) 时 golden 会给出算子实际
    不会产出的值。"""
    q = torch.div(a, b)
    return torch.where((q != 0) & (q.abs() < _F32_TINY), torch.zeros_like(q), q)


def _t(x):
    """numpy.ndarray -> float32 torch tensor; numpy is kept for I/O and dtype conversion only."""
    return torch.from_numpy(np.asarray(x).astype("float32"))


def lamb_apply_weight_assign_golden(
    input0, input1, input2, input3, input_param, **kwargs
):
    """Golden for LambApplyWeightAssign. Params follow lamb_apply_weight_assign_def.cpp (without outputs). All inputs are numpy.ndarray.

    Computed by composing torch tensor arithmetic instead of a hand-written numpy formula:
    red line R3 requires the golden to be a competitor-operator composition, and a naive
    numpy expression tends to make exactly the same rounding mistakes as the kernel under
    test, which would disguise a precision shortfall as a pass.
    """
    dt = input3.dtype
    wn, gn, lr = _scalars(input0, input1, input2)
    upd, param = _t(input3), _t(input_param)
    one = torch.tensor(1.0, dtype=torch.float32)
    inner = _div_ftz(wn, gn) if gn > 0 else one
    ratio = inner if wn > 0 else one
    # 运算序列对齐实现：A2 (tbe.vmul(update, lr) 再 vmul(ratio, ·)) 与 arch35 DAG
    # (UpdLr = Update*Lr; RatioUpdLr = Ratio*UpdLr) 都是先算 update*lr。按 README 原先的
    # 字面顺序写成 (lr*ratio)*upd 在浮点下不等价：update 与 lr 同时较大时实现会先溢出成 inf，
    # 而 (lr*ratio) 往往是正常量级、再乘 update 不溢出，两者可以差出 inf 与有限值。
    return [(param - ratio * (upd * lr)).numpy().astype(dt)]
