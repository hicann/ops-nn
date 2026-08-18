#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ActULQClampMinGrad 独立 Golden：raw、torch 小算子拼接。

ULQ 量化感知训练中 clamp 下界截断的反向梯度：
    mask'          = cast(clamp_min_mask, dtype(y_grad))
    signal         = 1 - mask'             # 被截断位置为 1，未截断为 0
    x_min_grad     = signal - x_clamped_loss   # loss 负号（与 Max 的 vadd 相反）
    clamp_min_grad = sum(y_grad * x_min_grad)   # 全轴求和，标量输出

fp16 输入升 fp32 计算（对齐 kernel fp32 累加器）；无单一 torch API 对应，
用 torch 原生小算子链式拼接作独立参考实现。
"""

import torch

__spec__ = {"act_ulq_clamp_min_grad": "ActUlqClampMinGradTestSpec"}

_L0 = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
}


def _compute(y_grad, clamp_min_mask, x_clamped_loss):
    """numpy 入参转 torch 计算，返回 numpy list。"""
    out_dtype = torch.from_numpy(y_grad).dtype
    calc_dtype = torch.float32 if out_dtype == torch.float16 else out_dtype

    y_grad_c = torch.from_numpy(y_grad).to(calc_dtype)
    x_clamped_loss_c = torch.from_numpy(x_clamped_loss).to(calc_dtype)
    mask_c = torch.from_numpy(clamp_min_mask).to(calc_dtype)

    ones = torch.ones_like(mask_c)
    signal = torch.sub(ones, mask_c)
    x_min_grad = torch.sub(signal, x_clamped_loss_c)
    prod = torch.mul(y_grad_c, x_min_grad)
    clamp_min_grad = torch.sum(prod)

    result = clamp_min_grad.to(out_dtype).numpy().reshape(1).astype(y_grad.dtype)
    return [result]


def act_ulq_clamp_min_grad_golden(y_grad, clamp_min_mask, x_clamped_loss, **kwargs):
    del kwargs
    return _compute(y_grad, clamp_min_mask, x_clamped_loss)


class _TorchActUlqClampMinGrad:
    def __call__(self, y_grad, clamp_min_mask, x_clamped_loss, **kwargs):
        del kwargs
        out_dtype = y_grad.dtype
        mask_c = clamp_min_mask.to(out_dtype)
        signal = torch.sub(torch.ones_like(mask_c), mask_c)
        result = torch.sum(torch.mul(y_grad, torch.sub(signal, x_clamped_loss)))
        return [result.reshape(1)]


class ActUlqClampMinGradTestSpec:
    golden = staticmethod(act_ulq_clamp_min_grad_golden)
    third_party = {"torch": _TorchActUlqClampMinGrad}
    tolerance = _L0
