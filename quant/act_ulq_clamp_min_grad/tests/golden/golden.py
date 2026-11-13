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
"""ActULQClampMinGrad: 910B dynamic semantics.

    clamp_min_grad = sum(y_grad * (x_clamped_loss + abs(clamp_min_mask)))

CPU Golden preserves the dtype supplied by TTK Promote (FP32 cases use FP64).
Native FP16 inputs use FP32 intermediates.
"""

__spec__ = {"act_ulq_clamp_min_grad": "ActUlqClampMinGradTestSpec"}

_L0 = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
}


def _compute(y_grad, clamp_min_mask, x_clamped_loss):
    import torch

    y = torch.from_numpy(y_grad)
    out_dtype = y.dtype
    calc_dtype = torch.float32 if out_dtype == torch.float16 else out_dtype
    y = y.to(calc_dtype)
    mask = torch.from_numpy(clamp_min_mask).to(calc_dtype)
    loss = torch.from_numpy(x_clamped_loss).to(calc_dtype)
    value = torch.sum(torch.mul(y, torch.add(loss, torch.abs(mask))))
    return [value.to(out_dtype).numpy().reshape(1)]


def act_ulq_clamp_min_grad_golden(y_grad, clamp_min_mask, x_clamped_loss, **kwargs):
    return _compute(y_grad, clamp_min_mask, x_clamped_loss)


class _TorchActUlqClampMinGrad:
    def __call__(self, y_grad, clamp_min_mask, x_clamped_loss, **kwargs):
        import torch

        out_dtype = y_grad.dtype
        calc_dtype = torch.float32 if out_dtype == torch.float16 else out_dtype
        mask = clamp_min_mask.to(calc_dtype)
        weighted = torch.mul(
            y_grad.to(calc_dtype),
            torch.add(x_clamped_loss.to(calc_dtype), torch.abs(mask)),
        )
        return [torch.sum(weighted).to(out_dtype).reshape(1)]


class ActUlqClampMinGradTestSpec:
    golden = staticmethod(act_ulq_clamp_min_grad_golden)
    third_party = {"torch": _TorchActUlqClampMinGrad}
    tolerance = _L0
