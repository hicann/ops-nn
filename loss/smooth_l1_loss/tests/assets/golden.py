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

"""SmoothL1Loss 算子测试规范（TestSpec）— GEIR / kernel 流程

公式（README §计算公式）:
  z = predict - label
  loss = |z| < sigma ? 0.5 * z^2 / sigma : |z| - 0.5 * sigma

特性:
  - FP16 / BF16 输入在计算前统一升精度到 FP32，计算完成后降回原精度
  - ThirdPartyImpl 不做升精度，直接用 torch 原生 dtype 计算
"""

import numpy as np
import torch

__spec__ = {
    "smooth_l1_loss": "SmoothL1LossTestSpec",
    "SmoothL1Loss": "SmoothL1LossTestSpec",
}


def _golden_torch(predict, label, sigma=1.0):
    """torch 实现（数值稳定：二次项先除后乘，避免 |z| 较大时 z^2 中间量溢出）"""
    p = torch.from_numpy(predict)
    label_t = torch.from_numpy(label)
    z = p - label_t
    az = z.abs()
    quad = 0.5 * (z / sigma) * z
    lin = az - 0.5 * sigma
    return torch.where(az < sigma, quad, lin).numpy()


class SmoothL1LossTestSpec:
    """SmoothL1Loss 算子测试规范（kernel / GEIR 流程）"""

    def golden(predict, label, sigma=1.0, **kwargs):
        if predict.size == 0:
            return [np.zeros(predict.shape, dtype=predict.dtype)]

        if isinstance(sigma, str):
            sigma = float(sigma)

        # FP16 / BF16 在计算前统一升精度到 FP32
        orig_dtype = predict.dtype
        need_downcast = "float16" in str(orig_dtype) or "bfloat16" in str(orig_dtype)
        if need_downcast:
            predict = predict.astype(np.float32)
            label = label.astype(np.float32)

        # 在 FP32 下完成全部计算
        result = _golden_torch(predict, label, sigma)

        # 降回原精度
        if need_downcast:
            result = result.astype(orig_dtype)

        return [result]

    class ThirdPartyImpl:
        """三方标杆实现（不做升精度，直接用 torch 原生 dtype 计算）"""

        def __init__(self, *, sigma=1.0, **kwargs):
            self.sigma = sigma

        def __call__(self, predict, label, **kwargs):
            if predict.size == 0:
                return [torch.zeros(predict.shape, dtype=predict.dtype)]
            beta = self.sigma
            if isinstance(beta, str):
                beta = float(beta)
            result = torch.nn.functional.smooth_l1_loss(
                predict, label, reduction="none", beta=beta
            )
            return [result]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
