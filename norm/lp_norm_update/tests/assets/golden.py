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
"""LpNormUpdate golden reference.

y = max(x^(1/p), eps_eff)

The operator computes only the root step: x is the LpNormReduce output.
Kernel and GEIR share the same operator name, so one spec covers both.
"""

import numpy as np
import torch

__spec__ = {
    "lp_norm_update": "LpNormUpdateTestSpec",
    "LpNormUpdate": "LpNormUpdateTestSpec",
}

P_POS_INF = 2147483647
P_NEG_INF = -2147483648
FP16_EPS_THRESHOLD = 1e-7


class TorchLpNormUpdate:
    """Native Torch reference used by cross_check/third_party.

    ``__init__`` receives the input tensor so dtype/device-dependent
    preparation can be done once; ``__call__`` keeps only the formula.
    """

    def __init__(self, x, *, p=2, epsilon=1e-12, **kwargs):
        self.p = int(p)
        self.epsilon = float(epsilon)
        if self.epsilon == 0.0:
            self.epsilon_fp16 = 0.0
        elif self.epsilon <= FP16_EPS_THRESHOLD:
            self.epsilon_fp16 = FP16_EPS_THRESHOLD
        else:
            self.epsilon_fp16 = self.epsilon
        self.identity = self.p in (0, 1, P_POS_INF, P_NEG_INF)
        self.inv_p = None if self.identity or self.p == 2 else 1.0 / float(self.p)

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        self.dtype = x.dtype
        self.calc = (
            x.to(torch.float32) if self.dtype in (torch.float16, torch.bfloat16) else x
        )
        eps_eff = self.epsilon_fp16 if self.dtype == torch.float16 else self.epsilon
        self.std_no = torch.tensor(
            eps_eff, dtype=self.calc.dtype, device=self.calc.device
        )

    def __call__(self, x=None, **kwargs):
        if self.identity:
            y = torch.max(self.calc, self.std_no)
        else:
            clamped = torch.clamp(self.calc, min=0.0)
            if self.p == 2:
                y = torch.max(torch.sqrt(clamped), self.std_no)
            else:
                y = torch.max(torch.pow(clamped, self.inv_p), self.std_no)
        return [y.to(self.dtype)]


def lp_norm_update_golden(x, p, epsilon):
    """LpNormUpdate golden: y = max(x^(1/p), eps_eff).

    Args:
        x: input tensor (numpy array or torch tensor), theoretically non-negative
        p: norm order (int), any integer or +/-inf (INT_MAX/INT_MIN)
        epsilon: numerical stability constant (float >= 0)

    Returns:
        y: output tensor, same shape and dtype as x
    """
    is_numpy = isinstance(x, np.ndarray)
    x_torch = torch.from_numpy(x) if is_numpy else x
    y = TorchLpNormUpdate(x_torch, p=p, epsilon=epsilon)()[0]
    return y.numpy() if is_numpy else y


class LpNormUpdateTestSpec:
    """CPU NumPy/Torch reference for the operator README formula."""

    @staticmethod
    def golden(x, p=2, epsilon=1e-12, **kwargs):
        return [lp_norm_update_golden(x, p, epsilon)]

    third_party = {"torch": TorchLpNormUpdate}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }
