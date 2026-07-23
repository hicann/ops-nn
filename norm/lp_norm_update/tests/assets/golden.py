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


def lp_norm_update_golden(x, p, epsilon):
    """LpNormUpdate golden: y = max(x^(1/p), eps_eff)

    Args:
        x: input tensor (numpy array or torch tensor), theoretically non-negative
        p: norm order (int), p != 0 or ±inf (INT_MAX/INT_MIN)
        epsilon: numerical stability constant (float >= 0)

    Returns:
        y: output tensor, same shape and dtype as x
    """
    if isinstance(x, np.ndarray):
        x_torch = torch.from_numpy(x)
    else:
        x_torch = x

    orig_dtype = x_torch.dtype

    # FP16 adaptive epsilon
    if orig_dtype == torch.float16:
        if epsilon <= 1e-7:
            eps_eff = 0.0 if epsilon == 0.0 else 1e-7
        else:
            eps_eff = epsilon
    else:
        eps_eff = epsilon

    # Compute in FP32 for precision
    x_fp32 = x_torch.to(torch.float32)

    # Map ±inf via INT_MAX/INT_MIN
    P_POS_INF = 2147483647
    P_NEG_INF = -2147483648

    if p == P_POS_INF or p == P_NEG_INF or p == 1:
        # Identity path: y = max(x, eps_eff)
        y_fp32 = torch.max(x_fp32, torch.tensor(eps_eff, dtype=torch.float32))
    elif p == 2:
        # Sqrt path: y = max(sqrt(max(x, 0)), eps_eff)
        x_clamped = torch.clamp(x_fp32, min=0.0)
        y_fp32 = torch.max(
            torch.sqrt(x_clamped), torch.tensor(eps_eff, dtype=torch.float32)
        )
    else:
        # Power path: y = max(max(x, 0)^(1/p), eps_eff)
        x_clamped = torch.clamp(x_fp32, min=0.0)
        inv_p = 1.0 / float(p)
        y_fp32 = torch.max(
            torch.pow(x_clamped, inv_p), torch.tensor(eps_eff, dtype=torch.float32)
        )

    # Cast back to original dtype
    y = y_fp32.to(orig_dtype)
    return y.numpy() if isinstance(x, np.ndarray) else y
