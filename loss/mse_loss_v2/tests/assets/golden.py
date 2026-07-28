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
"""
TTK custom golden for MSELossV2 (arch35 / Ascend950).

Golden via the competitor interface torch.nn.functional.mse_loss (the reference itself,
NOT a numpy re-implementation of the formula). Positional args follow mse_loss_v2_def.cpp
order (input, target); attr reduction in {none, sum, mean}.

    l = (input - target) ** 2
    reduction=none -> l (same shape as input)
    reduction=sum  -> sum(l)   (scalar)
    reduction=mean -> mean(l)  (scalar)

The kernel casts fp16/bf16 -> fp32 for compute/reduce, then casts the result back to the
input dtype (round-to-nearest-even). The golden mirrors this: compute in fp32, cast back.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

__golden__ = {"kernel": {"mse_loss_v2": "mse_loss_v2_golden"}}


def _to_f32_tensor(arr):
    # arr may be float16/float32/bfloat16(ml_dtypes). Upcast to float32 for compute
    # (mirrors the NPU kernel which upcasts fp16/bf16 to fp32).
    a = np.ascontiguousarray(arr)
    return torch.from_numpy(a.astype(np.float32)).to(torch.float32)


def mse_loss_v2_golden(input0, input1, *, reduction="mean", **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    if output_dtypes is not None and len(output_dtypes) > 0:
        target_dt = str(output_dtypes[0])
    else:
        target_dt = str(np.asarray(input0).dtype)

    inp = _to_f32_tensor(input0)
    tgt = _to_f32_tensor(input1)
    y = F.mse_loss(inp, tgt, reduction=reduction)
    out = y.detach().cpu().numpy().astype(np.float32)

    if target_dt == "bfloat16":
        out = out.astype(_bf16) if _bf16 is not None else out
    else:
        out = out.astype(target_dt)
    return [out]
