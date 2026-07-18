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
"""
TTK custom golden for hard_swish_grad_v2 (HardSwish backward).

Inputs (positional, in op-def order):
    grad_output : numpy array  (gradOutput)
    self_x      : numpy array  (self / x), same shape as grad_output
Output:
    out : gradInput, same shape, NON-inplace.

Formula:
    hardswish(x) = x * relu6(x + 3) / 6
    d(x) = d hardswish / dx:
        x <= -3 -> 0
        x >=  3 -> 1
        else    -> x/3 + 0.5   (== (2x+3)/6)
    gradInput = grad_output * d(x)

Kernel parity (op_kernel/hard_swish_grad_v2_100.h):
    The kernel does NOT use the naive piecewise where(); it computes a value
    val = x * oneThird + oneHalf, then applies TWO Selects driven by STRICT
    Compares:
        maskGreater = (x  > -3)   (CMPMODE::GT)
        maskLessThan = (x <  3)   (CMPMODE::LT)
        val = Select(maskGreater, val, 0.0)   # not (x>-3)  -> 0
        val = Select(maskLessThan, val, 1.0)  # not (x< 3)  -> 1
        out = grad * val
    Reproducing this exact ordering is what makes nan/inf self propagate the
    same way as the NPU:
        x == nan  : both masks False -> val=0 then val=1 -> out = grad * 1 = grad
        x == +inf : >  -3 True (keep), < 3 False -> val=1 -> out = grad
        x == -inf : >  -3 False     -> val=0      -> out = 0
    Boundaries (matches GE/LE result semantics):
        x == -3 : maskGreater False -> 0 ; maskLessThan True keeps 0 -> 0
        x ==  3 : maskGreater True keeps 1.5 ; maskLessThan False -> 1 -> grad*1
    grad nan/inf propagation: out = grad * val carries it (inf*0 -> nan, etc).
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

# Match the kernel's exact fp32 constants (oneThird = 0.33333334, oneHalf = 0.5).
_ONE_THIRD = np.float32(0.33333334)
_ONE_HALF = np.float32(0.5)


def __golden_hard_swish_grad_v2(grad_output, self_x, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    if output_dtypes is not None and len(output_dtypes) > 0:
        target = str(output_dtypes[0])
    else:
        target = str(np.asarray(grad_output).dtype)

    # Upcast to fp32 for compute (fp16/bf16 path on NPU also upcasts to fp32).
    g = np.asarray(grad_output).astype(np.float32)
    x = np.asarray(self_x).astype(np.float32)

    val = x * _ONE_THIRD + _ONE_HALF

    # Strict compares, mirroring the kernel's two ordered Selects.
    with np.errstate(invalid="ignore"):
        mask_greater = x > np.float32(-3.0)  # CMPMODE::GT
        mask_less = x < np.float32(3.0)  # CMPMODE::LT
    val = np.where(mask_greater, val, np.float32(0.0))  # x <= -3 (or nan) -> 0
    val = np.where(mask_less, val, np.float32(1.0))  # x >=  3 (or nan) -> 1

    with np.errstate(invalid="ignore"):
        out = (g * val).astype(np.float32)  # grad inf*0 -> nan (matches kernel Mul)

    if target == "bfloat16":
        out = out.astype(_bf16) if _bf16 is not None else out
    else:
        out = out.astype(target)
    return [out]


__golden__ = {"kernel": {"hard_swish_grad_v2": "__golden_hard_swish_grad_v2"}}
