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
import torch

__spec__ = {
    "hard_swish_grad_v2": "HardSwishGradV2KernelSpec",
    "aclnnHardswishBackwardV2": "HardSwishGradV2AclnnSpec",
}

_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}
_TOL_LOCAL = {
    "float32": {"standard": "stat_rel_err"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
}


def _numpy_to_torch_tensor(array):
    if "bfloat16" in array.dtype.name:
        return torch.from_numpy(array.view(dtype=np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy_tensor(tensor):
    if tensor.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return tensor.view(torch.int16).numpy().view(dtype=bfloat16)
    return tensor.numpy()


def _dtype_string(value):
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def _hard_swish_grad_v2_compute(grad_output, self_x, target=None):
    source_tensor = (
        _numpy_to_torch_tensor(np.asarray(grad_output))
        if not isinstance(grad_output, torch.Tensor)
        else grad_output
    )
    if target is None:
        target = (
            "bfloat16"
            if source_tensor.dtype == torch.bfloat16
            else str(source_tensor.dtype).replace("torch.", "")
        )
    compute_dtype = (
        torch.float32
        if source_tensor.dtype in (torch.float16, torch.bfloat16)
        else source_tensor.dtype
    )
    # Use PyTorch tensor operations as the third-party reference, while keeping
    # the same strict compare ordering required by the operator contract.
    g = source_tensor.to(compute_dtype)
    x_source = (
        _numpy_to_torch_tensor(np.asarray(self_x))
        if not isinstance(self_x, torch.Tensor)
        else self_x
    )
    x = x_source.to(compute_dtype)

    one_third = torch.tensor(0.33333334, dtype=compute_dtype, device=x.device)
    one_half = torch.tensor(0.5, dtype=compute_dtype, device=x.device)
    val = x * one_third + one_half

    # Strict compares, mirroring the kernel's two ordered Selects.
    mask_greater = x > torch.tensor(-3.0, dtype=compute_dtype, device=x.device)
    mask_less = x < torch.tensor(3.0, dtype=compute_dtype, device=x.device)
    val = torch.where(
        mask_greater, val, torch.tensor(0.0, dtype=compute_dtype, device=x.device)
    )
    val = torch.where(
        mask_less, val, torch.tensor(1.0, dtype=compute_dtype, device=x.device)
    )

    out = g * val
    if target == "bfloat16":
        out = out.to(torch.bfloat16)
    else:
        out = out.to(source_tensor.dtype)
    return out


class _HardSwishGradV2Compose:
    def __init__(self, **kwargs):
        pass

    def __call__(self, *inputs, **kwargs):
        grad_output, self_x = inputs[:2]
        return [_hard_swish_grad_v2_compute(grad_output, self_x)]


class HardSwishGradV2KernelSpec:
    @staticmethod
    def golden(gradOutput, self, **kwargs):
        output_dtypes = kwargs.get("output_dtypes")
        target = (
            _dtype_string(output_dtypes[0])
            if output_dtypes
            else str(np.asarray(gradOutput).dtype)
        )
        out = _hard_swish_grad_v2_compute(gradOutput, self, target)
        return [_torch_to_numpy_tensor(out)]

    third_party = {"torch": _HardSwishGradV2Compose}
    tolerance = _TOL_KERNEL


class HardSwishGradV2AclnnSpec:
    @staticmethod
    def golden(gradOutput, self, **kwargs):
        return [_hard_swish_grad_v2_compute(gradOutput, self)]

    third_party = {"torch": _HardSwishGradV2Compose}
    tolerance = _TOL_LOCAL


def __golden_hard_swish_grad_v2(grad_output, self_x, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    if output_dtypes is not None and len(output_dtypes) > 0:
        target = _dtype_string(output_dtypes[0])
    else:
        target = str(np.asarray(grad_output).dtype)
    out = _hard_swish_grad_v2_compute(grad_output, self_x, target)
    return [_torch_to_numpy_tensor(out)]


__golden__ = {"kernel": {"hard_swish_grad_v2": "__golden_hard_swish_grad_v2"}}


# Not registered in __spec__:
# - e2e/TensorFlow/ONNX/fusion: no torch_npu binding, parser, or graph pass is delivered.
