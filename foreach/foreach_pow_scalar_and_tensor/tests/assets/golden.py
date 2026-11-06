#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""foreach_pow_scalar_and_tensor 三方 Golden — 标量底数张量指数幂。"""

__spec__ = {
    "foreach_pow_scalar_and_tensor": "ForeachPowScalarAndTensorKernelSpec",
    "aclnnForeachPowScalarAndTensor": "AclnnForeachPowScalarAndTensorSpec",
    "torch._foreach_pow": "TorchForeachPowScalarAndTensorSpec",
}

import numpy as np
import torch
from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


def _pow_reference(s, t):
    """Match the operator: scalar keeps its own precision, x is cast to compute type.

    - float paths: scalar is float32; compute in float64 and cast to x dtype.
    - int32 path: scalar is int64; exponent is int32; overflow wraps to int64
      during the multiply and to int32 on the final store, matching IntPow.
    """
    if t.dtype in (torch.int32, torch.int64):
        base = int(s)
        exp = t.to(torch.int64)
        result = torch.zeros_like(exp, dtype=torch.int32)
        pos = exp >= 0
        if pos.any():
            result[pos] = torch.pow(torch.tensor(base, dtype=torch.int64), exp[pos]).to(
                torch.int32
            )
        neg = ~pos
        if neg.any():
            if base == 1:
                result[neg] = 1
            elif base == -1:
                result[neg] = torch.where(exp[neg] % 2 == 0, 1, -1).to(torch.int32)
            else:
                result[neg] = 0
        return result
    base = torch.tensor(float(s), dtype=torch.float64)
    return torch.pow(base, t.to(torch.float64)).to(t.dtype)


def _to_scalar(v):
    if isinstance(v, np.ndarray):
        return v.reshape(-1)[0].item()
    if isinstance(v, torch.Tensor):
        return v.reshape(-1)[0].item()
    return v


class _TorchForeachPowScalarAndTensorImpl:
    def __init__(self, scalar, x, **kwargs):
        value = _to_scalar(scalar)
        self._bases = [torch.tensor(value, dtype=t.dtype, device=t.device) for t in x]

    def __call__(self, x, **kwargs):
        return [torch.pow(base, exponent) for base, exponent in zip(self._bases, x)]


class _TorchForeachPowScalarAndTensorE2EImpl:
    def __init__(self, *args, exponent=None, **kwargs):
        value = _to_scalar(args[0])
        self._bases = [
            torch.tensor(value, dtype=t.dtype, device=t.device) for t in exponent
        ]

    def __call__(self, exponent, *args):
        return [torch.pow(base, value) for base, value in zip(self._bases, exponent)]


_ACLNN_THIRD_PARTY = {"torch": _TorchForeachPowScalarAndTensorImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachPowScalarAndTensorE2EImpl}


class ForeachPowScalarAndTensorKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。公式: y[i]=scalar**x[i]"""

    def golden(scalar, x, **kwargs):
        s = _to_scalar(scalar)
        x_list = [numpy_to_torch_tensor(np.ascontiguousarray(t)) for t in x]
        result = [_pow_reference(s, t) for t in x_list]
        return [torch_to_numpy_tensor(t) for t in result]

    third_party = _ACLNN_THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachPowScalarAndTensorSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: aclnnForeachPowScalarAndTensorGetWorkspaceSize(scalar, x, out, workspaceSize, executor)
    """

    def golden(scalar, x, out=None, **kwargs):
        s = _to_scalar(scalar)
        result = [_pow_reference(s, t) for t in x]
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _ACLNN_THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachPowScalarAndTensorSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(exponent=None, self=None, scalar=None, x=None, **kwargs):
        actual_scalar = self if self is not None else scalar
        actual_x = exponent if exponent is not None else x
        s = _to_scalar(actual_scalar)
        result = [torch.pow(torch.tensor(s, dtype=t.dtype), t) for t in actual_x]
        return list(result)

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
