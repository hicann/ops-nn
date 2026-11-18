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
"""foreach_pow_scalar 三方 Golden — 标量幂运算。"""

__spec__ = {
    "foreach_pow_scalar": "ForeachPowScalarKernelSpec",
    "aclnnForeachPowScalar": "AclnnForeachPowScalarSpec",
    "aclnnForeachPowScalarV2": "AclnnForeachPowScalarSpec",
    "torch._foreach_pow": "TorchForeachPowScalarSpec",
}

import ml_dtypes
import numpy as np
import torch

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


def _to_scalar(v):
    if isinstance(v, np.ndarray):
        return v.reshape(-1)[0].item()
    if isinstance(v, torch.Tensor):
        return v.reshape(-1)[0].item()
    return v


def _to_torch(t):
    if isinstance(t, torch.Tensor):
        return t, t.dtype
    a = np.ascontiguousarray(t)
    if a.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16), torch.bfloat16
    tensor = torch.from_numpy(a)
    return tensor, tensor.dtype


def _from_torch(t, dtype):
    if dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return t.to(dtype).numpy()


_INT_NP_DTYPES = {
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
}
_INT_DTYPES = tuple(_INT_NP_DTYPES)


def _int_pow(t, exp_scalar):
    """Integer powi per kernel PowiImpl: negative exponent -> 1/-1/0 by base, else wrapped fast power.

    The arithmetic is performed in the input dtype so that wraparound matches the
    input width (int32 for the current kernel, and the native dtype for others).
    """
    exponent = int(exp_scalar)
    np_dtype = _INT_NP_DTYPES[t.dtype]
    bases = t.to(torch.int64).cpu().numpy()
    if exponent < 0:
        res = np.where(
            bases == 1,
            1,
            np.where(bases == -1, 1 if exponent % 2 == 0 else -1, 0),
        ).astype(np_dtype)
        return torch.from_numpy(res).to(t.dtype)
    res = np.ones(bases.shape, dtype=np_dtype)
    b = bases.astype(np_dtype)
    e = exponent
    while e > 0:
        if e % 2 == 1:
            res = res * b
        b = b * b
        e //= 2
    return torch.from_numpy(res).to(t.dtype)


def _pow(t, base):
    if t.dtype in _INT_DTYPES:
        return _int_pow(t, base)
    return torch.pow(t, base)


class _TorchForeachPowScalarImpl:
    def __init__(self, scalar, **kwargs):
        self._exponent = _to_scalar(scalar)

    def __call__(self, x, **kwargs):
        return list(torch._foreach_pow(x, self._exponent))


class _TorchForeachPowScalarE2EImpl:
    def __init__(self, exponent, **kwargs):
        self._exponent = _to_scalar(exponent)

    def __call__(self, *args, exponent=None):
        return list(torch._foreach_pow(args[0], self._exponent))


_THIRD_PARTY = {"torch": _TorchForeachPowScalarImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachPowScalarE2EImpl}


class ForeachPowScalarKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x, scalar, **kwargs):
        x_list, out_dtypes = [], []
        for t in x:
            tensor, dtype = _to_torch(t)
            x_list.append(tensor)
            out_dtypes.append(dtype)
        base = _to_scalar(scalar)
        return [
            _from_torch(_pow(t, base), dtype) for t, dtype in zip(x_list, out_dtypes)
        ]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachPowScalarSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachPowScalarGetWorkspaceSize(x, scalar, out, workspaceSize, executor)
    """

    def golden(x, scalar, out=None, **kwargs):
        s = _to_scalar(scalar)
        result = [_pow(t, s) for t in x]
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachPowScalarSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, scalar=None, exponent=None, **kwargs):
        actual_exponent = exponent if exponent is not None else scalar
        s = _to_scalar(actual_exponent)
        return list(torch._foreach_pow(x, s))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
