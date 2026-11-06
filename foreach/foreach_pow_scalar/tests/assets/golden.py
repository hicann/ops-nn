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
    "torch._foreach_pow": "TorchForeachPowScalarSpec",
}

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
        x_list = [torch.from_numpy(np.ascontiguousarray(t)) for t in x]
        s = _to_scalar(scalar)
        result = torch._foreach_pow(x_list, s)
        return [t.numpy() for t in result]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachPowScalarSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachPowScalarGetWorkspaceSize(x, scalar, out, workspaceSize, executor)
    """

    def golden(x, scalar, out=None, **kwargs):
        s = _to_scalar(scalar)
        result = torch._foreach_pow(x, s)
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
