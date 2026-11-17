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
"""foreach_maximum_scalar Golden — TensorList 与标量逐元素最大值（竞品 torch._foreach_clamp_min）。"""

__spec__ = {
    "foreach_maximum_scalar": "ForeachMaximumScalarKernelSpec",
    "aclnnForeachMaximumScalar": "AclnnForeachMaximumScalarSpec",
    "aclnnForeachMaximumScalarV2": "AclnnForeachMaximumScalarSpec",
    "torch._foreach_clamp_min": "TorchForeachClampMinScalarSpec",
}

import numpy as np
import ml_dtypes
import torch

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
}
# int32 未声明 -> TTK dtype 路由自动 binary_equal


def _to_scalar(v):
    """v1/kernel/GEIR 的标量为 0-d/1 元素张量或 ndarray，拆出数值；python 值原样。"""
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return v.reshape(-1)[0].item()
    return v


def _compute(xs, scalar):
    return list(torch._foreach_clamp_min(list(xs), scalar))


def _to_torch(t):
    if isinstance(t, torch.Tensor):
        return t, t.dtype
    a = np.ascontiguousarray(t)
    if a.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16), torch.bfloat16
    tensor = torch.from_numpy(a)
    return tensor, tensor.dtype


class _TorchForeachClampMinScalarImpl:
    """Kernel/GEIR/ACLNN 三方参考（服务端 GPU）：对标竞品 torch._foreach_clamp_min。"""

    def __init__(self, scalar, **kwargs):
        self._scalar = _to_scalar(scalar)

    def __call__(self, x, **kwargs):
        return _compute(x, self._scalar)


class _TorchForeachClampMinScalarE2EImpl:
    """E2E 三方参考：self TensorList 位置入参 + scalar 关键字。"""

    def __call__(self, *args, scalar=None, **kwargs):
        if scalar is None:
            scalar = kwargs.get("scalar")
        return _compute(args[0], _to_scalar(scalar))


_THIRD_PARTY = {"torch": _TorchForeachClampMinScalarImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachClampMinScalarE2EImpl}


class ForeachMaximumScalarKernelSpec:
    """Kernel/GEIR 路径 — (x: list[np.ndarray], scalar: np.ndarray) -> list[np.ndarray]。"""

    def golden(x, scalar, **kwargs):
        x_list, out_dtypes = [], []
        for t in x:
            tt, dt = _to_torch(t)
            x_list.append(tt)
            out_dtypes.append(dt)
        result = _compute(x_list, _to_scalar(scalar))
        out = []
        for r, dt in zip(result, out_dtypes):
            if dt == torch.bfloat16:
                out.append(r.view(torch.uint16).numpy().view(ml_dtypes.bfloat16))
            else:
                out.append(r.to(dt).numpy())
        return out

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachMaximumScalarSpec:
    """ACLNN 路径 — v1: (x, scalar: aclTensor*, out)；V2: (x, scalar: aclScalar*, out)。"""

    def golden(x, scalar, out=None, **kwargs):
        result = _compute(list(x), _to_scalar(scalar))
        if out is not None:
            if len(out) != len(result):
                raise ValueError(
                    f"out length {len(out)} != result length {len(result)}"
                )
            for o, r in zip(out, result):
                if o.numel() != r.numel():
                    raise ValueError(
                        f"out numel {o.numel()} != result numel {r.numel()}"
                    )
                o.copy_(r)
            return list(out)
        return result

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachClampMinScalarSpec:
    """E2E 路径 — torch._foreach_clamp_min(self: TensorList, scalar: Scalar)。"""

    def golden(*args, self=None, scalar=None, **kwargs):
        x = self if self is not None else (args[0] if args else None)
        if scalar is None:
            scalar = args[1] if len(args) > 1 else kwargs.get("scalar")
        return _compute(list(x), _to_scalar(scalar))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
