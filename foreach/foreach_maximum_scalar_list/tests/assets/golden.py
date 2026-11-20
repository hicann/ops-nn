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
"""foreach_maximum_scalar_list Golden — TensorList 与标量列表逐元素最大值（竞品 torch._foreach_clamp_min）。"""

__spec__ = {
    "foreach_maximum_scalar_list": "ForeachMaximumScalarListKernelSpec",
    "aclnnForeachMaximumScalarList": "AclnnForeachMaximumScalarListSpec",
    "torch._foreach_maximum": "TorchForeachMaximumScalarListSpec",
    "torch._foreach_maximum.ScalarList": "TorchForeachMaximumScalarListSpec",
    "torch._foreach_clamp_min": "TorchForeachMaximumScalarListSpec",
}

import numpy as np
import ml_dtypes
import torch

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


def _to_scalars(v):
    """kernel/GEIR 的 scalars 槽为张量/ndarray，拆出数值列表；python 列表原样。"""
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return v.reshape(-1).tolist()
    return list(v)


def _compute(xs, scalars):
    return list(torch._foreach_clamp_min(list(xs), list(scalars)))


def _copy_to_out(o, r):
    """写回 out 视图。torch copy_ 拒绝 stride=0 广播视图：广播轴钉 0 号切片、
    其余轴 slice，组合索引一次批量写回（物理存储唯一）；空张量直接返回。"""
    if o.numel() == 0:
        return
    if not any(s == 0 for s in o.stride()):
        o.copy_(r)
        return
    idx = tuple(0 if s == 0 else slice(None) for s in o.stride())
    o[idx].copy_(r[idx])


def _to_torch(t):
    if isinstance(t, torch.Tensor):
        return t, t.dtype
    a = np.ascontiguousarray(t)
    if a.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16), torch.bfloat16
    tensor = torch.from_numpy(a)
    return tensor, tensor.dtype


class _TorchForeachClampMinScalarListImpl:
    """Kernel/GEIR/ACLNN 三方参考（服务端 GPU）：对标竞品 torch._foreach_clamp_min。"""

    def __init__(self, scalars, **kwargs):
        self._scalars = _to_scalars(scalars)

    def __call__(self, x, **kwargs):
        return _compute(list(x), self._scalars)


class _TorchForeachClampMinScalarListE2EImpl:
    """E2E 三方参考：self TensorList 位置入参 + scalars 关键字。"""

    def __call__(self, *args, scalars=None, **kwargs):
        if scalars is None:
            scalars = kwargs.get("scalars")
        return _compute(list(args[0]), _to_scalars(scalars))


_THIRD_PARTY = {"torch": _TorchForeachClampMinScalarListImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachClampMinScalarListE2EImpl}


class ForeachMaximumScalarListKernelSpec:
    """Kernel/GEIR 路径 — (x: list[np.ndarray], scalars: np.ndarray) -> list[np.ndarray]。"""

    def golden(x, scalars, **kwargs):
        s_list = _to_scalars(scalars)
        x_list, out_dtypes = [], []
        for t in x:
            tt, dt = _to_torch(t)
            x_list.append(tt)
            out_dtypes.append(dt)
        result = _compute(x_list, s_list)
        out = []
        for r, dt in zip(result, out_dtypes):
            if dt == torch.bfloat16:
                out.append(r.view(torch.uint16).numpy().view(ml_dtypes.bfloat16))
            else:
                out.append(r.to(dt).numpy())
        return out

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachMaximumScalarListSpec:
    """ACLNN 路径 — aclnnForeachMaximumScalarListGetWorkspaceSize(x, scalars, out, ...)。"""

    def golden(x, scalars, out=None, **kwargs):
        result = _compute(list(x), _to_scalars(scalars))
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
                _copy_to_out(o, r)
            return list(out)
        return result

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachMaximumScalarListSpec:
    """E2E 路径 — torch._foreach_clamp_min(self: TensorList, scalars: Scalar[])。"""

    def golden(*args, self=None, scalars=None, **kwargs):
        x = self if self is not None else (args[0] if args else None)
        if scalars is None:
            scalars = args[1] if len(args) > 1 else kwargs.get("scalars")
        return _compute(list(x), _to_scalars(scalars))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
