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
"""foreach_maximum_list Golden — TensorList 逐元素最大值（竞品 torch._foreach_clamp_min）。"""

__spec__ = {
    "foreach_maximum_list": "ForeachMaximumListKernelSpec",
    "aclnnForeachMaximumList": "AclnnForeachMaximumListSpec",
    "torch._foreach_clamp_min": "TorchForeachClampMinListSpec",
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


def _compute(x1s, x2s):
    return list(torch._foreach_clamp_min(list(x1s), list(x2s)))


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
    """numpy(含 ml_dtypes.bfloat16) -> (torch.Tensor, torch.dtype)；已是 torch 则原样返回。"""
    if isinstance(t, torch.Tensor):
        return t, t.dtype
    a = np.ascontiguousarray(t)
    if a.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16), torch.bfloat16
    tensor = torch.from_numpy(a)
    return tensor, tensor.dtype


def _to_numpy(t, torch_dtype):
    if torch_dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return t.to(torch_dtype).numpy()


class _TorchForeachClampMinListImpl:
    """Kernel/GEIR/ACLNN 三方参考（服务端 GPU）：对标竞品 torch._foreach_clamp_min。"""

    def __call__(self, x1, x2, **kwargs):
        return _compute(x1, x2)


class _TorchForeachClampMinListE2EImpl:
    """E2E 三方参考：self/other 两个 TensorList 位置入参。

    TTK 服务端按位置传参（self 为 __call__ 绑定实例，不与 op 参数名冲突）；
    other 亦可用关键字传入。位置形态缺第二个参数时报错而非静默 IndexError。
    """

    def __call__(self, *args, other=None, **kwargs):
        if not args:
            raise TypeError("expected positional TensorList input")
        x1 = args[0]
        if other is None:
            if len(args) < 2:
                raise TypeError("expected 'other' as second positional or keyword")
            other = args[1]
        return _compute(x1, other)


_THIRD_PARTY = {"torch": _TorchForeachClampMinListImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachClampMinListE2EImpl}


class ForeachMaximumListKernelSpec:
    """Kernel/GEIR 路径 — (x1, x2: list[np.ndarray]) -> list[np.ndarray]。"""

    def golden(x1, x2, **kwargs):
        x1_list, x2_list, out_dtypes = [], [], []
        for a, b in zip(x1, x2):
            ta, da = _to_torch(a)
            tb, _ = _to_torch(b)
            x1_list.append(ta)
            x2_list.append(tb)
            out_dtypes.append(da)
        result = _compute(x1_list, x2_list)
        return [_to_numpy(r, dt) for r, dt in zip(result, out_dtypes)]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachMaximumListSpec:
    """ACLNN 路径 — aclnnForeachMaximumListGetWorkspaceSize(x1, x2, out, ...)。"""

    def golden(x1, x2, out=None, **kwargs):
        result = _compute(x1, x2)
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


class TorchForeachClampMinListSpec:
    """E2E 路径 — torch._foreach_clamp_min(self: TensorList, other: TensorList)。

    .List 重载 self/other 为位置参数，但 TTK e2e 按参数名装配，两种形态都兼容。
    """

    def golden(*args, self=None, other=None, **kwargs):
        x1 = self if self is not None else (args[0] if args else None)
        if other is None:
            other = args[1] if len(args) > 1 else kwargs.get("other")
        return _compute(x1, other)

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
