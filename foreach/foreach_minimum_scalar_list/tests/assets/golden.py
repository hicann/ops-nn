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
"""foreach_minimum_scalar_list 三方 Golden — 标量列表最小值。

逐对逐元素语义: y_i = min(x_i, scalars_i), i 为张量列表索引, 每个标量 0-d
广播到所配对张量全部元素位。比较语义: 等值 tie 取 x 侧(x <= scalars 时取 x,
含 ±0 tie); NaN 任一侧传播(x 元素为 NaN → 对应位 NaN; 配对标量为 NaN →
该张量全部元素 NaN); fp16/bf16 输入按升精度(float)比较后 RNE 回写 x.dtype,
与标量先 RNE cast 到输入精度再比较逐位等价; int32 输入按 int64 比较后回写。
"""

__spec__ = {
    "foreach_minimum_scalar_list": "ForeachMinimumScalarListKernelSpec",
    "ForeachMinimumScalarList": "ForeachMinimumScalarListKernelSpec",
    "aclnnForeachMinimumScalarList": "AclnnForeachMinimumScalarListSpec",
    "torch._foreach_minimum": "TorchForeachMinimumScalarListSpec",
    "torch._foreach_minimum.ScalarList": "TorchForeachMinimumScalarListSpec",
}

import numpy as np

_TOLERANCE = {
    "float16": {"standard": "binary_equal", "level": "L1"},
    "float32": {"standard": "binary_equal", "level": "L1"},
    "bfloat16": {"standard": "binary_equal", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}


def _to_scalar(v):
    if isinstance(v, np.ndarray):
        return v.reshape(-1)[0].item()
    import torch

    if isinstance(v, torch.Tensor):
        return v.reshape(-1)[0].item()
    return v


def _to_scalars(v):
    if isinstance(v, np.ndarray):
        return v.reshape(-1).tolist()
    import torch

    if isinstance(v, torch.Tensor):
        return v.reshape(-1).tolist()
    return [_to_scalar(t) for t in v]


def _foreach_minimum(x, scalars):
    import torch

    s_list = _to_scalars(scalars)
    if x and all(t.dtype == torch.int32 for t in x):
        result = torch._foreach_minimum([t.to(torch.int64) for t in x], s_list)
        return [t.to(torch.int32) for t in result]
    return list(torch._foreach_minimum(x, s_list))


class _TorchForeachMinimumScalarListImpl:
    def __init__(self, x, scalars, **kwargs):
        import torch

        self._scalars = _to_scalars(scalars)
        self._is_int32 = bool(x) and all(t.dtype == torch.int32 for t in x)
        self._x = [t.to(torch.int64) for t in x] if self._is_int32 else x

    def __call__(self, **kwargs):
        import torch

        result = torch._foreach_minimum(self._x, self._scalars)
        return [t.to(torch.int32) for t in result] if self._is_int32 else list(result)


class _TorchForeachMinimumScalarListE2EImpl:
    def __init__(self, *args, scalars=None, **kwargs):
        import torch

        x = args[0]
        self._scalars = _to_scalars(scalars)
        self._is_int32 = bool(x) and all(t.dtype == torch.int32 for t in x)
        self._x = [t.to(torch.int64) for t in x] if self._is_int32 else x

    def __call__(self, *args, scalars=None):
        import torch

        result = torch._foreach_minimum(self._x, self._scalars)
        return [t.to(torch.int32) for t in result] if self._is_int32 else list(result)


_THIRD_PARTY = {"torch": _TorchForeachMinimumScalarListImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachMinimumScalarListE2EImpl}


class ForeachMinimumScalarListKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x, scalars, **kwargs):
        import torch

        x_list = []
        for t in x:
            arr = np.ascontiguousarray(t)
            if arr.dtype.name == "bfloat16":
                # bf16 无 numpy 原生 dtype: uint16 位视图携带位型投递 torch(不经 f32 中转)
                x_list.append(
                    torch.from_numpy(arr.view("uint16"))
                    .view(torch.bfloat16)
                    .reshape(t.shape)
                )
            else:
                x_list.append(torch.from_numpy(arr).reshape(t.shape))
        result = _foreach_minimum(x_list, scalars)
        # 输出侧对称: bf16 经 uint16 位视图回 numpy, 按输入数组自带 dtype 还原位型
        return [
            r.view(torch.uint16).numpy().view(t.dtype)
            if r.dtype == torch.bfloat16
            else r.numpy()
            for r, t in zip(result, x)
        ]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachMinimumScalarListSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachMinimumScalarListGetWorkspaceSize(x, scalars, out, workspaceSize, executor)
    """

    def golden(x, scalars, out=None, **kwargs):
        result = _foreach_minimum(x, scalars)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachMinimumScalarListSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, scalars, **kwargs):
        return _foreach_minimum(x, scalars)

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
