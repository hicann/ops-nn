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
"""foreach_minimum_list 三方 Golden — 逐元素最小值。"""

__spec__ = {
    "foreach_minimum_list": "ForeachMinimumListKernelSpec",
    "aclnnForeachMinimumList": "AclnnForeachMinimumListSpec",
    "torch._foreach_minimum": "TorchForeachMinimumListSpec",
}

import inspect

import numpy as np
import torch


_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


def _to_torch_bf16_safe(t):
    """numpy → torch，兼容裸 bf16 槽。

    torch.from_numpy 不支持 ml_dtypes.bfloat16 数组；裸 bf16 槽（比对判据不
    触发框架浮点抬升时）经 uint16 位视图转入 torch.bfloat16，值逐位保持。
    torch.Tensor 直通（ACLNN/E2E 通路）。"""
    if not isinstance(t, np.ndarray):
        return t
    if "bfloat16" in str(t.dtype):
        return (
            torch.from_numpy(np.ascontiguousarray(t).view(np.uint16))
            .view(torch.bfloat16)
            .reshape(t.shape)
        )
    return torch.from_numpy(np.ascontiguousarray(t)).reshape(t.shape)


def _to_numpy_bf16_safe(t, like):
    """torch → numpy，bfloat16 结果经 uint16 位视图回投为 like.dtype。

    like 为同 dtype 的输入数组（bf16 结果仅在裸 bf16 输入下产生，其 dtype
    对象即正确目标）；其余 dtype 走 t.numpy()。"""
    if t.dtype == torch.bfloat16 and "bfloat16" in str(getattr(like, "dtype", "")):
        return t.contiguous().view(torch.uint16).numpy().view(like.dtype)
    return t.numpy()


def _remote_third_party(x1, x2, **kwargs):
    """Kernel/ACLNN 通路的三方实现，参数名与算子接口保持一致。

    Kernel/GEIR 通路到达为 list[np.ndarray]，而 torch._foreach_minimum 只接受
    Tensor 列表；先经位保持桥接转入 torch（裸 bf16 槽 torch.from_numpy 不支
    持），否则 cross_check 会在三方入口抛 TypeError 被判为 third_party 缺失。
    返回的 Tensor 列表由测试框架统一转回 numpy 参与比对。"""
    x1_list = [_to_torch_bf16_safe(t) for t in x1]
    x2_list = [_to_torch_bf16_safe(t) for t in x2]
    return list(torch._foreach_minimum(x1_list, x2_list))


class _TorchForeachMinimumListE2EImpl:
    """E2E 三方实现。

    torch aten schema 使用 ``self``/``other`` 作为输入名，而 kernel/ACLNN
    golden 使用 ``x1``/``x2``。TTK 会按 ``__call__`` 签名绑定输入，因此显式
    声明一个可绑定的签名，再按位置转发，避免 x2 被当成 kwargs 丢失。
    """

    def __call__(self, *args, **kwargs):
        return _remote_third_party(*args, **kwargs)


# inspect.signature(bound_method) 会剥离首个 receiver；剩余的 self/other
# 正好对应 TTK E2E torch API 下发的 aten 输入名。
_TorchForeachMinimumListE2EImpl.__call__.__signature__ = inspect.Signature(
    [
        inspect.Parameter("receiver", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("other", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
    ]
)


_THIRD_PARTY = {"torch": _remote_third_party}
_E2E_THIRD_PARTY = {"torch": _TorchForeachMinimumListE2EImpl}


class ForeachMinimumListKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x1, x2, **kwargs):
        x1_list = [_to_torch_bf16_safe(t) for t in x1]
        x2_list = [_to_torch_bf16_safe(t) for t in x2]
        result = torch._foreach_minimum(x1_list, x2_list)
        return [_to_numpy_bf16_safe(r, o) for r, o in zip(result, x1)]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachMinimumListSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x1, x2, out=None, **kwargs):
        result = torch._foreach_minimum(x1, x2)
        if out is not None:
            for output, expected in zip(out, result):
                output.copy_(expected)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachMinimumListSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x1, x2, **kwargs):
        return list(torch._foreach_minimum(x1, x2))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
