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
"""foreach_log10 三方 Golden — 常用对数。"""

__spec__ = {
    "foreach_log10": "ForeachLog10KernelSpec",
    # kernel CaseSet 的 op_name 使用 GE 算子名（ForeachLog10），与 opInterface 名（foreach_log10）并存注册
    "ForeachLog10": "ForeachLog10KernelSpec",
    "aclnnForeachLog10": "AclnnForeachLog10Spec",
    "torch._foreach_log10": "TorchForeachLog10Spec",
}

import inspect as _inspect

import numpy as np
import torch

try:
    import ml_dtypes

    _BF16 = ml_dtypes.bfloat16
except ImportError:
    _BF16 = None

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _numpy_to_torch(t):
    # np.ascontiguousarray 会把 rank-0 提升为 1 维，破坏 rank-0 输入的 shape 契约（InferShape 原样复制）；
    # 仅在 ndim>0 时做连续化。
    a = np.asarray(t)
    if a.ndim > 0:
        a = np.ascontiguousarray(a)
    if a.dtype.kind == "V" and _BF16 is not None:
        a = a.view(_BF16)
    if _BF16 is not None and a.dtype == _BF16:
        # bf16 载体 ml_dtypes/tf 的 void dtype torch 不接受：按 fp32 计算、结果还原 bf16
        return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(a)


def _torch_to_numpy(t):
    if t.dtype == torch.bfloat16:
        arr = t.to(torch.float32).numpy()
        return arr.astype(_BF16) if _BF16 is not None else arr
    return t.numpy()


def _remote_third_party(x, **kwargs):
    # Kernel/GEIR 通路 third_party 输入容器为 numpy（与配对 golden 一致）：先做 numpy→torch 转换；
    # 已是 torch.Tensor 的容器（ACLNN/E2E 通路）原样透传。
    return list(
        torch._foreach_log10(
            [t if isinstance(t, torch.Tensor) else _numpy_to_torch(t) for t in x]
        )
    )


class _ForeachLog10E2ECompose:
    def __call__(self, x, **kwargs):
        return list(torch._foreach_log10(x))


class _TorchForeachLog10E2EImpl:
    def __call__(self, *args, **kwargs):
        return _ForeachLog10E2ECompose()(*args, **kwargs)


# TTK 服务端按 __call__ 的签名做入参绑定，且按 torch API 的形参名下发张量池：
# torch._foreach_log10 的首参名为 'self'（aten 惯例）。包装类方法接收者会被 inspect 剥离，
# 故显式构造带残留 'self' 形参的签名（首参 receiver 在绑定时被剥离），使池中的 'self' 正确对位。
try:
    _TorchForeachLog10E2EImpl.__call__.__signature__ = _inspect.Signature(
        [
            _inspect.Parameter("receiver", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
            _inspect.Parameter("self", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
            _inspect.Parameter("kwargs", _inspect.Parameter.VAR_KEYWORD),
        ]
    )
except (ValueError, TypeError):  # 无法构造时保持原样
    pass


_THIRD_PARTY = {"torch": _remote_third_party}
_E2E_THIRD_PARTY = {"torch": _TorchForeachLog10E2EImpl}


class ForeachLog10KernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x, **kwargs):
        x_list = [_numpy_to_torch(t) for t in x]
        result = torch._foreach_log10(x_list)
        return [_torch_to_numpy(t) for t in result]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachLog10Spec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachLog10GetWorkspaceSize(x, out, workspaceSize, executor)
    """

    def golden(x, out=None, **kwargs):
        result = torch._foreach_log10(x)
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachLog10Spec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, **kwargs):
        return list(torch._foreach_log10(x))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
