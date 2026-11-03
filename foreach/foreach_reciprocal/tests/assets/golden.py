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
"""foreach_reciprocal 三方 Golden — 倒数。"""

__spec__ = {
    "foreach_reciprocal": "ForeachReciprocalKernelSpec",
    # kernel CaseSet 的 op_name 使用 GE 算子名（ForeachReciprocal），与 opInterface 名（foreach_reciprocal）并存注册
    "ForeachReciprocal": "ForeachReciprocalKernelSpec",
    "aclnnForeachReciprocal": "AclnnForeachReciprocalSpec",
    "torch._foreach_reciprocal": "TorchForeachReciprocalSpec",
}

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
    # bf16 载体 ml_dtypes/tf 的 void dtype torch 不接受：按 fp32 计算、结果还原 bf16
    a = np.asarray(t)
    if a.dtype.kind == "V" and _BF16 is not None:
        a = a.view(_BF16)
    if _BF16 is not None and a.dtype == _BF16:
        return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(a)


def _torch_to_numpy(t):
    if t.dtype == torch.bfloat16:
        arr = t.to(torch.float32).numpy()
        return arr.astype(_BF16) if _BF16 is not None else arr
    return t.numpy()


def _remote_third_party(x, **kwargs):
    return list(torch._foreach_reciprocal(x))


class _TorchForeachReciprocalE2EImpl:
    def __call__(self, *args, **kwargs):
        return _remote_third_party(*args, **kwargs)


_THIRD_PARTY = {"torch": _remote_third_party}
_E2E_THIRD_PARTY = {"torch": _TorchForeachReciprocalE2EImpl}


class ForeachReciprocalKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x, **kwargs):
        x_list = [_numpy_to_torch(t) for t in x]
        result = torch._foreach_reciprocal(x_list)
        return [_torch_to_numpy(t) for t in result]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachReciprocalSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachReciprocalGetWorkspaceSize(x, out, workspaceSize, executor)
    """

    def golden(x, out=None, **kwargs):
        result = torch._foreach_reciprocal(x)
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachReciprocalSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, **kwargs):
        return list(torch._foreach_reciprocal(x))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
