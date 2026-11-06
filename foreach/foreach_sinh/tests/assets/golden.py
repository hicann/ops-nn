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
"""foreach_sinh 三方 Golden — 双曲正弦。"""

__spec__ = {
    "foreach_sinh": "ForeachSinhKernelSpec",
    "aclnnForeachSinh": "AclnnForeachSinhSpec",
    "torch._foreach_sinh": "TorchForeachSinhSpec",
}

import numpy as np
import torch
from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor


def _sinh_high_precision(tensors):
    # PyTorch CPU vectorized float32 sinh overflows at ~89.0 instead of the
    # mathematically correct asinh(FLT_MAX)~=89.416. Use float64 as the
    # reference and cast back so the overflow boundary matches the operator.
    return [torch.sinh(t.to(torch.float64)).to(t.dtype) for t in tensors]


_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
}


def _remote_third_party(x, **kwargs):
    return list(torch._foreach_sinh(x))


class _TorchForeachSinhE2EImpl:
    def __call__(self, *args, **kwargs):
        return _remote_third_party(*args, **kwargs)


_THIRD_PARTY = {"torch": _remote_third_party}
_E2E_THIRD_PARTY = {"torch": _TorchForeachSinhE2EImpl}


class ForeachSinhKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x, **kwargs):
        x_list = [numpy_to_torch_tensor(np.ascontiguousarray(t)) for t in x]
        result = _sinh_high_precision(x_list)
        return [torch_to_numpy_tensor(t) for t in result]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachSinhSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachSinhGetWorkspaceSize(x, out, workspaceSize, executor)
    """

    def golden(x, out=None, **kwargs):
        result = _sinh_high_precision(x)
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachSinhSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, **kwargs):
        return list(torch._foreach_sinh(x))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
