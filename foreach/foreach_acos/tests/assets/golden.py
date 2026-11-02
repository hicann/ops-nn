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
"""foreach_acos 三方 Golden — 反余弦。"""

__spec__ = {
    "foreach_acos": "ForeachAcosKernelSpec",
    "aclnnForeachAcos": "AclnnForeachAcosSpec",
    "torch._foreach_acos": "TorchForeachAcosSpec",
}

import numpy as np
import torch

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _remote_third_party(x, **kwargs):
    return list(torch._foreach_acos(x))


class _TorchForeachAcosE2EImpl:
    def __call__(self, *args, **kwargs):
        return list(torch._foreach_acos(args[0]))


_THIRD_PARTY = {"torch": _remote_third_party}
_E2E_THIRD_PARTY = {"torch": _TorchForeachAcosE2EImpl}


class ForeachAcosKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x, **kwargs):
        x_list = [torch.from_numpy(np.ascontiguousarray(t)) for t in x]
        result = torch._foreach_acos(x_list)
        return [t.numpy() for t in result]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachAcosSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachAcosGetWorkspaceSize(x, out, workspaceSize, executor)
    """

    def golden(x, out=None, **kwargs):
        result = torch._foreach_acos(x)
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachAcosSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, **kwargs):
        return list(torch._foreach_acos(x))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
