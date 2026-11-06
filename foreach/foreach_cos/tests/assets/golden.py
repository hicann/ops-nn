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
"""foreach_cos 三方 Golden — Kernel/ACLNN/E2E 三路径。"""

__spec__ = {
    "foreach_cos": "ForeachCosKernelSpec",
    "aclnnForeachCos": "AclnnForeachCosSpec",
    "torch._foreach_cos": "TorchForeachCosSpec",
}

import numpy as np
import torch
from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
}


def _remote_third_party(x, **kwargs):
    return list(torch._foreach_cos(x))


class _TorchForeachCosE2EImpl:
    def __call__(self, *args, **kwargs):
        return _remote_third_party(*args, **kwargs)


_THIRD_PARTY = {"torch": _remote_third_party}
_E2E_THIRD_PARTY = {"torch": _TorchForeachCosE2EImpl}


class ForeachCosKernelSpec:
    def golden(x, **kwargs):
        x_list = [numpy_to_torch_tensor(np.ascontiguousarray(t)) for t in x]
        result = torch._foreach_cos(x_list)
        return [torch_to_numpy_tensor(t) for t in result]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachCosSpec:
    def golden(x, out=None, **kwargs):
        result = torch._foreach_cos(x)
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachCosSpec:
    def golden(x, **kwargs):
        return list(torch._foreach_cos(x))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
