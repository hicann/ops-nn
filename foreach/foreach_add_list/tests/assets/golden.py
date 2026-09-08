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
"""foreach_add_list 三方 Golden — 逐元素加法(alpha系数)。"""

__spec__ = {
    "foreach_add_list": "ForeachAddListKernelSpec",
    "aclnnForeachAddList": "AclnnForeachAddListSpec",
    "torch._foreach_add": "TorchForeachAddListSpec",
    "torch._foreach_add.List": "TorchForeachAddListSpec",
}

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


def _to_scalar(v):
    if isinstance(v, np.ndarray):
        return v.reshape(-1)[0].item()
    if isinstance(v, torch.Tensor):
        return v.reshape(-1)[0].item()
    return v


def _reduce_alpha(dtype, alpha):
    s = str(dtype)
    if "uint8" in s:
        return int(alpha % 256)
    if "int8" in s:
        return int(((alpha + 128) % 256) - 128)
    if "int16" in s:
        return int(((alpha + 32768) % 65536) - 32768)
    return alpha


def _tp_foreach_add_list(x1, x2, alpha=1.0, **kwargs):
    return list(
        torch._foreach_add(x1, x2, alpha=_reduce_alpha(x1[0].dtype, _to_scalar(alpha)))
    )


_THIRD_PARTY = {"torch": _tp_foreach_add_list}


class ForeachAddListKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x1, x2, alpha=1.0, **kwargs):
        x1_list = [torch.from_numpy(np.ascontiguousarray(t)) for t in x1]
        x2_list = [torch.from_numpy(np.ascontiguousarray(t)) for t in x2]
        result = torch._foreach_add(
            x1_list, x2_list, alpha=_reduce_alpha(x1_list[0].dtype, _to_scalar(alpha))
        )
        return [t.numpy() for t in result]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachAddListSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachAddListGetWorkspaceSize(x1, x2, alpha=1.0, out, workspaceSize, executor)
    """

    def golden(x1, x2, alpha=1.0, out=None, **kwargs):
        result = torch._foreach_add(
            x1, x2, alpha=_reduce_alpha(x1[0].dtype, _to_scalar(alpha))
        )
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachAddListSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x1, x2, alpha=1.0, **kwargs):
        return list(
            torch._foreach_add(
                x1, x2, alpha=_reduce_alpha(x1[0].dtype, _to_scalar(alpha))
            )
        )

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE
