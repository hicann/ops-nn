#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""foreach_tan 三方 Golden — 正切。"""

__spec__ = {
    "foreach_tan": "ForeachTanKernelSpec",
    "ForeachTan": "ForeachTanKernelSpec",
    "aclnnForeachTan": "AclnnForeachTanSpec",
    "torch._foreach_tan": "TorchForeachTanSpec",
}

import numpy as np

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


_THIRD_PARTY = {"torch": "torch._foreach_tan"}


class ForeachTanKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x, **kwargs):
        import torch

        bf16NumpyDtype = None
        xList = []
        for t in x:
            a = np.ascontiguousarray(t)
            if a.dtype.name == "bfloat16":
                # bf16 无 numpy 原生 dtype，经 int16 位视图桥接到 torch
                # （reshape 回 t.shape：ascontiguousarray 会把 0 维数组升为 (1,)）
                bf16NumpyDtype = a.dtype
                xList.append(
                    torch.from_numpy(a.view(np.int16))
                    .view(torch.bfloat16)
                    .reshape(t.shape)
                )
            else:
                xList.append(torch.from_numpy(a).reshape(t.shape))
        result = torch._foreach_tan(xList)
        out = []
        for t in result:
            if t.dtype == torch.bfloat16:
                # 桥接回源 numpy bf16 dtype（复用输入侧 dtype 对象，保持位级一致）
                out.append(
                    t.contiguous().view(torch.int16).numpy().view(bf16NumpyDtype)
                )
            else:
                out.append(t.numpy())
        return out

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachTanSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachTanGetWorkspaceSize(x, out, workspaceSize, executor)
    out 槽按签名接收但仅作输出位透传: 参考值直接返回, 不回写
    out(保持高精度锚点, 且不向框架输出缓冲预置参考值)。
    """

    def golden(x, out=None, **kwargs):
        import torch

        return list(torch._foreach_tan(x))

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachTanSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, **kwargs):
        import torch

        return list(torch._foreach_tan(x))

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE
