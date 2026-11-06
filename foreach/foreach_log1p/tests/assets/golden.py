#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""foreach_log1p 三方 Golden — log(1+x)。"""

__spec__ = {
    "foreach_log1p": "ForeachLog1pKernelSpec",
    "ForeachLog1p": "ForeachLog1pKernelSpec",
    "aclnnForeachLog1p": "AclnnForeachLog1pSpec",
    "torch._foreach_log1p": "TorchForeachLog1pSpec",
}

import numpy as np

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_THIRD_PARTY = {"torch": "torch._foreach_log1p"}


class ForeachLog1pKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x, **kwargs):
        import torch

        # 0 维数组 ascontiguousarray 会升维为 (1,),此处按原 shape 传递
        x_list = [
            torch.from_numpy(np.ascontiguousarray(t) if t.ndim > 0 else t) for t in x
        ]
        result = torch._foreach_log1p(x_list)
        return [t.numpy() for t in result]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachLog1pSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachLog1pGetWorkspaceSize(x, out, workspaceSize, executor)
    out 槽按签名接收但仅作输出位透传: 参考值直接返回升档精度结果, 不回写
    out(保持高精度锚点, 且不向框架输出缓冲预置参考值)。
    """

    def golden(x, out=None, **kwargs):
        import torch

        return list(torch._foreach_log1p(x))

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachLog1pSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, **kwargs):
        import torch

        return list(torch._foreach_log1p(x))

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE
