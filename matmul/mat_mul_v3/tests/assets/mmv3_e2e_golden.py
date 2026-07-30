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
"""E2E 层 golden 实现。

委托 bmmv3_aclnn_golden 完成计算。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mmv3_aclnn_golden as _aclnn


class TorchMatmulTestSpec:
    @staticmethod
    def golden(input, other, *, out=None, **kwargs):
        """torch.matmul."""
        return _aclnn.AclnnMatmulTestSpec.golden(input, other, out=out, **kwargs)


class TorchMmTestSpec:
    @staticmethod
    def golden(input, mat2, *, out=None, **kwargs):
        """torch.mm."""
        return _aclnn.AclnnMatmulTestSpec.golden(input, mat2, out=out, **kwargs)


class TorchAddmmTestSpec:
    @staticmethod
    def golden(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None, **kwargs):
        """torch.addmm."""
        return _aclnn.AclnnAddmmTestSpec.golden(
            input, mat1, mat2, beta=beta, alpha=alpha, out=out, **kwargs
        )


__spec__ = {
    "torch.matmul": "TorchMatmulTestSpec",
    "torch.mm": "TorchMmTestSpec",
    "torch.addmm": "TorchAddmmTestSpec",
}
