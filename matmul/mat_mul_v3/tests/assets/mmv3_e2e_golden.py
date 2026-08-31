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
"""E2E golden specs: torch API tests delegate to the aclnn golden implementations."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import mmv3_aclnn_golden as _aclnn


class TorchMatmulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(input, other, *, out=None, **kwargs):
        return _aclnn.AclnnMatmulTestSpec.golden(input, other, out=out, **kwargs)


class TorchMmTestSpec(TorchMatmulTestSpec):
    """torch.mm delegates to the same aclnn mm golden."""


class TorchAddmmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None, **kwargs):
        return _aclnn.AclnnAddmmTestSpec.golden(
            input, mat1, mat2, beta=beta, alpha=alpha, out=out, **kwargs
        )


__spec__ = {
    "torch.matmul": "TorchMatmulTestSpec",
    "torch.mm": "TorchMmTestSpec",
    "torch.addmm": "TorchAddmmTestSpec",
}
