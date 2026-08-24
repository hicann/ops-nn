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
"""E2E-level goldens for torch bmm-family APIs.

Delegates to bmmv3_aclnn_golden for computation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import bmmv3_aclnn_golden as _aclnn


class TorchMatmulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(input_tensor, other, *, out=None, **kwargs):
        """torch.matmul."""
        return _aclnn.AclnnBatchMatMulTestSpec.golden(
            input_tensor, other, out=out, **kwargs
        )


class TorchBmmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(input_tensor, mat2, *, out=None, **kwargs):
        """torch.bmm."""
        return _aclnn.AclnnBatchMatMulTestSpec.golden(
            input_tensor, mat2, out=out, **kwargs
        )


class TorchAddbmmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        input_tensor, batch1, batch2, *, beta=1.0, alpha=1.0, out=None, **kwargs
    ):
        """torch.addbmm."""
        return _aclnn.AclnnAddbmmTestSpec.golden(
            input_tensor, batch1, batch2, beta=beta, alpha=alpha, out=out, **kwargs
        )


class TorchBaddbmmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        input_tensor, batch1, batch2, *, beta=1.0, alpha=1.0, out=None, **kwargs
    ):
        """torch.baddbmm."""
        return _aclnn.AclnnBaddbmmTestSpec.golden(
            input_tensor, batch1, batch2, beta=beta, alpha=alpha, out=out, **kwargs
        )


__spec__ = {
    "torch.matmul": "TorchMatmulTestSpec",
    "torch.bmm": "TorchBmmTestSpec",
    "torch.addbmm": "TorchAddbmmTestSpec",
    "torch.baddbmm": "TorchBaddbmmTestSpec",
}
