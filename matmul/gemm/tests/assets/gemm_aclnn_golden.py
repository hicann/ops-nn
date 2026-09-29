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
"""ACLNN-level golden for aclnnGemm (Ascend950).

On Ascend950 aclnnGemm shares the aclnnAddmm computation graph:
    out = alpha * op(A) @ op(B) + beta * C,  op(X) = X.T if trans else X,
so the golden reuses AclnnAddmmTestSpec with pre-transposed inputs.
"""

__spec__ = {"aclnnGemm": "AclnnGemmTestSpec"}

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../mat_mul_v3/tests/assets")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
from mmv3_aclnn_golden import AclnnAddmmTestSpec


def _to_np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return _util.torch_to_numpy(tensor)


class AclnnGemmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        A,
        B,
        C,
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
        out=None,
        cubeMathType=0,
        **kwargs,
    ):
        a_np, b_np, c_np = _to_np(A), _to_np(B), _to_np(C)
        if int(transA):
            a_np = a_np.T
        if int(transB):
            b_np = b_np.T
        return AclnnAddmmTestSpec.golden(
            c_np,
            a_np,
            b_np,
            beta=float(beta),
            alpha=float(alpha),
            out=out,
            cubeMathType=cubeMathType,
            **kwargs,
        )
