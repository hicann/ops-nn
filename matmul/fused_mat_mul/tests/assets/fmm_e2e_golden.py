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
"""E2E-level golden for torch_npu.npu_fused_matmul."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util
from fmm_aclnn_golden import torch_fused_matmul_core


class TorchNpuFusedMatmulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(x1, x2, *, bias=None, x3=None, fused_op_type="", **kwargs):
        """torch_npu.npu_fused_matmul: y = FUSED_OP(x1 @ x2 + bias, x3)."""
        out_dtype = torch.float32 if fused_op_type == "16cast32" else x1.dtype
        return torch_fused_matmul_core(
            x1,
            x2,
            bias,
            x3,
            fused_op_type=fused_op_type,
            out_dtype=out_dtype,
            cube_math_type=None,
        )


__spec__ = {
    "torch_npu.npu_fused_matmul": "TorchNpuFusedMatmulTestSpec",
}
