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
"""E2E-level golden for TransposeBatchMatMul.

Delegates core computation to ``tbmm_aclnn_golden``.
Handles torch API parameter name mapping (perm_x1 -> permX1, etc.)
and scale output dtype conversion (int8 -> float16, matching torch API eager behavior).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util
import tbmm_aclnn_golden as _aclnn


class TorchNpuNpuTransposeBatchmatmulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def customize_inputs(
        input,
        weight,
        *,
        bias=None,
        scale=None,
        perm_x1=None,
        perm_x2=None,
        perm_y=None,
        batch_split_factor=1,
        **kwargs,
    ):
        import torch

        if (
            scale is not None
            and hasattr(scale, "dtype")
            and scale.dtype in (torch.int64, torch.int32)
        ):
            scale_shape = scale.shape
            fp32_scale = np.random.uniform(low=-5, high=5, size=scale_shape).astype(
                np.float32
            )
            u32 = np.ascontiguousarray(fp32_scale).view(np.uint32).copy()
            u32 &= np.uint32(0xFFFFE000)
            new_scale = np.zeros(scale_shape, np.uint64)
            new_scale |= u32.astype(np.uint64)
            new_scale |= np.uint64(1 << 46)
            scale.copy_(
                torch.from_numpy(new_scale.astype(np.int64).copy()).to(scale.device)
            )

    @staticmethod
    def golden(
        input,
        weight,
        *,
        bias=None,
        scale=None,
        perm_x1=None,
        perm_x2=None,
        perm_y=None,
        batch_split_factor=1,
        **kwargs,
    ):
        result = _aclnn.AclnnTransposeBatchMatMulTestSpec.golden(
            input,
            weight,
            bias=bias,
            scale=scale,
            permX1=perm_x1,
            permX2=perm_x2,
            permY=perm_y,
            cubeMathType=0,
            batchSplitFactor=batch_split_factor,
            **kwargs,
        )
        out = np.asarray(result[0])
        if out.dtype == np.int8:
            out = out.astype(np.float16)
        return [out]


__spec__ = {
    "torch_npu.npu_transpose_batchmatmul": "TorchNpuNpuTransposeBatchmatmulTestSpec",
}
