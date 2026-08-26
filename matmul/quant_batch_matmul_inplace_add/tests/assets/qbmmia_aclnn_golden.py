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
"""ACLNN 层 golden 实现。

op: aclnnQuantBatchMatmulInplaceAdd
流程: torch → numpy → 调 kernel golden 做核心计算 → 返回结果
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import matmul_quant_util as _quant
from qbmmia_kernel_golden import QuantBatchMatmulInplaceAddTestSpec


def _to_np(*tensors):
    result = []
    for t in tensors:
        if t is None:
            result.append(None)
        elif isinstance(t, np.ndarray):
            result.append(t)
        else:
            result.append(_util.torch_to_numpy(t))
    return tuple(result)


def _customize_inputs_impl(x1, x2, x1ScaleOptional, x2Scale, yRef, **kwargs):
    testcase_name = kwargs.get("testcase_name", "unknown")
    input_ranges = kwargs.get("input_ranges", None)

    for idx, tensor in enumerate([x1, x2, x1ScaleOptional, x2Scale, yRef]):
        if tensor is None:
            continue
        cleaned = _quant.sanitize_e8m0_scale(tensor, idx, input_ranges, testcase_name)
        if cleaned is not tensor:
            _util.write_back(tensor, cleaned)


class AclnnQuantBatchMatmulInplaceAddTestSpec(QuantBatchMatmulInplaceAddTestSpec):
    @classmethod
    def golden(
        cls,
        x1,
        x2,
        x1ScaleOptional,
        x2Scale,
        yRef,
        transposeX1=False,
        transposeX2=False,
        groupSize=0,
        **kwargs,
    ):
        x1_np, x2_np, x1_scale_np, x2_scale_np, y_np = _to_np(
            x1, x2, x1ScaleOptional, x2Scale, yRef
        )

        # 检测非连续 tensor，恢复存储布局并修正 transpose
        import torch

        if isinstance(x1, torch.Tensor) and not x1.is_contiguous():
            x1_np = np.ascontiguousarray(x1_np.T)
            transposeX1 = True
        if isinstance(x2, torch.Tensor) and not x2.is_contiguous():
            x2_np = np.ascontiguousarray(x2_np.T)
            transposeX2 = True

        return super().golden(
            x1_np,
            x2_np,
            x2_scale_np,
            y_np,
            x1_scale_np,
            transpose_x1=transposeX1,
            transpose_x2=transposeX2,
            group_size=groupSize,
            **kwargs,
        )

    @classmethod
    def customize_inputs(
        cls,
        x1,
        x2,
        x1ScaleOptional,
        x2Scale,
        yRef,
        transposeX1=False,
        transposeX2=False,
        groupSize=0,
        **kwargs,
    ):
        _customize_inputs_impl(x1, x2, x1ScaleOptional, x2Scale, yRef, **kwargs)


__spec__ = {
    "aclnnQuantBatchMatmulInplaceAdd": "AclnnQuantBatchMatmulInplaceAddTestSpec",
}
