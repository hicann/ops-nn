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

import en_dtypes
import matmul_golden_util as _util
from qbmmia_kernel_golden import QuantBatchMatmulInplaceAddTestSpec


def _is_fp4_tensor(t):
    """Check if t is a torch.float4_e2m1fn_x2 tensor."""
    import torch

    return (
        isinstance(t, torch.Tensor)
        and hasattr(torch, "float4_e2m1fn_x2")
        and t.dtype == torch.float4_e2m1fn_x2
    )


def _unpack_fp4_numpy(raw_uint8):
    """Unpack packed uint8 (2 FP4/byte) → en_dtypes.float4_e2m1 (1 FP4/byte).

    Unpacks along the last dimension: each byte → 2 FP4 values
    (low nibble first, high nibble second).
    """
    out = np.empty((raw_uint8.shape[0], raw_uint8.shape[1] * 2), dtype=np.uint8)
    out[:, 0::2] = raw_uint8 & 0x0F
    out[:, 1::2] = (raw_uint8 >> 4) & 0x0F
    return out.view(en_dtypes.float4_e2m1)


def _fp4_to_np(t):
    """Convert torch.float4_e2m1fn_x2 → en_dtypes.float4_e2m1 numpy (storage layout).

    1. view(uint8).numpy() → packed uint8 (preserves non-contiguous strides)
    2. Detect the transposed packed view from strides, including singleton K
    3. Unpack along last dim: (K, M_packed) → (K, M) = storage layout
    """
    import torch

    raw = t.view(torch.uint8).numpy()
    # A transposed singleton dimension may still be contiguous in PyTorch.
    # Match the TN view strides instead of using is_contiguous() as a transpose flag.
    if _util.detect_transpose_from_strides(t):
        raw = np.ascontiguousarray(raw.T)
    return _unpack_fp4_numpy(raw)


def _to_np(*tensors):
    result = []
    for t in tensors:
        if t is None:
            result.append(None)
        elif isinstance(t, np.ndarray):
            result.append(t)
        elif _is_fp4_tensor(t):
            result.append(_fp4_to_np(t))
        else:
            result.append(_util.torch_to_numpy(t))
    return tuple(result)


def _scale_to_np(t):
    import torch

    values = _util.torch_to_numpy(t)
    if isinstance(t, (torch.Tensor, np.ndarray)) and t.ndim == 3:
        strides = (
            t.stride()
            if isinstance(t, torch.Tensor)
            else tuple(s // t.itemsize for s in t.strides)
        )
        if (
            strides[-3] == t.shape[-1]
            and strides[-2] == t.shape[-1] * t.shape[-3]
            and not (t.shape[-2] == 1 and t.shape[-3] == 1)
        ):
            values = np.ascontiguousarray(values.swapaxes(-3, -2))
    return values


def _customize_inputs_impl(x1, x2, x1ScaleOptional, x2Scale, yRef, **kwargs):
    testcase_name = kwargs.get("testcase_name", "unknown")

    for tensor in [x1, x2, x1ScaleOptional, x2Scale, yRef]:
        if tensor is None:
            continue
        cleaned = _util.sanitize_e8m0_scale(tensor, testcase_name)
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
        x1_np, x2_np, y_np = _to_np(x1, x2, yRef)
        x1_scale_np = _scale_to_np(x1ScaleOptional)
        x2_scale_np = _scale_to_np(x2Scale)

        import torch

        if isinstance(
            x1, (torch.Tensor, np.ndarray)
        ) and _util.detect_transpose_from_strides(x1):
            if not _is_fp4_tensor(x1):
                x1_np = np.ascontiguousarray(x1_np.T)
            transposeX1 = not transposeX1
        if isinstance(
            x2, (torch.Tensor, np.ndarray)
        ) and _util.detect_transpose_from_strides(x2):
            if not _is_fp4_tensor(x2):
                x2_np = np.ascontiguousarray(x2_np.T)
            transposeX2 = not transposeX2

        # Convert int8 scale to float8_e8m0 for MX mode detection
        if x1_scale_np is not None and x1_scale_np.dtype == np.int8:
            x1_scale_np = x1_scale_np.view(en_dtypes.float8_e8m0)
        if x2_scale_np is not None and x2_scale_np.dtype == np.int8:
            x2_scale_np = x2_scale_np.view(en_dtypes.float8_e8m0)

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
