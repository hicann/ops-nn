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
"""ACLNN-level golden for TransposeQuantBatchMatMul.

Delegates core computation to ``tqbmm_kernel_golden.TransposeQuantBatchMatMulTestSpec``.
Handles torch -> numpy conversion, FRACTAL_NZ -> ND conversion for the WeightNz
variant, aclnn camelCase parameter mapping and output dtype inference.

Quantization semantics (MXFP8 layout, e8m0 NaN passthrough, uint64 decode)
all live in the kernel-level golden - no input data is regenerated here so
that the golden always consumes exactly what the NPU received.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import tqbmm_kernel_golden as _kernel


def _to_np(*tensors):
    """torch tensor -> numpy. None stays None, numpy arrays pass through.

    torch's custom float8_e8m0 scalar types (e.g. "torch.float8_e8m0fnu") are
    not convertible via tensor.numpy(); their raw 8-bit payload is carried
    through as en_dtypes.float8_e8m0 so the kernel-level golden can decode it
    with the same 2^(raw - 127) semantics.
    """
    result = []
    for t in tensors:
        if t is None:
            result.append(None)
        elif isinstance(t, np.ndarray):
            result.append(t)
        else:
            import torch  # imported lazily: only needed for torch inputs

            t = t.detach().cpu()
            if "e8m0" in str(t.dtype):
                raw = t.view(torch.uint8).numpy()
                t = np.ascontiguousarray(raw).view(_util.np_mx_scale)
            else:
                t = _util.torch_to_numpy(t)
            result.append(t)
    return tuple(result)


def _out_dtype(out, fallback):
    """Return the output dtype name, falling back to the given tensor dtype.

    When neither is available None is returned and the caller keeps its own
    output dtype resolution (e.g. the dtype enum at E2E level).
    """
    if out is not None:
        (out_np,) = _to_np(out)
        if out_np is not None:
            return _util.dtype_to_str(out_np.dtype)
    if fallback is None:
        return None
    if isinstance(fallback, np.ndarray):
        return _util.dtype_to_str(fallback.dtype)
    return _util.torch_dtype_to_str(fallback.dtype)


def _nz_to_nd_if_needed(x2_np, kwargs):
    """Convert x2 from FRACTAL_NZ storage back to logical ND (WeightNz variant)."""
    tensor_formats = kwargs.get("tensor_formats", ())
    if len(tensor_formats) <= 1 or tensor_formats[1] != "FRACTAL_NZ":
        return x2_np
    storage_shapes = kwargs.get("tensor_storage_shapes", ())
    ori_shape = storage_shapes[1] if len(storage_shapes) > 1 else None
    if ori_shape is not None and tuple(x2_np.shape) != tuple(ori_shape):
        x2_np = _util.nz_to_nd(x2_np, ori_shape)
    return x2_np


class AclnnTransposeQuantBatchMatMulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        x1Scale=None,
        x2Scale=None,
        dtype=0,
        groupSize=0,
        permX1=None,
        permX2=None,
        permY=None,
        batchSplitFactor=1,
        out=None,
        **kwargs,
    ):
        x1_np, x2_np, bias_np, s1_np, s2_np = _to_np(x1, x2, bias, x1Scale, x2Scale)
        x2_np = _nz_to_nd_if_needed(x2_np, kwargs)

        out_dtype = _out_dtype(out, None)

        perm_x1 = tuple(permX1) if permX1 is not None else (1, 0, 2)
        perm_x2 = tuple(permX2) if permX2 is not None else (0, 1, 2)
        perm_y = tuple(permY) if permY is not None else (1, 0, 2)

        temp_kwargs = dict(kwargs)
        if out_dtype:
            temp_kwargs["output_dtypes"] = [out_dtype]

        return _kernel.TransposeQuantBatchMatMulTestSpec.golden(
            x1_np,
            x2_np,
            bias_np,
            s1_np,
            s2_np,
            dtype=dtype,
            group_size=groupSize,
            perm_x1=perm_x1,
            perm_x2=perm_x2,
            perm_y=perm_y,
            batch_split_factor=batchSplitFactor,
            **temp_kwargs,
        )


class AclnnTransposeQuantBatchMatMulWeightNzTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        x1Scale=None,
        x2Scale=None,
        dtype=0,
        groupSize=0,
        permX1=None,
        permX2=None,
        permY=None,
        batchSplitFactor=1,
        out=None,
        **kwargs,
    ):
        return AclnnTransposeQuantBatchMatMulTestSpec.golden(
            x1,
            x2,
            bias=bias,
            x1Scale=x1Scale,
            x2Scale=x2Scale,
            dtype=dtype,
            groupSize=groupSize,
            permX1=permX1,
            permX2=permX2,
            permY=permY,
            batchSplitFactor=batchSplitFactor,
            out=out,
            **kwargs,
        )


__spec__ = {
    "aclnnTransposeQuantBatchMatMul": "AclnnTransposeQuantBatchMatMulTestSpec",
    "aclnnTransposeQuantBatchMatMulWeightNz": "AclnnTransposeQuantBatchMatMulWeightNzTestSpec",
}
