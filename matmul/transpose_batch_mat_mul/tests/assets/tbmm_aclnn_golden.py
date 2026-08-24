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
"""ACLNN-level golden for TransposeBatchMatMul.

Delegates core computation to ``tbmm_kernel_golden._kernel_compute``.
Handles torch->numpy conversion, cubeMathType mapping, NZ format conversion,
and output dtype inference.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import tbmm_kernel_golden as _kernel


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


def _enable_hf32(cubeMathType):
    if cubeMathType is None:
        return False
    return int(cubeMathType) in (1, 3)


def _apply_cube_type(x1, x2, cubeMathType):
    if cubeMathType is None:
        return x1, x2
    cmt = int(cubeMathType)
    if cmt != 2:
        return x1, x2
    if x1 is not None and x1.dtype == np.float32:
        x1 = x1.astype(np.float16)
    if x2 is not None and x2.dtype == np.float32:
        x2 = x2.astype(np.float16)
    return x1, x2


def _out_dtype(out, fallback):
    if out is not None:
        (out_np,) = _to_np(out)
        if out_np is not None:
            return _util.dtype_to_str(out_np.dtype)
    if isinstance(fallback, np.ndarray):
        return _util.dtype_to_str(fallback.dtype)
    return _util.torch_dtype_to_str(fallback.dtype)


def _nz_to_nd_if_needed(x2_np, kwargs):
    tensor_formats = kwargs.get("tensor_formats", ())
    if len(tensor_formats) <= 1 or tensor_formats[1] != "FRACTAL_NZ":
        return x2_np
    storage_shapes = kwargs.get("tensor_storage_shapes", ())
    ori_shape = storage_shapes[1] if len(storage_shapes) > 1 else None
    if ori_shape is not None and tuple(x2_np.shape) != tuple(ori_shape):
        x2_np = _util.nz_to_nd(x2_np, ori_shape)
    return x2_np


class AclnnTransposeBatchMatMulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        bias=None,
        scale=None,
        permX1=None,
        permX2=None,
        permY=None,
        cubeMathType=0,
        batchSplitFactor=1,
        out=None,
        **kwargs,
    ):
        import torch

        if (
            scale is not None
            and hasattr(scale, "dtype")
            and scale.dtype in (torch.int64, torch.int32)
        ):
            scale_np = scale.detach().cpu().numpy()
            fp32_scale = np.random.uniform(low=-5, high=5, size=scale_np.shape).astype(
                np.float32
            )
            u32 = np.ascontiguousarray(fp32_scale).view(np.uint32).copy()
            u32 &= np.uint32(0xFFFFE000)
            new_scale = np.zeros(scale_np.shape, np.uint64)
            new_scale |= u32.astype(np.uint64)
            new_scale |= np.uint64(1 << 46)
            scale.copy_(
                torch.from_numpy(new_scale.astype(np.int64).copy()).to(scale.device)
            )

    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        scale=None,
        permX1=None,
        permX2=None,
        permY=None,
        cubeMathType=0,
        batchSplitFactor=1,
        out=None,
        **kwargs,
    ):
        x1_np, x2_np, bias_np, scale_np = _to_np(x1, x2, bias, scale)
        x2_np = _nz_to_nd_if_needed(x2_np, kwargs)
        x1_np, x2_np = _apply_cube_type(x1_np, x2_np, cubeMathType)
        out_dtype = _out_dtype(out, x1)

        perm_x1 = tuple(permX1) if permX1 is not None else (0, 1, 2)
        perm_x2 = tuple(permX2) if permX2 is not None else (0, 1, 2)
        perm_y = tuple(permY) if permY is not None else (1, 0, 2)

        temp_kwargs = dict(kwargs)
        if out_dtype:
            temp_kwargs["output_dtypes"] = [out_dtype]

        return _kernel.TransposeBatchMatMulTestSpec.golden(
            x1_np,
            x2_np,
            bias_np,
            scale_np,
            perm_x1=perm_x1,
            perm_x2=perm_x2,
            perm_y=perm_y,
            enable_hf32=_enable_hf32(cubeMathType),
            batch_split_factor=batchSplitFactor,
            **temp_kwargs,
        )


class AclnnTransposeBatchMatMulWeightNzTestSpec:
    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        scale=None,
        permX1=None,
        permX2=None,
        permY=None,
        cubeMathType=0,
        batchSplitFactor=1,
        out=None,
        **kwargs,
    ):
        return AclnnTransposeBatchMatMulTestSpec.golden(
            x1,
            x2,
            bias=bias,
            scale=scale,
            permX1=permX1,
            permX2=permX2,
            permY=permY,
            cubeMathType=cubeMathType,
            batchSplitFactor=batchSplitFactor,
            out=out,
            **kwargs,
        )


__spec__ = {
    "aclnnTransposeBatchMatMul": "AclnnTransposeBatchMatMulTestSpec",
    "aclnnTransposeBatchMatMulWeightNz": "AclnnTransposeBatchMatMulWeightNzTestSpec",
}
