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
            and scale.dtype
            in (torch.int64, torch.int32, getattr(torch, "uint64", torch.int64))
        ):
            scale_np = scale.detach().cpu().numpy()
            fp32_scale = np.random.uniform(low=-5, high=5, size=scale_np.shape).astype(
                np.float32
            )
            new_scale = _util.pack_u64_scale(_util.scale_generate(fp32_scale))
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
        out_dtype = _util.dtype_to_str(out if out is not None else x1.dtype)

        perm_x1 = tuple(permX1) if permX1 is not None else (0, 1, 2)
        perm_x2 = tuple(permX2) if permX2 is not None else (0, 1, 2)
        perm_y = tuple(permY) if permY is not None else (1, 0, 2)

        temp_kwargs = dict(kwargs)
        if out_dtype:
            temp_kwargs["output_dtypes"] = [out_dtype]

        if str(kwargs.get("device", "cpu")).startswith("npu"):
            if cubeMathType is not None and int(cubeMathType) == 2:
                raise NotImplementedError(
                    "NPU TBMM+Add golden does not model USE_FP16 input conversion"
                )
            return _kernel._npu_compute(
                x1,
                x2,
                bias,
                scale,
                perm_x1=perm_x1,
                perm_x2=perm_x2,
                perm_y=perm_y,
                enable_hf32=_enable_hf32(cubeMathType),
                batch_split_factor=batchSplitFactor,
                **temp_kwargs,
            )

        x1_np, x2_np, bias_np, scale_np = _util.to_numpy(x1, x2, bias, scale)
        x1_np, x2_np = _apply_cube_type(x1_np, x2_np, cubeMathType)
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


class AclnnTransposeBatchMatMulWeightNzTestSpec(AclnnTransposeBatchMatMulTestSpec):
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
        if not str(kwargs.get("device", "cpu")).startswith("npu"):
            # ACLNN carries physical NZ storage; decode before the shared ND reference.
            x2 = _util.nz_input_to_nd(x2, kwargs.get("tensor_formats", ()), 1)
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
