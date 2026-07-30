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
"""ACLNN 层 golden 实现。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import mmv3_kernel_golden as _kernel


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


def _opImplMode(cubeMathType):
    if cubeMathType is None:
        return 0
    return 64 if int(cubeMathType) in (1, 3) else 0


def _out_dtype(out, fallback):
    if out is not None:
        (out_np,) = _to_np(out)
        if out_np is not None:
            return _util.dtype_to_str(out_np.dtype)
    if isinstance(fallback, np.ndarray):
        return _util.dtype_to_str(fallback.dtype)
    return _util.torch_dtype_to_str(fallback.dtype)


def _nz_to_nd_if_needed(mat2_np, kwargs):
    tensor_formats = kwargs.get("tensor_formats", ())
    if len(tensor_formats) <= 1 or tensor_formats[1] != "FRACTAL_NZ":
        return mat2_np
    storage_shapes = kwargs.get("tensor_storage_shapes", ())
    ori_shape = storage_shapes[1] if len(storage_shapes) > 1 else None
    if ori_shape is not None and tuple(mat2_np.shape) != tuple(ori_shape):
        mat2_np = _util.nz_to_nd(mat2_np, ori_shape)
    return mat2_np


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


def _kernel_bmm(x1, x2, *, cubeMathType, out_dtype, **kwargs):
    temp_kwargs = dict(kwargs)
    if out_dtype:
        temp_kwargs["output_dtypes"] = [out_dtype]
    return _kernel.MatMulV3TestSpec.golden(
        x1,
        x2,
        bias=None,
        transpose_x1=False,
        transpose_x2=False,
        opImplMode=_opImplMode(cubeMathType),
        **temp_kwargs,
    )[0]


class AclnnMatmulTestSpec:
    @staticmethod
    def golden(self, mat2, out=None, cubeMathType=0, **kwargs):
        self_np, mat2_np = _to_np(self, mat2)
        mat2_np = _nz_to_nd_if_needed(mat2_np, kwargs)
        self_np, mat2_np = _apply_cube_type(self_np, mat2_np, cubeMathType)
        out_dtype = _out_dtype(out, self)
        return _kernel_bmm(
            self_np, mat2_np, cubeMathType=cubeMathType, out_dtype=out_dtype, **kwargs
        )


class AclnnMatmulWeightNzTestSpec:
    @staticmethod
    def golden(self, mat2, out=None, cubeMathType=0, **kwargs):
        return AclnnMatmulTestSpec.golden(
            self, mat2, out=out, cubeMathType=cubeMathType, **kwargs
        )


class AclnnMmTestSpec:
    @staticmethod
    def golden(self, mat2, out=None, cubeMathType=0, **kwargs):
        return AclnnMatmulTestSpec.golden(
            self, mat2, out=out, cubeMathType=cubeMathType, **kwargs
        )


class AclnnAddmmTestSpec:
    @staticmethod
    def golden(
        self, mat1, mat2, beta=1.0, alpha=1.0, out=None, cubeMathType=0, **kwargs
    ):
        self_np, mat1_np, mat2_np = _to_np(self, mat1, mat2)
        mat1_np, mat2_np = _apply_cube_type(mat1_np, mat2_np, cubeMathType)
        out_dtype = _out_dtype(out, self)

        bmm = _kernel_bmm(
            mat1_np, mat2_np, cubeMathType=cubeMathType, out_dtype=out_dtype, **kwargs
        )

        self_f = (
            self_np.astype(np.float32)
            if _util.dtype_to_str(self_np.dtype) == "bfloat16"
            else self_np.astype(bmm.dtype)
        )
        promote_dtype = np.result_type(self_f, bmm)
        result = (float(beta) * self_f).astype(promote_dtype) + (
            float(alpha) * bmm
        ).astype(promote_dtype)
        return _util.cast_output_dtype(result, out_dtype) if out_dtype else result


class AclnnAddmmWeightNzTestSpec:
    @staticmethod
    def golden(
        self, mat1, mat2, beta=1.0, alpha=1.0, out=None, cubeMathType=0, **kwargs
    ):
        return AclnnAddmmTestSpec.golden(
            self,
            mat1,
            mat2,
            beta=beta,
            alpha=alpha,
            out=out,
            cubeMathType=cubeMathType,
            **kwargs,
        )


class AclnnInplaceAddmmTestSpec:
    @staticmethod
    def golden(selfRef, mat1, mat2, beta=1.0, alpha=1.0, cubeMathType=0, **kwargs):
        return AclnnAddmmTestSpec.golden(
            selfRef,
            mat1,
            mat2,
            beta=beta,
            alpha=alpha,
            out=None,
            cubeMathType=cubeMathType,
            **kwargs,
        )


__spec__ = {
    "aclnnMatmul": "AclnnMatmulTestSpec",
    "aclnnMatmulWeightNz": "AclnnMatmulWeightNzTestSpec",
    "aclnnMm": "AclnnMmTestSpec",
    "aclnnAddmm": "AclnnAddmmTestSpec",
    "aclnnAddmmWeightNz": "AclnnAddmmWeightNzTestSpec",
    "aclnnInplaceAddmm": "AclnnInplaceAddmmTestSpec",
}
