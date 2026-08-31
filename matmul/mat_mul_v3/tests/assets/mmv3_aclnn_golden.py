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
"""ACLNN golden specs backed by MatMulV3.

The addmm golden mirrors the host-side graph selection of aclnnAddmm: bias
foldable into the cube (or 16-in-32-out) runs in fp32 with a single final
rounding; other cases take the vector path where the mm result is rounded
once and then combined by one fused Axpy/Add rounding at the promoted dtype.

Naming rules: variables reuse the API parameter names (self/mat1/mat2/out/
cubeMathType) wherever applicable; numpy arrays carry the _np suffix; dtype
variables use the full _dtype suffix; booleans use is_/enable_ prefixes;
host-side semantic names are used for intermediate results (matmul_result,
bias_term, add_result).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import mmv3_kernel_golden as _kernel

_FP32_EPS = np.finfo(np.float32).eps
_OP_IMPL_HF32 = 0x40
_CUBE_MATH_FP32_DOWN = 1
_CUBE_MATH_USE_FP16 = 2
_CUBE_MATH_USE_HF32 = 3
_CUBE_MATH_USE_FP32_ADD = 4


def _to_np(*tensors):
    result = []
    for tensor in tensors:
        if tensor is None:
            result.append(None)
        elif isinstance(tensor, np.ndarray):
            result.append(tensor)
        else:
            result.append(_util.torch_to_numpy(tensor))
    return tuple(result)


def _out_dtype(out, fallback):
    if out is not None:
        (out_np,) = _to_np(out)
        if out_np is not None:
            return _util.dtype_to_str(out_np.dtype)
    if isinstance(fallback, np.ndarray):
        return _util.dtype_to_str(fallback.dtype)
    return _util.torch_dtype_to_str(fallback.dtype)


def _apply_cube_type(x1, x2, cubeMathType):
    """USE_FP16: fp32 inputs are truncated to fp16 before the mm."""
    if cubeMathType is None or int(cubeMathType) != _CUBE_MATH_USE_FP16:
        return x1, x2
    if x1 is not None and x1.dtype == np.float32:
        x1 = x1.astype(np.float16)
    if x2 is not None and x2.dtype == np.float32:
        x2 = x2.astype(np.float16)
    return x1, x2


def _kernel_bmm(x1, x2, *, cubeMathType, out_dtype, **kwargs):
    """Run the kernel-level mm golden; HF32 impl mode for cmt in {1, 3}."""
    kernel_kwargs = dict(kwargs)
    if out_dtype:
        kernel_kwargs["output_dtypes"] = [out_dtype]
    cube_math_type = 0 if cubeMathType is None else int(cubeMathType)
    op_impl_mode = (
        _OP_IMPL_HF32
        if cube_math_type in (_CUBE_MATH_FP32_DOWN, _CUBE_MATH_USE_HF32)
        else 0
    )
    return _kernel.MatMulV3TestSpec.golden(
        x1,
        x2,
        bias=None,
        transpose_x1=False,
        transpose_x2=False,
        opImplMode=op_impl_mode,
        **kernel_kwargs,
    )[0]


class AclnnMatmulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(self, mat2, out=None, cubeMathType=0, **kwargs):
        self_np, mat2_np = _to_np(self, mat2)
        tensor_formats = kwargs.get("tensor_formats", ())
        if len(tensor_formats) > 1 and tensor_formats[1] == "FRACTAL_NZ":
            storage_shapes = kwargs.get("tensor_storage_shapes", ())
            original_shape = storage_shapes[1] if len(storage_shapes) > 1 else None
            if original_shape is not None and tuple(mat2_np.shape) != tuple(
                original_shape
            ):
                mat2_np = _util.nz_to_nd(mat2_np, original_shape)
        self_np, mat2_np = _apply_cube_type(self_np, mat2_np, cubeMathType)
        return _kernel_bmm(
            self_np,
            mat2_np,
            cubeMathType=cubeMathType,
            out_dtype=_out_dtype(out, self),
            **kwargs,
        )


class AclnnMatmulWeightNzTestSpec(AclnnMatmulTestSpec):
    """aclnnMatmulWeightNz shares the plain mm semantics; the NZ conversion
    above is driven by the tensor_formats kwarg."""


class AclnnMmTestSpec(AclnnMatmulTestSpec):
    """aclnnMm shares the plain mm semantics."""


class AclnnAddmmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def _upper_dtype(mat1_dtype, mat2_dtype, cube_math_type):
        """Upper dtype of the mm op (host MatMulRule promote table)."""
        if cube_math_type == _CUBE_MATH_USE_FP16:
            return np.float16
        if mat1_dtype == np.float32 or mat2_dtype == np.float32:
            return np.float32
        if mat1_dtype == _util.np_bfloat16 and mat2_dtype == _util.np_bfloat16:
            return _util.np_bfloat16
        if mat1_dtype == np.float16 and mat2_dtype == np.float16:
            return np.float16
        return np.float32

    @staticmethod
    def _need_fp32_out(mat1_dtype, mat2_dtype, out_dtype, cube_math_type):
        """Mirrors NeedEnableFp32Output for the fused (bias-free) mm."""
        both_inputs_low_precision = (
            mat1_dtype == np.float16 and mat2_dtype == np.float16
        ) or (mat1_dtype == _util.np_bfloat16 and mat2_dtype == _util.np_bfloat16)
        return both_inputs_low_precision and (
            out_dtype == "float32" or cube_math_type == _CUBE_MATH_USE_FP32_ADD
        )

    @staticmethod
    def _bias_can_fold(bias_np, mat1_dtype, mat2_cols, alpha, beta):
        """Mirrors NeedToConvertBias/CheckDtypeSupportBias (no split-K on 3510)."""
        if abs(beta - 1.0) > _FP32_EPS or abs(alpha - 1.0) > _FP32_EPS:
            return False
        is_bias_vector = bias_np.ndim == 1 and bias_np.shape[0] == mat2_cols
        is_bias_row = (
            bias_np.ndim == 2
            and bias_np.shape[0] == 1
            and bias_np.shape[1] == mat2_cols
        )
        if not (is_bias_vector or is_bias_row):
            return False
        if mat1_dtype == _util.np_bfloat16:
            return bias_np.dtype in (_util.np_bfloat16, np.float32)
        return bias_np.dtype in (mat1_dtype, np.float32)

    @staticmethod
    def golden(
        self, mat1, mat2, beta=1.0, alpha=1.0, out=None, cubeMathType=0, **kwargs
    ):
        self_np, mat1_np, mat2_np = _to_np(self, mat1, mat2)
        mat1_dtype, mat2_dtype = mat1_np.dtype, mat2_np.dtype
        mat1_np, mat2_np = _apply_cube_type(mat1_np, mat2_np, cubeMathType)
        cube_math_type = 0 if cubeMathType is None else int(cubeMathType)
        out_dtype = _out_dtype(out, self)
        alpha = float(alpha)
        beta = float(beta)

        if abs(alpha) <= _FP32_EPS:
            # alpha == 0: output is beta * self broadcast to the mm shape
            bias_term = self_np
            if abs(beta - 1.0) > _FP32_EPS:
                bias_term = (bias_term.astype(np.float32) * np.float32(beta)).astype(
                    bias_term.dtype
                )
            matmul_shape = (mat1_np.shape[-2], mat2_np.shape[-1])
            bias_term = np.broadcast_to(bias_term, matmul_shape)
            return (
                _util.cast_output_dtype(bias_term, out_dtype)
                if out_dtype
                else bias_term
            )

        matmul_result = _kernel_bmm(
            mat1_np, mat2_np, cubeMathType=cubeMathType, out_dtype=None, **kwargs
        )
        enable_fp32_out = AclnnAddmmTestSpec._need_fp32_out(
            mat1_dtype, mat2_dtype, out_dtype, cube_math_type
        )
        matmul_out_dtype = (
            np.float32
            if enable_fp32_out
            else AclnnAddmmTestSpec._upper_dtype(mat1_dtype, mat2_dtype, cube_math_type)
        )

        if abs(beta - 0.0) <= _FP32_EPS:
            # beta == 0: output is alpha * mm, rounded once to the mm dtype
            matmul_out = matmul_result.astype(matmul_out_dtype)
            if abs(alpha - 1.0) > _FP32_EPS:
                matmul_out = (matmul_out.astype(np.float32) * np.float32(alpha)).astype(
                    matmul_out.dtype
                )
            return (
                _util.cast_output_dtype(matmul_out, out_dtype)
                if out_dtype
                else matmul_out
            )

        if (
            AclnnAddmmTestSpec._bias_can_fold(
                self_np, mat1_dtype, mat2_np.shape[-1], alpha, beta
            )
            or enable_fp32_out
        ):
            # bais inside the cube / 16-in-32-out: fp32 accumulation,
            # a single rounding to the output dtype
            add_result = matmul_result.astype(np.float32) + self_np.astype(np.float32)
            return (
                _util.cast_output_dtype(add_result, out_dtype)
                if out_dtype
                else add_result
            )

        # generic vector path: mm rounded once, then one fused Axpy/Add
        # rounding at the promoted dtype
        matmul_out = matmul_result.astype(matmul_out_dtype)
        bias_term = self_np
        if abs(beta - 1.0) > _FP32_EPS:
            bias_term = (bias_term.astype(np.float32) * np.float32(beta)).astype(
                bias_term.dtype
            )
        combined_fp32 = matmul_out.astype(np.float32) * np.float32(
            alpha
        ) + bias_term.astype(np.float32)
        if (
            bias_term.dtype == np.float32
            or matmul_out.dtype == np.float32
            or bias_term.dtype != matmul_out.dtype
        ):
            combined = combined_fp32
        else:
            combined = combined_fp32.astype(matmul_out.dtype)
        return _util.cast_output_dtype(combined, out_dtype) if out_dtype else combined


class AclnnAddmmWeightNzTestSpec(AclnnAddmmTestSpec):
    """aclnnAddmmWeightNz shares the addmm semantics; the NZ conversion is
    handled by the kernel golden via the tensor_formats kwarg."""


class AclnnInplaceAddmmTestSpec(AclnnAddmmTestSpec):
    @staticmethod
    def golden(selfRef, mat1, mat2, beta=1.0, alpha=1.0, cubeMathType=0, **kwargs):
        # selfRef is both the bias and the output; no separate out tensor
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
    "aclnnAddmm": "AclnnAddmmTestSpec",
    "aclnnAddmmWeightNz": "AclnnAddmmWeightNzTestSpec",
    "aclnnInplaceAddmm": "AclnnInplaceAddmmTestSpec",
    "aclnnMatmul": "AclnnMatmulTestSpec",
    "aclnnMatmulWeightNz": "AclnnMatmulWeightNzTestSpec",
    "aclnnMm": "AclnnMmTestSpec",
}
