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
        self_np, mat2_np = _util.to_numpy(self, mat2)
        self_np, mat2_np = _apply_cube_type(self_np, mat2_np, cubeMathType)
        return _kernel_bmm(
            self_np,
            mat2_np,
            cubeMathType=cubeMathType,
            out_dtype=_util.dtype_to_str(out if out is not None else self.dtype),
            **kwargs,
        )


class AclnnMatmulWeightNzTestSpec(AclnnMatmulTestSpec):
    """aclnnMatmulWeightNz: mat2 为 NZ（tensor index 1），先解码为 ND 再计算。"""

    @staticmethod
    def customize_inputs(self, mat2, out=None, cubeMathType=0, **kwargs):
        """aclnn 输入定制：NZ mat2 规范化（in-place 覆写 storage）。"""
        _util.prepare_nz_input(mat2, kwargs.get("tensor_formats", ()), 1)

    @staticmethod
    def golden(self, mat2, out=None, cubeMathType=0, **kwargs):
        mat2 = _util.nz_input_to_nd(mat2, kwargs.get("tensor_formats", ()), 1)
        return AclnnMatmulTestSpec.golden(
            self, mat2, out=out, cubeMathType=cubeMathType, **kwargs
        )


class AclnnMmTestSpec(AclnnMatmulTestSpec):
    """aclnnMm shares the plain mm semantics."""


def _is_950_soc(kwargs):
    """DAV_3510(Ascend950 系列)平台判定，与 baddbmm 侧 _is_950 同源。"""
    return str(kwargs.get("short_soc_version") or "").startswith("Ascend950")


def _fmm_add_supported(
    self_np, mat1_np, mat2_np, out, alpha, beta, cubeMathType, kwargs
):
    """镜像 aclnn_addmm.cpp CheckFmmAddSupport（950 fmm(add) 单kernel路由门禁）：
    α=β=1(eps 容差) + mat1/mat2/self 均为完整 2D 且 self 严格 (M,N)(不支持
    (n,)/(1,n) 广播) + K>0 + 四方同 dtype ∈ {fp16,bf16,fp32} + fp32 输入仅
    cmt=3(USE_HF32) + storage format ND。核配 IsAivNumDoubleOfAicNum(标准 950
    MIX_AIC_1_2)按满足处理。
    命中后 NPU 走 AddmmFmmGraph -> fmm(fusedOpType="add", x3=self, bias=null)
    单kernel(高性能模式): round_out(round_out(mm_acc) + self)，mm 的 acc 累加值
    先舍回 C 域再融合，不携带 alpha/beta(门禁 eps 内偏离 1 的缩放被设备丢弃，
    golden 同步不缩放——baddbmm review#1 同款结论)。"""
    if not _is_950_soc(kwargs):
        return False
    if abs(float(alpha) - 1.0) > _FP32_EPS or abs(float(beta) - 1.0) > _FP32_EPS:
        return False
    if mat1_np.ndim != 2 or mat2_np.ndim != 2 or self_np.ndim != 2:
        return False
    m_dim, n_dim = mat1_np.shape[0], mat2_np.shape[1]
    if tuple(self_np.shape) != (m_dim, n_dim):
        return False
    if mat1_np.shape[1] == 0 or mat2_np.shape[0] == 0:
        return False
    dtype_str = _util.dtype_to_str(mat1_np.dtype)
    if dtype_str not in ("float16", "bfloat16", "float32"):
        return False
    if (
        _util.dtype_to_str(mat2_np.dtype) != dtype_str
        or _util.dtype_to_str(self_np.dtype) != dtype_str
    ):
        return False
    out_dtype_str = (
        _util.dtype_to_str(out.dtype)
        if out is not None
        else _util.dtype_to_str(self_np.dtype)
    )
    if out_dtype_str != dtype_str:
        return False
    cube_math_type = 0 if cubeMathType is None else int(cubeMathType)
    if dtype_str == "float32" and cube_math_type != _CUBE_MATH_USE_HF32:
        return False
    fmts = kwargs.get("tensor_formats") or ()
    if any(f not in (None, "", "ND") for f in list(fmts)[:3]):
        return False
    return True


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
        self_np, mat1_np, mat2_np = _util.to_numpy(self, mat1, mat2)
        mat1_dtype, mat2_dtype = mat1_np.dtype, mat2_np.dtype
        mat1_np, mat2_np = _apply_cube_type(mat1_np, mat2_np, cubeMathType)
        cube_math_type = 0 if cubeMathType is None else int(cubeMathType)
        out_dtype = _util.dtype_to_str(out if out is not None else self.dtype)
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

        # 950 fmm(add) 单kernel路由：高性能模式(innerPrecise=HIGH_PERFORMANCE,
        # cmt=0 时) mm 的 acc 累加值先舍回 C 域(round 到 out dtype)再融合 self
        # ——对齐 956bae3c1 所述"fmm 高性能路径(acc 舍回 C 域融合)"与本次实测
        # (bf16 缺失该 round: mismatch 27.3% -> 精度 74%失败)。不携带 alpha/beta
        # (门禁已限 α=β=1 eps 内, 缩放被设备丢弃)。
        if _fmm_add_supported(
            self_np, mat1_np, mat2_np, out, alpha, beta, cubeMathType, kwargs
        ):
            matmul_result = _kernel_bmm(
                mat1_np,
                mat2_np,
                cubeMathType=cubeMathType,
                out_dtype=out_dtype,
                **kwargs,
            )
            add_result = matmul_result.astype(np.float32) + self_np.astype(np.float32)
            return (
                _util.cast_output_dtype(add_result, out_dtype)
                if out_dtype
                else add_result
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
        # 对齐 baddbmm d7d063bfd 实证结论: l0op::Axpy/Add 是融合算子(fp32 域
        # x + α·y 单 pass 单次 round), 不塌回低精度域再逐元素双 round
        # (旧版 combined_fp32.astype(matmul_out_dtype) 在 bf16 case 与 NPU 相差
        # 1 ULP 系统性偏差 —— fp16(ULP 0.049%) 侥幸被 rtol 0.001 掩盖, bf16
        # (ULP 0.39%) 超差失败)
        matmul_out = matmul_result.astype(matmul_out_dtype)
        bias_term = self_np
        if abs(beta - 1.0) > _FP32_EPS:
            bias_term = (bias_term.astype(np.float32) * np.float32(beta)).astype(
                bias_term.dtype
            )
        combined_fp32 = matmul_out.astype(np.float32) * np.float32(
            alpha
        ) + bias_term.astype(np.float32)
        return (
            _util.cast_output_dtype(combined_fp32, out_dtype)
            if out_dtype
            else combined_fp32
        )


class AclnnAddmmWeightNzTestSpec(AclnnAddmmTestSpec):
    """aclnnAddmmWeightNz: mat2 为 NZ（tensor index 2），先解码为 ND 再计算。"""

    @staticmethod
    def customize_inputs(
        self, mat1, mat2, beta=1.0, alpha=1.0, out=None, cubeMathType=0, **kwargs
    ):
        """aclnn 输入定制：NZ mat2 规范化（in-place 覆写 storage）。"""
        _util.prepare_nz_input(mat2, kwargs.get("tensor_formats", ()), 2)

    @staticmethod
    def golden(
        self, mat1, mat2, beta=1.0, alpha=1.0, out=None, cubeMathType=0, **kwargs
    ):
        mat2 = _util.nz_input_to_nd(mat2, kwargs.get("tensor_formats", ()), 2)
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
