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

各 API 的 golden 按统一流程组织：
  1. torch → numpy
  2. NZ→ND 格式转换（仅 BatchMatMul 需要）
  3. cubeMathType 精度模拟
  4. 调 kernel golden 做核心 matmul
  5. 叠加 ACLNN 层语义

参数名严格对齐 C API GetWorkspaceSize 声明。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import bmmv3_kernel_golden as _kernel


# ============================================================================
# shared helpers
# ============================================================================


def _to_np(*tensors):
    """torch tensor → numpy。None 保持 None，已是 numpy 则跳过。"""
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
    """cubeMathType → kernel opImplMode: KEEP_DTYPE=0, HF32=64。"""
    if cubeMathType is None:
        return 0
    return 64 if int(cubeMathType) in (1, 3) else 0


def _out_dtype(out, fallback):
    """返回 out tensor 的 dtype 名，若 out 为 None 则用 fallback 的 dtype。"""
    if out is not None:
        (out_np,) = _to_np(out)
        if out_np is not None:
            return _util.dtype_to_str(out_np.dtype)
    if isinstance(fallback, np.ndarray):
        return _util.dtype_to_str(fallback.dtype)
    return _util.torch_dtype_to_str(fallback.dtype)


def _nz_to_nd_if_needed(mat2_np, kwargs):
    """若 tensor_formats 指示 mat2 为 FRACTAL_NZ，转换为 ND。

    ACLNN 层在此完成转换，传给 kernel golden 的 mat2 始终为 ND。
    kernel golden 自身的 NZ→ND 守卫作为 kernel 模式的安全网保留。
    """
    tensor_formats = kwargs.get("tensor_formats", ())
    if len(tensor_formats) <= 1 or tensor_formats[1] != "FRACTAL_NZ":
        return mat2_np

    storage_shapes = kwargs.get("tensor_storage_shapes", ())
    ori_shape = storage_shapes[1] if len(storage_shapes) > 1 else None
    if ori_shape is not None and tuple(mat2_np.shape) != tuple(ori_shape):
        mat2_np = _util.nz_to_nd(mat2_np, ori_shape)
    return mat2_np


def _apply_cube_type(x1, x2, cubeMathType):
    """cubeMathType 精度模拟。

    - mode 2 (USE_FP16): fp32→fp16 截断，kernel golden 检测到 fp16 后
      自动提升到 fp32 累加，匹配 NPU 行为。
    - modes 1/3 (HF32): 由 kernel golden 通过 opImplMode=64 处理。
    - mode 0 (KEEP_DTYPE): 无操作。
    """
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
    """调 kernel golden 做核心 matmul，返回 numpy 结果。"""
    temp_kwargs = dict(kwargs)
    if out_dtype:
        temp_kwargs["output_dtypes"] = [out_dtype]
    return _kernel.BatchMatMulV3TestSpec.golden(
        x1,
        x2,
        bias=None,
        adj_x1=False,
        adj_x2=False,
        opImplMode=_opImplMode(cubeMathType),
        **temp_kwargs,
    )[0]


# ============================================================================
# TestSpec classes
# ============================================================================


class AclnnBatchMatMulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(self, mat2, out=None, cubeMathType=0, **kwargs):
        """aclnnBatchMatMul: self @ mat2。"""
        # 1) torch → numpy
        self_np, mat2_np = _to_np(self, mat2)

        # 2) NZ→ND 格式转换（若 mat2 为 FRACTAL_NZ）
        mat2_np = _nz_to_nd_if_needed(mat2_np, kwargs)

        # 3) cubeMathType 精度模拟
        self_np, mat2_np = _apply_cube_type(self_np, mat2_np, cubeMathType)

        # 4) 调 kernel golden 做核心 matmul
        out_dtype = _out_dtype(out, self)
        return _kernel_bmm(
            self_np, mat2_np, cubeMathType=cubeMathType, out_dtype=out_dtype, **kwargs
        )


class AclnnBatchMatMulWeightNzTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(self, mat2, out=None, cubeMathType=0, **kwargs):
        """aclnnBatchMatMulWeightNz — 委托 BatchMatMul，NZ→ND 已在其中处理。"""
        return AclnnBatchMatMulTestSpec.golden(
            self, mat2, out=out, cubeMathType=cubeMathType, **kwargs
        )


class AclnnAddbmmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        self, batch1, batch2, beta=1.0, alpha=1.0, out=None, cubeMathType=0, **kwargs
    ):
        """aclnnAddbmm: beta*self + alpha * sum_i(batch1_i @ batch2_i)。"""
        # 1) torch → numpy
        self_np, batch1_np, batch2_np = _to_np(self, batch1, batch2)

        # 2) cubeMathType 精度模拟
        batch1_np, batch2_np = _apply_cube_type(batch1_np, batch2_np, cubeMathType)

        out_dtype = _out_dtype(out, self)

        # 3) 调 kernel golden 做核心 matmul
        bmm = _kernel_bmm(
            batch1_np,
            batch2_np,
            cubeMathType=cubeMathType,
            out_dtype=out_dtype,
            **kwargs,
        )
        bmm_summed = np.sum(bmm.astype(np.float32), axis=0).astype(bmm.dtype)

        # 4) ACLNN 语义叠加：对齐 C++ SelfMulsBetaProcess + PromoteType + Add/Axpy + Cast
        #    Step 4a: SelfMulsBetaProcess — Muls 在 self 的 dtype 下计算
        #    C++: BF16 self → cast FP32 → Muls(beta) in FP32; else → Muls(beta) in self.dtype
        if _util.dtype_to_str(self_np.dtype) == "bfloat16":
            self_f = self_np.astype(np.float32)
            mulOut = (self_f * np.float32(beta)).astype(np.float32)
        else:
            beta_cast = self_np.dtype.type(float(beta))
            mulOut = (self_np * beta_cast).astype(self_np.dtype)

        #    Step 4b: PromoteType + Cast
        promote_dtype = np.result_type(mulOut, bmm_summed)
        mulOut_casted = mulOut.astype(promote_dtype)
        bmm_summed_casted = bmm_summed.astype(promote_dtype)

        #    Step 4c: Add 或 Axpy — 在 promote_dtype 下计算
        #    C++: alpha==1 → Add; alpha!=1 → Axpy (y + alpha*x, alpha cast 到 promote_dtype)
        if abs(float(alpha) - 1.0) <= np.finfo(np.float32).eps:
            result = mulOut_casted + bmm_summed_casted
        else:
            alpha_cast = promote_dtype.type(float(alpha))
            result = mulOut_casted + alpha_cast * bmm_summed_casted

        #    Step 4d: Cast to output dtype
        return _util.cast_output_dtype(result, out_dtype) if out_dtype else result


class AclnnInplaceAddbmmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(selfRef, batch1, batch2, beta=1.0, alpha=1.0, cubeMathType=0, **kwargs):
        """aclnnInplaceAddbmm — 无独立 out，selfRef 既是输入也是输出。"""
        return AclnnAddbmmTestSpec.golden(
            selfRef,
            batch1,
            batch2,
            beta=beta,
            alpha=alpha,
            out=None,
            cubeMathType=cubeMathType,
            **kwargs,
        )


class AclnnBaddbmmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        self, batch1, batch2, beta=1.0, alpha=1.0, out=None, cubeMathType=0, **kwargs
    ):
        """aclnnBaddbmm: beta*self + alpha*(batch1 @ batch2)。"""
        # 1) torch → numpy
        self_np, batch1_np, batch2_np = _to_np(self, batch1, batch2)

        # 2) cubeMathType 精度模拟
        batch1_np, batch2_np = _apply_cube_type(batch1_np, batch2_np, cubeMathType)

        out_dtype = _out_dtype(out, self)

        # 3) 调 kernel golden 做核心 matmul
        bmm = _kernel_bmm(
            batch1_np,
            batch2_np,
            cubeMathType=cubeMathType,
            out_dtype=out_dtype,
            **kwargs,
        )

        # 4) ACLNN 语义叠加：对齐 C++ SelfMulsBetaProcess + PromoteType + Add/Axpy + Cast
        #    Step 4a: SelfMulsBetaProcess — Muls 在 self 的 dtype 下计算
        if _util.dtype_to_str(self_np.dtype) == "bfloat16":
            self_f = self_np.astype(np.float32)
            mulOut = (self_f * np.float32(beta)).astype(np.float32)
        else:
            beta_cast = self_np.dtype.type(float(beta))
            mulOut = (self_np * beta_cast).astype(self_np.dtype)

        #    Step 4b: PromoteType + Cast
        promote_dtype = np.result_type(mulOut, bmm)
        mulOut_casted = mulOut.astype(promote_dtype)
        bmm_casted = bmm.astype(promote_dtype)

        #    Step 4c: Add 或 Axpy — 在 promote_dtype 下计算
        if abs(float(alpha) - 1.0) <= np.finfo(np.float32).eps:
            result = mulOut_casted + bmm_casted
        else:
            alpha_cast = promote_dtype.type(float(alpha))
            result = mulOut_casted + alpha_cast * bmm_casted

        #    Step 4d: Cast to output dtype
        return _util.cast_output_dtype(result, out_dtype) if out_dtype else result


class AclnnInplaceBaddbmmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(selfRef, batch1, batch2, beta=1.0, alpha=1.0, cubeMathType=0, **kwargs):
        """aclnnInplaceBaddbmm — 无独立 out，selfRef 既是输入也是输出。"""
        return AclnnBaddbmmTestSpec.golden(
            selfRef,
            batch1,
            batch2,
            beta=beta,
            alpha=alpha,
            out=None,
            cubeMathType=cubeMathType,
            **kwargs,
        )


class AclnnEinsumTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(tensors, equation, output=None, **kwargs):
        """aclnnEinsum: tensors 为 tensor 列表，equation 为字符串。"""
        np_tensors = _to_np(*tensors)
        return np.einsum(equation, *np_tensors)


__spec__ = {
    "aclnnBatchMatMul": "AclnnBatchMatMulTestSpec",
    "aclnnBatchMatMulWeightNz": "AclnnBatchMatMulWeightNzTestSpec",
    "aclnnAddbmm": "AclnnAddbmmTestSpec",
    "aclnnInplaceAddbmm": "AclnnInplaceAddbmmTestSpec",
    "aclnnBaddbmm": "AclnnBaddbmmTestSpec",
    "aclnnInplaceBaddbmm": "AclnnInplaceBaddbmmTestSpec",
    "aclnnEinsum": "AclnnEinsumTestSpec",
}
