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


def _opImplMode(cubeMathType):
    """cubeMathType → kernel opImplMode: KEEP_DTYPE=0, HF32=64。"""
    if cubeMathType is None:
        return 0
    return 64 if int(cubeMathType) in (1, 3) else 0


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
        self_np, mat2_np = _util.to_numpy(self, mat2)

        # 2) cubeMathType 精度模拟
        self_np, mat2_np = _apply_cube_type(self_np, mat2_np, cubeMathType)

        # 4) 调 kernel golden 做核心 matmul
        out_dtype = _util.dtype_to_str(out if out is not None else self.dtype)
        return _kernel_bmm(
            self_np, mat2_np, cubeMathType=cubeMathType, out_dtype=out_dtype, **kwargs
        )


class AclnnBatchMatMulWeightNzTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def customize_inputs(self, mat2, out=None, cubeMathType=0, **kwargs):
        """aclnn 输入定制：NZ mat2 规范化（in-place 覆写 storage）。"""
        _util.prepare_nz_input(mat2, kwargs.get("tensor_formats", ()), 1)

    @staticmethod
    def golden(self, mat2, out=None, cubeMathType=0, **kwargs):
        """aclnnBatchMatMulWeightNz: self @ mat2（mat2 为 NZ，先解码为 ND）。"""
        mat2 = _util.nz_input_to_nd(mat2, kwargs.get("tensor_formats", ()), 1)
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
        self_np, batch1_np, batch2_np = _util.to_numpy(self, batch1, batch2)

        # 2) cubeMathType 精度模拟
        batch1_np, batch2_np = _apply_cube_type(batch1_np, batch2_np, cubeMathType)

        out_dtype = _util.dtype_to_str(out if out is not None else self.dtype)

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


_F32_EPS = float(np.finfo(np.float32).eps)


def _is_std_transpose_last2(t):
    """镜像 Ops::NN::IsTransposeLastTwoDims(matmul_util.cpp)：仅识别"标准末两维
    转置"——内轴交换(strides[dim2]==1 且 strides[dim1]==shape[dim2])且所有 batch
    维 stride 按转置布局逐层连乘完全连续；1x1 退化不算。strides 为元素单位
    (torch .stride()/aclTensor GetViewStrides 同单位)。非标准转置(如 batch 维
    交换+内轴转置组合)、空洞/切片等非连续形态均返回 False(NPU 侧不挡 scale_add)。"""
    shape = tuple(t.shape)
    ndim = len(shape)
    if ndim < 2 or ndim > 6:
        return False
    strides = t.stride() if hasattr(t, "stride") else t.strides
    if hasattr(t, "stride"):  # torch: 元素单位
        pass
    else:  # numpy: 字节单位, 换算为元素单位
        itemsize = t.dtype.itemsize
        strides = tuple(s // itemsize for s in strides)
    dim1, dim2 = ndim - 1, ndim - 2
    if strides[dim2] == 1 and strides[dim1] == shape[dim2]:
        tmp = shape[dim1] * shape[dim2]
        for batch_dim in range(ndim - 3, -1, -1):
            if strides[batch_dim] != tmp:
                return False
            tmp *= shape[batch_dim]
        if shape[dim1] == 1 and shape[dim2] == 1:
            return False
        return True
    return False


def _tensor_meta(t):
    """torch/numpy 通用的张量元信息 (dim, shape, dtype_str, 末两维是否标准转置)。"""
    if t is None:
        return 0, (), None, False
    dim = t.dim() if hasattr(t, "dim") else t.ndim
    shape = tuple(t.shape)
    dtype_str = _util.dtype_to_str(t.dtype)
    transposed = _is_std_transpose_last2(t)
    return dim, shape, dtype_str, transposed


def _is_950(kwargs):
    """DAV_3510(Ascend950 系列)。920x/960 不满足前缀, 与 C++ 侧严格 == 判定对齐。"""
    return str(kwargs.get("short_soc_version") or "").startswith("Ascend950")


def _fmm_add_supported(self, batch1, batch2, out, alpha, beta, cubeMathType, kwargs):
    """镜像 CheckFmmBaddbmmSupport(aclnn_baddbmm.cpp) + DAV_3510 平台门禁(一级路由)：
    α=β=1(eps 容差) + batch1/batch2 3D 且 batch 维一致 + self ∈ {(b,m,n),(1,m,n),(m,n)}
    末两维严格 (M,N) + 四方同 dtype ∈ {fp16,bf16,fp32} + cmt≠4 + fp32 仅 cmt=3
    + 任一输入 FRACTAL_NZ 即回退 + K>0。核配 IsAivNumDoubleOfAicNum(标准 950
    MIX_AIC_1_2)按满足处理。"""
    if not _is_950(kwargs):
        return False
    if abs(float(alpha) - 1.0) > _F32_EPS or abs(float(beta) - 1.0) > _F32_EPS:
        return False
    b1_dim, b1_shape, b1_dt, _ = _tensor_meta(batch1)
    b2_dim, b2_shape, b2_dt, _ = _tensor_meta(batch2)
    if b1_dim != 3 or b2_dim != 3:
        return False
    if b1_shape[0] != b2_shape[0] or b1_shape[2] != b2_shape[1]:
        return False
    m_dim, n_dim = b1_shape[1], b2_shape[2]
    s_dim, s_shape, s_dt, _ = _tensor_meta(self)
    if s_dim == 3:
        if (
            (s_shape[0] != b1_shape[0] and s_shape[0] != 1)
            or s_shape[1] != m_dim
            or s_shape[2] != n_dim
        ):
            return False
    elif s_dim == 2:
        if s_shape != (m_dim, n_dim):
            return False
    else:
        return False
    o_dt = _tensor_meta(out)[2] if out is not None else s_dt
    if b1_dt not in ("float16", "bfloat16", "float32"):
        return False
    if b2_dt != b1_dt or s_dt != b1_dt or o_dt != b1_dt:
        return False
    cmt = 0 if cubeMathType is None else int(cubeMathType)
    if cmt == 4:
        return False
    if b1_dt == "float32" and cmt != 3:
        return False
    fmts = kwargs.get("tensor_formats") or ()
    if any(f == "FRACTAL_NZ" for f in list(fmts)[:3]):
        return False
    if b1_shape[2] == 0:
        return False
    return True


def _scale_add_supported(self, batch1, batch2, out, alpha, beta, cubeMathType, kwargs):
    """镜像 CheckScaleAddSupport + DAV_3510 门禁(二级路由, 承接一级未命中的 α/β
    任意缩放)：3D BMM + self 严格 (b,m,n) 完整 3D(不支持 (1,m,n)/(m,n) 广播)
    + 仅 fp16/bf16 + 四方同 dtype + batch1/batch2 末两维非转置 + 非 NZ + K>0。
    核配按满足处理。"""
    if not _is_950(kwargs):
        return False
    b1_dim, b1_shape, b1_dt, b1_tp = _tensor_meta(batch1)
    b2_dim, b2_shape, b2_dt, b2_tp = _tensor_meta(batch2)
    if b1_dim != 3 or b2_dim != 3:
        return False
    if b1_shape[0] != b2_shape[0] or b1_shape[2] != b2_shape[1]:
        return False
    m_dim, n_dim = b1_shape[1], b2_shape[2]
    s_dim, s_shape, s_dt, _ = _tensor_meta(self)
    if (
        s_dim != 3
        or s_shape[0] != b1_shape[0]
        or s_shape[1] != m_dim
        or s_shape[2] != n_dim
    ):
        return False
    if b1_dt not in ("float16", "bfloat16"):
        return False
    o_dt = _tensor_meta(out)[2] if out is not None else s_dt
    if b2_dt != b1_dt or s_dt != b1_dt or o_dt != b1_dt:
        return False
    if b1_tp or b2_tp:
        return False
    fmts = kwargs.get("tensor_formats") or ()
    if any(f == "FRACTAL_NZ" for f in list(fmts)[:3]):
        return False
    if b1_shape[2] == 0:
        return False
    return True


class AclnnBaddbmmTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        self, batch1, batch2, beta=1.0, alpha=1.0, out=None, cubeMathType=0, **kwargs
    ):
        """aclnnBaddbmm: beta*self + alpha*(batch1 @ batch2)。

        950(DAV_3510) 上对齐 aclnn_baddbmm.cpp 的 fmm 两级路由(升精度真值范式：
        累加域融合 + 单次截回，消除分解链 bmm 的中间 round)：
          一级 α=β=1 → fmm BMM+add 单 kernel 融合 (mm 累加域 + self 相加后单次截回)
          二级 α/β 任意 + fp16/bf16 完整 3D self → fmm scale_add (α·mm + β·self 累加域
          融合单次截回; innerPrecise 恒高性能, "scale_add" 不在 kSupportedX3OpTypes)
        其余平台/门禁未命中 → 既有分解链(Muls+BMM+Add 拼接)，行为零变化。
        """
        # 1) torch → numpy
        self_np, batch1_np, batch2_np = _util.to_numpy(self, batch1, batch2)

        # 2) cubeMathType 精度模拟
        batch1_np, batch2_np = _apply_cube_type(batch1_np, batch2_np, cubeMathType)

        out_dtype = _util.dtype_to_str(out if out is not None else self.dtype)

        # 3) 950 fmm 两级路由：累加域融合 + 单次截回（out_dtype=None 使 kernel
        #    golden 保留累加精度）。两级公式分开计算：
        #    一级 α=β=1(eps 容差) → ExecBaddbmmFmmOp 直调 aclnnFusedMatmul(bias=self)
        #      不携带 alpha/beta —— 门禁 eps 内偏离 1 的缩放系数被设备丢弃，
        #      golden 同步不缩放(直接 mm + self)，否则 (1+eps)·mm 的伪差异在
        #      抵消点被放大(如 mm=1024/self=-1024/a=1+2^-23: 设备 0, 旧统一公式
        #      2^-13=0.000122, 误报精度失败)；
        #    二级 scale_add → ExecBaddbmmScaleAddOp 携带实际 alpha/beta
        #      (FusedMatMulWithScaleAddNd)，按实际 α·mm + β·self 缩放
        if _fmm_add_supported(
            self, batch1, batch2, out, alpha, beta, cubeMathType, kwargs
        ):
            bmm = _kernel_bmm(
                batch1_np,
                batch2_np,
                cubeMathType=cubeMathType,
                out_dtype=None,
                **kwargs,
            )
            comp = np.result_type(bmm.dtype, np.float32)
            fused = bmm.astype(comp) + self_np.astype(comp)
            return _util.cast_output_dtype(fused, out_dtype) if out_dtype else fused
        if _scale_add_supported(
            self, batch1, batch2, out, alpha, beta, cubeMathType, kwargs
        ):
            bmm = _kernel_bmm(
                batch1_np,
                batch2_np,
                cubeMathType=cubeMathType,
                out_dtype=None,
                **kwargs,
            )
            comp = np.result_type(bmm.dtype, np.float32)
            fused = comp.type(float(alpha)) * bmm.astype(comp) + comp.type(
                float(beta)
            ) * self_np.astype(comp)
            return _util.cast_output_dtype(fused, out_dtype) if out_dtype else fused

        # 4) 既有分解链（Muls+BMM+Add 拼接，bmm 先 round 到 out_dtype）
        bmm = _kernel_bmm(
            batch1_np,
            batch2_np,
            cubeMathType=cubeMathType,
            out_dtype=out_dtype,
            **kwargs,
        )

        #    Step 4a: SelfMulsBetaProcess — 对齐 cann-asc-devkit 标杆(aclnnBaddbmm.py
        #    CPU golden 分支): β·self 在 fp32 域计算后 round 到 out dtype(标杆
        #    mid_dtype; fp32 输出不 round)。原 BF16 保持 fp32 无 round 的建模(按
        #    C++ SelfMulsBeta 字面语义)与 NPU 实际相差 self 项 1 ULP —— 分解链
        #    case 残差 2% 的根因。
        mulOut = self_np.astype(np.float32) * np.float32(beta)
        if out_dtype and out_dtype != "float32":
            mulOut = _util.cast_output_dtype(mulOut, out_dtype)

        #    Step 4b/4c: Axpy/Add 是融合算子(单 pass: 读 bf16 -> fp32 域 x + α·y ->
        #    单次 round 写出), 不能让 promote 落回低精度域后用 numpy 逐元素双
        #    round(55acbbd75 版 mulOut round 后 result_type(bf16,bf16)=bf16,
        #    Axpy 塌到 bf16 域双 round, 与 NPU 融合语义多 1 ULP —— 79.56% 根因);
        #    统一在 fp32 域融合, 单次截回 out dtype
        mul_out_f = mulOut.astype(np.float32)
        bmm_f = bmm.astype(np.float32)
        if abs(alpha - 1.0) <= _F32_EPS:
            fused = mul_out_f + bmm_f
        else:
            fused = mul_out_f + np.float32(alpha) * bmm_f

        #    Step 4d: Cast to output dtype(单次截回)
        return _util.cast_output_dtype(fused, out_dtype) if out_dtype else fused


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
        np_tensors = _util.to_numpy(*tensors)
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
