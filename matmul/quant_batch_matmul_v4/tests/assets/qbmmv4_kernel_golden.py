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
"""
quant_batch_matmul_v4 kernel golden 实现 (仅覆盖 Ascend 950 场景)。

本模块包含 V4 kernel 的 TestSpec 类和入口函数 qbmmv4_kernel_compute。

950 V4 支持的量化模式 (对应 arch35 tiling QuantType 枚举):
  - MX (4):               FP8×FP4 + E8M0 scale 预乘
  - T_CG (PER_GROUP 3):   FP8×FP4 + BF16/FP16 x2Scale + UINT64 yScale
  - PER_TILE (5):         INT8×INT8 + FLOAT32 多维 per-tile scale
  - INT4_ASYMMETRICAL (6): INT4×INT4 + FLOAT32 scale + FP16 x2Offset (per-group 非对称)
"""

import os
import sys
import warnings

import numpy as np

# Import common utilities
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../common/tests/st/arch35"
    ),
)
from matmul_golden_util import (
    nz_to_nd,
    dtype_to_str,
    cast_output_dtype,
    np_bfloat16,
)
from matmul_quant_util import (
    gen_axes_for_transpose,
    get_effective_group_size,
    u64_to_deq_scale,
    sanitize_e8m0_scale,
    generate_u64_scale,
    ceil_div,
    unpack_groupsize,
)

# FP8 输入 dtype (E4M3FN / E5M2)
FP8_INPUT_DTYPE = ("float8_e4m3fn", "float8_e5m2")
# FP4 输入 dtype
FP4_INPUT_DTYPE = ("float4_e2m1",)


# ============================================================================
# V4 量化模式判定 (仅 950, 对齐 arch35 tiling QuantType 枚举)
# ============================================================================


def determine_quant_mode(x1, x2, x1_scale, x2_scale, y_scale, x2_offset):
    """根据输入参数的 dtype 和 shape 判断量化模式。

    对应 arch35 tiling 的 AnalyzeQuantType + AnalyzeX2ScaleShape 决策树:
      1. MX                ── x1Scale 或 x2Scale 为 float8_e8m0
      2. INT4_ASYMMETRICAL ── INT4×INT4 + x2Offset (非对称 per-group)
      3. PER_TILE          ── INT8×INT8 + FLOAT32 多维 scales (perblock)
      4. T_CG              ── FP8×FP4 + yScale (tiling 中为 PER_GROUP)

    返回: "MX", "INT4_ASYMMETRICAL", "PER_TILE", "T_CG"
    """
    x1_dt = dtype_to_str(x1.dtype)
    x2_dt = dtype_to_str(x2.dtype)
    x1s_dt = dtype_to_str(x1_scale.dtype) if x1_scale is not None else None
    x2s_dt = dtype_to_str(x2_scale.dtype) if x2_scale is not None else None

    # 1. MX: E8M0 scale (对应 QuantType::MX)
    if x1s_dt == "float8_e8m0" or x2s_dt == "float8_e8m0":
        return "MX"

    # 2. INT4_ASYMMETRICAL: INT4×INT4 + x2Offset (对应 QuantType::INT4_ASYMMETRICAL)
    if x1_dt == "int4" and x2_dt == "int4" and x2_offset is not None:
        return "INT4_ASYMMETRICAL"

    # 3. PER_TILE: INT8×INT8 + FLOAT32 多维 scales (对应 QuantType::PER_TILE)
    #    kernel 要求 x1Scale.ndim == x1.ndim 且 x2Scale.ndim == x2.ndim
    if (
        x1_scale is not None
        and x2_scale is not None
        and x1_dt == "int8"
        and x2_dt == "int8"
        and x1s_dt == "float32"
        and x2s_dt == "float32"
        and x1_scale.ndim == x1.ndim
        and x2_scale.ndim == x2.ndim
    ):
        return "PER_TILE"

    # 4. T_CG: FP8×FP4 + yScale (对应 QuantType::PER_GROUP in tiling key)
    if y_scale is not None and x1_dt in FP8_INPUT_DTYPE and x2_dt in FP4_INPUT_DTYPE:
        return "T_CG"

    raise ValueError(
        f"Unsupported V4 quant mode: x1={x1_dt}, x2={x2_dt}, "
        f"x1Scale={x1s_dt}, x2Scale={x2s_dt}, "
        f"yScale={'not None' if y_scale is not None else 'None'}, "
        f"x2Offset={'not None' if x2_offset is not None else 'None'}"
    )


# ============================================================================
# V4 compute functions
# ============================================================================


def compute_mx(
    x1,
    x2,
    x1_scale,
    x2_scale,
    bias,
    transpose_x1,
    transpose_x2,
    group_size,
    out_dtype_str,
):
    """MX (Microscaling FP) 量化模式计算。

    量化模式 (对应 arch35 QuantType::MX):
    - x1/x2 为 FP8/FP4, x1Scale/x2Scale 为 float8_e8m0
    - 每 group_size(默认32) 个元素共享一个 scale

    计算流程:
    1. x1/x2/x1_scale/x2_scale 转 fp32
    2. 3D scale reshape 为 2D
    3. 转置 (若 transpose_x1/transpose_x2), scale 同步转置
    4. scale 沿 K 轴 repeat x group_size, 预乘到 x1/x2
    5. matmul(x1, x2)
    6. + bias (FP32 opt)
    7. cast 到 out_dtype
    """
    x1_f = x1.astype(np.float32)
    x2_f = x2.astype(np.float32)
    x1_scale_f = x1_scale.astype(np.float32)
    x2_scale_f = x2_scale.astype(np.float32)

    gs = get_effective_group_size(group_size)

    if x1_scale_f.ndim == 3:
        x1_scale_f = x1_scale_f.reshape(x1_scale_f.shape[0], -1)
    if x2_scale_f.ndim == 3:
        x2_scale_f = x2_scale_f.reshape(x2_scale_f.shape[0], -1)

    if transpose_x1:
        axes = gen_axes_for_transpose(len(x1_f.shape) - 2, [1, 0])
        x1_f = np.transpose(x1_f, axes)
        axes_s = gen_axes_for_transpose(len(x1_scale_f.shape) - 2, [1, 0])
        x1_scale_f = np.transpose(x1_scale_f, axes_s)
    if transpose_x2:
        axes = gen_axes_for_transpose(len(x2_f.shape) - 2, [1, 0])
        x2_f = np.transpose(x2_f, axes)
        axes_s = gen_axes_for_transpose(len(x2_scale_f.shape) - 2, [1, 0])
        x2_scale_f = np.transpose(x2_scale_f, axes_s)

    x1_scale_br = np.repeat(x1_scale_f, gs, axis=-1)
    k_dim_x1 = x1_f.shape[-1]
    if x1_scale_br.shape[-1] > k_dim_x1:
        x1_scale_br = x1_scale_br[..., :k_dim_x1]

    x2_scale_br = np.repeat(x2_scale_f, gs, axis=-2)
    k_dim_x2 = x2_f.shape[-2]
    if x2_scale_br.shape[-2] > k_dim_x2:
        x2_scale_br = x2_scale_br[..., :k_dim_x2, :]

    x1_f = x1_f * x1_scale_br
    x2_f = x2_f * x2_scale_br

    out = np.matmul(x1_f, x2_f)

    if bias is not None:
        out = out + bias.astype(np.float32)

    return cast_output_dtype(out, out_dtype_str)


def compute_t_cg(
    x1,
    x2,
    x2_scale,
    y_scale,
    bias,
    transpose_x1,
    transpose_x2,
    group_size,
    out_dtype_str,
):
    """T_CG (FP8 x FP4 + yScale) 量化模式计算。

    量化模式 (对应 arch35 QuantType::PER_GROUP in tiling key):
    - x1=FP8, x2=FP4, x1Scale=null, x2Scale=BF16/FP16, yScale=UINT64

    计算流程:
    1. x2(float4_e2m1) 转 fp32, x2_scale 转 fp32
    2. 转置 x2 和 x2_scale (若 transpose_x2)
    3. x2_scale 沿 K 轴 repeat x group_size
    4. vcvt 模拟 (对齐 kernel CastWeightF4ToF16, scaleType=out_dtype):
       x2 -> scale_mid(bf16/fp16) -> scale_mid * scale -> fp32 -> x1_dtype -> fp32
    5. 转置 x1 (若 transpose_x1)
    6. matmul(x1, x2_deq)
    7. y_scale 后乘 (fixpipe): u64_to_deq_scale 解码 uint64->float32
    8. + bias (FP32 opt)
    9. cast 到 out_dtype
    """
    x1_f = x1.astype(np.float32)
    x2_f = x2.astype(np.float32)
    x2_scale_f = x2_scale.astype(np.float32)

    gs = get_effective_group_size(group_size)

    if transpose_x2:
        axes = gen_axes_for_transpose(len(x2_f.shape) - 2, [1, 0])
        x2_f = np.transpose(x2_f, axes)
        axes_s = gen_axes_for_transpose(len(x2_scale_f.shape) - 2, [1, 0])
        x2_scale_f = np.transpose(x2_scale_f, axes_s)

    x2_scale_br = np.repeat(x2_scale_f, gs, axis=-2)
    k_dim = x2_f.shape[-2]
    if x2_scale_br.shape[-2] > k_dim:
        x2_scale_br = x2_scale_br[..., :k_dim, :]

    scale_mid = (
        np_bfloat16 if dtype_to_str(x2_scale.dtype) == "bfloat16" else np.float16
    )
    x2_cvt = x2_f.astype(scale_mid)
    x2_deq = (x2_cvt * x2_scale_br.astype(scale_mid)).astype(np.float32)
    x2_deq = x2_deq.astype(x1.dtype).astype(np.float32)

    if transpose_x1:
        axes = gen_axes_for_transpose(len(x1_f.shape) - 2, [1, 0])
        x1_f = np.transpose(x1_f, axes)

    out = np.matmul(x1_f, x2_deq)

    if y_scale is not None:
        deq_scale = u64_to_deq_scale(y_scale)
        deq_scale_br = deq_scale.reshape(1, -1)[:, : out.shape[-1]]
        out = out * deq_scale_br

    if bias is not None:
        out = out + bias.astype(np.float32)

    return cast_output_dtype(out, out_dtype_str)


def compute_per_tile(
    x1,
    x2,
    x1_scale,
    x2_scale,
    bias,
    transpose_x1,
    transpose_x2,
    group_size,
    out_dtype_str,
):
    """PER_TILE (INT8 x INT8 per-block) 量化模式计算。

    量化模式 (对应 arch35 QuantType::PER_TILE):
    - x1=INT8, x2=INT8, x1Scale=FLOAT32 [M, ceil(K/gK)], x2Scale=FLOAT32 [ceil(K/gK), ceil(N/gN)]
    - 950 约束: groupSizeM=1, groupSizeN=128, groupSizeK=128

    kernel (CMCT Pertile, block_epilogue_pertile.h):
    - cube: INT8xINT8->INT32 matmul per K-tile(128)
    - vector: FP32 scale 乘法 + K 累加, 最后加 bias

    公式: out = sum_g (matmul(x1_g, x2_g) * x1Scale[:,g] * x2Scale[g,:]) + bias
    """
    if group_size < 0:
        group_size_m, group_size_n, group_size_k = 1, 128, 128
    else:
        group_size_m, group_size_n, group_size_k = unpack_groupsize(group_size)

    x1_f = x1.astype(np.int32)
    x2_f = x2.astype(np.int32)
    x1_scale_f = x1_scale.astype(np.float32)
    x2_scale_f = x2_scale.astype(np.float32)

    if transpose_x1:
        axes = gen_axes_for_transpose(len(x1_f.shape) - 2, [1, 0])
        x1_f = np.transpose(x1_f, axes)
        s_axes = gen_axes_for_transpose(len(x1_scale_f.shape) - 2, [1, 0])
        x1_scale_f = np.transpose(x1_scale_f, s_axes)
    if transpose_x2:
        axes = gen_axes_for_transpose(len(x2_f.shape) - 2, [1, 0])
        x2_f = np.transpose(x2_f, axes)
        s_axes = gen_axes_for_transpose(len(x2_scale_f.shape) - 2, [1, 0])
        x2_scale_f = np.transpose(x2_scale_f, s_axes)

    m = x1_f.shape[-2]
    k = x1_f.shape[-1]
    n = x2_f.shape[-1]

    # x1Scale: [ceil(M/gM), ceil(K/gK)] -> repeat M -> [M, ceil(K/gK)]
    x1_scale_expanded = np.repeat(x1_scale_f, group_size_m, axis=-2)[..., :m, :]
    # x2Scale: [ceil(K/gK), ceil(N/gN)] -> repeat N -> [ceil(K/gK), N]
    x2_scale_expanded = np.repeat(x2_scale_f, group_size_n, axis=-1)[..., :n]

    k_tile_count = ceil_div(k, group_size_k)

    out = np.zeros((*x1_f.shape[:-2], m, n), dtype=np.float32)
    for k_idx in range(k_tile_count):
        k_start = k_idx * group_size_k
        k_end = min(k_start + group_size_k, k)
        mm_tile = np.matmul(x1_f[..., :, k_start:k_end], x2_f[..., k_start:k_end, :])
        tile_scale = np.expand_dims(
            x1_scale_expanded[..., :, k_idx], axis=-1
        ) * np.expand_dims(x2_scale_expanded[..., k_idx, :], axis=-2)
        out += mm_tile.astype(np.float32) * tile_scale

    if bias is not None:
        out = out + bias.astype(np.float32)
    return cast_output_dtype(out, out_dtype_str)


def compute_int4_asymmetrical(
    x1,
    x2,
    x1_scale,
    x2_scale,
    x2_offset,
    transpose_x1,
    transpose_x2,
    out_dtype_str,
):
    """INT4_ASYMMETRICAL (INT4 x INT4 per-group 非对称) 量化模式计算。

    量化模式 (对应 arch35 QuantType::INT4_ASYMMETRICAL):
    - x1=INT4, x2=INT4, x1Scale=FLOAT32 [M, 1], x2Scale=FLOAT32 [ceil(K/256), N],
      x2Offset=FLOAT16 [ceil(K/256), N]
    - 950 约束: groupSizeK=256, transA=false, transB=true, K 对齐 1024, N 对齐 256

    kernel (quant_batch_matmul_v4_pergroup.h, DequantAndAccu + X2OffsetProcess):
    - INT4->INT8 预处理后做 INT8xINT8->INT32 matmul per K-group(256)
    - 每 group: acc += (mm_g - sum(x1_g) * x2Offset_g) * x2Scale_g
    - 最后: out = acc * x1Scale

    公式: out = x1Scale * sum_g (matmul(x1_g, x2_g) - sum(x1_g)*x2Offset_g) * x2Scale_g
    """
    if transpose_x1:
        warnings.warn(
            "INT4_ASYMMETRICAL: transpose_x1=True not supported by kernel "
            "(transA=false constraint)",
            stacklevel=2,
        )

    GROUP_SIZE_K = 256

    x1_f = x1.astype(np.int32)
    x2_f = x2.astype(np.int32)

    if transpose_x1:
        axes = gen_axes_for_transpose(len(x1_f.shape) - 2, [1, 0])
        x1_f = np.transpose(x1_f, axes)
    if transpose_x2:
        axes = gen_axes_for_transpose(len(x2_f.shape) - 2, [1, 0])
        x2_f = np.transpose(x2_f, axes)

    m = x1_f.shape[-2]
    k = x1_f.shape[-1]
    n = x2_f.shape[-1]
    n_kgroups = ceil_div(k, GROUP_SIZE_K)

    x1_scale_f = x1_scale.astype(np.float32)
    x2_scale_f = x2_scale.astype(np.float32)
    x2_offset_f = x2_offset.astype(np.float32)

    out = np.zeros((*x1_f.shape[:-2], m, n), dtype=np.float32)
    for g in range(n_kgroups):
        k_start = g * GROUP_SIZE_K
        k_end = min(k_start + GROUP_SIZE_K, k)
        x1_g = x1_f[..., :, k_start:k_end]
        x2_g = x2_f[..., k_start:k_end, :]

        mm_g = np.matmul(x1_g, x2_g).astype(np.float32)
        x1_sum_g = np.sum(x1_g.astype(np.float32), axis=-1, keepdims=True)
        offset_g = x1_sum_g * x2_offset_f[..., g : g + 1, :]
        out += (mm_g - offset_g) * x2_scale_f[..., g : g + 1, :]

    # x1Scale: [M, 1] -> broadcast to [M, N] via last dim
    out = out * x1_scale_f[..., :, 0:1]

    return cast_output_dtype(out, out_dtype_str)


# ============================================================================
# Entry function
# ============================================================================


def qbmmv4_kernel_compute(
    x1,
    x2,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    y_scale=None,
    x1_offset=None,
    x2_offset=None,
    y_offset=None,
    x2_table=None,
    *,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    dtype: int = -1,
    compute_type: int = -1,
    group_size: int = -1,
    **kwargs,
):
    """quant_batch_matmul_v4 的 kernel 级别 numpy 模拟计算入口。

    根据量化模式分发到对应的计算函数 (仅 950 V4 场景):
    - MX:                compute_mx (E8M0 scale 预乘)
    - T_CG:              compute_t_cg (FP8 x FP4 per-group + yScale 后乘)
    - PER_TILE:          compute_per_tile (INT8 x INT8 per-tile 累加)
    - INT4_ASYMMETRICAL: compute_int4_asymmetrical (INT4 x INT4 per-group 非对称)
    """
    output_dtypes = kwargs.get("output_dtypes", ["float16"])
    out_dtype_str = output_dtypes[0]

    quant_mode = determine_quant_mode(x1, x2, x1_scale, x2_scale, y_scale, x2_offset)

    if quant_mode == "MX":
        out = compute_mx(
            x1,
            x2,
            x1_scale,
            x2_scale,
            bias,
            transpose_x1,
            transpose_x2,
            group_size,
            out_dtype_str,
        )
    elif quant_mode == "T_CG":
        out = compute_t_cg(
            x1,
            x2,
            x2_scale,
            y_scale,
            bias,
            transpose_x1,
            transpose_x2,
            group_size,
            out_dtype_str,
        )
    elif quant_mode == "PER_TILE":
        out = compute_per_tile(
            x1,
            x2,
            x1_scale,
            x2_scale,
            bias,
            transpose_x1,
            transpose_x2,
            group_size,
            out_dtype_str,
        )
    elif quant_mode == "INT4_ASYMMETRICAL":
        out = compute_int4_asymmetrical(
            x1,
            x2,
            x1_scale,
            x2_scale,
            x2_offset,
            transpose_x1,
            transpose_x2,
            out_dtype_str,
        )
    else:
        raise ValueError(f"Unsupported quant mode for kernel: {quant_mode}")

    return [out]


# ============================================================================
# TestSpec
# ============================================================================


class QuantBatchMatmulV4TestSpec:
    """quant_batch_matmul_v4 算子的 TestSpec 类。"""

    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        x1_scale=None,
        x2_scale=None,
        y_scale=None,
        x1_offset=None,
        x2_offset=None,
        y_offset=None,
        x2_table=None,
        *,
        transpose_x1: bool = False,
        transpose_x2: bool = False,
        dtype: int = -1,
        compute_type: int = -1,
        group_size: int = -1,
        **kwargs,
    ):
        """Kernel golden: NZ->ND 转换 x2 + 调用 kernel 计算。"""
        input_formats = kwargs.get("input_formats", ())
        input_ori_shapes = kwargs.get("input_ori_shapes", ())
        if len(input_formats) > 1 and input_formats[1] == "FRACTAL_NZ":
            ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
            x2 = nz_to_nd(x2, ori_shape)

        return qbmmv4_kernel_compute(
            x1,
            x2,
            bias,
            x1_scale,
            x2_scale,
            y_scale,
            x1_offset,
            x2_offset,
            y_offset,
            x2_table,
            transpose_x1=transpose_x1,
            transpose_x2=transpose_x2,
            dtype=dtype,
            compute_type=compute_type,
            group_size=group_size,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        bias=None,
        x1_scale=None,
        x2_scale=None,
        y_scale=None,
        x1_offset=None,
        x2_offset=None,
        y_offset=None,
        x2_table=None,
        *,
        transpose_x1: bool = False,
        transpose_x2: bool = False,
        dtype: int = -1,
        compute_type: int = -1,
        group_size: int = -1,
        **kwargs,
    ):
        """输入预处理: 清洗 E8M0 NaN + 生成 UINT64 scale。"""
        testcase_name = kwargs.get("testcase_name", "unknown")

        x1_scale = sanitize_e8m0_scale(x1_scale, testcase_name)
        x2_scale = sanitize_e8m0_scale(x2_scale, testcase_name)

        if y_scale is not None and y_scale.dtype.name == "uint64":
            y_scale = generate_u64_scale(y_scale.shape)

        if x2_scale is not None and x2_scale.dtype.name == "uint64":
            x2_scale = generate_u64_scale(x2_scale.shape)

        return (
            x1,
            x2,
            bias,
            x1_scale,
            x2_scale,
            y_scale,
            x1_offset,
            x2_offset,
            y_offset,
            x2_table,
        )

    @staticmethod
    def pre_compare(*outputs, **kwargs):
        return list(outputs)


__spec__ = {
    "quant_batch_matmul_v4": "QuantBatchMatmulV4TestSpec",
}
