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
import numpy as np
import en_dtypes
import ml_dtypes

np_fp4_e2m1 = en_dtypes.float4_e2m1
np_fp4_e1m2 = en_dtypes.float4_e1m2
np_hif8 = en_dtypes.hifloat8
np_mx_scale = en_dtypes.float8_e8m0
np_bfloat16 = ml_dtypes.bfloat16
np_fp8_e4m3 = ml_dtypes.float8_e4m3fn
np_fp8_e5m2 = ml_dtypes.float8_e5m2
np_int4 = ml_dtypes.int4

__golden__ = {"kernel": {"quant_batch_matmul_v4": "quant_batch_matmul_v4_golden"}}

# 参数命名与算子定义的对应关系:
#   x1          → 算子入参0 x1 (左矩阵/激活)         dtype: float8_e4m3fn/int8/int4
#   x2          → 算子入参1 x2 (右矩阵/权重)         dtype: float4_e2m1/int8/int4
#   bias        → 算子入参2 bias (optional)
#   x1_scale    → 算子入参3 x1Scale (x1量化scale)    dtype: float8_e8m0(MX)/int8(K_C)/None
#   x2_scale    → 算子入参4 x2Scale (x2量化scale)    dtype: float8_e8m0(MX)/bfloat16(T_CG)/int8(K_C)
#   y_scale     → 算子入参5 yScale (输出反量化scale)  dtype: uint64(T_CG)/None
#   x1_offset   → 算子入参6 x1Offset (optional)
#   x2_offset   → 算子入参7 x2Offset (optional)
#   y_offset    → 算子入参8 yOffset (optional)
#   x2_table    → 算子入参9 x2Table (optional)
#
# 量化模式命名 (对齐 tiling AnalyzeQuantType + AnalyzeX2ScaleShape 决策树):
#   MX            ── x1Scale或x2Scale为float8_e8m0, x1/x2预乘scale后matmul
#   PER_GROUP     ── groupSize>0 (非MX); fp8×fp4: K轴repeat反量化+vcvt+yScale后乘
#   PER_TENSOR    ── x2Scale.size==1 (非MX, 非PER_GROUP)
#   PER_CHANNEL   ── x2Scale.size==nSize (非MX, 非PER_GROUP)
# 计算路径由 mode+dtype 组合分发:
#   MX                    → _compute_mx (e8m0 scale预乘)
#   PER_GROUP+fp8×fp4     → _compute_t_cg (pergroup K轴repeat + vcvt + yScale后乘)
#   PER_TENSOR/CHANNEL+int8×int8 → _compute_k_c (int32 matmul + scale后乘 + bias位置)
#   PER_TENSOR/CHANNEL+fp8×fp4   → _compute_tc (标量/per-channel反量化 + vcvt + yScale后乘)
#
# 计算流程参考:
#   转置在所有路径中先于scale应用 (修复原golden的轴错误)
#   3D scale需先reshape为2D再转置
#   PER_GROUP/T_C路径vcvt使用x1的dtype而非硬编码fp8_e4m3


def quant_batch_matmul_v4_golden(
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
    (
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
    ) = customize_inputs(
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

    output_dtypes = kwargs.get("output_dtypes", ["float16"])
    out_dtype_str = output_dtypes[0]

    quant_mode = _determine_quant_mode(x1, x2, x1_scale, x2_scale, y_scale, group_size)
    x1_dtype = _dtype_to_str(x1.dtype)
    x2_dtype = _dtype_to_str(x2.dtype)

    if quant_mode == "MX":
        out = _compute_mx(
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
    elif quant_mode == "PER_GROUP":
        out = _compute_t_cg(
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
    elif quant_mode in ("PER_TENSOR", "PER_CHANNEL"):
        if x1_dtype in ("int8", "int4") and x2_dtype in ("int8", "int4"):
            out = _compute_k_c(
                x1,
                x2,
                x1_scale,
                x2_scale,
                bias,
                transpose_x1,
                transpose_x2,
                out_dtype_str,
            )
        else:
            out = _compute_tc(
                x1,
                x2,
                x2_scale,
                y_scale,
                bias,
                transpose_x1,
                transpose_x2,
                out_dtype_str,
            )
    else:
        raise ValueError(f"Unsupported quant mode: {quant_mode}")
    return [out]


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
    input_formats = kwargs.get("input_formats", ())
    input_ori_shapes = kwargs.get("input_ori_shapes", ())
    if len(input_formats) > 1 and input_formats[1] == "FRACTAL_NZ":
        x2_dtype_str = _dtype_to_str(x2.dtype)
        ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
        x2 = _nz_to_nd(x2, x2_dtype_str, ori_shape)
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


def pre_compare(*outputs, **kwargs):
    return list(outputs)


class QuantBatchMatmulV4Assets:
    golden = quant_batch_matmul_v4_golden
    customize_inputs = customize_inputs
    pre_compare = pre_compare

    tolerance = {
        "float32": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0,
            "max_re_rtol": 5.0,
            "rmse_rtol": 2.0,
            "small_value": 1e-6,
            "small_value_atol": 1e-9,
        },
        "float16": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0,
            "max_re_rtol": 10.0,
            "rmse_rtol": 2.0,
            "small_value": 0.001,
            "small_value_atol": 1e-5,
        },
        "bfloat16": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0,
            "max_re_rtol": 10.0,
            "rmse_rtol": 2.0,
            "small_value": 0.001,
            "small_value_atol": 1e-5,
        },
        "int32": {"standard": "BinaryMatch"},
        "int8": {"standard": "IsClose", "rtol": 0, "atol": 1},
    }


# ----------------------------------------------------------------------------
# 量化模式计算函数 — 每种量化类型一个函数, 函数前注释给出计算基础流程
# ----------------------------------------------------------------------------


def _compute_mx(
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
    # MX(mxFP, e8m0 scale) 计算流程 (参考mat_mul.py is_mx分支):
    # 1. x1/x2/x1_scale/x2_scale转fp32
    # 2. 3D scale reshape为2D: (d0, d1, d2) → (d0, d1*d2)
    # 3. 转置 (若transpose_x1/transpose_x2), scale同步转置
    # 4. scale沿K轴repeat×group_size, 预乘到x1/x2
    # 5. matmul(x1, x2)
    # 6. +bias(FP32 opt)
    # 7. cast到out_dtype
    x1_f = x1.astype(np.float32)
    x2_f = x2.astype(np.float32)
    x1_scale_f = x1_scale.astype(np.float32)
    x2_scale_f = x2_scale.astype(np.float32)

    gs = _get_effective_group_size(group_size)

    if x1_scale_f.ndim == 3:
        x1_scale_f = x1_scale_f.reshape(x1_scale_f.shape[0], -1)
    if x2_scale_f.ndim == 3:
        x2_scale_f = x2_scale_f.reshape(x2_scale_f.shape[0], -1)

    if transpose_x1:
        axes = _gen_axes_for_transpose(len(x1_f.shape) - 2, [1, 0])
        x1_f = np.transpose(x1_f, axes)
        axes_s = _gen_axes_for_transpose(len(x1_scale_f.shape) - 2, [1, 0])
        x1_scale_f = np.transpose(x1_scale_f, axes_s)
    if transpose_x2:
        axes = _gen_axes_for_transpose(len(x2_f.shape) - 2, [1, 0])
        x2_f = np.transpose(x2_f, axes)
        axes_s = _gen_axes_for_transpose(len(x2_scale_f.shape) - 2, [1, 0])
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

    return _cast_output_dtype(out, out_dtype_str)


def _compute_t_cg(
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
    # T_CG(FP8×FP4) 计算流程 (参考mat_mul.py else分支):
    # 1. x2(float4_e2m1)转fp32, x2_scale转fp32
    # 2. 转置x2和x2_scale (若transpose_x2)
    # 3. x2_scale沿K轴repeat×group_size, 预乘到x2: x2_deq = x2 * x2_scale_br
    # 4. vcvt模拟: x2_deq → bfloat16 → fp32 → x1_dtype → fp32 (模拟硬件vcvt指令)
    # 5. 转置x1 (若transpose_x1)
    # 6. matmul(x1, x2_deq)
    # 7. y_scale后乘(fixpipe): _u64_to_deq_scale解码uint64→float32(含高19位截断)
    # 8. +bias(FP32 opt)
    # 9. cast到out_dtype
    x1_f = x1.astype(np.float32)
    x2_f = x2.astype(np.float32)
    x2_scale_f = x2_scale.astype(np.float32)

    gs = _get_effective_group_size(group_size)

    if transpose_x2:
        axes = _gen_axes_for_transpose(len(x2_f.shape) - 2, [1, 0])
        x2_f = np.transpose(x2_f, axes)
        axes_s = _gen_axes_for_transpose(len(x2_scale_f.shape) - 2, [1, 0])
        x2_scale_f = np.transpose(x2_scale_f, axes_s)

    x2_scale_br = np.repeat(x2_scale_f, gs, axis=-2)
    k_dim = x2_f.shape[-2]
    if x2_scale_br.shape[-2] > k_dim:
        x2_scale_br = x2_scale_br[..., :k_dim, :]

    x2_deq = x2_f * x2_scale_br
    x2_deq = (
        x2_deq.astype(np_bfloat16)
        .astype(np.float32)
        .astype(x1.dtype)
        .astype(np.float32)
    )

    if transpose_x1:
        axes = _gen_axes_for_transpose(len(x1_f.shape) - 2, [1, 0])
        x1_f = np.transpose(x1_f, axes)

    out = np.matmul(x1_f, x2_deq)

    if y_scale is not None:
        deq_scale = _u64_to_deq_scale(y_scale)
        deq_scale_br = deq_scale.reshape(1, -1)[:, : out.shape[-1]]
        out = out * deq_scale_br

    if bias is not None:
        out = out + bias.astype(np.float32)

    return _cast_output_dtype(out, out_dtype_str)


def _compute_tc(
    x1, x2, x2_scale, y_scale, bias, transpose_x1, transpose_x2, out_dtype_str
):
    # PerTensor/PerChannel(fp8×fp4) 计算流程:
    # 1. x2(float4_e2m1)转fp32, x2_scale(bf16/fp16/fp32)转fp32
    # 2. 转置x2 (若transpose_x2)
    # 3. 反量化: x2_deq = x2 * x2_scale (标量广播 或 per-N-channel, 不做K轴repeat)
    # 4. vcvt模拟: x2_deq → bfloat16 → fp32 → x1_dtype → fp32
    # 5. 转置x1 (若transpose_x1)
    # 6. matmul(x1, x2_deq)
    # 7. y_scale后乘(fixpipe): _u64_to_deq_scale解码uint64→float32
    # 8. +bias(FP32 opt)
    # 9. cast到out_dtype
    x1_f = x1.astype(np.float32)
    x2_f = x2.astype(np.float32)
    x2_scale_f = x2_scale.astype(np.float32)

    if transpose_x2:
        axes = _gen_axes_for_transpose(len(x2_f.shape) - 2, [1, 0])
        x2_f = np.transpose(x2_f, axes)

    if x2_scale_f.size > 1:
        x2_deq = x2_f * x2_scale_f.reshape(1, -1)
    else:
        x2_deq = x2_f * x2_scale_f.reshape(-1)[0]

    x2_deq = (
        x2_deq.astype(np_bfloat16)
        .astype(np.float32)
        .astype(x1.dtype)
        .astype(np.float32)
    )

    if transpose_x1:
        axes = _gen_axes_for_transpose(len(x1_f.shape) - 2, [1, 0])
        x1_f = np.transpose(x1_f, axes)

    out = np.matmul(x1_f, x2_deq)

    if y_scale is not None:
        deq_scale = _u64_to_deq_scale(y_scale)
        deq_scale_br = deq_scale.reshape(1, -1)[:, : out.shape[-1]]
        out = out * deq_scale_br

    if bias is not None:
        out = out + bias.astype(np.float32)

    return _cast_output_dtype(out, out_dtype_str)


def _compute_k_c(
    x1, x2, x1_scale, x2_scale, bias, transpose_x1, transpose_x2, out_dtype_str
):
    # K_C(INT×INT) 计算流程:
    # 1. x1/x2转int32, matmul(int32累加)
    # 2. 转置 (若transpose_x1/transpose_x2)
    # 3. matmul → int32结果转float32
    # 4. x2_scale/x1_scale reshape对齐, out = matmul * x2_scale * x1_scale
    # 5. bias位置: int8时bias在scale前加, int4时bias在scale后加
    # 6. cast到out_dtype
    x1_dtype_str = _dtype_to_str(x1.dtype)
    is_int8 = x1_dtype_str == "int8"

    x1_f = x1.astype(np.int32)
    x2_f = x2.astype(np.int32)

    if transpose_x1:
        axes = _gen_axes_for_transpose(len(x1_f.shape) - 2, [1, 0])
        x1_f = np.transpose(x1_f, axes)
    if transpose_x2:
        axes = _gen_axes_for_transpose(len(x2_f.shape) - 2, [1, 0])
        x2_f = np.transpose(x2_f, axes)

    out = np.matmul(x1_f, x2_f)
    out_f = out.astype(np.float32)

    x1_scale_f = x1_scale.astype(np.float32)
    x2_scale_f = x2_scale.astype(np.float32)

    batch_dims = len(out_f.shape) - 2
    if len(x2_scale_f.shape) == 1:
        x2_scale_f = x2_scale_f.reshape(*([1] * batch_dims), 1, -1)
    if len(x1_scale_f.shape) == 1:
        x1_scale_f = x1_scale_f.reshape(*([1] * batch_dims), -1, 1)

    if is_int8:
        if bias is not None:
            out_f = out_f + bias.astype(np.float32)
        out_f = out_f * x2_scale_f * x1_scale_f
    else:
        out_f = out_f * x2_scale_f * x1_scale_f
        if bias is not None:
            out_f = out_f + bias.astype(np.float32)

    return _cast_output_dtype(out_f, out_dtype_str)


# ----------------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------------


def _determine_quant_mode(x1, x2, x1_scale, x2_scale, y_scale, group_size):
    # 对齐 tiling AnalyzeQuantType + AnalyzeX2ScaleShape 决策树:
    #   1. MX            ── x1Scale或x2Scale为float8_e8m0
    #   2. PER_GROUP     ── groupSize > 0 (非MX)
    #   3. PER_TENSOR    ── x2Scale.size == 1 (非MX, 非PER_GROUP)
    #   4. PER_CHANNEL   ── 其他 (非MX, 非PER_GROUP, scaleSize == nSize)
    if x1_scale is not None and _dtype_to_str(x1_scale.dtype) == "float8_e8m0":
        return "MX"
    if x2_scale is not None and _dtype_to_str(x2_scale.dtype) == "float8_e8m0":
        return "MX"

    if group_size > 0:
        return "PER_GROUP"

    x2_scale_size = int(np.prod(x2_scale.shape)) if x2_scale is not None else 0
    if x2_scale_size <= 1:
        return "PER_TENSOR"

    return "PER_CHANNEL"


def _get_effective_group_size(group_size):
    if group_size < 0:
        return 32
    gsK = group_size & 0xFFFF
    if gsK > 0:
        return gsK
    return 32


def _u64_to_deq_scale(u64_scale):
    shape = u64_scale.shape
    deq_u32 = u64_scale.astype(np.uint32).copy()
    deq_u32 &= np.uint32(0xFFFFE000)
    return deq_u32.view(np.float32).reshape(shape)


def _gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def _cast_output_dtype(arr, dtype_name):
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "int32": np.int32,
        "int8": np.int8,
        "bfloat16": np_bfloat16,
        "hifloat8": np_hif8,
        "float8_e4m3fn": np_fp8_e4m3,
    }
    target = dtype_map.get(dtype_name)
    if target is not None:
        return arr.astype(target)
    return arr.astype(dtype_name)


def _dtype_to_str(dtype):
    dtype_map = {
        np.float16: "float16",
        np.float32: "float32",
        np.float64: "float64",
        np.int8: "int8",
        np.int32: "int32",
        np_bfloat16: "bfloat16",
        np_int4: "int4",
        np_fp8_e4m3: "float8_e4m3fn",
        np_fp8_e5m2: "float8_e5m2",
        np_hif8: "hifloat8",
        np_mx_scale: "float8_e8m0",
        np_fp4_e2m1: "float4_e2m1",
        np_fp4_e1m2: "float4_e1m2",
    }
    return dtype_map.get(dtype, str(dtype))


def _nz_to_nd(data, dtype_str, ori_shape=None):
    shape = data.shape
    batch_dims = len(shape) - 4
    perm = list(range(batch_dims)) + [
        batch_dims + 1,
        batch_dims + 2,
        batch_dims + 0,
        batch_dims + 3,
    ]
    data = np.transpose(data, perm)
    m1 = shape[batch_dims + 1]
    m0_actual = shape[batch_dims + 2]
    n1 = shape[batch_dims + 0]
    n0_actual = shape[batch_dims + 3]
    data = data.reshape(*shape[:batch_dims], m1 * m0_actual, n1 * n0_actual)
    if ori_shape is not None:
        target_M = ori_shape[-2]
        target_N = ori_shape[-1]
        data = data[..., :target_M, :target_N]
    return data
