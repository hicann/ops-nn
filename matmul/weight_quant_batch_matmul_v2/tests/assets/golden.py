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

__golden__ = {
    "kernel": {"weight_quant_batch_matmul_v2": "weight_quant_batch_matmul_v2_golden"}
}

# 参数命名与算子定义的对应关系:
#   x                → 算子入参0 x (左矩阵/激活)            dtype: float16/bfloat16
#   weight           → 算子入参1 weight (右矩阵/权重)        dtype: int8/int4/int32/float8_e4m3fn/hifloat8/float4_e2m1
#   antiquant_scale  → 算子入参2 antiquantScale (反量化scale) dtype: float16/bfloat16/uint64/int64/float8_e8m0
#   antiquant_offset → 算子入参3 antiquantOffset (反量化offset, optional)
#   quant_scale      → 算子入参4 quantScale (输出量化scale, optional, uint64)
#   quant_offset     → 算子入参5 quantOffset (输出量化offset, optional, float32)
#   bias             → 算子入参6 bias (optional)
#   y                → 算子出参0 y (结果)                    dtype: float16/bfloat16
#
# 反量化模式命名 (与量化介绍.md对应):
#   T  = PerTensor   ── antiquantScale.shape==(1,) or (1,1)
#   C  = PerChannel  ── antiquantScale.shape==(n,) or (1,n)
#   G  = PerGroup    ── antiquantScale.shape[0]>1, antiquantGroupSize>0
#   MX = mxFP微缩放  ── antiquantScale.dtype==float8_e8m0 (weight=float4_e2m1)
#
# 计算流程总览:
#   y = x @ ANTIQUANT(weight) + bias
#   ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
#   ── fp16路径: 反量化在fp16精度下计算(模拟NPU逻辑)
#   ── bf16路径: 反量化在fp32精度下计算
#   ── 打包scale(uint64/int64): scale延后至matmul后乘(fixpipe), offset在matmul前加
#   ── MX路径: e8m0 scale沿K轴repeat×32后预乘到weight
#   ── bf16输出: 结果经bf16往返(round-trip)模拟硬件精度
#
# 注: int8输出requant路径暂未实现(binary config无int8输出用例), 输出仅支持float16/bfloat16


def weight_quant_batch_matmul_v2_golden(
    x,
    weight,
    antiquant_scale,
    antiquant_offset=None,
    quant_scale=None,
    quant_offset=None,
    bias=None,
    *,
    transpose_x: bool = False,
    transpose_weight: bool = False,
    antiquant_group_size: int = 0,
    dtype: int = -1,
    **kwargs,
):
    x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias = (
        customize_inputs(
            x,
            weight,
            antiquant_scale,
            antiquant_offset,
            quant_scale,
            quant_offset,
            bias,
            transpose_x=transpose_x,
            transpose_weight=transpose_weight,
            antiquant_group_size=antiquant_group_size,
            dtype=dtype,
            **kwargs,
        )
    )

    out_dtype_str = kwargs.get("output_dtypes", ["float16"])[0]
    weight_dtype = _dtype_to_str(weight.dtype)
    antiquant_scale_dtype = _dtype_to_str(antiquant_scale.dtype)
    x_dtype_str = _dtype_to_str(x.dtype)

    inter_dtype = _get_intermediate_dtype(x_dtype_str)
    is_f8_weight = weight_dtype in ("hifloat8", "float8_e5m2", "float8_e4m3fn")
    is_packed_scale = antiquant_scale_dtype in ("uint64", "int64")

    antiquant_mode = _determine_antiquant_mode(
        antiquant_scale_dtype, antiquant_scale, weight_dtype, antiquant_group_size
    )

    if is_packed_scale:
        y = _compute_packed_scale(
            x,
            weight,
            antiquant_scale,
            antiquant_offset,
            bias,
            inter_dtype,
            transpose_x,
            transpose_weight,
            x_dtype_str,
            out_dtype_str,
        )
    elif antiquant_mode == "MX":
        y = _compute_mx(
            x,
            weight,
            antiquant_scale,
            bias,
            inter_dtype,
            transpose_x,
            transpose_weight,
            x_dtype_str,
            out_dtype_str,
        )
    elif antiquant_mode == "G":
        y = _compute_pergroup(
            x,
            weight,
            antiquant_scale,
            antiquant_offset,
            bias,
            inter_dtype,
            is_f8_weight,
            transpose_x,
            transpose_weight,
            antiquant_group_size,
            x_dtype_str,
            out_dtype_str,
        )
    else:
        y = _compute_tc(
            x,
            weight,
            antiquant_scale,
            antiquant_offset,
            bias,
            inter_dtype,
            is_f8_weight,
            transpose_x,
            transpose_weight,
            x_dtype_str,
            out_dtype_str,
        )
    return [y]


# ----------------------------------------------------------------------------
# customize_inputs — golden侧输入预处理 (golden首行调用, 同时作为Assets类属性)
# 框架通过 kwargs 传入 tuple 类型的 input_formats/input_ori_shapes (按入参位置索引):
#   weight 在 index 1, antiquant_scale 在 index 2
# ----------------------------------------------------------------------------


def customize_inputs(
    x,
    weight,
    antiquant_scale,
    antiquant_offset=None,
    quant_scale=None,
    quant_offset=None,
    bias=None,
    *,
    transpose_x: bool = False,
    transpose_weight: bool = False,
    antiquant_group_size: int = 0,
    dtype: int = -1,
    **kwargs,
):
    input_formats = kwargs.get("input_formats", ())
    input_ori_shapes = kwargs.get("input_ori_shapes", ())
    weight_format = input_formats[1] if len(input_formats) > 1 else "ND"
    if weight_format == "FRACTAL_NZ":
        weight_dtype_str = _dtype_to_str(weight.dtype)
        ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
        weight = _nz_to_nd(weight, weight_dtype_str, ori_shape)
    return x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias


def pre_compare(*outputs, **kwargs):
    return list(outputs)


class WeightQuantBatchMatmulV2Assets:
    golden = weight_quant_batch_matmul_v2_golden
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
# 量化模式计算函数 — 每种反量化类型一个函数, 函数前注释给出计算基础流程
# ----------------------------------------------------------------------------


def _compute_tc(
    x,
    weight,
    antiquant_scale,
    antiquant_offset,
    bias,
    inter_dtype,
    is_f8_weight,
    transpose_x,
    transpose_weight,
    x_dtype_str,
    out_dtype_str,
):
    # T/C(PerTensor/PerChannel) 计算流程:
    # 1. weight转fp32, 经inter_dtype截断(模拟硬件中间精度); f8+bf16路径经bf16截断
    # 2. transpose weight (若transpose_weight)
    # 3. antiquant_scale转fp32 (f8+bf16路径经bf16截断)
    # 4. 反量化: (weight+offset)*scale, 经inter_dtype截断; 无offset则weight*scale
    # 5. transpose x, x/weight经inter_dtype截断
    # 6. matmul -> _finalize(bias, bf16往返, 输出cast)
    x_f = x.astype(np.float32)
    weight_f = _prepare_weight(weight, inter_dtype, is_f8_weight, transpose_weight)

    antiquant_scale_f = antiquant_scale.astype(np.float32)
    if is_f8_weight and inter_dtype == np_bfloat16:
        antiquant_scale_f = antiquant_scale_f.astype(np_bfloat16).astype(np.float32)
    if transpose_weight and antiquant_scale_f.ndim >= 2:
        antiquant_scale_f = np.transpose(
            antiquant_scale_f,
            _gen_axes_for_transpose(len(antiquant_scale_f.shape) - 2, [1, 0]),
        )

    if antiquant_offset is not None:
        offset_f = _truncate(antiquant_offset.astype(np.float32), inter_dtype)
        if transpose_weight and offset_f.ndim >= 2:
            offset_f = np.transpose(
                offset_f, _gen_axes_for_transpose(len(offset_f.shape) - 2, [1, 0])
            )
        weight_f = _truncate(weight_f + offset_f, inter_dtype)
        weight_f = _truncate(weight_f * antiquant_scale_f, inter_dtype)
    else:
        weight_f = _truncate(weight_f * antiquant_scale_f, inter_dtype)

    if transpose_x:
        x_f = np.transpose(x_f, _gen_axes_for_transpose(len(x_f.shape) - 2, [1, 0]))
    x_f = _truncate(x_f, inter_dtype)
    weight_f = _truncate(weight_f, inter_dtype)

    out = np.matmul(x_f, weight_f)
    return _finalize(out, bias, x_dtype_str, out_dtype_str)


def _compute_pergroup(
    x,
    weight,
    antiquant_scale,
    antiquant_offset,
    bias,
    inter_dtype,
    is_f8_weight,
    transpose_x,
    transpose_weight,
    antiquant_group_size,
    x_dtype_str,
    out_dtype_str,
):
    # G(PerGroup) 计算流程:
    # 1. weight转fp32, 经inter_dtype截断; f8+bf16路径经bf16截断
    # 2. transpose weight (若transpose_weight); 2D scale/offset同步转置
    # 3. antiquant_scale转fp32 (f8+bf16路径经bf16截断)
    # 4. 按antiquant_group_size分K轴循环: (weight[k_slice]+offset[g])*scale[g], 经inter_dtype截断
    # 5. transpose x, x/weight经inter_dtype截断
    # 6. matmul -> _finalize
    x_f = x.astype(np.float32)
    weight_f = _prepare_weight(weight, inter_dtype, is_f8_weight, transpose_weight)

    antiquant_scale_f = antiquant_scale.astype(np.float32)
    if is_f8_weight and inter_dtype == np_bfloat16:
        antiquant_scale_f = antiquant_scale_f.astype(np_bfloat16).astype(np.float32)
    if transpose_weight and antiquant_scale_f.ndim >= 2:
        antiquant_scale_f = np.transpose(
            antiquant_scale_f,
            _gen_axes_for_transpose(len(antiquant_scale_f.shape) - 2, [1, 0]),
        )

    weight_f = weight_f.copy()
    antiquant_offset_f = None
    if antiquant_offset is not None:
        antiquant_offset_f = _truncate(antiquant_offset.astype(np.float32), inter_dtype)
        if transpose_weight and antiquant_offset_f.ndim >= 2:
            antiquant_offset_f = np.transpose(
                antiquant_offset_f,
                _gen_axes_for_transpose(len(antiquant_offset_f.shape) - 2, [1, 0]),
            )
    k_size = weight_f.shape[-2]
    num_groups = _ceil_div(k_size, antiquant_group_size)
    for g_idx in range(num_groups):
        k_start = g_idx * antiquant_group_size
        k_end = min((g_idx + 1) * antiquant_group_size, k_size)
        if antiquant_offset_f is not None:
            add_result = _truncate(
                weight_f[..., k_start:k_end, :]
                + antiquant_offset_f[g_idx : g_idx + 1, :],
                inter_dtype,
            )
            weight_f[..., k_start:k_end, :] = _truncate(
                add_result * antiquant_scale_f[g_idx : g_idx + 1, :], inter_dtype
            )
        else:
            weight_f[..., k_start:k_end, :] = _truncate(
                weight_f[..., k_start:k_end, :]
                * antiquant_scale_f[g_idx : g_idx + 1, :],
                inter_dtype,
            )

    if transpose_x:
        x_f = np.transpose(x_f, _gen_axes_for_transpose(len(x_f.shape) - 2, [1, 0]))
    x_f = _truncate(x_f, inter_dtype)
    weight_f = _truncate(weight_f, inter_dtype)

    out = np.matmul(x_f, weight_f)
    return _finalize(out, bias, x_dtype_str, out_dtype_str)


def _compute_mx(
    x,
    weight,
    antiquant_scale,
    bias,
    inter_dtype,
    transpose_x,
    transpose_weight,
    x_dtype_str,
    out_dtype_str,
):
    # MX(mxFP, e8m0 scale) 计算流程:
    # 1. weight(float4_e2m1)转fp32, 经inter_dtype截断(fp16路径经fp16, bf16路径不截断)
    # 2. transpose weight (若transpose_weight)
    # 3. antiquant_scale(e8m0)转fp32, fp16路径经fp16截断
    # 4. scale沿K轴repeat×32, 奇数group裁剪末尾; weight按需padding对齐
    # 5. weight*scale, 经inter_dtype截断
    # 6. transpose x, x/weight经inter_dtype截断
    # 7. matmul -> _finalize
    # 注: MX模式(weight=float4_e2m1)不支持antiquant_offset
    x_f = x.astype(np.float32)
    weight_f = _prepare_weight(weight, inter_dtype, False, transpose_weight)

    antiquant_scale_f = antiquant_scale.astype(np.float32)
    if inter_dtype == np.float16:
        antiquant_scale_f = antiquant_scale_f.astype(np.float16).astype(np.float32)
    if transpose_weight and antiquant_scale_f.ndim >= 2:
        antiquant_scale_f = np.transpose(
            antiquant_scale_f,
            _gen_axes_for_transpose(len(antiquant_scale_f.shape) - 2, [1, 0]),
        )

    antiquant_scale_br = np.repeat(antiquant_scale_f, 32, axis=-2)
    k_dim = weight_f.shape[-2]
    if _ceil_div(k_dim, 32) % 2 != 0:
        antiquant_scale_br = antiquant_scale_br[..., :-1, :]
    weight_dims = len(weight_f.shape)
    weight_pad_len = antiquant_scale_br.shape[-2] - weight_f.shape[-2]
    if weight_pad_len > 0:
        weight_f = np.pad(
            weight_f,
            [(0, 0)] * (weight_dims - 2) + [(0, weight_pad_len)] + [(0, 0)],
            mode="constant",
            constant_values=0,
        )
    weight_f = _truncate(weight_f * antiquant_scale_br, inter_dtype)

    if transpose_x:
        x_f = np.transpose(x_f, _gen_axes_for_transpose(len(x_f.shape) - 2, [1, 0]))
    x_f = _truncate(x_f, inter_dtype)
    weight_f = _truncate(weight_f, inter_dtype)

    out = np.matmul(x_f, weight_f)
    return _finalize(out, bias, x_dtype_str, out_dtype_str)


def _compute_packed_scale(
    x,
    weight,
    antiquant_scale,
    antiquant_offset,
    bias,
    inter_dtype,
    transpose_x,
    transpose_weight,
    x_dtype_str,
    out_dtype_str,
):
    # PackedScale(uint64/int64 fixpipe) 计算流程:
    # 1. weight转fp32, 经inter_dtype截断
    # 2. transpose weight (若transpose_weight)
    # 3. pre-matmul: (weight+offset)经fp16截断; scale延后至matmul后(fixpipe模板)
    # 4. transpose x, x/weight经inter_dtype截断
    # 5. matmul
    # 6. 后乘deq_scale: _u64_to_deq_scale解码uint64→float32(含高19位截断), 经inter_dtype截断
    # 7. _finalize(bias, bf16往返, 输出cast)
    # 注: scale单次应用(post-matmul), 与mat_mul.py在uint64+offset共存时的重复应用故意不同
    x_f = x.astype(np.float32)
    weight_f = _prepare_weight(weight, inter_dtype, False, transpose_weight)

    if antiquant_offset is not None:
        offset_f = antiquant_offset.astype(np.float32)
        weight_f = _truncate(weight_f + offset_f, np.float16)
    else:
        weight_f = _truncate(weight_f, np.float16)

    if transpose_x:
        x_f = np.transpose(x_f, _gen_axes_for_transpose(len(x_f.shape) - 2, [1, 0]))
    x_f = _truncate(x_f, inter_dtype)
    weight_f = _truncate(weight_f, inter_dtype)

    out = np.matmul(x_f, weight_f)

    deq_scale_f = _u64_to_deq_scale(antiquant_scale)
    deq_scale_f = deq_scale_f.astype(np.float16).astype(np.float32)
    deq_scale_br = deq_scale_f.reshape(1, -1)[:, : out.shape[-1]]
    out = out * deq_scale_br

    return _finalize(out, bias, x_dtype_str, out_dtype_str)


# ----------------------------------------------------------------------------
# 共享辅助函数
# ----------------------------------------------------------------------------


def _prepare_weight(weight, inter_dtype, is_f8_weight, transpose_weight):
    weight_f = weight.astype(np.float32)
    if inter_dtype == np.float16:
        weight_f = weight_f.astype(np.float16).astype(np.float32)
    elif inter_dtype == np_bfloat16 and is_f8_weight:
        weight_f = weight_f.astype(np_bfloat16).astype(np.float32)
    if transpose_weight:
        weight_f = np.transpose(
            weight_f, _gen_axes_for_transpose(len(weight_f.shape) - 2, [1, 0])
        )
    return weight_f


def _finalize(out, bias, x_dtype_str, out_dtype_str):
    if bias is not None:
        out = out + bias.astype(np.float32)
    if x_dtype_str == "bfloat16" and out_dtype_str != "int8":
        out = out.astype(np_bfloat16).astype(np.float32)
    return _cast_output_dtype(out, out_dtype_str)


def _determine_antiquant_mode(
    antiquant_scale_dtype, antiquant_scale, weight_dtype, antiquant_group_size
):
    # 量化模式命名遵循量化介绍.md:
    #   T = PerTensor, C = PerChannel, G = PerGroup, MX = mxFP微缩放(float8_e8m0)
    #
    # 优先级与算子tiling一致(op_tiling/...tiling.cpp ConfigureAntiQuant):
    #   1. MX   ── antiquantScale==float8_e8m0
    #   2. T    ── antiquantScale.size==1
    #   3. G    ── antiquant_group_size>0
    #   4. C    ── 其他(scaleSize>1且groupSize==0)
    if antiquant_scale_dtype == "float8_e8m0":
        return "MX"

    shape = antiquant_scale.shape if hasattr(antiquant_scale, "shape") else ()
    shape_size = int(np.prod(shape)) if shape else 0

    if shape_size <= 1:
        return "T"
    if antiquant_group_size > 0:
        return "G"
    return "C"


def _adjust_antiquant_offset_range(offset, weight_dtype_name):
    if weight_dtype_name == "int8":
        offset = np.clip(np.round(offset), -128, 127).astype(offset.dtype)
    elif weight_dtype_name == "int4":
        offset = np.clip(np.round(offset), -8, 7).astype(offset.dtype)
    return offset


def _get_intermediate_dtype(x_dtype_str):
    if x_dtype_str == "bfloat16":
        return np_bfloat16
    elif x_dtype_str == "float16":
        return np.float16
    return None


def _truncate(arr, inter_dtype):
    if inter_dtype is None:
        return arr
    return arr.astype(inter_dtype).astype(np.float32)


def _u64_to_deq_scale(u64_scale):
    shape = u64_scale.shape
    deq_u32 = u64_scale.astype(np.uint32).copy()
    deq_u32 &= np.uint32(0xFFFFE000)
    return deq_u32.view(np.float32).reshape(shape)


def _ceil_div(a, b):
    if b == 0:
        return 0
    return (a + b - 1) // b


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
        "float8_e5m2": np_fp8_e5m2,
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
        np.uint64: "uint64",
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
