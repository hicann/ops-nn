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
weight_quant_batch_matmul_v2 kernel golden implementation.

This module contains V2 kernel compute functions and TestSpec class.
"""

import os
import sys
import numpy as np
from importlib import import_module

_common = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../common/tests/st/arch35"
)
if _common not in sys.path:
    sys.path.insert(0, _common)

_util = import_module("matmul_golden_util")
_quant_util = import_module("matmul_quant_util")


# ============================================================================
# V2-specific utilities
# ============================================================================


def determine_antiquant_mode(
    antiquant_scale_dtype, antiquant_scale, antiquant_group_size
):
    """根据 scale dtype 和 shape 判断反量化模式（MX/T/G/C）。"""
    if antiquant_scale_dtype == "float8_e8m0":
        return "MX"
    shape = antiquant_scale.shape if hasattr(antiquant_scale, "shape") else ()
    shape_size = int(np.prod(shape)) if shape else 0
    if shape_size <= 1:
        return "T"
    if antiquant_group_size > 0:
        return "G"
    return "C"


def prepare_weight(weight, inter_dtype, is_f8_weight, transpose_weight):
    """准备 weight：转换为 fp32，按需截断到中间精度，可选转置。"""
    weight_f = weight.astype(np.float32)
    if inter_dtype == np.float16:
        weight_f = weight_f.astype(np.float16).astype(np.float32)
    elif inter_dtype == _util.np_bfloat16 and is_f8_weight:
        weight_f = weight_f.astype(_util.np_bfloat16).astype(np.float32)
    if transpose_weight:
        weight_f = _util.transpose_last_two_dims(weight_f)
    return weight_f


def prepare_scale(antiquant_scale, inter_dtype, is_f8_weight, transpose_weight):
    """准备 antiquant scale：转为 fp32，f8+bf16 时额外截断，可选转置。"""
    scale_f = antiquant_scale.astype(np.float32)
    if is_f8_weight and inter_dtype == _util.np_bfloat16:
        scale_f = scale_f.astype(_util.np_bfloat16).astype(np.float32)
    if transpose_weight and scale_f.ndim >= 2:
        scale_f = _util.transpose_last_two_dims(scale_f)
    return scale_f


def prepare_offset(antiquant_offset, inter_dtype, transpose_weight):
    """准备 antiquant offset：截断到中间精度，可选转置。"""
    offset_f = antiquant_offset.astype(np.float32)
    if inter_dtype is not None:
        offset_f = offset_f.astype(inter_dtype).astype(np.float32)
    if transpose_weight and offset_f.ndim >= 2:
        offset_f = _util.transpose_last_two_dims(offset_f)
    return offset_f


def matmul_and_finalize(
    x_f, weight_f, bias, inter_dtype, transpose_x, x_dtype_str, out_dtype_str
):
    """公共尾部：转置 x，截断 x/weight，矩阵乘法，后处理。"""
    if transpose_x:
        x_f = _util.transpose_last_two_dims(x_f)
    x_f = _util.truncate(x_f, inter_dtype)
    weight_f = _util.truncate(weight_f, inter_dtype)
    out = np.matmul(x_f, weight_f)
    return finalize(out, bias, x_dtype_str, out_dtype_str)


def finalize(out, bias, x_dtype_str, out_dtype_str):
    """后处理：加 bias，bf16 往返截断，转换为输出 dtype。"""
    if bias is not None:
        out = out + bias.astype(np.float32)
    if x_dtype_str == "bfloat16" and out_dtype_str != "int8":
        out = out.astype(_util.np_bfloat16).astype(np.float32)
    return _util.cast_output_dtype(out, out_dtype_str)


def wqbmm_kernel_compute(
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
    """Weight quantized batch matmul 的 kernel 级别 numpy 模拟计算。

    根据量化模式分发到对应的计算函数：
    - Packed Scale (uint64/int64): _compute_packed_scale
    - MX (float8_e8m0): _compute_mx
    - PerGroup: _compute_pergroup
    - PerTensor/PerChannel: _compute_tc
    """
    out_dtype_str = kwargs.get("output_dtypes", ["float16"])[0]
    weight_dtype = _util.dtype_to_str(weight.dtype)
    antiquant_scale_dtype = _util.dtype_to_str(antiquant_scale.dtype)
    x_dtype_str = _util.dtype_to_str(x.dtype)

    inter_dtype = _util.get_intermediate_dtype(x_dtype_str)
    is_f8_weight = weight_dtype in ("hifloat8", "float8_e5m2", "float8_e4m3fn")
    is_packed_scale = antiquant_scale_dtype in ("uint64", "int64")

    antiquant_mode = determine_antiquant_mode(
        antiquant_scale_dtype, antiquant_scale, antiquant_group_size
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
    """PerTensor/PerChannel 量化模式计算。

    量化模式：
    - PerTensor (T): 整个 weight 使用单个 scale/offset
    - PerChannel (C): weight 的每个输出通道使用独立的 scale/offset

    计算流程：
    1. 准备输入：x, weight, scale, offset 转为 fp32 并按需截断
    2. 反量化 weight：weight_f = truncate((weight_f + offset_f) * scale_f)
    3. 矩阵乘法：out = x_f @ weight_f
    4. 后处理：加 bias，按 x_dtype 截断，转为输出 dtype
    """
    x_f = x.astype(np.float32)
    weight_f = prepare_weight(weight, inter_dtype, is_f8_weight, transpose_weight)
    scale_f = prepare_scale(
        antiquant_scale, inter_dtype, is_f8_weight, transpose_weight
    )

    if antiquant_offset is not None:
        offset_f = prepare_offset(antiquant_offset, inter_dtype, transpose_weight)
        weight_f = _util.truncate(weight_f + offset_f, inter_dtype)
        weight_f = _util.truncate(weight_f * scale_f, inter_dtype)
    else:
        weight_f = _util.truncate(weight_f * scale_f, inter_dtype)

    return matmul_and_finalize(
        x_f, weight_f, bias, inter_dtype, transpose_x, x_dtype_str, out_dtype_str
    )


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
    """PerGroup 量化模式计算（分组量化）。

    量化模式：
    - PerGroup (G): weight 的 K 维度按 group_size 分组，每组使用独立的 scale/offset

    计算流程：
    1. 准备输入：x, weight, scale, offset 转为 fp32 并按需截断
    2. 分组反量化 weight：对 K 维度的每组应用独立的 scale/offset
    3. 矩阵乘法：out = x_f @ weight_f
    4. 后处理：加 bias，按 x_dtype 截断，转为输出 dtype
    """
    x_f = x.astype(np.float32)
    weight_f = prepare_weight(weight, inter_dtype, is_f8_weight, transpose_weight)
    scale_f = prepare_scale(
        antiquant_scale, inter_dtype, is_f8_weight, transpose_weight
    )

    weight_f = weight_f.copy()
    offset_f = None
    if antiquant_offset is not None:
        offset_f = prepare_offset(antiquant_offset, inter_dtype, transpose_weight)

    k_size = weight_f.shape[-2]
    num_groups = _quant_util.ceil_div(k_size, antiquant_group_size)
    for g_idx in range(num_groups):
        k_start = g_idx * antiquant_group_size
        k_end = min((g_idx + 1) * antiquant_group_size, k_size)
        w_slice = weight_f[..., k_start:k_end, :]
        s_slice = scale_f[g_idx : g_idx + 1, :]
        if offset_f is not None:
            w_slice = _util.truncate(
                _util.truncate(w_slice + offset_f[g_idx : g_idx + 1, :], inter_dtype)
                * s_slice,
                inter_dtype,
            )
        else:
            w_slice = _util.truncate(w_slice * s_slice, inter_dtype)
        weight_f[..., k_start:k_end, :] = w_slice

    return matmul_and_finalize(
        x_f, weight_f, bias, inter_dtype, transpose_x, x_dtype_str, out_dtype_str
    )


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
    """MX (Microscaling FP) 量化模式计算。

    量化模式：
    - MX: 使用 float8_e8m0 格式的 scale，每 32 个元素共享一个 scale

    计算流程：
    1. 准备输入：x, weight 转为 fp32 并按需截断
    2. 处理 scale：转为 fp32，沿 K 维度 repeat 32 次与 weight 对齐
    3. 反量化 weight：weight_f = truncate(weight_f * scale_br)
    4. 矩阵乘法：out = x_f @ weight_f
    5. 后处理：加 bias，按 x_dtype 截断，转为输出 dtype
    """
    x_f = x.astype(np.float32)
    weight_f = prepare_weight(weight, inter_dtype, False, transpose_weight)

    scale_f = antiquant_scale.astype(np.float32)
    if inter_dtype == np.float16:
        scale_f = scale_f.astype(np.float16).astype(np.float32)
    if transpose_weight and scale_f.ndim >= 2:
        scale_f = _util.transpose_last_two_dims(scale_f)

    scale_br = np.repeat(scale_f, 32, axis=-2)
    k_dim = weight_f.shape[-2]
    scale_k = scale_br.shape[-2]

    if scale_k > k_dim:
        if _quant_util.ceil_div(k_dim, 32) % 2 != 0:
            scale_br = scale_br[..., :-1, :]
        weight_dims = len(weight_f.shape)
        weight_pad_len = scale_br.shape[-2] - weight_f.shape[-2]
        if weight_pad_len > 0:
            weight_f = np.pad(
                weight_f,
                [(0, 0)] * (weight_dims - 2) + [(0, weight_pad_len)] + [(0, 0)],
                mode="constant",
                constant_values=0,
            )
    weight_f = _util.truncate(weight_f * scale_br, inter_dtype)

    return matmul_and_finalize(
        x_f, weight_f, bias, inter_dtype, transpose_x, x_dtype_str, out_dtype_str
    )


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
    """Packed Scale (uint64/int64 fixpipe) 量化模式计算。

    量化模式：
    - Packed Scale: scale 以 uint64/int64 格式打包存储，包含 deq_scale 和 offset

    计算流程：
    1. 准备输入：x, weight 转为 fp32 并按需截断
    2. 应用 offset（如果有）：weight_f = truncate(weight_f + offset_f, fp16)
    3. 矩阵乘法：out = x_f @ weight_f
    4. 解码并应用 deq_scale：从 uint64 提取 deq_scale，out = out * deq_scale
    5. 后处理：加 bias，按 x_dtype 截断，转为输出 dtype
    """
    x_f = x.astype(np.float32)
    weight_f = prepare_weight(weight, inter_dtype, False, transpose_weight)

    if antiquant_offset is not None:
        weight_f = _util.truncate(
            weight_f + antiquant_offset.astype(np.float32), np.float16
        )
    else:
        weight_f = _util.truncate(weight_f, np.float16)

    if transpose_x:
        x_f = _util.transpose_last_two_dims(x_f)
    x_f = _util.truncate(x_f, inter_dtype)
    weight_f = _util.truncate(weight_f, inter_dtype)

    out = np.matmul(x_f, weight_f)

    deq_scale_f = _quant_util.u64_to_deq_scale(antiquant_scale)
    deq_scale_f = deq_scale_f.astype(np.float16).astype(np.float32)
    deq_scale_br = deq_scale_f.reshape(1, -1)[:, : out.shape[-1]]
    out = out * deq_scale_br

    return finalize(out, bias, x_dtype_str, out_dtype_str)


class WeightQuantBatchMatmulV2TestSpec:
    @staticmethod
    def golden(
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
        """Kernel golden: NZ→ND格式转换 + 调用kernel计算。"""
        input_formats = kwargs.get("input_formats", ())
        input_ori_shapes = kwargs.get("input_ori_shapes", ())
        weight_format = input_formats[1] if len(input_formats) > 1 else "ND"
        if weight_format == "FRACTAL_NZ":
            ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
            if ori_shape is not None and tuple(weight.shape) != tuple(ori_shape):
                weight = _util.nz_to_nd(weight, ori_shape)

        return wqbmm_kernel_compute(
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
        input_ranges = kwargs.get("input_ranges", None)
        testcase_name = kwargs.get("testcase_name", "unknown")

        antiquant_scale = _quant_util.sanitize_e8m0_scale(
            antiquant_scale, 2, input_ranges, testcase_name
        )

        _quant_util.validate_wqbmmv2_constraints(
            x,
            weight,
            antiquant_scale,
            antiquant_offset,
            quant_scale,
            quant_offset,
            bias,
            **kwargs,
        )

        return (
            x,
            weight,
            antiquant_scale,
            antiquant_offset,
            quant_scale,
            quant_offset,
            bias,
        )

    def pre_compare(*outputs, **kwargs):
        return list(outputs)


__spec__ = {
    "weight_quant_batch_matmul_v2": "WeightQuantBatchMatmulV2TestSpec",
}
