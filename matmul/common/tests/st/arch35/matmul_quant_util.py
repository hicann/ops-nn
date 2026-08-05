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
Quant-specific utilities for matmul golden implementations.

Extracted from matmul_golden_util for use by quant operators (qbmmv3, qbmmv4, wqbmmv2).
"""

import warnings
import numpy as np
import en_dtypes

# ============================================================================
# dtype constants
# ============================================================================

np_mx_scale = en_dtypes.float8_e8m0

# ============================================================================
# Generic utilities (used only by quant operators)
# ============================================================================


def ceil_div(a, b):
    """向上取整除法，返回 ceil(a/b)。"""
    if b == 0:
        return 0
    return (a + b - 1) // b


def gen_axes_for_transpose(offset, base):
    """生成多维转置的轴序列，保留前 offset 个维度，交换最后两个维度。"""
    return [x for x in range(offset)] + [x + offset for x in base]


# ============================================================================
# Scale/offset utilities
# ============================================================================


def scale_generate(fp32_array):
    """生成 scale（高 19 位截断）。"""
    u32 = fp32_array.view(np.uint32).copy()
    u32 &= np.uint32(0xFFFFE000)
    return u32.view(np.float32)


def f32_2_s9(array):
    """将 float32 转换为 s9 格式（9-bit signed integer）。"""
    return np.clip(np.round(array), -256, 255)


def needs_scale_generate(x1_dtype, x2_scale, bias_dtype, scale_dtype):
    """判断是否需要生成 scale（高 19 位截断）。"""
    if x1_dtype not in ("int8", "int4"):
        return False
    if scale_dtype in ("uint64", "int64"):
        return False
    if x2_scale.shape[0] != 1:
        return False
    if bias_dtype in ("bfloat16", "float32"):
        return False
    return True


def sanitize_e8m0_scale(tensor, input_index, input_ranges, testcase_name):
    """Ensure float8_e8m0 scale tensor has no NaN values.

    float8_e8m0 is unsigned (8 exponent bits, no sign bit), so negative values
    and zero become NaN. This function regenerates with a positive range if
    NaN is detected.
    """
    if tensor is None or "float8_e8m0" not in str(tensor.dtype):
        return tensor
    nan_count = int(np.isnan(tensor.astype(np.float32)).sum())
    if nan_count == 0:
        return tensor
    orig_low, orig_high = -10, 10
    if input_ranges is not None:
        try:
            rng = input_ranges[input_index] if input_index < len(input_ranges) else None
            if rng is not None and len(rng) >= 2:
                orig_low = rng[0] if rng[0] is not None else -10
                orig_high = rng[1] if rng[1] is not None else 10
        except (TypeError, IndexError):
            pass
    new_high = max(abs(float(orig_low)), abs(float(orig_high)), 1.0)
    new_low = max(new_high * 0.001, 1e-6)
    warnings.warn(
        f"[{testcase_name}] Input {input_index} dtype=float8_e8m0 contains {nan_count} NaN "
        f"values (original range ({orig_low}, {orig_high}) includes negatives). "
        f"Regenerating with positive range ({new_low}, {new_high})."
    )
    return (
        np.random.uniform(new_low, new_high, tensor.shape)
        .astype(np.float32)
        .astype(tensor.dtype)
    )


def get_input_range(input_ranges, index):
    """从 input_ranges 列表中提取指定索引的 (low, high) 范围。"""
    if input_ranges is None:
        return None
    try:
        if index < len(input_ranges):
            rng = input_ranges[index]
            if rng is not None and len(rng) >= 2:
                return rng[0], rng[1]
    except (TypeError, IndexError):
        pass
    return None


# ============================================================================
# UINT64 scale packing/unpacking
# ============================================================================


def u64_to_deq_scale(u64_scale):
    """从 uint64 打包格式解码出 float32 dequantization scale。"""
    shape = u64_scale.shape
    deq_u32 = u64_scale.astype(np.uint32).copy()
    deq_u32 &= np.uint32(0xFFFFE000)
    return deq_u32.view(np.float32).reshape(shape)


def u64_to_offset(u64_scale):
    """从 uint64 scale 中提取 offset（bits 37-45）。"""
    raw = (u64_scale.astype(np.uint64) >> np.uint64(37)) & np.uint64(0x1FF)
    raw = raw.astype(np.int64)
    sign_mask = np.int64(0x100)
    raw = np.where(raw & sign_mask, raw - np.int64(0x200), raw)
    return raw.astype(np.float32).reshape(u64_scale.shape)


def pack_u64_scale(fp32_scale, fp32_offset=None):
    """将 fp32 scale（和可选 offset）打包为 uint64 格式。"""
    u32_scale = np.ascontiguousarray(fp32_scale).view(np.uint32).copy()
    u64 = u32_scale.astype(np.uint64)
    u64 |= np.uint64(1 << 46)
    if fp32_offset is not None:
        s9 = np.clip(np.round(fp32_offset), -256, 255).astype(np.int64)
        s9 = s9.astype(np.uint64) & np.uint64(0x1FF)
        u64 |= s9 << np.uint64(37)
    return u64


def generate_u64_scale(shape):
    """生成随机 uint64 scale，模拟硬件 fixpipe 精度。"""
    fp32_scale = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)
    u32 = np.ascontiguousarray(fp32_scale).view(np.uint32).copy()
    u32 &= np.uint32(0xFFFFE000)
    u64 = u32.astype(np.uint64)
    return u64


def quant_scale_generate(scale_shape):
    """为 int8 输出量化生成随机 scale 和 offset 并打包为 uint64 格式。"""
    fp32_scale = np.random.uniform(low=-5, high=5, size=scale_shape).astype(np.float32)
    fp32_offset = np.random.uniform(low=-5, high=5, size=scale_shape).astype(np.float32)
    return pack_u64_scale(fp32_scale, fp32_offset)


# ============================================================================
# Group size utilities
# ============================================================================


def unpack_groupsize(group_size):
    """解包 group_size 为 (gs_m, gs_n, gs_k)。"""
    gs_m = (group_size >> 32) & 0xFFFF
    gs_n = (group_size >> 16) & 0xFFFF
    gs_k = group_size & 0xFFFF
    if gs_m == 0:
        gs_m = 1
    if gs_n == 0:
        gs_n = 1
    if gs_k == 0:
        gs_k = 1
    return gs_m, gs_n, gs_k


def get_effective_group_size(group_size):
    """从编码的 groupSize 中提取 K 方向的分组大小。

    groupSize 编码格式: (groupSizeM << 32) | (groupSizeN << 16) | groupSizeK
    每个字段占 16 位，共占用 int64 的低 48 位。
    """
    if group_size < 0:
        return 32
    gsK = group_size & 0xFFFF
    if gsK > 0:
        return gsK
    return 32


def decode_group_size(group_size):
    """从编码的 groupSize 中提取 (gsM, gsN, gsK) 三元组。"""
    if group_size <= 0:
        return 128, 128, 128
    gsK = group_size & 0xFFFF
    gsN = (group_size >> 16) & 0xFFFF
    gsM = (group_size >> 32) & 0xFFFF
    if gsM == 0:
        gsM = 128
    if gsN == 0:
        gsN = 128
    if gsK == 0:
        gsK = 128
    return gsM, gsN, gsK


# ============================================================================
# Packed weight packing utilities (CPU, matching op-plugin C++ implementation)
# ============================================================================


def pack_int32_from_int4(weight_int4, axis=-1):
    """将 INT4 打包为 INT32 (INT4Pack)，每 8 个 int4 值打包进 1 个 int32。

    匹配 op-plugin ConvertWeightToINT4PackKernelNpuOpApi.cpp 的 convert_to_int4_pack：
      packed[i] = (a&0xF) | (b&0xF)<<4 | (c&0xF)<<8 | ... | (h&0xF)<<28

    Args:
        weight_int4: int32 数组，每个元素值在 [-8, 7] 范围内，最后一维长度必须是 8 的倍数
        axis: 打包轴，-1 表示最后一维

    Returns:
        int32 数组，打包后指定轴长度变为原来的 1/8
    """
    if weight_int4 is None:
        return None

    orig_shape = weight_int4.shape
    if len(orig_shape) < 2:
        return weight_int4

    if axis == -1:
        axis = len(orig_shape) - 1

    n_unpacked = orig_shape[axis]
    if n_unpacked % 8 != 0:
        raise ValueError(f"pack axis {axis} size {n_unpacked} is not a multiple of 8")
    n_packed = n_unpacked // 8

    if axis == len(orig_shape) - 1:
        w = weight_int4.reshape(*orig_shape[:-1], n_packed, 8).astype(np.int64)
        packed = np.zeros((*orig_shape[:-1], n_packed), dtype=np.int64)
        for i in range(8):
            packed |= (w[..., i] & 0xF) << (i * 4)
        new_shape = list(orig_shape)
        new_shape[axis] = n_packed
        return packed.astype(np.int32).reshape(new_shape)
    elif axis == 0:
        w = weight_int4.reshape(n_packed, 8, *orig_shape[1:]).astype(np.int64)
        packed = np.zeros((n_packed, *orig_shape[1:]), dtype=np.int64)
        for i in range(8):
            packed |= (w[:, i, ...] & 0xF) << (i * 4)
        new_shape = list(orig_shape)
        new_shape[axis] = n_packed
        return packed.astype(np.int32).reshape(new_shape)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def pack_int32_from_fp4(fp4_indices, axis=-1):
    """将 FP4 索引打包为 INT32 (FP4Pack)，每 8 个 fp4 索引打包进 1 个 int32。

    匹配 op-plugin ConvertWeightToINT4PackKernelNpuOpApi.cpp 的打包逻辑，
    位操作与 int4 相同：packed[i] = (a&0xF) | (b&0xF)<<4 | ... | (h&0xF)<<28

    Args:
        fp4_indices: int32 数组，每个元素值在 [0, 15] 范围内，最后一维长度必须是 8 的倍数
        axis: 打包轴，-1 表示最后一维

    Returns:
        int32 数组，打包后指定轴长度变为原来的 1/8
    """
    if fp4_indices is None:
        return None

    orig_shape = fp4_indices.shape
    if len(orig_shape) < 2:
        return fp4_indices

    if axis == -1:
        axis = len(orig_shape) - 1

    n_unpacked = orig_shape[axis]
    if n_unpacked % 8 != 0:
        raise ValueError(f"pack axis {axis} size {n_unpacked} is not a multiple of 8")
    n_packed = n_unpacked // 8

    if axis == len(orig_shape) - 1:
        w = fp4_indices.reshape(*orig_shape[:-1], n_packed, 8)
        packed = np.zeros((*orig_shape[:-1], n_packed), dtype=np.int64)
        for i in range(8):
            packed |= (w[..., i].astype(np.int64) & 0xF) << (i * 4)
        new_shape = list(orig_shape)
        new_shape[axis] = n_packed
        return packed.astype(np.int32).reshape(new_shape)
    elif axis == 0:
        w = fp4_indices.reshape(n_packed, 8, *orig_shape[1:])
        packed = np.zeros((n_packed, *orig_shape[1:]), dtype=np.int64)
        for i in range(8):
            packed |= (w[:, i, ...].astype(np.int64) & 0xF) << (i * 4)
        new_shape = list(orig_shape)
        new_shape[axis] = n_packed
        return packed.astype(np.int32).reshape(new_shape)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


# ============================================================================
# Packed weight unpacking utilities
# ============================================================================


FP4_E2M1_TABLE = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def unpack_int32_to_int4(weight_int32, axis=-1):
    """将 INT32 (INT4Pack) 解包为 INT4，每个 int32 包含 8 个 int4 值。"""
    if weight_int32 is None:
        return None

    orig_shape = weight_int32.shape
    if len(orig_shape) < 2:
        return weight_int32

    if not weight_int32.flags["C_CONTIGUOUS"]:
        weight_int32 = np.ascontiguousarray(weight_int32)

    if axis == -1:
        axis = len(orig_shape) - 1

    n_packed = orig_shape[axis]
    n_unpacked = n_packed * 8

    if axis == len(orig_shape) - 1:
        flat_shape = (-1, n_packed)
        weight_flat = weight_int32.reshape(flat_shape)
        result = np.zeros((weight_flat.shape[0], n_unpacked), dtype=np.int8)
        for i in range(8):
            shift = i * 4
            mask = 0xF
            int4_val = (weight_flat >> shift) & mask
            int4_val = np.where(int4_val >= 8, int4_val - 16, int4_val)
            result[:, i::8] = int4_val
        new_shape = list(orig_shape)
        new_shape[axis] = n_unpacked
        return result.reshape(new_shape)
    elif axis == 0:
        flat_shape = (n_packed, -1)
        weight_flat = weight_int32.reshape(flat_shape)
        result = np.zeros((n_unpacked, weight_flat.shape[1]), dtype=np.int8)
        for i in range(8):
            shift = i * 4
            mask = 0xF
            int4_val = (weight_flat >> shift) & mask
            int4_val = np.where(int4_val >= 8, int4_val - 16, int4_val)
            result[i::8, :] = int4_val
        new_shape = list(orig_shape)
        new_shape[axis] = n_unpacked
        return result.reshape(new_shape)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def unpack_float_to_fp4(weight_float, axis=-1):
    """将 FLOAT (FLOAT4Pack) 解包为 FLOAT4_E2M1，每个 float32 包含 8 个 fp4 值。"""
    if weight_float is None:
        return None

    orig_shape = weight_float.shape
    if len(orig_shape) < 2:
        return weight_float

    if not weight_float.flags["C_CONTIGUOUS"]:
        weight_float = np.ascontiguousarray(weight_float)

    if axis == -1:
        axis = len(orig_shape) - 1

    n_packed = orig_shape[axis]
    n_unpacked = n_packed * 8

    weight_f32 = weight_float.astype(np.float32)

    if axis == len(orig_shape) - 1:
        flat_shape = (-1, n_packed)
        weight_flat = weight_f32.reshape(flat_shape)
        result = np.zeros((weight_flat.shape[0], n_unpacked), dtype=np.float32)
        for i in range(8):
            shift = i * 4
            fp4_bits = (weight_flat.view(np.uint32) >> shift) & 0xF
            result[:, i::8] = np.vectorize(lambda b: FP4_E2M1_TABLE[b])(fp4_bits)
        new_shape = list(orig_shape)
        new_shape[axis] = n_unpacked
        return result.reshape(new_shape)
    elif axis == 0:
        flat_shape = (n_packed, -1)
        weight_flat = weight_f32.reshape(flat_shape)
        result = np.zeros((n_unpacked, weight_flat.shape[1]), dtype=np.float32)
        for i in range(8):
            shift = i * 4
            fp4_bits = (weight_flat.view(np.uint32) >> shift) & 0xF
            result[i::8, :] = np.vectorize(lambda b: FP4_E2M1_TABLE[b])(fp4_bits)
        new_shape = list(orig_shape)
        new_shape[axis] = n_unpacked
        return result.reshape(new_shape)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


# ============================================================================
# Dtype name utility
# ============================================================================


def get_dtype_name(tensor):
    """Get dtype string from torch.Tensor or numpy.ndarray."""
    if tensor is None:
        return None
    return str(tensor.dtype).replace("torch.", "")


# ============================================================================
# 950 (DAV_3510) parameter validation
# ============================================================================


def validate_wqbmmv2_constraints(
    x,
    weight,
    antiquant_scale,
    antiquant_offset=None,
    quant_scale=None,
    quant_offset=None,
    bias=None,
    **kwargs,
):
    """Validate WQBMMV2 950 (DAV_3510) parameter constraints and emit warnings.

    Does not modify any tensors. Only warns about configurations that would
    cause NPU parameter validation failures on Ascend 950.

    Supports both kernel (input_formats) and aclnn (tensor_formats) kwargs.
    """
    testcase_name = kwargs.get("testcase_name", "unknown")
    x_dtype = get_dtype_name(x)
    weight_dtype = get_dtype_name(weight)
    scale_dtype = get_dtype_name(antiquant_scale)
    offset_dtype = get_dtype_name(antiquant_offset)
    bias_dtype = get_dtype_name(bias)

    formats = kwargs.get("input_formats") or kwargs.get("tensor_formats")
    weight_format = formats[1] if formats and len(formats) > 1 else "ND"

    is_fp4_weight = weight_dtype in ("float32",)
    is_fp8_weight = weight_dtype in ("float8_e4m3fn", "hifloat8")
    is_int8_weight = weight_dtype == "int8"

    if quant_scale is not None:
        warnings.warn(
            f"[{testcase_name}] 950 does not support quantScale, must be null. "
            f"Got dtype={get_dtype_name(quant_scale)}"
        )
    if quant_offset is not None:
        warnings.warn(
            f"[{testcase_name}] 950 does not support quantOffset, must be null. "
            f"Got dtype={get_dtype_name(quant_offset)}"
        )

    if (is_fp4_weight or is_fp8_weight) and antiquant_offset is not None:
        warnings.warn(
            f"[{testcase_name}] 950 requires antiquantOffset=null when weight is "
            f"{weight_dtype}, got dtype={offset_dtype}"
        )

    if antiquant_scale is not None and scale_dtype != "float8_e8m0":
        if x_dtype is not None and scale_dtype != x_dtype:
            warnings.warn(
                f"[{testcase_name}] 950 requires antiquantScale dtype ({scale_dtype}) "
                f"to match x dtype ({x_dtype}) when not float8_e8m0"
            )

    if antiquant_offset is not None and x_dtype is not None:
        if offset_dtype != x_dtype:
            warnings.warn(
                f"[{testcase_name}] 950 requires antiquantOffset dtype ({offset_dtype}) "
                f"to match x dtype ({x_dtype})"
            )

    if bias is not None and x_dtype == "bfloat16":
        if (is_fp4_weight or is_fp8_weight) and bias_dtype != "bfloat16":
            warnings.warn(
                f"[{testcase_name}] 950 requires bias dtype=bfloat16 when weight is "
                f"{weight_dtype} and x=bfloat16, got bias dtype={bias_dtype}"
            )
        if is_int8_weight and weight_format == "FRACTAL_NZ" and bias_dtype != "float32":
            warnings.warn(
                f"[{testcase_name}] 950 requires bias dtype=float32 when weight is "
                f"int8+FRACTAL_NZ and x=bfloat16, got bias dtype={bias_dtype}"
            )
