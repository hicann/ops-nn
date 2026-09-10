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

import re

import numpy as np

__golden__ = {
    "kernel": {"grouped_dynamic_mx_quant": "grouped_dynamic_mx_quant_golden"},
    "aclnn": {"aclnnGroupedDynamicMxQuantV2": "grouped_dynamic_mx_quant_v2_golden"},
    "e2e": {
        "torch_npu.npu_grouped_dynamic_mx_quant": "npu_grouped_dynamic_mx_quant_golden"
    },
}

__input__ = {
    "aclnn": {"aclnnGroupedDynamicMxQuantV2": "grouped_dynamic_mx_quant_v2_input"},
    "e2e": {
        "torch_npu.npu_grouped_dynamic_mx_quant": "npu_grouped_dynamic_mx_quant_input"
    },
}


DATA_TYPE_INT_TO_STR = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    11: "double",
    12: "bool",
    16: "complex64",
    17: "complex128",
    18: "qint8",
    19: "qint16",
    20: "qint32",
    21: "quint8",
    22: "quint16",
    23: "resource",
    25: "dual",
    26: "variant",
    27: "bfloat16",
    29: "int4",
    30: "uint1",
    31: "int2",
    32: "uint2",
    33: "complex32",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    37: "float8_e8m0",
    40: "float4_e2m1",
    41: "float4_e1m2",
    42: "hifloat4",
}

# torch_scalarType 枚举值 -> CANN dtype 字符串（E2E 走 torch_npu，CSV 里 dst_type 是 torch 枚举）。
# 值对齐 DATA_TYPE_INT_TO_STR；16(quint4x2)/21(bits8) 在 CANN 表无对应项，省略。
TORCH_DTYPE_ENUM_TO_CANN_STR = {
    0: "uint8",
    1: "int8",
    2: "int16",
    3: "int32",
    4: "int64",
    5: "float16",
    6: "float32",
    7: "double",
    8: "complex32",
    9: "complex64",
    10: "complex128",
    11: "bool",
    12: "qint8",
    13: "quint8",
    14: "qint32",
    15: "bfloat16",
    23: "float8_e5m2",
    24: "float8_e4m3fn",
    285: "int4",
    290: "hifloat8",
    291: "float8_e5m2",
    292: "float8_e4m3fn",
    296: "float4_e2m1",
    297: "float4_e1m2",
}


def _numpy_mx_dtype(mx_ele_dtype):
    if mx_ele_dtype == "float4_e2m1":
        from en_dtypes import float4_e2m1

        return float4_e2m1
    if mx_ele_dtype == "float4_e1m2":
        from en_dtypes import float4_e1m2

        return float4_e1m2
    if mx_ele_dtype == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    if mx_ele_dtype == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    if mx_ele_dtype == "float8_e8m0":
        from en_dtypes import float8_e8m0

        return float8_e8m0
    raise ValueError(f"Unsupported mx dtype: {mx_ele_dtype}")


def _get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "uint4" in str(dt):
        return 0, 15
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in str(dt):
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    if "complex32" in str(dt):
        dt = "float16"
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


def _mx_calculate_share_exp(fp_array, scale_axis, mx_ele_dtype):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = _get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(np.log2(max_norm))
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True)
    res = (
        np.floor(
            np.log2(fp_abs_max.astype(np.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
        )
        - ele_emax
    )
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_calculate_share_exp_nv(
    fp_array, scale_axis, mx_ele_dtype, max_norm, subnormal, max_low_bound=0
):
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True).astype(
        np.float32
    )
    fp_abs_max_orig = fp_abs_max.copy()
    if max_low_bound != 0:
        fp_abs_max = np.maximum(fp_abs_max, max_low_bound)
    s_fp32 = fp_abs_max / max_norm
    binary_ints = np.array(s_fp32.view(np.uint32))
    exponent_mask = np.uint32(0x7F800000)
    mantissa_mask = np.uint32(0x007FFFFF)
    exponents = (binary_ints & exponent_mask) >> 23
    exponents_int16 = exponents.astype(np.int16)
    mantissas = binary_ints & mantissa_mask
    condition_1 = (exponents_int16 > 0) & (exponents_int16 < 254) & (mantissas > 0)
    if subnormal:
        condition_2 = (exponents_int16 == 0) & (mantissas > 2**22)
    else:
        condition_2 = False
    exponents_int16 = np.where(
        (condition_1 | condition_2), exponents_int16 + 1, exponents_int16
    )
    res = (exponents_int16 - 127).astype(np.float32)
    res[fp_abs_max_orig == 0] = -float("inf")
    return res


def _mx_calculate_share_exp_dynamic_dtype_range(
    fp_array, scale_axis, mx_ele_dtype, max_norm, subnormal
):
    from ml_dtypes import bfloat16

    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True).astype(
        bfloat16
    )
    binary_ints = np.array(fp_abs_max.view(np.uint16))
    exponent_mask = np.uint16(0x7F80)
    mantissa_mask = np.uint16(0x007F)
    exponents = (binary_ints & exponent_mask) >> 7
    exponents_int16 = exponents.astype(np.int16)
    mantissas = binary_ints & mantissa_mask
    mantissas = mantissas.astype(np.uint16)
    threshold = np.uint16(
        np.array([max_norm], dtype=bfloat16).view(np.uint16)[0] & 0x007F
    )
    condition = mantissas > threshold
    exponents_int16_1 = np.where(condition, exponents_int16 + 1, exponents_int16)
    ele_emax = int(np.floor(np.log2(max_norm)))
    exponents_int16_1 -= ele_emax
    res = (exponents_int16_1 - 127).astype(np.float32)
    res[exponents_int16 == 255] = float("inf")
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_round_mantissa(fp_array, round_mode):
    if round_mode in ("rint", "even"):
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):
        sign = np.signbit(fp_array)
        rounded_abs = np.floor(np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
        fp_array = np.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":
        fp_array = np.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array


def _mx_quantize_to_element_format(
    fp_array, share_exp, mx_ele_dtype, round_mode, scale_alg=0, dst_type_max=0.0
):
    mx_dtype = str(mx_ele_dtype)
    match = re.search(r"e(\d+)m(\d+)", mx_dtype)
    if match:
        exp_bits = int(match.group(1))
        mantissa_bits = int(match.group(2))
    else:
        raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

    ret = np.where(
        share_exp == -127,
        np.where(np.signbit(fp_array), -0.0, 0.0),
        fp_array / (2**share_exp),
    )
    private_exp = np.floor(np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))).astype(
        fp_array.dtype, copy=False
    )
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    ret = ret / (2**private_exp) * (2**mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    ret = ret / (2**mantissa_bits) * (2**private_exp)
    max_norm = _get_dtype_range(mx_dtype)[1]
    np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret


def _grouped_mx_undo_reshape_to_blocks(
    fp_array, group_index, axis, padded_group_index, padded_shape
):
    # Step 1: Undo reshape to blocks
    fp_array = fp_array.reshape(padded_shape)

    # Step 2: Remove the expanded dimension (axis+1)
    fp_array = np.squeeze(fp_array, axis=axis + 1)

    # Step 3: Split into padded groups
    split_indices = padded_group_index[:-1]  # 获取分割点（排除最后一个总长度）
    groups = np.split(fp_array, split_indices, axis=axis)

    # Step 4: Trim padding from each group
    trimmed_groups = []
    prev = 0
    for i, end in enumerate(group_index):
        # 计算原始组长度
        original_length = end - prev
        # 沿 axis 轴截取原始数据（去掉 Pad 部分）
        slices = [slice(None)] * fp_array.ndim
        slices[axis] = slice(0, original_length)
        trimmed_group = groups[i][tuple(slices)]
        trimmed_groups.append(trimmed_group)
        prev = end

    # Step 5: Concatenate trimmed groups
    restored_array = np.concatenate(trimmed_groups, axis=axis)
    return restored_array


def _grouped_mx_reshape_to_blocks(fp_array, group_index, axis, block_size):
    # Step 1: Split into groups based on group_index
    groups = []
    padded_group_index = []
    prev = 0

    for end in group_index:
        # 沿 axis 轴切片
        slices = [slice(None)] * fp_array.ndim
        slices[axis] = slice(prev, end)
        group = fp_array[tuple(slices)]
        groups.append(group)
        prev = end

    # Step 2: Pad each group to align with block_size
    padded_groups = []
    padded_index = 0
    for group in groups:
        group_len = group.shape[axis]
        pad_size = (block_size - (group_len % block_size)) % block_size
        if pad_size > 0:
            pad_width = [(0, 0)] * group.ndim
            pad_width[axis] = (0, pad_size)
            padded_group = np.pad(group, pad_width, mode="constant")
        else:
            padded_group = group
        padded_index = padded_index + group_len + pad_size
        padded_groups.append(padded_group)
        padded_group_index.append(padded_index)

    # Step 3: Concatenate all groups along axis
    padded_array = np.concatenate(padded_groups, axis=axis)
    # Step 4: Expand dimensions as in original function
    expanded_array = np.expand_dims(padded_array, axis=axis + 1)
    # Step 5: Reshape into blocks
    # 总块数 = 总长度（已对齐）// block_size
    total_blocks = padded_array.shape[axis] // block_size
    reshape = list(expanded_array.shape)
    reshape[axis] = total_blocks  # 替换原轴为块数
    reshape.insert(axis + 1, block_size)  # 插入块大小维度

    reshaped_array = expanded_array.reshape(reshape)
    padded_shape = expanded_array.shape  # 补 Pad 后的形状（包含 expand_dims）

    return reshaped_array, padded_group_index, padded_shape


def _reshape_scale_array_pad(scale_array, group_index, axis):
    import math

    scale_array = np.squeeze(scale_array, 1)
    cur_idx = 0
    pre_element = 0
    for idx, element in enumerate(group_index):
        next_group_start = (element // 64 + idx + 1) * 2  # 下一个group的初始地址
        real_idx = cur_idx + math.ceil((element - pre_element) / 32)  # 实际计算到多少行
        pad_idx = math.ceil(real_idx / 2) * 2  # 需要pad到多少行
        zero_row = np.full((1, scale_array.shape[1]), 2**-127)
        one_row = np.full((1, scale_array.shape[1]), 1)
        for i in range(real_idx, pad_idx):
            scale_array = np.insert(scale_array, i, zero_row, axis=0)
        for i in range(pad_idx, next_group_start):
            scale_array = np.insert(scale_array, i, one_row, axis=0)
        pre_element = element  # 前一个element
        cur_idx = next_group_start  # 当前计算到多少行

    scale_array = (
        scale_array.reshape(int(scale_array.shape[0] / 2), 2, scale_array.shape[1])
        .transpose(0, 2, 1)
        .reshape(int(scale_array.shape[0] / 2), scale_array.shape[1], 2)
    )

    return scale_array


def _grouped_mx_quantize(
    fp_array,
    group_index,
    mx_ele_dtype="float8_e5m2",
    axis=-2,
    block_size=32,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
):
    if not isinstance(fp_array, np.ndarray):
        raise RuntimeError(
            f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}"
        )
    if fp_array.dtype.name not in ("bfloat16", "float16"):
        raise RuntimeError(
            f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}"
        )
    if mx_ele_dtype not in (
        "float4_e2m1",
        "float4_e1m2",
        "float8_e4m3fn",
        "float8_e5m2",
    ):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")
    if (
        scale_alg == 2
        and mx_ele_dtype == "float4_e2m1"
        and (
            dst_type_max < 0
            or (dst_type_max > 0 and dst_type_max < 6)
            or dst_type_max > 12
        )
    ):
        raise RuntimeError(
            f"For float4_e2m1, dst_type_max[{dst_type_max}] should be in [6,12] or equal 0"
        )
    if (
        scale_alg == 2
        and mx_ele_dtype == "float4_e1m2"
        and (
            dst_type_max < 0
            or (dst_type_max > 0 and dst_type_max < 1.75)
            or dst_type_max >= 3.5
        )
    ):
        raise RuntimeError(
            f"For float4_e1m2, dst_type_max[{dst_type_max}] should be in [1.75,3.5) or equal 0"
        )

    def is_non_reverse_order(arr):
        if len(arr) <= 1:
            return True  # 空数组或单元素数组视为逆序
        diff = np.diff(arr)
        return np.all(diff >= 0)

    if not is_non_reverse_order(group_index):
        raise RuntimeError("Input tensor group_index should be non-reverse order.")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    if axis != -2 and axis != 0:
        raise RuntimeError(f"Not support {axis} yet!")

    if group_index[-1] != fp_array.shape[axis]:
        raise RuntimeError(
            "The last element of group_index should match the dimension size of the input x axis."
        )

    # padding & reshape to block_size
    fp_array, padded_group_index, padded_shape = _grouped_mx_reshape_to_blocks(
        fp_array, group_index, axis, block_size
    )
    # get mx scale exponents
    if scale_alg == 0:
        share_exp = _mx_calculate_share_exp(
            fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype
        )
    elif scale_alg == 1:
        share_exp = _mx_calculate_share_exp_nv(
            fp_array,
            scale_axis=axis + 1,
            mx_ele_dtype=mx_ele_dtype,
            max_norm=_get_dtype_range(mx_ele_dtype)[1],
            subnormal=True,
        )
    elif scale_alg == 2:
        if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2"):
            raise RuntimeError(
                "scale_alg = 2 is only supported by float4_e2m1 and float4_e1m2"
            )
        if dst_type_max == 0:
            dst_type_max = 6 if mx_ele_dtype == "float4_e2m1" else 1.75
        is_optimized = (mx_ele_dtype == "float4_e2m1" and dst_type_max in (6, 7)) or (
            mx_ele_dtype == "float4_e1m2" and dst_type_max == 1.875
        )
        if is_optimized:
            share_exp = _mx_calculate_share_exp_dynamic_dtype_range(
                fp_array,
                scale_axis=axis + 1,
                mx_ele_dtype=mx_ele_dtype,
                max_norm=dst_type_max,
                subnormal=False,
            )
        else:
            share_exp = _mx_calculate_share_exp_nv(
                fp_array,
                scale_axis=axis + 1,
                mx_ele_dtype=mx_ele_dtype,
                max_norm=dst_type_max,
                subnormal=False,
            )
    else:
        raise RuntimeError(f"scale_alg is not supported: {scale_alg}")

    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    # quantize mx element
    ele_array = _mx_quantize_to_element_format(
        fp_array, share_exp, mx_ele_dtype, round_mode, scale_alg, dst_type_max
    )
    # undo reshape
    ele_array = _grouped_mx_undo_reshape_to_blocks(
        ele_array, group_index, axis, padded_group_index, padded_shape
    )
    share_exp = np.squeeze(share_exp, axis=axis + 1)
    # convert to fp8_e8m0 & fp8 dtype
    ele_dtype_np = _numpy_mx_dtype(mx_ele_dtype)
    # share_exp is always float32
    scale_array = 2**share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array = _reshape_scale_array_pad(scale_array, group_index, axis)
    scale_array = scale_array.astype(_numpy_mx_dtype("float8_e8m0"), copy=False)

    return scale_array, ele_array


def grouped_dynamic_mx_quant_golden(
    x,
    group_index,
    *,
    round_mode="rint",
    dst_type=35,
    blocksize=32,
    scale_alg=0,
    dst_type_max=0.0,
    **kwargs,
):
    """
    Golden function for grouped_dynamic_mx_quant.
    All the parameters (names and order) follow @grouped_dynamic_mx_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    group_idx = group_index
    block_size = blocksize
    dst_type_str = (
        dst_type if isinstance(dst_type, str) else DATA_TYPE_INT_TO_STR[dst_type]
    )

    ret = _grouped_mx_quantize(
        x,
        group_idx,
        mx_ele_dtype=dst_type_str,
        axis=0,
        block_size=block_size,
        round_mode=round_mode,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
    )

    return ret[1], ret[0]


def _pick_attr(kwargs, *names, default=None):
    for n in names:
        if n in kwargs and kwargs[n] is not None:
            return kwargs[n]
    return default


def npu_grouped_dynamic_mx_quant_input(x, group_index, *extra, **kwargs):
    """E2E input normalize for torch_npu.npu_grouped_dynamic_mx_quant."""
    dim_s = int(x.shape[0])
    if hasattr(group_index, "copy_"):
        import torch

        group_index.copy_(torch.sort(group_index).values)
        group_index.clamp_(0, dim_s)
        group_index[-1] = dim_s
    else:
        import numpy as np

        gi = np.sort(group_index)
        gi = np.minimum(np.maximum(gi, 0), dim_s)
        gi[-1] = dim_s
        group_index[...] = gi


def npu_grouped_dynamic_mx_quant_golden(x, group_index, *extra, **kwargs):
    """E2E golden for torch_npu.npu_grouped_dynamic_mx_quant.

    Computed on CPU numpy. Inputs may be torch.Tensor or numpy.ndarray.
    Returns [y, mxscale]. mxscale is returned as uint8 (bit-view of the
    float8_e8m0 scale) because torch_npu's device-side mxscale output is uint8
    (torch has no native float8_e8m0), so the comparator sees the same bit
    representation. The quantization math is identical to the ACLNN golden.
    FP4 (dst_type 40/41) element output is returned packed to uint8 (last dim
    halved) to match the torch_npu device-side representation, since E2E does
    not run the framework's unpack_4bit_outputs.
    """
    import numpy as np

    round_mode = _pick_attr(kwargs, "roundMode", "round_mode", default="rint")
    dst_type = _pick_attr(kwargs, "dstType", "dst_type", default=35)
    dst_type = TORCH_DTYPE_ENUM_TO_CANN_STR.get(dst_type, dst_type)
    blocksize = _pick_attr(kwargs, "blocksize", "block_size", default=32)
    scale_alg = _pick_attr(kwargs, "scaleAlg", "scale_alg", default=0)
    dst_type_max = _pick_attr(kwargs, "dstTypeMax", "dst_type_max", default=0.0)

    x_np = _to_numpy(x)
    gi_np = _to_numpy(group_index)

    ele, scale = grouped_dynamic_mx_quant_golden(
        x_np,
        gi_np,
        round_mode=round_mode,
        dst_type=dst_type,
        blocksize=blocksize,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
    )
    ele = _pack_fp4_to_uint8(ele)
    return [np.ascontiguousarray(ele), np.ascontiguousarray(scale).view(np.uint8)]


def grouped_dynamic_mx_quant_v2_input(x, group_index, *args, **kwargs):
    """Normalize group_index in place for aclnnGroupedDynamicMxQuantV2.

    Works for both torch.Tensor (torch-native dtypes) and numpy.ndarray (when
    any output dtype is non-torch-native, e.g. float8). Mirrors the kernel-side
    input.py: sort, clamp to [0, x.shape[0]], and force the last element to equal
    the group axis dim so that group_index[-1] == x.shape[group_axis].
    """
    dim_s = int(x.shape[0])
    if hasattr(group_index, "copy_"):
        import torch

        group_index.copy_(torch.sort(group_index).values)
        group_index.clamp_(0, dim_s)
        group_index[-1] = dim_s
    else:
        import numpy as np

        gi = np.sort(group_index)
        gi = np.minimum(np.maximum(gi, 0), dim_s)
        gi[-1] = dim_s
        group_index[...] = gi


def _pack_fp4_to_uint8(ele):
    """Pack a native FP4 (float4_e2m1 / float4_e1m2) element tensor into the
    E2E host-side (op-plugin) representation of the y output.

    torch_npu.npu_grouped_dynamic_mx_quant stores y as a uint8 tensor with the
    last dim halved, packing 2 consecutive FP4 elements per byte: the first
    (even) element goes to the low nibble, the second (odd) element to the high
    nibble. (TTK's unpack_4bit_outputs only runs for kernel/aclnn which expose
    flat_output_dtypes; E2E has none, so the golden must pack explicitly.)
    """
    import numpy as np

    if not isinstance(ele, np.ndarray) or "float4" not in str(ele.dtype):
        return ele
    if ele.shape[-1] % 2:
        raise ValueError(
            "FP4 output requires the last dim of x to be divisible by 2 for packing"
        )
    raw = ele.view(np.uint8) & 0x0F
    pairs = raw.reshape(raw.shape[:-1] + (-1, 2))
    packed = (
        pairs[..., 0].astype(np.uint16) | (pairs[..., 1].astype(np.uint16) << 4)
    ).astype(np.uint8)
    return packed


def _to_numpy(tensor):
    """ACLNN golden receives torch.Tensor; convert back to numpy for the math.

    torch bfloat16 cannot be converted via .numpy() on this (CPU) build, so it is
    routed through float32 and back to ml_dtypes.bfloat16 (lossless round trip).
    """
    if hasattr(tensor, "detach"):
        import torch

        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            import ml_dtypes

            return t.to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
        return t.numpy()
    return tensor


def grouped_dynamic_mx_quant_v2_golden(
    x,
    group_index,
    round_mode,
    dst_type,
    blocksize,
    scale_alg,
    dst_type_max,
    y=None,
    mxscale=None,
    **kwargs,
):
    """Golden for aclnnGroupedDynamicMxQuantV2.

    Positional args follow aclnnGroupedDynamicMxQuantV2GetWorkspaceSize, without
    workspaceSize/executor. x, groupIndex, y, mxscale are torch.Tensor; the
    scalar/attr params (roundMode, dstType, blocksize, scaleAlg, dstTypeMax) are
    passed positionally from the testcase attributes.

    **kwargs: tensor_dtypes, tensor_formats, scalar_dtypes,
             short_soc_version, testcase_name, ...

    Returns:
        [y, mxscale] -> quantized element tensor, then the mx scale tensor.
    """
    x_np = _to_numpy(x)
    gi_np = _to_numpy(group_index)

    ele, scale = grouped_dynamic_mx_quant_golden(
        x_np,
        gi_np,
        round_mode=round_mode,
        dst_type=dst_type,
        blocksize=blocksize,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
    )

    return [ele, scale]
