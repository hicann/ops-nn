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
flat_quant kernel golden 实现。

本模块包含 flat_quant 算子的 MX 量化 numpy 实现和 TestSpec 类。
"""

import copy
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util
import numeric_ulp

np_bfloat16 = _util.np_bfloat16
np_fp4_e2m1 = _util.np_fp4_e2m1
np_fp4_e1m2 = _util.np_fp4_e1m2
np_fp8_e4m3 = _util.np_fp8_e4m3
np_fp8_e5m2 = _util.np_fp8_e5m2
np_mx_scale = _util.np_mx_scale
np_int4 = _util.np_int4
numeric_ulp_compare = numeric_ulp.numeric_ulp_compare

MXFP4_DST_DTYPE = 40

_MX_DTYPE_MAP = {
    "float4_e2m1": np_fp4_e2m1,
    "float4_e1m2": np_fp4_e1m2,
    "float8_e4m3fn": np_fp8_e4m3,
    "float8_e5m2": np_fp8_e5m2,
}


# ============================================================================
# MX quantize (numpy implementation)
# ============================================================================


def mx_quantize(
    fp_array: np.ndarray,
    mx_ele_dtype: str = "float4_e2m1",
    axis: int = -1,
    block_size: int = 32,
    round_mode: str = "rint",
    scale_alg: int = 0,
    dst_type_max: float = 0,
    max_low_bound: float = 0,
) -> tuple:
    """
    quantize BFP16/FP16/FP32 to MX dtypes
    :parameter fp_array: input numpy array with dtype BFP16/FP16/FP32
    :parameter mx_ele_dtype: dtype of element in MX dtype. support float4_e2m1/float4_e1m2/float8_e4m3fn/float8_e5m2
    :parameter axis: specify the axis across which shared scales/exponents are calculated.
    :parameter block_size: each block_size shares the same mx scale along the axis
    :parameter round_mode: round mode. support rint/floor/round/nearest
    :parameter scale_alg: The calculation method for scale.Support MxFP8(OCP , count 0) or MxFP8(nvidia-cuBLAS , count 1)
    :return: mx-elements & mx-scale-exponents

    NOTE: Scenarios below should be considered and tested when TRYING to modify this code:
    1. block with only subnormal floats
    2. block with only one nan
    3. block with only one inf or -inf
    """

    def _mx_reshape_to_blocks(fp_array, axis, block_size):
        fp_array = np.expand_dims(fp_array, axis=axis + 1)
        orig_shape = fp_array.shape
        pad = [[0, 0] for _ in range(len(orig_shape))]
        pad_size = orig_shape[axis] % block_size
        pad[axis][1] = block_size - pad_size
        if pad_size > 0:
            fp_array = np.pad(fp_array, pad, "constant")
        padded_shape = fp_array.shape
        reshape = list(padded_shape)
        reshape[axis + 1] = block_size
        reshape[axis] = reshape[axis] // block_size
        fp_array = fp_array.reshape(reshape)
        return fp_array, orig_shape, padded_shape

    def _mx_round_mantissa(fp_array, round_mode):
        """
        For example:
        fp_array  = [-4.5, -3.5, -2.5, -2.0, -1.7, -1.5, -1.4, -0.5, -0.2, 0.2, 0.5, 1.4, 1.5, 1.7, 2.0, 2.5, 3.4, 4.5]
        - rint    = [-4.,  -4.,  -2.,  -2.,  -2.,  -2.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  3.,  4.]
        - nearest = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  0.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  5.]
        - floor   = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -2.,  -1.,  -1.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
        - ceil    = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  4.,  5.]
        - trunc   = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
        """
        if round_mode in ("rint", "even"):  # tie to even(c language rint)
            fp_array = np.rint(fp_array)
        elif round_mode in (
            "round",
            "nearest",
        ):  # tie away from zero(c language round).
            sign = np.signbit(fp_array)
            rounded_abs = np.floor(
                np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype)
            )
            fp_array = np.where(sign, -rounded_abs, rounded_abs)
        elif round_mode == "floor":  # round to minus infinity(c language floor)
            fp_array = np.floor(fp_array)
        elif round_mode == "ceil":  # round to positive infinity(c language ceil)
            fp_array = np.ceil(fp_array)
        elif round_mode == "trunc":  # round to zero(c language truncation)
            fp_array = np.trunc(fp_array)
        else:
            raise Exception(f"Unrecognized round method {round_mode}")
        return fp_array

    def _mx_calculate_share_exp(fp_array, scale_axis, mx_ele_dtype):
        FP32_EXPONENT_BIAS = 127
        FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
        ele_emax = 0
        mx_dtype = str(mx_ele_dtype)
        if "float4_e2m1" in mx_dtype:
            ele_emax = 2
        elif "float8_e4m3fn" in mx_dtype:
            ele_emax = 8
        elif "float8_e5m2" in mx_dtype:
            ele_emax = 15
        fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True)
        res = (
            np.floor(
                np.log2(
                    fp_abs_max.astype(np.float64) + FP32_MIN_NORMAL * (fp_abs_max == 0)
                )
            ).astype(np.float32)
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
        exponent_mask = np.uint32(
            0x7F800000
        )  # 二进制：01111111100000000000000000000000
        mantissa_mask = np.uint32(
            0x007FFFFF
        )  # 二进制：00000000011111111111111111111111
        # 提取指数部分并转换为uint16
        exponents = (binary_ints & exponent_mask) >> 23
        exponents_int16 = exponents.astype(np.int16)
        # 提取尾数部分并转换为float
        mantissas = binary_ints & mantissa_mask
        condition_1 = (exponents_int16 > 0) & (exponents_int16 < 254) & (mantissas > 0)
        # 2 ** 23 fp32的尾数位值0.5，即：二进制：0 00000000 10000000000000000000000
        # condition_2 = (exponents_int16 == 0) & (mantissas > 2**22)
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
        fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True).astype(
            np_bfloat16
        )

        binary_ints = np.array(fp_abs_max.view(np.uint16))
        exponent_mask = np.uint16(0x7F80)  # 0111111110000000
        mantissa_mask = np.uint16(0x007F)  # 0000000001111111
        exponents = (binary_ints & exponent_mask) >> 7
        exponents_int16 = exponents.astype(np.int16)
        mantissas = binary_ints & mantissa_mask
        mantissas = mantissas.astype(np.uint16)
        threshold = np.uint16(0x0040) if max_norm == 6 else np.uint16(0x0060)
        condition = mantissas > threshold
        exponents_int16_1 = np.where((condition), exponents_int16 + 1, exponents_int16)
        exponents_int16_1 -= 2
        res = (exponents_int16_1 - 127).astype(np.float32)
        res[exponents_int16 == 255] = float("inf")
        res[fp_abs_max == 0] = -float("inf")
        return res

    def get_dtype_range(dt):
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

    def _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode):
        mx_dtype = str(mx_ele_dtype)
        exp_bits = 0
        mantissa_bits = 0
        if "float4_e2m1" in mx_dtype:
            exp_bits = 2
            mantissa_bits = 1
        elif "float4_e1m2" in mx_dtype:
            exp_bits = 1
            mantissa_bits = 2
        elif "float8_e4m3fn" in mx_dtype:
            exp_bits = 4
            mantissa_bits = 3
        elif "float8_e5m2" in mx_dtype:
            exp_bits = 5
            mantissa_bits = 2

        max_norm = get_dtype_range(mx_dtype)[1]
        if scale_alg == 1 or (
            scale_alg == 2 and dst_type_max != 6 and dst_type_max != 7
        ):
            ret = fp_array / (2**share_exp)
        else:
            ret = np.where(
                share_exp == -127,
                np.where(fp_array >= 0, 0.0, -0.0),
                fp_array / (2**share_exp),
            )
        private_exp = np.floor(
            np.log2(np.abs(ret.astype(np.float64)) + (ret == 0))
        ).astype(fp_array.dtype, copy=False)
        # The minimum representable exponent for 8 exp bits is -126
        # 5bit exp  2^4-1 = 15  or 2 ** (exp_bits - 1) -1
        if "float8_e4m3fn" in mx_dtype or "float8_e5m2" in mx_dtype:
            min_exp = -(2 ** (exp_bits - 1)) + 2  # 指数位 -2^3+4 = -4
        else:
            min_exp = -(2 ** (exp_bits - 1)) + exp_bits
        private_exp = private_exp.clip(min=min_exp)
        # Scale up so appropriate number of bits are in the integer portion of the number
        ret = ret / (2**private_exp) * (2**mantissa_bits)
        ret = _mx_round_mantissa(ret, round_mode)
        # Undo scaling
        ret = ret / (2**mantissa_bits) * (2**private_exp)
        # Set values > max_norm to Inf if desired, else clamp them
        np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
        return ret

    def pad_to_even(tensor, axis):
        if not isinstance(tensor, np.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
            )

        shape = tensor.shape
        length = shape[axis]

        # 如果已经是偶数，直接返回原数组
        if length % 2 == 0:
            return tensor

        # 构造 pad_width：仅对目标 axis 补一个 0
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[axis] = (0, 1)  # 在 axis 维度末尾补一个 0

        padded_tensor = np.pad(
            tensor, pad_width, mode="constant", constant_values=2**-127
        )
        return padded_tensor

    def _mx_undo_reshape_to_blocks(fp_array, axis, orig_shape, padded_shape):
        # Undo tile reshaping
        fp_array = fp_array.reshape(padded_shape)
        # Undo padding
        if tuple(padded_shape) != tuple(orig_shape):
            slices = [slice(0, x) for x in orig_shape]
            fp_array = fp_array[tuple(slices)]
        # Remove extra dimension
        fp_array = np.squeeze(fp_array, axis=axis + 1)
        return fp_array

    def interleave(tensor, axis, n_group: int = 2):
        if not isinstance(tensor, np.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
            )
        # 获取目标轴的长度
        length = tensor.shape[axis]
        # 检查是否可整除
        if length % n_group != 0:
            raise ValueError(
                f"Axis length ({length}) must be divisible by n_group ({n_group})"
            )

        group_length = length // n_group  # 每组长度
        shape = list(tensor.shape)

        # 重塑形状：在目标轴后插入组维度
        new_shape = shape[:axis] + [group_length, 2] + shape[axis + 1 :]
        reshaped = tensor.reshape(new_shape)

        # 构建转置顺序：交换组维度和组内维度
        transpose_order = (
            list(range(0, axis + 1))  # 目标轴之前的维度
            + list(range(axis + 2, len(new_shape)))
            + [
                axis + 1,
            ]
        )  # 后续维度

        # 执行转置
        transposed = reshaped.transpose(transpose_order)

        return transposed

    if not isinstance(fp_array, np.ndarray):
        raise RuntimeError(
            f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}"
        )
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
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

    if scale_alg != 1 and max_low_bound != 0:
        raise RuntimeError(
            f"max_low_bound must be 0 when scale_alg != 1, got scale_alg={scale_alg}, max_low_bound={max_low_bound}"
        )
    axis = len(fp_array.shape) + axis if axis < 0 else axis
    postAxis = 1
    for dim in fp_array.shape[axis + 1 :]:
        postAxis = postAxis * dim
    # padding & reshape to block_size
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(
        fp_array, axis, block_size
    )
    # get mx scale exponents
    if scale_alg == 0:
        share_exp = _mx_calculate_share_exp(
            fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype
        )
    elif scale_alg == 1:
        if mx_ele_dtype in ("float4_e2m1", "float4_e1m2"):
            raise RuntimeError("scale_alg = 1 is only supported by float8")
        share_exp = _mx_calculate_share_exp_nv(
            fp_array,
            scale_axis=axis + 1,
            mx_ele_dtype=mx_ele_dtype,
            max_norm=get_dtype_range(mx_ele_dtype)[1],
            subnormal=True,
            max_low_bound=max_low_bound,
        )
    elif scale_alg == 2:
        if mx_ele_dtype not in ("float4_e2m1"):
            raise RuntimeError("scale_alg = 2 is only supported by float4_e2m1")
        # dst_type_max为0时，当成6处理
        if dst_type_max == 0:
            dst_type_max = 6
        if dst_type_max == 6 or dst_type_max == 7:
            if (
                postAxis < 64
                and axis != (len(orig_shape) - 2)
                and fp_array.dtype.name in ("float16")
            ):
                share_exp = _mx_calculate_share_exp_nv(
                    fp_array,
                    scale_axis=axis + 1,
                    mx_ele_dtype=mx_ele_dtype,
                    max_norm=dst_type_max,
                    subnormal=False,
                )
            else:
                share_exp = _mx_calculate_share_exp_dynamic_dtype_range(
                    fp_array,
                    scale_axis=axis + 1,
                    mx_ele_dtype=mx_ele_dtype,
                    max_norm=dst_type_max,
                    subnormal=False,
                )
        else:
            # 当dst_type_max=6/7时，为性能优化考虑，在OCP标准基础上新增判断尾数位值的逻辑，大于1.75即尾数位的前2位是1，且后面的不全为0，
            # 举例BF16的数值0B0 01010101 11000010，即表示满足条件，此刻是对scale做+1的处理。
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
        fp_array, share_exp, mx_ele_dtype, round_mode
    )
    # undo reshape
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)
    # convert to fp8_e8m0 & fp4/fp8 dtype
    ele_dtype_np = _MX_DTYPE_MAP[mx_ele_dtype]
    # share_exp is always float32
    scale_array = 2**share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array_pad = pad_to_even(scale_array, axis=axis)

    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2

    # when axis is -1, do not need interleave
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)

    scale_array = scale_array_pad.astype(np_mx_scale, copy=False)

    return ele_array, scale_array


# ============================================================================
# flat_quant compute functions
# ============================================================================


def compute_mx_fp4(x, p1, p2):
    """
    MXFP4 (float4_e2m1) 量化模式计算。

    计算流程:
    1. Kronecker 分解: x1 = x @ p2, x2 = p1 @ x1
    2. x2 转 bfloat16 并 flatten 末两维
    3. 调用 mx_quantize 生成 MX scale 和量化元素
    """
    x_f = x.astype(np.float32)
    p1_f = p1.astype(np.float32)
    p2_f = p2.astype(np.float32)
    x1 = np.matmul(x_f, p2_f).astype(x.dtype)
    x2 = np.matmul(p1_f, x1.astype(np.float32)).astype(np_bfloat16)
    x2 = x2.reshape(x2.shape[0], -1)

    ele_array, scale_array = mx_quantize(
        x2,
        mx_ele_dtype="float4_e2m1",
        axis=-1,
        block_size=32,
        round_mode="rint",
        scale_alg=0,
    )
    return [ele_array, scale_array]


def compute_int4(x, p1, p2, clip_ratio):
    """
    INT4 量化模式计算。

    计算流程:
    1. Kronecker 分解: x1 = x @ p2, x2 = p1 @ x1
    2. x2 转 float16 并 flatten 末两维
    3. 计算 per-row max abs → quant_scale = max / (7 / clip_ratio)
    4. 量化到 int4, clamp[-8, 7]
    """
    x_f = x.astype(np.float32)
    p1_f = p1.astype(np.float32)
    p2_f = p2.astype(np.float32)
    x1 = np.matmul(x_f, p2_f).astype(x.dtype)
    x2 = np.matmul(p1_f, x1.astype(np.float32)).astype(np.float16)

    x_shape = x.shape
    x2 = x2.reshape(x2.shape[0], -1)

    qscale = np.max(np.abs(x2), axis=-1, keepdims=True).astype(np.float32)
    ratio = np.ones_like(qscale) * 7 / clip_ratio
    qscale2 = ratio / qscale
    out = x2 * qscale2.astype(np.float16)
    qscale = (qscale / ratio).flatten()
    out = np.clip(out, -8, 7)
    out = np.round(out).astype(np_int4).reshape(x_shape)

    return [out, qscale]


# ============================================================================
# TestSpec
# ============================================================================


class FlatQuantTestSpec:
    """flat_quant 算子的 TestSpec 类。"""

    @staticmethod
    def compare(*outputs, **kwargs):
        """多输出精度比对：ttk 传入顺序为 npu0, npu1, ..., golden0, golden1, ...
        前半为 NPU 输出，后半为 golden，按位置配对比较。
        """
        n = len(outputs) // 2
        results = []
        for i in range(n):
            r = numeric_ulp_compare(outputs[i], outputs[n + i])
            results.append(r)
        return results

    @staticmethod
    def golden(
        x,
        kronecker_p1,
        kronecker_p2,
        group_list=None,
        *,
        clip_ratio: float = 1.0,
        dst_dtype: int = MXFP4_DST_DTYPE,
        **kwargs,
    ):
        """
        flat_quant kernel golden: Kronecker 分解 + 量化。

        根据 dst_dtype 分发:
        - MXFP4 (dst_dtype == 40): compute_mx_fp4 → [ele_array, scale_array]
        - INT4 (dst_dtype == 29): compute_int4 → [out_32, quant_scale]
        """
        if dst_dtype == MXFP4_DST_DTYPE:
            return compute_mx_fp4(x, kronecker_p1, kronecker_p2)
        else:
            return compute_int4(x, kronecker_p1, kronecker_p2, clip_ratio)


__spec__ = {
    "flat_quant": "FlatQuantTestSpec",
}
