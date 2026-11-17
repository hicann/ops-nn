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

import copy
import re

import numpy as np
from ml_dtypes import bfloat16

__golden__ = {"kernel": {"dynamic_mx_quant": "dynamic_mx_quant_golden"}}


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


def _mx_undo_reshape_to_blocks(fp_array, axis, orig_shape, padded_shape):
    fp_array = fp_array.reshape(padded_shape)
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array


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
    fp_array, scale_axis, mx_ele_dtype, max_low_bound=0.0, dst_type_max=0.0
):
    max_norm = dst_type_max or _get_dtype_range(mx_ele_dtype)[1]
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True).astype(
        np.float32
    )
    zero_blocks = fp_abs_max == 0
    fp_abs_max = np.maximum(fp_abs_max, np.float32(max_low_bound))
    s_fp32 = fp_abs_max / max_norm
    binary_ints = np.array(s_fp32.view(np.uint32))
    exponent_mask = np.uint32(0x7F800000)
    mantissa_mask = np.uint32(0x007FFFFF)
    exponents = (binary_ints & exponent_mask) >> 23
    exponents_int16 = exponents.astype(np.int16)
    mantissas = binary_ints & mantissa_mask
    condition_1 = (exponents_int16 > 0) & (exponents_int16 < 254) & (mantissas > 0)
    condition_2 = (exponents_int16 == 0) & (mantissas > 2**22)
    exponents_int16 = np.where(
        (condition_1 | condition_2), exponents_int16 + 1, exponents_int16
    )
    res = (exponents_int16 - 127).astype(np.float32)
    # max_low_bound must not turn an originally all-zero block into a nonzero scale.
    res[zero_blocks] = -float("inf")
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


def _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode):
    mx_dtype = str(mx_ele_dtype)
    match = re.search(r"e(\d+)m(\d+)", mx_dtype)
    if match:
        exp_bits = int(match.group(1))
        mantissa_bits = int(match.group(2))
    else:
        raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

    ret = fp_array / (2**share_exp)
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


def _pad_to_even(tensor, axis):
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
        )

    shape = tensor.shape
    length = shape[axis]

    if length % 2 == 0:
        return tensor

    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)

    padded_tensor = np.pad(tensor, pad_width, mode="constant", constant_values=2**-127)
    return padded_tensor


def _interleave(tensor, axis, n_group=2):
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
        )
    length = tensor.shape[axis]
    if length % n_group != 0:
        raise ValueError(
            f"Axis length ({length}) must be divisible by n_group ({n_group})"
        )

    group_length = length // n_group
    shape = list(tensor.shape)

    new_shape = shape[:axis] + [group_length, 2] + shape[axis + 1 :]
    reshaped = tensor.reshape(new_shape)

    transpose_order = (
        list(range(0, axis + 1))
        + list(range(axis + 2, len(new_shape)))
        + [
            axis + 1,
        ]
    )

    transposed = reshaped.transpose(transpose_order)
    return transposed


def _mx_quantize(
    fp_array,
    mx_ele_dtype="float4_e2m1",
    axis=-1,
    block_size=32,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
    max_low_bound=0.0,
):
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

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(
        fp_array, axis, block_size
    )
    if scale_alg == 0 or (
        scale_alg != 2 and mx_ele_dtype in ("float4_e2m1", "float4_e1m2")
    ):
        share_exp = _mx_calculate_share_exp(
            fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype
        )
    else:
        share_exp = _mx_calculate_share_exp_nv(
            fp_array,
            scale_axis=axis + 1,
            mx_ele_dtype=mx_ele_dtype,
            max_low_bound=max_low_bound if scale_alg == 1 else 0.0,
            dst_type_max=dst_type_max if scale_alg == 2 else 0.0,
        )
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    ele_array = _mx_quantize_to_element_format(
        fp_array, share_exp, mx_ele_dtype, round_mode
    )
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)
    ele_dtype_np = _numpy_mx_dtype(mx_ele_dtype)
    scale_array = 2**share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array_pad = _pad_to_even(scale_array, axis=axis)

    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = _interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)

    scale_array = scale_array_pad.astype(_numpy_mx_dtype("float8_e8m0"), copy=False)

    return scale_array, ele_array


def _e6m2_rec(x):
    # input: E6M2, 8-bit, >= 0
    # output: BF16, 16-bit
    # Compute: 1 / E6M2
    if np.isnan(x):
        return bfloat16(np.nan)
    else:
        E6 = np.floor(np.log2(x + 2 ** (-1000)))
        M2 = x * 2 ** (-E6 + 2) - 4

        if M2 == 0:  # 1.00
            M7 = 0.0  # binary: 000-0000
        elif M2 == 1:  # 1.25
            M7 = 77  # binary: 100-1101
        elif M2 == 2:  # 1.50
            M7 = 43  # binary: 010-1011
        elif M2 == 3:  # 1.75
            M7 = 18  # binary: 001-0010
        else:
            print("Unexpected Input")
            exit()

        if M2 == 0:
            E8 = -E6
        else:
            E8 = -E6 - 1

        res = 2 ** (E8) * (1 + M7 * 2 ** (-7))
        return res


def _bf16_to_e6m2(x):
    # input Non-negative BF16
    # output E6M2
    # E6M2: 2^[-48, 15] * 1.M2, E6_offset = 48, 2^15 * 1.75 is NaN
    if (
        np.isinf(x)
        or np.isnan(x)
        or (x >= 1.625 * (2**15))
        or (x == 0)
        or (x < 2 ** (-48))
    ):
        return bfloat16(np.nan)
    else:
        x = 2 ** (-48) if (x < 2 ** (-48)) else x  # underflow handling
        E = np.floor(np.log2(x))  # Exponent
        E6M2 = np.round(x * 2 ** (-E + 2)) * 2 ** (
            -2 + E
        )  # Round Half tie to even or away
    return E6M2


def _float_to_e6m2_int(x):
    if np.isnan(x):
        return 0xFF  # 11111111 in binary
    else:
        E6 = np.floor(np.log2(x + 2 ** (-1000)))
        M2 = x * (2 ** (-E6 + 2)) - 4.0
        E6 = E6 + 48
        B8_E6M2 = np.uint8(int(E6) * 4) | np.uint8(int(M2))

        return B8_E6M2


def _dynamic_quant_hifp4(x):
    G = 64
    shape = x.shape
    last_dim = shape[-1]
    Ncnt = np.ceil(last_dim / G).astype(int)
    # 将输入 reshape 成二维张量，以便复用当前逻辑
    x_2d = x.reshape(-1, last_dim)
    Mi = x_2d.shape[0]
    Ni = x_2d.shape[1]
    res = np.zeros((Mi, Ni))
    scale = np.zeros((Mi, Ncnt)).astype(np.float32)
    scale_uint8 = scale.view(np.uint8).reshape(Mi * Ncnt * 4)
    ksi = 0

    from en_dtypes import float4_e1m2

    for i in range(Mi):
        for j in range(Ncnt):
            ori = x_2d[i, j * G : j * G + G]
            S = np.ones(G)  # sign of grp values
            S[ori < 0] = -1
            S = S.T
            tmpG = np.abs(ori)  # abs of grp values

            V16 = np.zeros(16)
            for k in range(16):
                V16[k] = np.max(tmpG[k * 4 : k * 4 + 4])

            V8 = np.zeros(8)
            for k in range(8):
                V8[k] = np.max(V16[k * 2 : k * 2 + 2])

            Vmax = np.max(V8).astype(bfloat16)

            # level-1 exp
            ##TODO: const_rec using a const with dtype of bfloat16
            Const_rec = np.uint16(0x3E12).view(bfloat16)

            SF = Vmax * Const_rec

            E6M2 = _bf16_to_e6m2(SF)
            E6M2_code = _float_to_e6m2_int(E6M2)

            # level-2 exp
            REC_E6M2 = _e6m2_rec(E6M2)

            E1_8 = (V8 * REC_E6M2) >= 4

            # level-3 exp
            E1_8x2 = np.zeros(16)
            for k in range(8):
                E1_8x2[k * 2 : k * 2 + 2] = E1_8[k]
            E1_16 = (V16 * REC_E6M2 * 2 ** (-E1_8x2)) >= 2

            e6m2_uint8 = np.uint8(E6M2_code)

            scale_uint8[ksi] = e6m2_uint8
            ksi += 1

            E1_8_int = E1_8.astype(np.int32)
            E1_8_bit = np.array(
                [
                    np.packbits(E1_8_int[:8][::-1])[
                        0
                    ],  # 第一个 uint8 packbit bit序在小端机要反过来
                ],
                dtype=np.uint8,
            )
            scale_uint8[ksi] = E1_8_bit[0]
            ksi += 1

            E1_16_int = E1_16.astype(np.int32)
            E1_16_bit = np.array(
                [
                    np.packbits(E1_16_int[:8][::-1])[
                        0
                    ],  # 第一个 uint8 packbit bit序在小端机要反过来
                    np.packbits(E1_16_int[8:][::-1])[
                        0
                    ],  # 第二个 uint8 packbit bit序在小端机要反过来
                ],
                dtype=np.uint8,
            )
            scale_uint8[ksi] = E1_16_bit[0]
            ksi += 1
            scale_uint8[ksi] = E1_16_bit[1]
            ksi += 1

            DE16 = E1_16 + E1_8x2  # Fused 16 Exp offsets
            DE64 = np.zeros(G)  # Fused 64 Exp offsets
            for k in range(16):
                DE64[k * 4 : k * 4 + 4] = DE16[k]
            in_grp = (
                bfloat16(tmpG) * bfloat16(REC_E6M2) * 2 ** (-DE64)
            )  # Round Half tie to away or even

            res[i, j * G : j * G + G] = S * in_grp
    res = res.astype(float4_e1m2)

    scale_shape = list(shape[:-1]) + [Ncnt]
    scale = scale.reshape(scale_shape)
    output_data = [res.reshape(shape), scale]
    return output_data


# Frozen FP32 reciprocal-of-six coefficient; multiply after maxLowBound.
_NVFP4_INV_SIX = np.float32(float.fromhex("0x1.555556p-3"))

_NVFP4_FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
_NVFP4_FP8 = np.array(
    [
        i * 2.0**-9 if i < 8 else (1 + (i & 7) / 8) * 2.0 ** ((i >> 3) - 7)
        for i in range(127)
    ],
    dtype=np.float32,
)


def _nvfp4_encode_nearest(values, table, mode="rint"):
    """Vectorized enumerated nearest value, ties even/away, or signed floor."""
    values = np.asarray(values, dtype=np.float32)
    # Clamp finite magnitudes before selecting adjacent representable values.
    magnitude = np.minimum(np.abs(values), table[-1])
    hi = np.minimum(np.searchsorted(table, magnitude, side="left"), len(table) - 1)
    lo = np.maximum(hi - 1, 0)
    # FP64 distances avoid loss of tiny source differences; arithmetic is FP32.
    dl = magnitude.astype(np.float64) - table[lo]
    dh = table[hi].astype(np.float64) - magnitude
    if mode == "floor":
        below = np.maximum(np.searchsorted(table, magnitude, side="right") - 1, 0)
        below = np.minimum(below, len(table) - 1)
        codes = np.where(np.signbit(values), hi, below)
    else:
        tie_hi = True if mode == "round" else (hi % 2 == 0)
        codes = np.where((dh < dl) | ((dh == dl) & tie_hi), hi, lo)
    codes = np.where(np.isinf(magnitude), len(table) - 1, codes).astype(np.uint8)
    sign_bit = 8 if len(table) == 8 else 128
    codes |= np.where(np.signbit(values), sign_bit, 0).astype(np.uint8)
    return np.where(np.isnan(values), 7 if len(table) == 8 else 127, codes).astype(
        np.uint8
    )


def _nvfp4_reference(
    x,
    *,
    axis=-1,
    blocksize=16,
    round_mode="rint",
    max_low_bound=0.0,
    promoted_source=False,
):
    """Return packed y, raw E4M3 scale, and exact-result mask, all flat.

    promoted_source=True attests that float32 values are an exact widening of
    an already-rounded FP16/BF16 tensor. It does not enable native FP32 inputs.
    """
    x = np.asarray(x)
    if x.dtype.name not in ("float16", "bfloat16") and not (
        promoted_source and x.dtype.name == "float32"
    ):
        raise ValueError("NVFP4 source must be FP16 or BF16")
    if not 1 <= x.ndim <= 7 or axis not in (-1, x.ndim - 1):
        raise ValueError("NVFP4 requires rank 1..7 and the tail axis")
    if blocksize != 16 or x.shape[-1] % 2:
        raise ValueError("NVFP4 requires block16 and even K")
    mode = "rint" if round_mode is None else round_mode
    if mode not in ("rint", "round", "floor"):
        raise ValueError("Unsupported rounding mode")
    # Host validates the FP32 attribute, not just its Python float carrier.
    with np.errstate(over="ignore", invalid="ignore"):
        max_low_bound = np.float32(max_low_bound)
    if not np.isfinite(max_low_bound) or max_low_bound < 0:
        raise ValueError("Invalid maxLowBound after FP32 conversion")
    k = x.shape[-1]
    rows = int(np.prod(x.shape[:-1]))
    blocks = (k + 15) // 16
    scales = np.zeros((rows, (blocks + 1) // 2 * 2), np.uint8)
    if x.size == 0:
        return np.empty(0, np.uint8), scales.ravel(), np.empty(0, np.uint8)
    # Native sources: abs/max in FP16/BF16, then widen amax to FP32.
    # Trusted Promote sources already contain exactly those source values;
    # abs/max selects the same value after widening, with the NaN rules below.
    # Never round promoted FP16 values to BF16.
    source = x.reshape(rows, k)
    source_padded = np.full((rows, blocks * 16), np.nan, dtype=x.dtype)
    source_padded[:, :k] = source
    source_part = source_padded.reshape(rows, blocks, 16)
    all_nan = np.all(np.isnan(source_part), axis=-1)
    source_zero = np.array(0, dtype=x.dtype)
    amax_source = np.max(
        np.where(np.isnan(source_part), source_zero, np.abs(source_part)), axis=-1
    )
    amax = amax_source.astype(np.float32)

    values = source.astype(np.float32)
    padded = np.full((rows, blocks * 16), np.nan, np.float32)
    padded[:, :k] = values
    part = padded.reshape(rows, blocks, 16)
    with np.errstate(all="ignore"):
        scale32 = np.multiply(
            np.maximum(amax, np.float32(max_low_bound)),
            _NVFP4_INV_SIX,
            dtype=np.float32,
        )
        scale32 = np.minimum(scale32, np.float32(448))
        scale_codes = _nvfp4_encode_nearest(scale32, _NVFP4_FP8)
        scale_codes = np.where(amax == 0, 0, scale_codes).astype(np.uint8)
        scale_codes[all_nan] = 127
        back = _NVFP4_FP8[np.minimum(scale_codes, 126)]
        inv = np.divide(np.float32(1), back, dtype=np.float32)
        inv[amax == 0] = np.float32(0)
        product = np.multiply(part, inv[..., None], dtype=np.float32)
        codes = _nvfp4_encode_nearest(product, _NVFP4_FP4, mode).reshape(
            rows, blocks * 16
        )[:, :k]
    exact = ((~np.isfinite(product)) | (product == 0)).reshape(rows, blocks * 16)[:, :k]
    scales[:, :blocks] = scale_codes
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    return packed.ravel(), scales.ravel(), exact.astype(np.uint8).ravel()


def _nvfp4_ttk_outputs(x, **params):
    """TTK expects logical dtype arrays; the local ACLNN runner uses raw bytes."""
    from en_dtypes import float4_e2m1
    from ml_dtypes import float8_e4m3fn

    packed, scales, _ = _nvfp4_reference(x, **params)
    codes = np.stack((packed & 15, packed >> 4), axis=-1).ravel()
    values = _NVFP4_FP4[codes & 7] * np.where(codes & 8, np.float32(-1), np.float32(1))
    blocks = (x.shape[-1] + 15) // 16
    scale_shape = (*x.shape[:-1], (blocks + 1) // 2, 2)
    return values.astype(float4_e2m1).reshape(x.shape), scales.view(
        float8_e4m3fn
    ).reshape(scale_shape)


def dynamic_mx_quant_golden(
    x,
    *,
    axis=-1,
    round_mode="rint",
    dst_type=40,
    blocksize=32,
    scale_alg=0,
    dst_type_max=0.0,
    max_low_bound=0.0,
    **kwargs,
):
    """
    Golden function for dynamic_mx_quant.
    All the parameters (names and order) follow @dynamic_mx_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    if scale_alg == 3:
        if dst_type != 40 or dst_type_max != 0.0:
            raise ValueError("NVFP4 requires dst_type=40 and dst_type_max=0")

        # Promote widens already-rounded source values exactly. Current TTK
        # also replaces input_dtypes, so do not guess BF16 from float32.
        promoted_source = False
        if x.dtype.name == "float32":
            if kwargs.get("golden_mode") == "Promote":
                promoted_source = True
            else:
                from ml_dtypes import bfloat16

                input_dtypes = kwargs.get("input_dtypes")
                if isinstance(input_dtypes, (tuple, list)):
                    source_dtype = input_dtypes[0] if input_dtypes else None
                else:
                    source_dtype = input_dtypes
                source_dtype = (
                    str(source_dtype).lower() if source_dtype is not None else ""
                )
                if source_dtype in ("bfloat16", "bf16"):
                    x = x.astype(bfloat16)
                elif source_dtype in ("float16", "fp16"):
                    x = x.astype(np.float16)
                else:
                    raise ValueError(
                        "NVFP4 float32 carrier needs FP16/BF16 source metadata "
                        "or trusted TTK golden_mode=Promote"
                    )

        params = dict(
            axis=axis,
            blocksize=blocksize,
            round_mode=round_mode,
            max_low_bound=max_low_bound,
            promoted_source=promoted_source,
        )
        if kwargs.get("raw_output", False):
            return _nvfp4_reference(x, **params)
        return _nvfp4_ttk_outputs(x, **params)

    output0 = kwargs["output_dtypes"][0]
    output1 = kwargs["output_dtypes"][1]
    block_size = blocksize
    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    if x.dtype.name == "bfloat16" and output0 == "hifloat4" and output1 == "float32":
        return _dynamic_quant_hifp4(x)
    else:
        ret = _mx_quantize(
            x,
            mx_ele_dtype=dst_type_str,
            axis=axis,
            block_size=block_size,
            round_mode=round_mode,
            scale_alg=scale_alg,
            dst_type_max=dst_type_max,
            max_low_bound=max_low_bound,
        )
        return ret[1], ret[0]
