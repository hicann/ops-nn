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
TTK custom golden for swiglu_mx_quant_with_dual_axis.

Fused operator = SwiGLU activation (mode 0) followed by dual-axis (-1 / -2)
dynamic MX block quantization. It produces FOUR outputs:

    y1        : -1 axis (column direction) quantized elements (FP4/FP8)
    mx_scale1 : -1 axis block scales (FLOAT8_E8M0)
    y2        : -2 axis (row direction) quantized elements (FP4/FP8)
    mx_scale2 : -2 axis block scales (FLOAT8_E8M0)

This golden is an INDEPENDENT CPU reference. It never invokes the NPU operator
under test. Every function is composed from already-merged sibling goldens that
share one MX element-quantization core, so there is no algorithm drift:

    - SwiGLU (mode 0) body    <- torch.nn.functional.silu (Torch SiLU/Swish),
                                 semantics per quant/swiglu_mx_quant/golden.py::_swiglu
    - -1 / -2 axis quant      <- quant/dynamic_mx_quant_with_dual_axis/golden.py::_mx_quantize
    - grouped -2 axis quant   <- quant/grouped_dynamic_mx_quant/golden.py::_grouped_mx_quantize

MX/FP4 quantization keeps the standard NumPy implementation on purpose: the
fused op is NOT required to be a single Torch API. Only the SwiGLU activation
body is expressed with Torch F.silu; block quant, E8M0 scale encoding and the
dual-axis / grouped packing remain the merged NumPy sibling-golden code above.

Inputs (positional, in op-def order):
    x           : numpy array (fp16/bf16), 2-D [M, 2N]
    group_index : numpy int64 array, OPTIONAL (absent -> None), shape [G],
                  cumsum row boundaries, each > 0 and last == M

Attributes (keyword, names match swiglu_mx_quant_with_dual_axis_def.cpp):
    activate_left   : bool  (default True)  True -> Swish(front half) * back half
    round_mode      : str   (default "rint")  rint / round / floor
    scale_alg       : int   (default 0)     0 = OCP, 1 = cuBLAS (FP8 only)
    dst_type        : int   (default 35)    35=E5M2, 36=E4M3FN, 40=E2M1, 41=E1M2
    max_dtype_value : float (default 0.0)   reserved, only 0.0 supported

Semantics note (documented conflict, resolved by evidence):
    README stage-1 text writes the Swish target on the wrong half. The kernel
    (regbase.h ComputeVfSwiglu), docs/aclnnSwigluMxQuantWithDualAxis.md, and the
    sibling golden swiglu_mx_quant all agree that activate_left=True applies
    SiLU/Swish to the FRONT half: res = silu(front) * back, where
    silu(x) = x * sigmoid(x). This golden follows the kernel/aclnn semantics,
    not the README prose.
"""

import copy
import re

import numpy as np

__golden__ = {
    "kernel": {
        "swiglu_mx_quant_with_dual_axis": "swiglu_mx_quant_with_dual_axis_golden"
    }
}


DATA_TYPE_INT_TO_STR = {
    0: "float32",
    1: "float16",
    27: "bfloat16",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    37: "float8_e8m0",
    40: "float4_e2m1",
    41: "float4_e1m2",
    42: "hifloat4",
}

BLOCK_SIZE = 32


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
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


# ---------------------------------------------------------------------------
# SwiGLU (mode 0) activation body -- torch.nn.functional.silu
#   The gated half is activated with the standard Torch SiLU/Swish op
#   (silu(x) = x * sigmoid(x)) rather than a hand-rolled sigmoid, per the
#   reference standard "SwiGLU body uses Torch F.silu". The half-selection and
#   the element-wise product with the linear half stay in fp32. Semantics match
#   the kernel (regbase.h ComputeVfSwiglu) and the sibling golden
#   quant/swiglu_mx_quant/golden.py::_swiglu (mode 0): silu(gate) * linear.
# ---------------------------------------------------------------------------
def _swiglu_mode0(x_fp32, activate_left):
    """SwiGLU mode 0 on a 2-D [M, 2N] tensor, split on the last axis.

    activate_left=True  -> silu(front) * back   (gate = front half)
    activate_left=False -> silu(back)  * front  (gate = back half)

    The activation is computed by ``torch.nn.functional.silu`` in fp32; the
    result is returned as a fp32 numpy array so the downstream MX quantization
    (NumPy) is untouched.
    """
    import torch
    import torch.nn.functional as F

    xf = np.ascontiguousarray(x_fp32.astype(np.float32))
    h = xf.shape[-1] // 2
    a = xf[:, :h]
    b = xf[:, h:]
    t_a = torch.from_numpy(np.ascontiguousarray(a))
    t_b = torch.from_numpy(np.ascontiguousarray(b))
    if activate_left:
        res = F.silu(t_a) * t_b
    else:
        res = F.silu(t_b) * t_a
    return res.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Shared MX element-quantization core
# (identical across dynamic_mx_quant_with_dual_axis / grouped_dynamic_mx_quant)
# ---------------------------------------------------------------------------
def _mx_calculate_share_exp(fp_array, scale_axis, mx_ele_dtype):
    """OCP shared exponent: floor(log2(amax)) - emax."""
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


def _mx_calculate_share_exp_nv(fp_array, scale_axis, mx_ele_dtype):
    """cuBLAS shared exponent (mantissa-aware round up). FP8 only."""
    max_norm = _get_dtype_range(mx_ele_dtype)[1]
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True).astype(
        np.float32
    )
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
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_round_mantissa(fp_array, round_mode):
    if round_mode in ("rint", "even"):  # tie to even (C rint)
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):  # tie away from zero (C round)
        sign = np.signbit(fp_array)
        rounded_abs = np.floor(np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
        fp_array = np.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":  # round to minus infinity (C floor)
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":
        fp_array = np.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array


def _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode):
    """Cast V / 2^share_exp into the destination MX element format.

    Handles the private (per-element) exponent normalization, minimum
    representable exponent clipping and max_norm saturation. The share_exp==-127
    guard avoids dividing by the clipped minimum scale.
    """
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
    # minimum representable exponent for the element format
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    ret = ret / (2**private_exp) * (2**mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    ret = ret / (2**mantissa_bits) * (2**private_exp)
    max_norm = _get_dtype_range(mx_dtype)[1]
    np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret


def _select_share_exp(fp_array, scale_axis, mx_ele_dtype, scale_alg):
    """Pick OCP / cuBLAS shared exponent, then clip to E8M0 range."""
    if scale_alg == 0 or (mx_ele_dtype in ("float4_e2m1", "float4_e1m2")):
        share_exp = _mx_calculate_share_exp(fp_array, scale_axis, mx_ele_dtype)
    elif scale_alg == 1:
        share_exp = _mx_calculate_share_exp_nv(fp_array, scale_axis, mx_ele_dtype)
    else:
        raise RuntimeError(f"scale_alg is not supported: {scale_alg}")
    scale_emax = 2 ** (8 - 1) - 1  # 127 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax
    return share_exp


# ---------------------------------------------------------------------------
# Non-grouped dual-axis quant -- from dynamic_mx_quant_with_dual_axis/golden.py
# ---------------------------------------------------------------------------
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


def _pad_to_even(tensor, axis):
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
        )
    length = tensor.shape[axis]
    if length % 2 == 0:
        return tensor
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)
    return np.pad(tensor, pad_width, mode="constant", constant_values=2**-127)


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
        list(range(0, axis + 1)) + list(range(axis + 2, len(new_shape))) + [axis + 1]
    )
    return reshaped.transpose(transpose_order)


def _mx_quantize(fp_array, mx_ele_dtype, axis, block_size, round_mode, scale_alg):
    """Single-axis dynamic MX quantization. Returns (scale_array, ele_array)."""
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
    share_exp = _select_share_exp(fp_array, axis + 1, mx_ele_dtype, scale_alg)

    ele_array = _mx_quantize_to_element_format(
        fp_array, share_exp, mx_ele_dtype, round_mode
    )
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)

    ele_dtype_np = _numpy_mx_dtype(mx_ele_dtype)
    scale_array = 2**share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    # NPU casts NaN (with/without sign) to positive zero
    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)

    scale_array_pad = _pad_to_even(scale_array, axis=axis)
    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)
    result_shape[axis] = scale_array_pad.shape[axis] // 2
    # -1 axis (last) does not interleave; -2 axis interleaves every two rows
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = _interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)
    scale_array = scale_array_pad.astype(_numpy_mx_dtype("float8_e8m0"), copy=False)
    return scale_array, ele_array


# ---------------------------------------------------------------------------
# Grouped -2 axis quant -- from quant/grouped_dynamic_mx_quant/golden.py
# ---------------------------------------------------------------------------
def _grouped_mx_reshape_to_blocks(fp_array, group_index, axis, block_size):
    groups = []
    padded_group_index = []
    prev = 0
    for end in group_index:
        slices = [slice(None)] * fp_array.ndim
        slices[axis] = slice(prev, end)
        groups.append(fp_array[tuple(slices)])
        prev = end

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

    padded_array = np.concatenate(padded_groups, axis=axis)
    expanded_array = np.expand_dims(padded_array, axis=axis + 1)
    total_blocks = padded_array.shape[axis] // block_size
    reshape = list(expanded_array.shape)
    reshape[axis] = total_blocks
    reshape.insert(axis + 1, block_size)
    reshaped_array = expanded_array.reshape(reshape)
    padded_shape = expanded_array.shape
    return reshaped_array, padded_group_index, padded_shape


def _grouped_mx_undo_reshape_to_blocks(
    fp_array, group_index, axis, padded_group_index, padded_shape
):
    fp_array = fp_array.reshape(padded_shape)
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    split_indices = padded_group_index[:-1]
    groups = np.split(fp_array, split_indices, axis=axis)
    trimmed_groups = []
    prev = 0
    for i, end in enumerate(group_index):
        original_length = end - prev
        slices = [slice(None)] * fp_array.ndim
        slices[axis] = slice(0, original_length)
        trimmed_groups.append(groups[i][tuple(slices)])
        prev = end
    return np.concatenate(trimmed_groups, axis=axis)


def _reshape_scale_array_pad(scale_array, group_index, axis):
    """Pack grouped -2 axis scales into [floor(M/64)+G, N, 2] with every-two-row
    interleave. Group g is placed at pair offset (cumsum_g//64 + g); intra-group
    rows are padded to even with 2**-127, inter-group gaps filled with 1.
    """
    import math

    scale_array = np.squeeze(scale_array, 1)
    cur_idx = 0
    pre_element = 0
    for idx, element in enumerate(group_index):
        next_group_start = (element // 64 + idx + 1) * 2
        real_idx = cur_idx + math.ceil((element - pre_element) / 32)
        pad_idx = math.ceil(real_idx / 2) * 2
        zero_row = np.full((1, scale_array.shape[1]), 2**-127)
        one_row = np.full((1, scale_array.shape[1]), 1)
        for i in range(real_idx, pad_idx):
            scale_array = np.insert(scale_array, i, zero_row, axis=0)
        for i in range(pad_idx, next_group_start):
            scale_array = np.insert(scale_array, i, one_row, axis=0)
        pre_element = element
        cur_idx = next_group_start

    scale_array = (
        scale_array.reshape(int(scale_array.shape[0] / 2), 2, scale_array.shape[1])
        .transpose(0, 2, 1)
        .reshape(int(scale_array.shape[0] / 2), scale_array.shape[1], 2)
    )
    return scale_array


def _grouped_mx_quantize(
    fp_array, group_index, mx_ele_dtype, axis, block_size, round_mode, scale_alg
):
    """Grouped single-axis MX quantization (axis=-2). Returns (scale, ele)."""
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

    def is_non_reverse_order(arr):
        if len(arr) <= 1:
            return True
        return np.all(np.diff(arr) >= 0)

    if not is_non_reverse_order(group_index):
        raise RuntimeError("Input tensor group_index should be non-reverse order.")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    if axis != 0:
        raise RuntimeError(f"grouped quant only supports the -2 axis, got axis={axis}")
    if int(group_index[-1]) != fp_array.shape[axis]:
        raise RuntimeError(
            "The last element of group_index should match the -2 axis dimension of x."
        )

    fp_array, padded_group_index, padded_shape = _grouped_mx_reshape_to_blocks(
        fp_array, group_index, axis, block_size
    )
    share_exp = _select_share_exp(fp_array, axis + 1, mx_ele_dtype, scale_alg)

    ele_array = _mx_quantize_to_element_format(
        fp_array, share_exp, mx_ele_dtype, round_mode
    )
    ele_array = _grouped_mx_undo_reshape_to_blocks(
        ele_array, group_index, axis, padded_group_index, padded_shape
    )
    share_exp = np.squeeze(share_exp, axis=axis + 1)

    ele_dtype_np = _numpy_mx_dtype(mx_ele_dtype)
    scale_array = 2**share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)
    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array = _reshape_scale_array_pad(scale_array, group_index, axis)
    scale_array = scale_array.astype(_numpy_mx_dtype("float8_e8m0"), copy=False)
    return scale_array, ele_array


# ---------------------------------------------------------------------------
# Top-level golden
# ---------------------------------------------------------------------------
def swiglu_mx_quant_with_dual_axis_golden(
    x,
    group_index=None,
    *,
    activate_left=True,
    round_mode="rint",
    scale_alg=0,
    dst_type=35,
    max_dtype_value=0.0,
    **kwargs,
):
    """Golden for swiglu_mx_quant_with_dual_axis.

    Parameter names/order follow swiglu_mx_quant_with_dual_axis_def.cpp (inputs
    positional, attributes keyword). All input tensors are numpy.ndarray.

    Returns:
        (y1, mx_scale1, y2, mx_scale2)
    """
    x = np.asarray(x)
    if group_index is not None:
        group_index = np.asarray(group_index).reshape(-1).astype(np.int64)

    activate_left = bool(activate_left)
    round_mode = str(round_mode)
    scale_alg = int(scale_alg)
    dst_type = int(dst_type)
    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]

    # 1) SwiGLU in fp32
    x_fp32 = x.astype(np.float32)
    swiglu_fp32 = _swiglu_mode0(x_fp32, activate_left)

    # 2) round the SwiGLU intermediate to x precision (fp16/bf16) then back to
    #    fp32, mirroring the kernel's intermediate storage precision.
    x_dtype_name = str(x.dtype)
    if "bfloat16" in x_dtype_name:
        try:
            from ml_dtypes import bfloat16 as _bf16

            swiglu_fp32 = swiglu_fp32.astype(_bf16).astype(np.float32)
        except ImportError:
            pass
    elif "float16" in x_dtype_name:
        swiglu_fp32 = swiglu_fp32.astype(np.float16).astype(np.float32)

    # 3) -1 axis quant (group-independent): y1 / mx_scale1
    scale1, y1 = _mx_quantize(
        swiglu_fp32, dst_type_str, -1, BLOCK_SIZE, round_mode, scale_alg
    )

    # 4) -2 axis quant: y2 / mx_scale2 (grouped when group_index is present)
    if group_index is None:
        scale2, y2 = _mx_quantize(
            swiglu_fp32, dst_type_str, -2, BLOCK_SIZE, round_mode, scale_alg
        )
    else:
        scale2, y2 = _grouped_mx_quantize(
            swiglu_fp32,
            group_index,
            dst_type_str,
            -2,
            BLOCK_SIZE,
            round_mode,
            scale_alg,
        )

    return y1, scale1, y2, scale2
