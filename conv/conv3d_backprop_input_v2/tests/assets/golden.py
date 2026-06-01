#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file under compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
Golden function for conv3d_backprop_input_v2 kernel.

Based on conv3d_backprop_input_v2_def.cpp:
- Inputs: input_size (REQUIRED), filter (REQUIRED), out_backprop (REQUIRED)
- Output: y (REQUIRED)

Format and DataType support based on SoC version:
- Ascend910B/Ascend910_93:
    - input_size: Format=ND, DataType=INT32/INT64
    - filter: Format=FRACTAL_Z_3D, DataType=FLOAT16/BF16/FLOAT
    - out_backprop: Format=NDC1HWC0, DataType=FLOAT16/BF16/FLOAT
    - y(output): Format=NCDHW/NDC1HWC0, DataType=FLOAT16/BF16/FLOAT
- Ascend950:
    - input_size: Format=ND, DataType=INT32/INT64
    - filter: Format=NCDHW/NDHWC/DHWCN, DataType=FLOAT16/BF16/FLOAT
    - out_backprop: Format=NCDHW/NDHWC, DataType=FLOAT16/BF16/FLOAT
    - y(output): Format=NCDHW/NDHWC, DataType=FLOAT16/BF16/FLOAT

This computes the gradient with respect to the input of a 3D convolution.
"""

import numpy as np
import torch
import torch.nn.functional as F
import math

__golden__ = {
    "kernel": {
        "conv3d_backprop_input_v2": "conv3d_backprop_input_v2_golden"
    }
}

NCDHW_FORMAT = "NCDHW"
NDHWC_FORMAT = "NDHWC"
FP32_STR = "float32"


def ceil_div(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


def _lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def determine_c0(dtype):
    """Determine C0 value based on dtype."""
    if dtype in ["float16", "bfloat16"]:
        return 16
    elif dtype in ["float32", "float"]:
        return 8  # FP32 uses C0=8 for some configs
    else:
        return 16


def due_fp16_overflow(data):
    """Overflow interception for float16"""
    FP16_MAX_FINITE = 65504
    FP16_MIN_FINITE = -65504
    data = np.maximum(data, FP16_MIN_FINITE)
    data = np.minimum(data, FP16_MAX_FINITE)
    return data


def simulate_hf32_precision(data, short_soc_version=None):
    """
    Simulate HF32 (Half Float 32) precision.
    Ascend910B: truncates lower 12 bits of float32 mantissa.
    """
    if data.dtype == np.dtype(np.float32):
        input_hf32 = data.view(np.int32)
        if short_soc_version in ("Ascend910B", "Ascend910B3"):
            input_hf32 = np.right_shift(np.right_shift(input_hf32, 11) + 1, 1)
            input_hf32 = np.left_shift(input_hf32, 12)
        else:
            input_hf32 = np.right_shift(np.right_shift(input_hf32, 12) + 1, 1)
            input_hf32 = np.left_shift(input_hf32, 13)
        return input_hf32.view(np.float32)
    return data


def convert_output_dtype(out, output_dtype, enable_hf32=False, short_soc_version=None):
    """Convert output to target dtype with overflow handling."""
    dtype_map = {
        "float16": (np.float16, True),
        "float32": (np.float32, False),
        "bfloat16": ("ml_dtypes.bfloat16", True),
        "float": (np.float32, False),
    }

    dtype_info = dtype_map.get(output_dtype)
    if dtype_info is None:
        return out.astype(np.float32)

    dtype_ref, need_overflow = dtype_info
    if need_overflow:
        out = due_fp16_overflow(out)

    if isinstance(dtype_ref, str):
        module_name, dtype_name = dtype_ref.split(".")
        try:
            dtype_cls = getattr(__import__(module_name, fromlist=[dtype_name]), dtype_name)
        except (ImportError, AttributeError):
            # Fallback to float32 if ml_dtypes not available
            return out.astype(np.float32)
        out = out.astype(dtype_cls)
    else:
        out = out.astype(dtype_ref)

    if output_dtype == FP32_STR and enable_hf32:
        out = simulate_hf32_precision(out, short_soc_version)

    return out


def to_NCDHW_from_NDC1HWC0(data, ori_shape):
    """Convert from NDC1HWC0 to NCDHW format."""
    n, c, d, h, w = ori_shape
    c0 = determine_c0(data.dtype.name if hasattr(data.dtype, 'name') else str(data.dtype))
    c1 = ceil_div(c, c0)

    # NDC1HWC0 shape: (n, d, c1, h, w, c0)
    # transpose to (n, c1, c0, d, h, w) then reshape to (n, c1*c0, d, h, w)
    data = data.transpose(0, 2, 5, 1, 3, 4)
    data = data.reshape((n, c1 * c0, d, h, w))
    if c1 * c0 > c:
        data = data[:, :c, :, :, :]
    return data


def to_NCDHW_from_FRACTAL_Z_3D(data, filter_ori_shape, groups=1):
    """Convert from FRACTAL_Z_3D to NCDHW format.

    filter_ori_shape: (c_out, c_in_per_group, kd, kh, kw) - CANN filter original shape
    result shape: (c_out, c_in_per_group, kd, kh, kw) - weight in NCDHW format
    """
    c_out, c_in_per_group, kd, kh, kw = filter_ori_shape
    c0 = 16
    c_in = c_in_per_group * groups

    cin_ori = c_in_per_group  # c_in per group
    cout_ori = c_out // groups  # c_out per group

    mag_factor0 = _lcm(cin_ori, c0) // cin_ori
    mag_factor1 = _lcm(cout_ori, c0) // cout_ori
    mag_factor = min(_lcm(mag_factor0, mag_factor1), groups)

    cin_g = align(mag_factor * cin_ori, c0)
    cout_g = align(mag_factor * cout_ori, c0)
    real_g = ceil_div(groups, mag_factor)
    cin1_g = cin_g // c0
    cout1_g = cout_g // c0

    # FRACTAL_Z_3D shape: (real_g * kd * cin1_g * kh * kw, cout1_g, c0, c0)
    data = data.reshape((real_g, kd, cin1_g, kh, kw, cout_g, c0))

    result = np.zeros((c_out, c_in_per_group, kd, kh, kw), dtype=data.dtype)
    for g in range(groups):
        for ci in range(c_in_per_group):
            for co in range(c_out // groups):
                e = g % mag_factor
                dst_cin = e * cin_ori + ci
                dst_cout = e * cout_ori + co
                src_cout = g * cout_ori + co
                result[src_cout, ci, :, :, :] = \
                    data[g // mag_factor, :, dst_cin // c0, :, :, dst_cout, dst_cin % c0]

    return result


def to_NDC1HWC0_from_NCDHW(data, ori_shape):
    """Convert from NCDHW to NDC1HWC0 format."""
    n, c, d, h, w = ori_shape
    c0 = determine_c0(data.dtype.name if hasattr(data.dtype, 'name') else str(data.dtype))
    c1 = ceil_div(c, c0)

    if c1 * c0 > c:
        num_2_padding_in_c = c1 * c0 - c
        zero_padding_array = np.zeros((n, num_2_padding_in_c, d, h, w), dtype=data.dtype)
        data = np.concatenate((data, zero_padding_array), axis=1)

    data = data.reshape((n, c1, c0, d, h, w))
    data = data.transpose(0, 3, 1, 4, 5, 2)
    return data


def to_FRACTAL_Z_3D_from_NCDHW(data, filter_ori_shape, groups=1):
    """Convert from NCDHW to FRACTAL_Z_3D format.

    filter_ori_shape: (c_out, c_in_per_group, kd, kh, kw) - weight original shape in NCDHW
    result shape: (real_g * kd * cin1_g * kh * kw, cout1_g, c0, c0) - FRACTAL_Z_3D format
    """
    c_out, c_in_per_group, kd, kh, kw = filter_ori_shape
    c0 = 16
    c_in = c_in_per_group * groups

    cin_ori = c_in_per_group  # c_in per group
    cout_ori = c_out // groups  # c_out per group

    mag_factor0 = _lcm(cin_ori, c0) // cin_ori
    mag_factor1 = _lcm(cout_ori, c0) // cout_ori
    mag_factor = min(_lcm(mag_factor0, mag_factor1), groups)

    cin_g = align(mag_factor * cin_ori, c0)
    cout_g = align(mag_factor * cout_ori, c0)
    real_g = ceil_div(groups, mag_factor)
    cin1_g = cin_g // c0
    cout1_g = cout_g // c0

    weight_group = np.zeros((real_g, kd, cin1_g, kh, kw, cout_g, c0), dtype=data.dtype)

    for g in range(groups):
        for ci in range(c_in_per_group):
            for co in range(c_out // groups):
                e = g % mag_factor
                dst_cin = e * cin_ori + ci
                dst_cout = e * cout_ori + co
                src_cout = g * cout_ori + co
                weight_group[g // mag_factor, :, dst_cin // c0, :, :, dst_cout, dst_cin % c0] = \
                    data[src_cout, ci, :, :, :]

    weight_group = weight_group.reshape((real_g * kd * cin1_g * kh * kw, cout1_g, c0, c0))
    return weight_group


def to_NCDHW_from_NDHWC(data):
    """Convert from NDHWC to NCDHW format.

    NDHWC: (...leading, D, H, W, C) -> NCDHW: (...leading, C, D, H, W)
    Works for both filter (C_out, kD, kH, kW, C_in) and data (N, D, H, W, C).
    """
    ndim = data.ndim
    if ndim == 5:
        return np.transpose(data, (0, 4, 1, 2, 3))
    return data


def to_NDHWC_from_NCDHW(data):
    """Convert from NCDHW to NDHWC format.

    NCDHW: (...leading, C, D, H, W) -> NDHWC: (...leading, D, H, W, C)
    Works for both filter and data tensors.
    """
    ndim = data.ndim
    if ndim == 5:
        return np.transpose(data, (0, 2, 3, 4, 1))
    return data


def to_NCDHW_from_DHWCN(data):
    """Convert from DHWCN to NCDHW format for conv filter.

    DHWCN: (kD, kH, kW, C_in_per_group, C_out) -> NCDHW: (C_out, C_in_per_group, kD, kH, kW)
    """
    if data.ndim == 5:
        return np.transpose(data, (4, 3, 0, 1, 2))
    return data


def parse_pads(pads):
    """Parse padding parameter."""
    if isinstance(pads, (list, tuple)):
        if len(pads) == 6:
            return int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3]), int(pads[4]), int(pads[5])
        elif len(pads) == 3:
            return int(pads[0]), int(pads[0]), int(pads[1]), int(pads[1]), int(pads[2]), int(pads[2])
        else:
            val = int(pads[0])
            return val, val, val, val, val, val
    else:
        val = int(pads)
        return val, val, val, val, val, val


def parse_strides_dilations(param):
    """Parse strides or dilations parameter."""
    if isinstance(param, (list, tuple)):
        if len(param) == 5:
            return int(param[2]), int(param[3]), int(param[4])
        elif len(param) == 3:
            return int(param[0]), int(param[1]), int(param[2])
        else:
            val = int(param[0])
            return val, val, val
    else:
        val = int(param)
        return val, val, val


def conv3d_backprop_input_v2_golden(input_size, filter, out_backprop,
                                    *,
                                    strides: list,
                                    pads: list,
                                    dilations: list = [1, 1, 1, 1, 1],
                                    groups: int = 1,
                                    data_format: str = NDHWC_FORMAT,
                                    enable_hf32: bool = False,
                                    **kwargs):
    """
    Kernel golden for conv3d_backprop_input_v2.

    This computes the gradient with respect to the input of a 3D convolution.
    It's equivalent to conv_transpose3d in PyTorch.

    Args:
        input_size: output shape (N, C_in, D, H, W) as const input (int32 tensor)
        filter: weight tensor in FRACTAL_Z_3D format
        out_backprop: gradient from next layer in NDC1HWC0 format

    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_dtypes, output_dtypes.
    """
    input_formats = kwargs.get("input_formats", ["ND", "FRACTAL_Z_3D", "NDC1HWC0"])
    output_formats = kwargs.get("output_formats", ["NDC1HWC0"])
    short_soc_version = kwargs.get("short_soc_version", "")
    input_ori_shapes = kwargs.get("input_ori_shapes", None)
    output_ori_shapes = kwargs.get("output_ori_shapes", None)
    input_dtypes = kwargs.get("input_dtypes", ["int32", "float16", "float16"])
    output_dtypes = kwargs.get("output_dtypes", ["float16"])

    # Determine dtype
    filter_dtype_str = input_dtypes[1] if len(input_dtypes) > 1 else "float16"
    out_backprop_dtype_str = input_dtypes[2] if len(input_dtypes) > 2 else "float16"
    output_dtype_str = output_dtypes[0]

    # Dynamic format conversion based on input_formats
    filter_format = input_formats[1] if len(input_formats) > 1 else "FRACTAL_Z_3D"
    out_backprop_format = input_formats[2] if len(input_formats) > 2 else "NDC1HWC0"

    if input_ori_shapes is not None:
        filter_ori_shape = input_ori_shapes[1]
        out_backprop_ori_shape = input_ori_shapes[2]
    else:
        filter_ori_shape = None
        out_backprop_ori_shape = None

    # Convert filter to NCDHW based on its storage format
    if filter_format == "FRACTAL_Z_3D":
        filter_ncdhw = to_NCDHW_from_FRACTAL_Z_3D(filter, filter_ori_shape, groups) if filter_ori_shape is not None else filter
    elif filter_format == "NCDHW":
        filter_ncdhw = filter
    elif filter_format == "NDHWC":
        filter_ncdhw = to_NCDHW_from_NDHWC(filter)
    elif filter_format == "DHWCN":
        filter_ncdhw = to_NCDHW_from_DHWCN(filter)
    else:
        filter_ncdhw = filter

    # Convert out_backprop to NCDHW based on its storage format
    if out_backprop_format == "NDC1HWC0":
        out_backprop_ncdhw = to_NCDHW_from_NDC1HWC0(out_backprop, out_backprop_ori_shape) if out_backprop_ori_shape is not None else out_backprop
    elif out_backprop_format == "NCDHW":
        out_backprop_ncdhw = out_backprop
    elif out_backprop_format == "NDHWC":
        out_backprop_ncdhw = to_NCDHW_from_NDHWC(out_backprop)
    else:
        out_backprop_ncdhw = out_backprop

    # Parse parameters
    stride_d, stride_h, stride_w = parse_strides_dilations(strides)
    dilation_d, dilation_h, dilation_w = parse_strides_dilations(dilations)
    pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right = parse_pads(pads)

    if enable_hf32 and filter_dtype_str == FP32_STR:
        calc_dtype = np.float32
        filter_calc = simulate_hf32_precision(filter_ncdhw.astype(np.float32), short_soc_version)
        out_backprop_calc = simulate_hf32_precision(out_backprop_ncdhw.astype(np.float32), short_soc_version)
    else:
        calc_dtype = np.float64 if filter_dtype_str == FP32_STR else np.float32
        filter_calc = filter_ncdhw.astype(calc_dtype)
        out_backprop_calc = out_backprop_ncdhw.astype(calc_dtype)

    # Prepare for PyTorch conv_transpose3d
    # conv3d_backprop_input is equivalent to conv_transpose3d
    # CANN filter format (C_out, C_in/groups, kD, kH, kW) can be used directly
    # in conv_transpose3d without transposing
    filter_torch = filter_calc

    # Convert to PyTorch tensors
    out_backprop_torch = torch.from_numpy(out_backprop_calc)
    weight_torch = torch.from_numpy(filter_torch)

    # Handle asymmetric padding - use minimum padding for symmetric part
    sym_pad_d = min(pad_d_front, pad_d_back)
    sym_pad_h = min(pad_top, pad_bottom)
    sym_pad_w = min(pad_left, pad_right)

    # Compute output_padding from output_ori_shapes to match target output shape
    # conv_transpose3d output_size = (input_size - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    # So: output_padding = target_size - ((input_size - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1)
    output_padding_d, output_padding_h, output_padding_w = 0, 0, 0
    target_d, target_h, target_w = None, None, None
    if output_ori_shapes is not None and len(output_ori_shapes) > 0:
        target_shape = output_ori_shapes[0]
        n_index, d_index = data_format.index("N"), data_format.index("D")
        h_index, w_index, c_index = data_format.index("H"), data_format.index("W"), data_format.index("C")
        target_d = int(target_shape[d_index])
        target_h = int(target_shape[h_index])
        target_w = int(target_shape[w_index])
        _, _, kd, kh, kw = filter_calc.shape
        _, _, d_in, h_in, w_in = out_backprop_calc.shape
        base_d = (d_in - 1) * stride_d - 2 * sym_pad_d + dilation_d * (kd - 1) + 1
        base_h = (h_in - 1) * stride_h - 2 * sym_pad_h + dilation_h * (kh - 1) + 1
        base_w = (w_in - 1) * stride_w - 2 * sym_pad_w + dilation_w * (kw - 1) + 1
        output_padding_d = max(0, target_d - base_d)
        output_padding_h = max(0, target_h - base_h)
        output_padding_w = max(0, target_w - base_w)

    # Call conv_transpose3d (equivalent to conv3d_backprop_input)
    result = F.conv_transpose3d(
        out_backprop_torch,
        weight_torch,
        bias=None,
        stride=(stride_d, stride_h, stride_w),
        padding=(sym_pad_d, sym_pad_h, sym_pad_w),
        output_padding=(output_padding_d, output_padding_h, output_padding_w),
        dilation=(dilation_d, dilation_h, dilation_w),
        groups=groups,
    ).numpy()

    # Slice to target shape if needed (handles both asymmetric padding and size ambiguity)
    if target_d is not None:
        n, c, d, h, w = result.shape
        slice_d = min(d, target_d)
        slice_h = min(h, target_h)
        slice_w = min(w, target_w)
        result = result[:, :, :slice_d, :slice_h, :slice_w]

    # Handle asymmetric padding - slice to remove extra output from asymmetric part
    extra_d_front = max(0, pad_d_front - pad_d_back)
    extra_d_back = max(0, pad_d_back - pad_d_front)
    extra_top = max(0, pad_top - pad_bottom)
    extra_bottom = max(0, pad_bottom - pad_top)
    extra_left = max(0, pad_left - pad_right)
    extra_right = max(0, pad_right - pad_left)

    if any([extra_d_front, extra_d_back, extra_top, extra_bottom, extra_left, extra_right]):
        n, c, d, h, w = result.shape
        slice_d_start = extra_d_front
        slice_d_end = d - extra_d_back if extra_d_back > 0 else d
        slice_h_start = extra_top
        slice_h_end = h - extra_bottom if extra_bottom > 0 else h
        slice_w_start = extra_left
        slice_w_end = w - extra_right if extra_right > 0 else w
        result = result[:, :, slice_d_start:slice_d_end,
                               slice_h_start:slice_h_end,
                               slice_w_start:slice_w_end]

    # Convert output dtype with overflow handling
    result = convert_output_dtype(result, output_dtype_str, enable_hf32, short_soc_version)

    # Convert output format if needed
    output_format = output_formats[0] if output_formats else "NDC1HWC0"
    if output_format == "NDC1HWC0" and output_ori_shapes is not None:
        result = to_NDC1HWC0_from_NCDHW(result, output_ori_shapes[0])
    elif output_format == "NDHWC":
        result = to_NDHWC_from_NCDHW(result)

    return result