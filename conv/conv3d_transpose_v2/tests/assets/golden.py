#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

"""
Golden function for conv3d_transpose_v2 kernel.

Based on conv3d_transpose_v2_def.cpp:
- Inputs: input_size (REQUIRED), x (REQUIRED), filter (REQUIRED), bias (OPTIONAL), offset_w (OPTIONAL)
- Output: y (REQUIRED)

Format and DataType support based on SoC version:
- Ascend910B/Ascend910_93:
    - input_size: Format=ND, DataType=INT32/INT64
    - x: Format=NDC1HWC0, DataType=FLOAT16/BF16/FLOAT
    - filter: Format=FRACTAL_Z_3D, DataType=FLOAT16/BF16/FLOAT
    - bias: Format=ND, DataType=FLOAT16/FLOAT
    - offset_w: Format=ND, DataType=INT8
    - y(output): Format=NDC1HWC0, DataType=FLOAT16/BF16/FLOAT
- Ascend950:
    - input_size: Format=ND, DataType=INT32/INT64
    - x: Format=NCDHW/NDHWC, DataType=多种
    - filter: Format=NCDHW/NDHWC/DHWCN, DataType=多种
    - y(output): Format=NCDHW/NDHWC, DataType=多种

This computes the transposed 3D convolution (conv3d_backward).
"""

__golden__ = {
    "kernel": {
        "conv3d_transpose_v2": "conv3d_transpose_v2_golden"
    }
}

NCDHW_FORMAT = "NCDHW"
NDHWC_FORMAT = "NDHWC"
FP32_STR = "float32"


def due_fp16_overflow(data):
    """Overflow interception for float16
    Clips values to the finite range of float16: [-65504, 65504]
    """
    FP16_MAX_FINITE = 65504
    FP16_MIN_FINITE = -65504
    data = np.maximum(data, FP16_MIN_FINITE)
    data = np.minimum(data, FP16_MAX_FINITE)
    return data


def simulate_hf32_precision(data, short_soc_version=None):
    """
    Simulate HF32 (Half Float 32) precision.
    Ascend910B: truncates lower 12 bits of float32 mantissa, keeping 20 bits with rounding.
    Default: truncates lower 13 bits of float32 mantissa, keeping 19 bits with rounding.
    """
    if data.dtype == np.dtype(np.float32):
        input_hf32 = data.view(np.int32)
        if short_soc_version in ("Ascend910B",):
            input_hf32 = np.right_shift(np.right_shift(input_hf32, 11) + 1, 1)
            input_hf32 = np.left_shift(input_hf32, 12)
        else:
            input_hf32 = np.right_shift(np.right_shift(input_hf32, 12) + 1, 1)
            input_hf32 = np.left_shift(input_hf32, 13)
        return input_hf32.view(np.float32)
    return data


def convert_output_dtype(out, output_dtype, enable_hf32=False, short_soc_version=None):
    dtype_map = {
        "float16": (np.float16, True),
        "float32": (np.float32, False),
        "bfloat16": ("ml_dtypes.bfloat16", True),
        "hifloat8": ("en_dtypes.hifloat8", False),
        "float8_e4m3fn": ("ml_dtypes.float8_e4m3fn", False),
        "int8": (np.int8, False),
        "int32": (np.int32, False),
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
            raise RuntimeError(f"{module_name} is required for {output_dtype}. "
                               f"Install: pip install {module_name}")
        out = out.astype(dtype_cls)
    else:
        out = out.astype(dtype_ref)

    if output_dtype == FP32_STR and enable_hf32:
        out = simulate_hf32_precision(out, short_soc_version)

    return out


def ceil_div(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


def _lcm(a, b):
    import math
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def determine_c0(dtype):
    if dtype in ["float16", "bfloat16"]:
        return 16
    elif dtype in [
        "float8_e4m3fn",
        "float8_e5m2",
        "float4_e2m1",
        "float4_e1m2",
        "hifloat8",
    ]:
        return 32
    else:
        return 16


def is_ascend950(short_soc_version):
    """Check if the target is Ascend 950PR/950DT"""
    return short_soc_version == "Ascend950"


def to_NCDHW_from_NDC1HWC0(data, ori_shape):
    """Convert from NDC1HWC0 to NCDHW format"""
    # ori_shape is in NCDHW format: (n, c, d, h, w)
    n, c, d, h, w = ori_shape
    c0 = determine_c0(data.dtype.name)
    c1 = ceil_div(c, c0)

    # NDC1HWC0 shape: (n, d, c1, h, w, c0)
    # We want to convert to (n, c1, c0, d, h, w) first, then (n, c1*c0, d, h, w)
    # transpose (0, 2, 5, 1, 3, 4): (n, d, c1, h, w, c0) -> (n, c1, c0, d, h, w)
    data = data.transpose(0, 2, 5, 1, 3, 4)
    # Reshape to (n, c1*c0, d, h, w)
    data = data.reshape((n, c1 * c0, d, h, w))
    # Slice to (n, c, d, h, w) if there's padding
    if c1 * c0 > c:
        data = data[:, :c, :, :, :]

    return data


def to_NCDHW_from_FRACTAL_Z_3D(data, ori_shape, groups=1):
    """Convert from FRACTAL_Z_3D to NCDHW format"""
    n, c_in, d, h, w = ori_shape
    c0 = 16
    cin_ori = c_in // groups
    cout_ori = n // groups

    mag_factor0 = _lcm(cin_ori, c0) // cin_ori
    mag_factor1 = _lcm(cout_ori, c0) // cout_ori
    mag_factor = min(_lcm(mag_factor0, mag_factor1), groups)

    cin_g = align(mag_factor * cin_ori, c0)
    cout_g = align(mag_factor * cout_ori, c0)
    real_g = ceil_div(groups, mag_factor)
    cin1_g = cin_g // c0
    cout1_g = cout_g // c0

    # FRACTAL_Z_3D shape: (real_g * d * cin1_g * h * w, cout1_g, c0, c0)
    data = data.reshape((real_g, d, cin1_g, h, w, cout_g, c0))

# Reverse the mapping
    c_in_per_group = c_in // groups
    result = np.zeros((n, c_in_per_group, d, h, w), dtype=data.dtype)
    for g in range(groups):
        for ci in range(c_in_per_group):
            for co in range(cout_ori):
                e = g % mag_factor
                dst_cin = e * cin_ori + ci
                dst_cout = e * cout_ori + co
                src_cout = g * cout_ori + co
                result[src_cout, ci, :, :, :] = \
                    data[g // mag_factor, :, dst_cin // c0, :, :, dst_cout, dst_cin % c0]

    return result


def to_NDC1HWC0_from_NCDHW(data, ori_shape):
    """Convert from NCDHW to NDC1HWC0 format"""
    n, c, d, h, w = ori_shape
    c0 = determine_c0(data.dtype.name)
    c1 = ceil_div(c, c0)

    # NCDHW -> (n, c1, c0, d, h, w) -> (n, d, c1, h, w, c0)
    if c1 * c0 > c:
        num_2_padding_in_c = c1 * c0 - c
        zero_padding_array = np.zeros((n, num_2_padding_in_c, d, h, w), dtype=data.dtype)
        data = np.concatenate((data, zero_padding_array), axis=1)

    data = data.reshape((n, c1, c0, d, h, w))
    data = data.transpose(0, 3, 1, 4, 5, 2)

    return data


def to_FRACTAL_Z_3D_from_NCDHW(data, ori_shape, groups=1):
    """Convert from NCDHW to FRACTAL_Z_3D format"""
    n, c_in, d, h, w = ori_shape
    c0 = 16
    cin_ori = c_in // groups
    cout_ori = n // groups

    mag_factor0 = _lcm(cin_ori, c0) // cin_ori
    mag_factor1 = _lcm(cout_ori, c0) // cout_ori
    mag_factor = min(_lcm(mag_factor0, mag_factor1), groups)

    cin_g = align(mag_factor * cin_ori, c0)
    cout_g = align(mag_factor * cout_ori, c0)
    real_g = ceil_div(groups, mag_factor)
    cin1_g = cin_g // c0
    cout1_g = cout_g // c0

    weight_group = np.zeros((real_g, d, cin1_g, h, w, cout_g, c0), dtype=data.dtype)

    c_in_per_group = c_in // groups
    for g in range(groups):
        for ci in range(c_in_per_group):
            for co in range(cout_ori):
                e = g % mag_factor
                dst_cin = e * cin_ori + ci
                dst_cout = e * cout_ori + co
                src_cout = g * cout_ori + co
                weight_group[g // mag_factor, :, dst_cin // c0, :, :, dst_cout, dst_cin % c0] = \
                    data[src_cout, ci, :, :, :]

    weight_group = weight_group.reshape((real_g * d * cin1_g * h * w, cout1_g, c0, c0))

    return weight_group


def process_formats_a2_a3(x, filter, input_formats, input_ori_shapes, groups):
    """
    Process format conversion for A2/A3 series (ascend910b, ascend910_93).

    Supported formats:
    - x: NDC1HWC0
    - filter: FRACTAL_Z_3D

    Note: input_formats follows the operator's input order (input_size, x, filter, ...),
    so x format is at index 1 and filter format is at index 2.
    """
    input_data_format = input_formats[1] if len(input_formats) > 1 else NCDHW_FORMAT
    input_filter_format = input_formats[2] if len(input_formats) > 2 else NCDHW_FORMAT

    if input_data_format == "NDC1HWC0":
        if input_ori_shapes is not None and len(input_ori_shapes) > 1:
            x = to_NCDHW_from_NDC1HWC0(x, input_ori_shapes[1])

    if input_filter_format == "FRACTAL_Z_3D":
        if input_ori_shapes is not None and len(input_ori_shapes) > 2:
            filter = to_NCDHW_from_FRACTAL_Z_3D(filter, input_ori_shapes[2], groups)

    return x, filter


def process_formats_a5(x, filter, input_formats, input_ori_shapes=None, groups=1):
    """
    Process format conversion for Ascend 950PR/950DT (A5).

    Supported formats:
    - x: NCDHW, NDHWC, NDC1HWC0 (6D physical)
    - filter: NCDHW, NDHWC, DHWCN, FRACTAL_Z_3D (4D physical)

    Note: input_formats follows the operator's input order (input_size, x, filter, ...),
    so x format is at index 1 and filter format is at index 2.
    """
    input_data_format = input_formats[1] if len(input_formats) > 1 else NCDHW_FORMAT
    input_filter_format = input_formats[2] if len(input_formats) > 2 else NCDHW_FORMAT

    if input_data_format == NDHWC_FORMAT:
        # NDHWC -> NCDHW: (N, D, H, W, C) -> (N, C, D, H, W)
        x = x.transpose(0, 4, 1, 2, 3)
    elif input_data_format == NCDHW_FORMAT and x.ndim == 6 and input_ori_shapes and len(input_ori_shapes) > 1:
        # NDC1HWC0 (6D physical) -> NCDHW (5D logical)
        x = to_NCDHW_from_NDC1HWC0(x, input_ori_shapes[1])

    if input_filter_format == "DHWCN":
        # DHWCN -> NCDHW: (D, H, W, C, N) -> (N, C, D, H, W)
        filter = filter.transpose(4, 3, 0, 1, 2)
    elif input_filter_format == NDHWC_FORMAT:
        # NDHWC -> NCDHW: (N, D, H, W, C) -> (N, C, D, H, W)
        filter = filter.transpose(0, 4, 1, 2, 3)
    elif input_filter_format == NCDHW_FORMAT and filter.ndim == 4 and input_ori_shapes and len(input_ori_shapes) > 2:
        # FRACTAL_Z_3D (4D physical) -> NCDHW (5D logical)
        filter = to_NCDHW_from_FRACTAL_Z_3D(filter, input_ori_shapes[2], groups)

    return x, filter


def process_output_format_a2_a3(out, output_format, output_ori_shapes):
    """
    Process output format conversion for A2/A3 series.

    Supported formats:
    - y: NDC1HWC0
    """
    if output_format == "NDC1HWC0":
        if output_ori_shapes is not None:
            target_shape = output_ori_shapes[0]
            out = to_NDC1HWC0_from_NCDHW(out, target_shape)

    return out


def process_output_format_a5(out, output_format, output_ori_shapes=None, input_was_6d=False):
    """
    Process output format conversion for Ascend 950PR/950DT (A5).

    PyTorch conv_transpose3d always outputs NCDHW; convert to the kernel's
    physical output_format. Conversion direction depends only on output_format.
    Supported:
    - y: NCDHW, NDHWC, NDC1HWC0 (6D physical)
    """
    if output_format == NDHWC_FORMAT:
        # NCDHW -> NDHWC: (N, C, D, H, W) -> (N, D, H, W, C)
        out = out.transpose((0, 2, 3, 4, 1))
    elif output_format == NCDHW_FORMAT and input_was_6d and output_ori_shapes and len(output_ori_shapes) > 0:
        # NCDHW (5D logical) -> NDC1HWC0 (6D physical) to match kernel output
        out = to_NDC1HWC0_from_NCDHW(out, output_ori_shapes[0])

    return out


def parse_pads(pads):
    """
    Parse padding parameter into tuple format.

    Args:
        pads: padding value, can be int, or list/tuple with 1, 3, or 6 elements.

    Returns:
        tuple: (pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right)
    """
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


def parse_output_padding(output_padding):
    """
    Parse output_padding parameter.

    Args:
        output_padding: can be int, or list/tuple with 1, 3, or 5 elements.

    Returns:
        tuple: (output_padding_d, output_padding_h, output_padding_w)
    """
    if isinstance(output_padding, (list, tuple)):
        if len(output_padding) == 5:
            return int(output_padding[2]), int(output_padding[3]), int(output_padding[4])
        elif len(output_padding) == 3:
            return int(output_padding[0]), int(output_padding[1]), int(output_padding[2])
        else:
            val = int(output_padding[0])
            return val, val, val
    else:
        val = int(output_padding)
        return val, val, val


def conv3d_transpose_v2_golden(input_size, x, filter, bias=None, offset_w=None,
                               *,
                               strides: list,
                               pads: list,
                               dilations: list = [1, 1, 1, 1, 1],
                               groups: int = 1,
                               data_format: str = NDHWC_FORMAT,
                               output_padding: list = [0, 0, 0, 0, 0],
                               offset_x: int = 0,
                               enable_hf32: bool = False,
                               **kwargs,
                               ):
    """
    Kernel golden for conv3d_transpose_v2.
    All parameters follow @conv3d_transpose_v2_def.cpp without outputs.

    Format and DataType support based on SoC version:
    - ascend910b/ascend910_93: x/filter/y use NDC1HWC0/FRACTAL_Z_3D, dtype: float16/bfloat16/float
    - ascend950/ascend910_55: x/filter/y use NCDHW/NDHWC/DHWCN, dtype: float16/bfloat16/float/hifloat8/float8_e4m3fn

    All input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch
    import torch.nn.functional as F

    input_formats = kwargs.get("input_formats", [NCDHW_FORMAT, NCDHW_FORMAT])
    short_soc_version = kwargs.get("short_soc_version", "")
    input_ori_shapes = kwargs.get("input_ori_shapes", None)
    is_950 = is_ascend950(short_soc_version)

    x_dtype_str = x.dtype.name
    x_was_6d = x.ndim == 6

    # Process format conversion based on SoC version
    if is_950:
        # ascend950/ascend910_55: support NCDHW, NDHWC, DHWCN, NDC1HWC0, FRACTAL_Z_3D
        x_np, filter_np = process_formats_a5(x, filter, input_formats, input_ori_shapes, groups)
    else:
        # ascend910b/ascend910_93: support NDC1HWC0, FRACTAL_Z_3D
        x_np, filter_np = process_formats_a2_a3(x, filter, input_formats, input_ori_shapes, groups)

    # Handle dtype and precision
    if enable_hf32 and x_dtype_str == FP32_STR:
        calc_dtype = np.float32
        x_np = simulate_hf32_precision(x_np.astype(np.float32), short_soc_version)
        filter_np = simulate_hf32_precision(filter_np.astype(np.float32), short_soc_version)
    else:
        calc_dtype = np.float64 if x_dtype_str == FP32_STR else np.float32
        x_np = x_np.astype(calc_dtype)
        filter_np = filter_np.astype(calc_dtype)

    if bias is not None:
        bias_np = bias.astype(calc_dtype)
    else:
        bias_np = None

    # Parse strides
    if isinstance(strides, (list, tuple)):
        if len(strides) == 5:
            stride_d, stride_h, stride_w = int(strides[2]), int(strides[3]), int(strides[4])
        elif len(strides) == 3:
            stride_d, stride_h, stride_w = int(strides[0]), int(strides[1]), int(strides[2])
        else:
            stride_d = stride_h = stride_w = int(strides[0])
    else:
        stride_d = stride_h = stride_w = int(strides)

    # Parse dilations
    if isinstance(dilations, (list, tuple)):
        if len(dilations) == 5:
            dilation_d, dilation_h, dilation_w = int(dilations[2]), int(dilations[3]), int(dilations[4])
        elif len(dilations) == 3:
            dilation_d, dilation_h, dilation_w = int(dilations[0]), int(dilations[1]), int(dilations[2])
        else:
            dilation_d = dilation_h = dilation_w = int(dilations[0])
    else:
        dilation_d = dilation_h = dilation_w = int(dilations)

    # Parse pads
    pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right = parse_pads(pads)

    # Parse output_padding
    out_pad_d, out_pad_h, out_pad_w = parse_output_padding(output_padding)

    # Prepare tensors for conv_transpose3d
    input_torch = torch.from_numpy(x_np)
    
    # NOTE: No weight transpose needed!
    # PyTorch conv_transpose3d weight format: (in_channels, out_channels/groups, kD, kH, kW)
    # CANN filter format:                      (C_out,       C_in/groups,         kD, kH, kW)
    # Since in_channels=C_out and out_channels=C_in, the CANN filter is ALREADY in
    # the correct format for PyTorch's conv_transpose3d. No transpose required.
    weight_torch = torch.from_numpy(filter_np)
    bias_torch = torch.from_numpy(bias_np) if bias_np is not None else None

    # conv_transpose3d uses symmetric padding (pad_d, pad_h, pad_w)
    # For asymmetric padding, we need to handle it separately
    sym_pad_d = min(pad_d_front, pad_d_back)
    sym_pad_h = min(pad_top, pad_bottom)
    sym_pad_w = min(pad_left, pad_right)

    torch_padding = (sym_pad_d, sym_pad_h, sym_pad_w)
    torch_output_padding = (out_pad_d, out_pad_h, out_pad_w)

    # Call conv_transpose3d
    out = torch.nn.functional.conv_transpose3d(
        input_torch,
        weight_torch,
        bias=bias_torch,
        stride=(stride_d, stride_h, stride_w),
        padding=torch_padding,
        output_padding=torch_output_padding,
        dilation=(dilation_d, dilation_h, dilation_w),
        groups=groups,
    ).numpy()

    # Handle asymmetric padding - remove extra padding from output
    extra_pad_d_front = max(0, pad_d_front - pad_d_back)
    extra_pad_d_back = max(0, pad_d_back - pad_d_front)
    extra_pad_top = max(0, pad_top - pad_bottom)
    extra_pad_bottom = max(0, pad_bottom - pad_top)
    extra_pad_left = max(0, pad_left - pad_right)
    extra_pad_right = max(0, pad_right - pad_left)

    if any([extra_pad_d_front, extra_pad_d_back, extra_pad_top, extra_pad_bottom, extra_pad_left, extra_pad_right]):
        # out shape is (N, C, D, H, W)
        n, c, d, h, w = out.shape
        slice_d_start = extra_pad_d_front
        slice_d_end = d - extra_pad_d_back
        slice_h_start = extra_pad_top
        slice_h_end = h - extra_pad_bottom
        slice_w_start = extra_pad_left
        slice_w_end = w - extra_pad_right
        out = out[:, :, slice_d_start:slice_d_end, slice_h_start:slice_h_end, slice_w_start:slice_w_end]

    # Convert output dtype
    output_dtypes = kwargs.get("output_dtypes", [FP32_STR])
    output_dtype = output_dtypes[0]
    output_formats = kwargs.get("output_formats", [NCDHW_FORMAT])
    output_format = output_formats[0]
    output_ori_shapes = kwargs.get("output_ori_shapes", None)

    out = convert_output_dtype(out, output_dtype, enable_hf32, short_soc_version)

    # Process output format conversion based on SoC version
    if is_950:
        out = process_output_format_a5(out, output_format, output_ori_shapes, x_was_6d)
    else:
        out = process_output_format_a2_a3(out, output_format, output_ori_shapes)

    return out