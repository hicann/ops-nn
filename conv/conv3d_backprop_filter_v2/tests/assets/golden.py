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
"""
Golden function for conv3d_backprop_filter_v2 kernel.

Based on conv3d_backprop_filter_v2_def.cpp:
- Inputs: x (REQUIRED), filter_size (REQUIRED), out_backprop (REQUIRED)
- Output: y (REQUIRED)

Format and DataType support based on SoC version:
- Ascend910B/Ascend910_93:
    - x: Format=NDC1HWC0/NCDHW, DataType=BF16/FLOAT/FLOAT16
    - filter_size: Format=ND, DataType=INT32
    - out_backprop: Format=NDC1HWC0, DataType=BF16/FLOAT/FLOAT16
    - y(output): Format=FRACTAL_Z_3D, DataType=FLOAT
- Ascend950:
    - x: Format=NCDHW/NDHWC, DataType=BF16/FLOAT16/FLOAT
    - filter_size: Format=ND, DataType=INT32/INT64
    - out_backprop: Format=NCDHW/NDHWC, DataType=BF16/FLOAT16/FLOAT
    - y(output): Format=NCDHW/NDHWC/DHWCN, DataType=FLOAT

This computes the gradient with respect to the filter weights of a 3D convolution.
"""

import numpy as np
import torch
import torch.nn.functional as F
import math

__golden__ = {
    "kernel": {
        "conv3d_backprop_filter_v2": "conv3d_backprop_filter_v2_golden"
    }
}

NCDHW_FORMAT = "NCDHW"
NDHWC_FORMAT = "NDHWC"
DHWCN_FORMAT = "DHWCN"
FP32_STR = "float32"


def ceil_div(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


def _lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a and b else 0


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
            return out.astype(np.float32)
        out = out.astype(dtype_cls)
    else:
        out = out.astype(dtype_ref)

    if output_dtype == FP32_STR and enable_hf32:
        out = simulate_hf32_precision(out, short_soc_version)

    return out


def parse_pads(pads):
    """Parse padding parameter into 6-element format [d_front, d_back, top, bottom, left, right]."""
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
    """Parse strides or dilations parameter for 3D convolution."""
    if isinstance(param, (list, tuple)):
        if len(param) == 5:
            # N, C, D, H, W format - use D, H, W positions
            return int(param[2]), int(param[3]), int(param[4])
        elif len(param) == 3:
            return int(param[0]), int(param[1]), int(param[2])
        else:
            val = int(param[0])
            return val, val, val
    else:
        val = int(param)
        return val, val, val


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


def process_format_to_ncdhw(data, data_format, ori_shape=None):
    """Convert data from original format to NCDHW for computation."""
    if data_format == NDHWC_FORMAT:
        # NDHWC: (N, D, H, W, C) -> NCDHW: (N, C, D, H, W)
        data = data.transpose(0, 4, 1, 2, 3)
    elif data_format == "NDC1HWC0" and ori_shape is not None:
        data = to_NCDHW_from_NDC1HWC0(data, ori_shape)
    return data


def determine_c0(dtype):
    """Determine C0 value based on dtype."""
    if dtype in ["float16", "bfloat16"]:
        return 16
    elif dtype in ["float32", "float"]:
        return 16  # FP32 also uses C0=16 for FRACTAL_Z_3D
    else:
        return 16


def to_FRACTAL_Z_3D_from_NCDHW(data, filter_ori_shape, groups=1):
    """Convert from NCDHW to FRACTAL_Z_3D format.

    filter_ori_shape: (c_out, c_in_per_group, kd, kh, kw) - weight original shape in NCDHW
    result shape: (real_g * kd * cin1_g * kh * kw, cout1_g, c0, c0) - FRACTAL_Z_3D format
    """
    c_out, c_in_per_group, kd, kh, kw = filter_ori_shape
    c0 = 16
    c_in = c_in_per_group * groups

    cin_ori = c_in_per_group
    cout_ori = c_out // groups

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


def is_ascend950(short_soc_version):
    """Check if the target is Ascend 950PR/950DT"""
    return short_soc_version == "Ascend950"


def process_output_format(data, output_format, input_format, short_soc_version=None,
                           output_ori_shape=None, groups=1):
    """Convert output from NCDHW to target output format."""
    if output_format == "FRACTAL_Z_3D":
        # For Ascend910B series, output weight in FRACTAL_Z_3D format
        if output_ori_shape is not None:
            data = to_FRACTAL_Z_3D_from_NCDHW(data, output_ori_shape, groups)
    elif output_format == NDHWC_FORMAT:
        # NCDHW: (N, C, D, H, W) -> NDHWC: (N, D, H, W, C)
        data = data.transpose(0, 2, 3, 4, 1)
    elif output_format == DHWCN_FORMAT:
        # NCDHW: (N, C, D, H, W) -> DHWCN: (D, H, W, C, N)
        data = data.transpose(1, 2, 3, 4, 0)
    return data


def conv3d_backprop_filter_v2_golden(x, filter_size, out_backprop,
                                      *,
                                      strides: list,
                                      pads: list,
                                      dilations: list = [1, 1, 1, 1, 1],
                                      groups: int = 1,
                                      data_format: str = NDHWC_FORMAT,
                                      enable_hf32: bool = False,
                                      **kwargs):
    """
    Kernel golden for conv3d_backprop_filter_v2.

    This computes the gradient with respect to the filter weights of a 3D convolution.
    Formula: dL/dw = correlation(x, out_backprop) using appropriate strides and dilations.

    Args:
        x: Input feature map tensor
        filter_size: Shape of filter tensor as const input (C_out, C_in/groups, kD, kH, kW)
        out_backprop: Gradient from next layer

    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_dtypes, output_dtypes.
    """
    input_formats = kwargs.get("input_formats", [NDHWC_FORMAT, "ND", NDHWC_FORMAT])
    output_formats = kwargs.get("output_formats", [NCDHW_FORMAT])
    short_soc_version = kwargs.get("short_soc_version", "")
    input_dtypes = kwargs.get("input_dtypes", ["float16", "int32", "float16"])
    output_dtypes = kwargs.get("output_dtypes", ["float32"])
    output_ori_shapes = kwargs.get("output_ori_shapes", None)
    input_ori_shapes = kwargs.get("input_ori_shapes", None)

    x_dtype_str = input_dtypes[0] if len(input_dtypes) > 0 else "float16"
    output_dtype_str = output_dtypes[0] if len(output_dtypes) > 0 else "float32"

    x_format = input_formats[0] if len(input_formats) > 0 else NDHWC_FORMAT
    out_backprop_format = input_formats[2] if len(input_formats) > 2 else NDHWC_FORMAT
    output_format = output_formats[0] if len(output_formats) > 0 else NCDHW_FORMAT

    # Get ori_shapes for format conversion
    x_ori_shape = input_ori_shapes[0] if input_ori_shapes and len(input_ori_shapes) > 0 else None
    out_backprop_ori_shape = input_ori_shapes[2] if input_ori_shapes and len(input_ori_shapes) > 2 else None

    # Convert formats to NCDHW for computation
    x_ncdhw = process_format_to_ncdhw(x, x_format, x_ori_shape)
    out_backprop_ncdhw = process_format_to_ncdhw(out_backprop, out_backprop_format, out_backprop_ori_shape)

    # Parse filter_size to get filter shape
    # Try to get from attributes first (more reliable), fallback to input
    filter_size_attr = kwargs.get("filter_size", None)
    if filter_size_attr is not None:
        filter_shape = tuple(filter_size_attr)
    elif isinstance(filter_size, np.ndarray):
        filter_shape = tuple(filter_size.astype(int).flatten())
        # If filter_shape contains invalid values (zeros), try to infer from output_ori_shapes
        if any(s <= 0 for s in filter_shape):
            if output_ori_shapes and len(output_ori_shapes) > 0:
                filter_shape = tuple(output_ori_shapes[0])
    else:
        filter_shape = tuple(filter_size)
        if any(s <= 0 for s in filter_shape):
            if output_ori_shapes and len(output_ori_shapes) > 0:
                filter_shape = tuple(output_ori_shapes[0])

    # filter_size should be 5-element: (C_out, C_in/groups, kD, kH, kW) for NCDHW
    # or (C_out, kD, kH, kW, C_in/groups) for DHWCN
    c_out = filter_shape[0]
    if len(filter_shape) == 5:
        c_in_per_group = filter_shape[1]
        kd = filter_shape[2]
        kh = filter_shape[3]
        kw = filter_shape[4]
    else:
        raise ValueError(f"filter_size must be 5-element tuple/list, got {filter_shape}")

    # Parse convolution parameters
    stride_d, stride_h, stride_w = parse_strides_dilations(strides)
    dilation_d, dilation_h, dilation_w = parse_strides_dilations(dilations)
    pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right = parse_pads(pads)

    # Use float64 for better precision in gradient computation
    calc_dtype = np.float64
    x_calc = x_ncdhw.astype(calc_dtype)
    out_backprop_calc = out_backprop_ncdhw.astype(calc_dtype)

    if enable_hf32 and x_dtype_str == FP32_STR:
        x_calc = simulate_hf32_precision(x_calc.astype(np.float32), short_soc_version)
        out_backprop_calc = simulate_hf32_precision(out_backprop_calc.astype(np.float32), short_soc_version)

    # Convert to PyTorch tensors
    x_torch = torch.from_numpy(x_calc).float()
    grad_output_torch = torch.from_numpy(out_backprop_calc).float()

    # Handle asymmetric padding - use symmetric padding for torch conv
    sym_pad_d = min(pad_d_front, pad_d_back)
    sym_pad_h = min(pad_top, pad_bottom)
    sym_pad_w = min(pad_left, pad_right)

    # Apply extra asymmetric padding to input
    extra_pad_d_front = max(0, pad_d_front - pad_d_back)
    extra_pad_d_back = max(0, pad_d_back - pad_d_front)
    extra_pad_top = max(0, pad_top - pad_bottom)
    extra_pad_bottom = max(0, pad_bottom - pad_top)
    extra_pad_left = max(0, pad_left - pad_right)
    extra_pad_right = max(0, pad_right - pad_left)

    if any([extra_pad_d_front, extra_pad_d_back, extra_pad_top, extra_pad_bottom,
            extra_pad_left, extra_pad_right]):
        x_padded = np.pad(
            x_torch.numpy(),
            (
                (0, 0),
                (0, 0),
                (extra_pad_d_front, extra_pad_d_back),
                (extra_pad_top, extra_pad_bottom),
                (extra_pad_left, extra_pad_right),
            ),
            "constant",
            constant_values=0,
        )
        x_torch = torch.from_numpy(x_padded).float()

    # Use PyTorch's convolution_backward to compute grad_weight
    # torch.ops.aten.convolution_backward returns (grad_input, grad_weight, grad_bias)
    # output_mask = [False, True, False] means we only want grad_weight
    weight_shape = (c_out, c_in_per_group, kd, kh, kw)

    grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
        grad_output_torch,
        x_torch,
        torch.zeros(weight_shape, dtype=x_torch.dtype),  # dummy weight for shape inference
        None,  # bias_sizes
        [stride_d, stride_h, stride_w],
        [sym_pad_d, sym_pad_h, sym_pad_w],
        [dilation_d, dilation_h, dilation_w],
        False,  # transposed
        [0, 0, 0],  # output_padding
        groups,
        [False, True, False]  # output_mask: don't need grad_input, need grad_weight, don't need grad_bias
    )

    grad_weight_result = grad_weight.numpy()

    # Convert output dtype
    grad_weight_result = convert_output_dtype(grad_weight_result, output_dtype_str,
                                               enable_hf32, short_soc_version)

    # Convert to output format (output is filter weight)
    # Output format for weight gradient: NCDHW, NDHWC, DHWCN, or FRACTAL_Z_3D
    # grad_weight is always in NCDHW format (C_out, C_in/groups, kD, kH, kW) from PyTorch
    output_ori_shape = output_ori_shapes[0] if output_ori_shapes and len(output_ori_shapes) > 0 else None
    grad_weight_result = process_output_format(
        grad_weight_result, output_format, x_format,
        short_soc_version, output_ori_shape, groups
    )

    return grad_weight_result