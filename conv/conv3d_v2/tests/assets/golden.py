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

# Control whether to use TensorFlow golden (default: False)
# Set to True to use TensorFlow implementation (same as opstest)
# Set to False to use PyTorch implementation

__golden__ = {
    "kernel": {
        "conv3dv2": "conv3d_v2_golden",
        "conv3d_v2": "conv3d_v2_golden"
    }
}


NCDHW_FORMAT = "NCDHW"
NDHWC_FORMAT = "NDHWC"
FP32_STR = "float32"


def due_fp16_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def simulate_hf32_precision(data):
    """
    Simulate HF32 (Half Float 32) precision.
    HF32 truncates lower 12 bits of float32 mantissa (keeping 11 bits with rounding).
    """
    if data.dtype == np.dtype(np.float32):
        input_hf32 = data.view(np.int32)
        input_hf32 = np.right_shift(np.right_shift(input_hf32, 11) + 1, 1)
        input_hf32 = np.left_shift(input_hf32, 12)
        return input_hf32.view(np.float32)
    return data


def convert_output_dtype(out, output_dtype, enable_hf32=False):
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
        out = simulate_hf32_precision(out)

    return out


def ceil_div(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


def _lcm(a, b):
    import math
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def determine_c0(dtype, target_shape):
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


def to_NDC1HWC0(data, ori_format, target_shape):
    ori_shape = data.shape
    if len(ori_shape) != 5:
        raise RuntimeError("to_NDC1HWC0: ori_format must be 5D format")

    n_index, d_index, h_index, w_index, c_index = (
        ori_format.index("N"),
        ori_format.index("D"),
        ori_format.index("H"),
        ori_format.index("W"),
        ori_format.index("C"),
    )
    n, d, h, w, c = (
        ori_shape[n_index],
        ori_shape[d_index],
        ori_shape[h_index],
        ori_shape[w_index],
        ori_shape[c_index],
    )

    c0 = determine_c0(data.dtype.name, target_shape)
    c1 = ceil_div(c, c0)

    data = data.reshape((n, d, h, w, c1, c0))
    data = data.transpose(n_index, d_index, h_index, w_index, c1, c0)

    if c1 * c0 > c:
        num_2_padding_in_c = c1 * c0 - c
        zero_padding_array = np.zeros(
            (n, d, h, w, num_2_padding_in_c), dtype=data.dtype
        )
        data = np.concatenate((data, zero_padding_array), axis=1)

    return data


def is_ascend950(short_soc_version):
    """Check if the target is Ascend 950PR/950DT"""
    return (short_soc_version == "Ascend950")


def process_formats_a2_a3(x, filter, input_formats, input_ori_shapes, groups):
    """
    Process format conversion for A2/A3 series.

    Constraints:
    - x supports: NCDHW, NDC1HWC0
    - filter supports: NCDHW, FRACTAL_Z_3D
    - DHWCN format is not supported
    """
    input_data_format, input_filter_format = input_formats[0], input_formats[1]
    
    if input_data_format == "NDC1HWC0":
        if input_ori_shapes is not None:
            x = to_NCDHW_from_NDC1HWC0(x, input_ori_shapes[0])
    
    if input_filter_format == "FRACTAL_Z_3D":
        if input_ori_shapes is not None:
            filter = to_NCDHW_from_FRACTAL_Z_3D(filter, input_ori_shapes[1], groups)
    
    return x, filter


def process_formats_a5(x, filter, input_formats):
    """
    Process format conversion for Ascend 950PR/950DT (A5).

    Constraints:
    - x supports: NCDHW, NDHWC
    - filter supports: NCDHW, DHWCN
    """
    input_data_format, input_filter_format = input_formats[0], input_formats[1]
    
    if input_data_format == "NDHWC":
        x = x.transpose(0, 4, 1, 2, 3)
    
    if input_filter_format == "DHWCN":
        filter = filter.transpose(4, 3, 0, 1, 2)
    
    return x, filter


def process_output_format_a2_a3(out, output_format, input_format, output_ori_shapes):
    """
    Process output format conversion for A2/A3 series.

    Constraints:
    - y supports: NCDHW, NDC1HWC0
    """
    if output_format == "NDC1HWC0":
        if output_ori_shapes is not None:
            target_shape = output_ori_shapes[0]
            out = to_NDC1HWC0_from_NCDHW(out, target_shape)
    elif output_format == NCDHW_FORMAT and input_format == NDHWC_FORMAT:
        out = out.transpose((0, 4, 1, 2, 3))
    
    return out


def process_output_format_a5(out, output_format, input_format):
    """
    Process output format conversion for Ascend 950PR/950DT (A5).

    Constraints:
    - y supports: NCDHW, NDHWC, DHWCN
    """
    if output_format == NDHWC_FORMAT and input_format == NCDHW_FORMAT:
        out = out.transpose((0, 2, 3, 4, 1))
    elif output_format == NCDHW_FORMAT and input_format == NDHWC_FORMAT:
        out = out.transpose((0, 4, 1, 2, 3))
    
    return out


def to_NCDHW_from_NDC1HWC0(data, ori_shape):
    """Convert from NDC1HWC0 to NCDHW format"""
    n, d, h, w, c = ori_shape
    c0 = determine_c0(data.dtype.name, None)
    c1 = ceil_div(c, c0)
    
    # NDC1HWC0 shape: (n, d, c1, h, w, c0)
    # Transpose to (n, c1, d, h, w, c0)
    data = data.transpose(0, 2, 1, 3, 4, 5)
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
    result = np.zeros((n, c_in, d, h, w), dtype=data.dtype)
    for g in range(groups):
        for ci in range(c_in):
            for co in range(n // groups):
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
    c0 = determine_c0(data.dtype.name, None)
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
    
    for g in range(groups):
        for ci in range(c_in):
            for co in range(n // groups):
                e = g % mag_factor
                dst_cin = e * cin_ori + ci
                dst_cout = e * cout_ori + co
                src_cout = g * cout_ori + co
                weight_group[g // mag_factor, :, dst_cin // c0, :, :, dst_cout, dst_cin % c0] = \
                    data[src_cout, ci, :, :, :]
    
    weight_group = weight_group.reshape((real_g * d * cin1_g * h * w, cout1_g, c0, c0))
    
    return weight_group


def _ceil_div(a, b):
    """Ceiling division: returns ceil(a / b)."""
    return -(-a // b)


def _parse_padding(pads):
    """
    Parse padding parameter into 6-element format [pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right].

    Args:
        pads: padding value, can be int, or list/tuple with 1, 3, or 6 elements.

    Returns:
        tuple: (pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right)
    """
    if isinstance(pads, (list, tuple)):
        if len(pads) == 6:
            # Already in 6-element format: [d_front, d_back, top, bottom, left, right]
            return pads[0], pads[1], pads[2], pads[3], pads[4], pads[5]
        elif len(pads) == 3:
            # 3-element format: symmetric padding for [d, h, w]
            return pads[0], pads[0], pads[1], pads[1], pads[2], pads[2]
        else:
            # Single or other element count: use first value for all dimensions
            val = pads[0]
            return val, val, val, val, val, val
    else:
        # Scalar value: use for all dimensions
        val = int(pads)
        return val, val, val, val, val, val


def _apply_pad_mode(pad_mode, input_shape, filter_shape, stride_d, stride_h, stride_w,
                    dilation_d, dilation_h, dilation_w):
    """
    Calculate padding based on pad_mode, aligned with C++ GetOriPadFromPadMode and ApplySamesPad logic.

    Supported modes:
        - VALID: no padding, all zeros
        - SAME: same output size, extra padding distributed equally (extra goes to tail/bottom/right)
        - SAME_UPPER: same output size, extra padding goes to tail/bottom/right
        - SAME_LOWER: same output size, extra padding goes to head/top/left

    Args:
        pad_mode: padding mode string
        input_shape: input tensor shape (N, C, D, H, W)
        filter_shape: filter tensor shape (C_out, C_in, kD, kH, kW)
        stride_d/h/w: stride values for each spatial dimension
        dilation_d/h/w: dilation values for each spatial dimension

    Returns:
        tuple: (pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right)
    """
    mode = pad_mode.upper()

    if mode == "VALID":
        # VALID mode: no padding
        return 0, 0, 0, 0, 0, 0

    if mode == "SPECIFIC":
        # SPECIFIC mode: padding provided explicitly, return zeros here
        return 0, 0, 0, 0, 0, 0

    # Get input spatial dimensions
    di, hi, wi = input_shape[2], input_shape[3], input_shape[4]
    # Get kernel spatial dimensions
    kd, kh, kw = filter_shape[2], filter_shape[3], filter_shape[4]

    # Calculate total padding needed for each dimension
    # Formula: pad = (ceil(in / stride) - 1) * stride + dilation * (kernel - 1) - in + 1
    pad_d = (_ceil_div(di, stride_d) - 1) * stride_d + dilation_d * (kd - 1) - di + 1
    pad_h = (_ceil_div(hi, stride_h) - 1) * stride_h + dilation_h * (kh - 1) - hi + 1
    pad_w = (_ceil_div(wi, stride_w) - 1) * stride_w + dilation_w * (kw - 1) - wi + 1

    if mode == "SAME":
        # SAME mode: clamp negative padding to 0, extra goes to tail/bottom/right
        pad_d = max(0, pad_d)
        pad_h = max(0, pad_h)
        pad_w = max(0, pad_w)
        pad_d_back = _ceil_div(pad_d, 2)
        pad_d_front = pad_d - pad_d_back
        pad_bottom = _ceil_div(pad_h, 2)
        pad_top = pad_h - pad_bottom
        pad_right = _ceil_div(pad_w, 2)
        pad_left = pad_w - pad_right
    elif mode == "SAME_UPPER":
        # SAME_UPPER: extra padding goes to tail/bottom/right
        pad_d_back = _ceil_div(pad_d, 2)
        pad_d_front = pad_d - pad_d_back
        pad_bottom = _ceil_div(pad_h, 2)
        pad_top = pad_h - pad_bottom
        pad_right = _ceil_div(pad_w, 2)
        pad_left = pad_w - pad_right
    elif mode == "SAME_LOWER":
        # SAME_LOWER: extra padding goes to head/top/left
        pad_d_front = _ceil_div(pad_d, 2)
        pad_d_back = pad_d - pad_d_front
        pad_top = _ceil_div(pad_h, 2)
        pad_bottom = pad_h - pad_top
        pad_left = _ceil_div(pad_w, 2)
        pad_right = pad_w - pad_left
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    return pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right


def _process_conv3d_padding(x_np, pads, pad_mode, filter_shape, stride_d, stride_h, stride_w,
                            dilation_d, dilation_h, dilation_w):
    """
    Process padding for conv3d operation.

    This function handles both explicit padding (SPECIFIC mode) and automatic padding
    modes (VALID, SAME, SAME_UPPER, SAME_LOWER). It applies asymmetric padding by
    splitting it into symmetric padding (for torch.conv3d) and extra padding (applied to input).

    Args:
        x_np: input numpy array (N, C, D, H, W)
        pads: explicit padding values
        pad_mode: padding mode string
        filter_shape: filter tensor shape
        stride_d/h/w: stride values
        dilation_d/h/w: dilation values

    Returns:
        tuple: (padded_input_np, torch_pad_list)
            - padded_input_np: input with extra asymmetric padding applied
            - torch_pad_list: symmetric padding [d, h, w] for torch.conv3d
    """
    # Parse explicit padding into 6-element format
    pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right = _parse_padding(pads)

    # Apply pad_mode if not SPECIFIC
    if pad_mode.upper() != "SPECIFIC":
        pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right = _apply_pad_mode(
            pad_mode, x_np.shape, filter_shape, stride_d, stride_h, stride_w,
            dilation_d, dilation_h, dilation_w
        )

    # Calculate symmetric padding (minimum of front/back, top/bottom, left/right)
    sym_pad_d = min(pad_d_front, pad_d_back)
    sym_pad_h = min(pad_top, pad_bottom)
    sym_pad_w = min(pad_left, pad_right)

    # Calculate extra asymmetric padding to be applied directly to input
    extra_pad_d_front = max(0, pad_d_front - pad_d_back)
    extra_pad_d_back = max(0, pad_d_back - pad_d_front)
    extra_pad_top = max(0, pad_top - pad_bottom)
    extra_pad_bottom = max(0, pad_bottom - pad_top)
    extra_pad_left = max(0, pad_left - pad_right)
    extra_pad_right = max(0, pad_right - pad_left)

    # Apply extra padding to input (N, C, D, H, W dimensions)
    input_pad = np.pad(
        x_np,
        (
            (0, 0),  # N dimension: no padding
            (0, 0),  # C dimension: no padding
            (extra_pad_d_front, extra_pad_d_back),  # D dimension
            (extra_pad_top, extra_pad_bottom),  # H dimension
            (extra_pad_left, extra_pad_right),  # W dimension
        ),
        "constant",
        constant_values=(0, 0),
    )

    # Symmetric padding for torch.nn.functional.conv3d
    torch_pad = [sym_pad_d, sym_pad_h, sym_pad_w]

    return input_pad, torch_pad


def conv3d_v2_golden(x, filter, bias=None, scale=None, offset=None, offset_w=None,
                    *,
                    strides,
                    pads: list=[0, 0, 0, 0, 0, 0],
                    dilations: list=[1, 1, 1, 1, 1],
                    groups: int=1,
                    data_format:str=NCDHW_FORMAT,
                    offset_x: int= 0,
                    pad_mod: str='SPECIFIC',
                    enable_hf32: bool=False,
                    **kwargs,
                    ):
    """
    Kernel golden for conv3d_v2.
    All parameters follow @conv3d_v2_def.cpp without outputs.
    All input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch
    import torch.nn.functional as F
    
    short_soc_version = kwargs.get("short_soc_version", "")
    is_950 = is_ascend950(short_soc_version)
    
    input_formats = kwargs.get("input_formats", [NCDHW_FORMAT, NCDHW_FORMAT])
    input_ori_shapes = kwargs.get("input_ori_shapes", None)

    x_dtype_str = x.dtype.name 
    
    if is_950:
        x_np, filter_np = process_formats_a5(x, filter, input_formats)
    else:
        x_np, filter_np = process_formats_a2_a3(x, filter, input_formats, input_ori_shapes, groups)
    
    if enable_hf32 and x_dtype_str == FP32_STR:
        calc_dtype = np.float32
        x_np = simulate_hf32_precision(x_np.astype(np.float32))
        filter_np = simulate_hf32_precision(filter_np.astype(np.float32))
    else:
        calc_dtype = np.float64 if x_dtype_str == FP32_STR else np.float32
        x_np = x_np.astype(calc_dtype)
        filter_np = filter_np.astype(calc_dtype)
    
    if bias is not None:
        bias_np = bias.astype(calc_dtype)
    else:
        bias_np = None
    
    if scale is not None:
        scale_arr = scale if isinstance(scale, np.ndarray) else np.array(scale)
        scale_np = scale_arr.astype(np.float32)
    else:
        scale_np = None
    
    if isinstance(strides, (list, tuple)):
        if len(strides) == 5:
            stride_d, stride_h, stride_w = strides[2], strides[3], strides[4]
        elif len(strides) == 3:
            stride_d, stride_h, stride_w = strides[0], strides[1], strides[2]
        else:
            stride_d = stride_h = stride_w = strides[0]
    else:
        stride_d = stride_h = stride_w = int(strides)
    
    if isinstance(dilations, (list, tuple)):
        if len(dilations) == 5:
            dilation_d, dilation_h, dilation_w = dilations[2], dilations[3], dilations[4]
        elif len(dilations) == 3:
            dilation_d, dilation_h, dilation_w = dilations[0], dilations[1], dilations[2]
        else:
            dilation_d = dilation_h = dilation_w = dilations[0]
    else:
        dilation_d = dilation_h = dilation_w = int(dilations)
        
    output_dtypes = kwargs.get("output_dtypes", [FP32_STR])
    output_dtype = output_dtypes[0]
    output_formats = kwargs.get("output_formats", [NCDHW_FORMAT])
    output_format = output_formats[0]
    input_format = input_formats[0]

    # Process padding and prepare tensors for conv3d
    input_pad, pad_torch = _process_conv3d_padding(
        x_np, pads, pad_mod, filter_np.shape, stride_d, stride_h, stride_w,
        dilation_d, dilation_h, dilation_w
    )

    input_torch = torch.from_numpy(input_pad)
    weight_torch = torch.from_numpy(filter_np)
    bias_torch = torch.from_numpy(bias_np) if bias_np is not None else None

    stridedhw = [stride_d, stride_h, stride_w]
    dilationdhw = [dilation_d, dilation_h, dilation_w]
    
    out = torch.nn.functional.conv3d(
        input_torch,
        weight_torch,
        bias_torch,
        stride=stridedhw,
        padding=pad_torch,
        dilation=dilationdhw,
        groups=groups,
    )
    
    if scale_np is not None:
        if scale_np.ndim == 1:
            scale_tensor = torch.from_numpy(scale_np.reshape(1, scale_np.shape[0], 1, 1, 1))
        elif scale_np.ndim == 5:
            scale_tensor = torch.from_numpy(scale_np)
        else:
            raise ValueError(f"scale must be 1D or 5D, got {scale_np.ndim}D with shape {scale_np.shape}")
        out = torch.multiply(out, scale_tensor).numpy()
    else:
        out = out.numpy()
    
    out = convert_output_dtype(out, output_dtype, enable_hf32)
    
    output_ori_shapes = kwargs.get("output_ori_shapes", None)
    
    if not is_950:
        out = process_output_format_a2_a3(out, output_format, input_format, output_ori_shapes)
    else:
        out = process_output_format_a5(out, output_format, input_format)
    
    return out
