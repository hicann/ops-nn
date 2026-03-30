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

__golden__ = {
    "kernel": {
        "conv2dv2": "conv2d_v2_golden",
        "conv2d_v2": "conv2d_v2_golden"
    }
}

FP32_STR = "float32"
NCHW_FORMAT = "NCHW"


def due_fp16_overflow(data):
    """Overflow interception for float16
    Clips values to the finite range of float16: [-65504, 65504]
    """
    # Maximum finite value for float16 (FP16)
    FP16_MAX_FINITE = 65504
    # Minimum finite value for float16 (FP16)
    FP16_MIN_FINITE = -65504
    data = np.maximum(data, FP16_MIN_FINITE)
    data = np.minimum(data, FP16_MAX_FINITE)
    return data


def simulate_hf32_precision(data):
    """
    Simulate HF32 (Half Float 32) precision.
    HF32 truncates lower 12 bits of float32 mantissa (keeping 11 bits with rounding).
    """
    if data.dtype == np.float32:
        # HF32 format details:
        # - Sign bit: 1 bit
        # - Exponent: 5 bits  
        # - Mantissa: 11 bits (stored) + 1 implicit bit = 12 bits precision
        # - Total: 1 + 5 + 12 = 18 bits used (but stored in 16 bits)
        # When converting from FP32 to HF32:
        # 1. Right shift by 11 to remove lower 11 bits of mantissa (keep sign + exp + upper mantissa)
        # 2. Add 1 for rounding (to nearest even)
        # 3. Right shift by 1 to complete the rounding operation
        # 4. Left shift by 12 to restore the original bit alignment
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


def process_formats_a5(x, filter, input_formats):
    """
    Process format conversion for Ascend 950PR/950DT (A5).
    Constraints:
    - x supports: NCHW, NHWC
    - filter supports: NCHW (when x is NCHW) or HWCN (when x is NHWC)
    """
    input_data_format, input_filter_format = input_formats[0], input_formats[1]
    
    if input_data_format == "NHWC":
        x = x.transpose(0, 3, 1, 2)
    
    if input_filter_format == "HWCN":
        filter = filter.transpose(3, 2, 0, 1)
    
    return x, filter


def process_output_format_a5(out, output_format):
    """
    Process output format conversion for Ascend 950PR/950DT (A5).
    Constraints:
    - y supports: NCHW (when x is NCHW) or NHWC (when x is NHWC)
    """
    if output_format == "NHWC":
        out = out.transpose((0, 2, 3, 1))
    
    return out


def get_ori_pad_from_pad_mode(x_np, filter_np, pads, pad_mode, stride_h, stride_w, dilation_h, dilation_w):
    """
    Calculate original padding values based on pad_mode, corresponding to C++ GetOriPadFromPadMode.
    
    Args:
        x_np: Input feature map numpy array
        filter_np: Weight filter numpy array
        pads: Padding value, can be None, scalar, list of 2 [V, V] or list of 4 [top, bottom, left, right]
        pad_mode: Padding mode, supports "SPECIFIC", "VALID", "SAME", "SAME_UPPER", "SAME_LOWER"
        stride_h: Stride in H dimension
        stride_w: Stride in W dimension
        dilation_h: Dilation in H dimension
        dilation_w: Dilation in W dimension
    
    Returns:
        Tuple (pad_top, pad_bottom, pad_left, pad_right)
    """
    _, _, in_h, in_w = x_np.shape
    _, _, k_h, k_w = filter_np.shape

    def conv_ceil_div(a, b):
        """Ceiling division: (a + b - 1) // b"""
        return (a + b - 1) // b if b != 0 else 0

    pad_mode_upper = pad_mode.upper()
    if pad_mode_upper == "SPECIFIC":
        if isinstance(pads, (list, tuple)):
            if len(pads) == 4:
                pad_top, pad_bottom, pad_left, pad_right = int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])
            elif len(pads) == 2:
                pad_top, pad_bottom, pad_left, pad_right = int(pads[0]), int(pads[0]), int(pads[1]), int(pads[1])
            else:
                pad_val = int(pads[0])
                pad_top = pad_bottom = pad_left = pad_right = pad_val
        else:
            pad_val = int(pads)
            pad_top = pad_bottom = pad_left = pad_right = pad_val
    elif pad_mode_upper == "VALID":
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        pad_h = (conv_ceil_div(in_h, stride_h) - 1) * stride_h + dilation_h * (k_h - 1) - in_h + 1
        pad_w = (conv_ceil_div(in_w, stride_w) - 1) * stride_w + dilation_w * (k_w - 1) - in_w + 1
        if pad_mode_upper == "SAME" or pad_mode_upper == "SAME_UPPER":
            if pad_mode_upper == "SAME":
                pad_h = max(0, pad_h)
                pad_w = max(0, pad_w)
            pad_bottom = conv_ceil_div(pad_h, 2)
            pad_top = pad_h - pad_bottom
            pad_right = conv_ceil_div(pad_w, 2)
            pad_left = pad_w - pad_right
        else:
            # SAME_LOWER
            pad_top = conv_ceil_div(pad_h, 2)
            pad_bottom = pad_h - pad_top
            pad_left = conv_ceil_div(pad_w, 2)
            pad_right = pad_w - pad_left

    return pad_top, pad_bottom, pad_left, pad_right


def conv2d_v2_golden(x, filter, bias=None, offset_w=None,
                    *,
                    strides: list,
                    pads: list = [0, 0, 0, 0],
                    dilations: list = [1, 1, 1, 1],
                    groups: int = 1,
                    data_format: str = NCHW_FORMAT,
                    offset_x: int = 0,
                    pad_mode: str = "SPECIFIC",
                    enable_hf32: bool = False,
                    **kwargs,
                ):
    """
    Kernel golden for conv2d_v2.
    All parameters follow @conv2d_v2_def.cpp without outputs.
    All input Tensors are numpy.ndarray.
    kwargs may contain: input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch
    import torch.nn.functional as F
    
    input_formats = kwargs.get("input_formats", [NCHW_FORMAT, NCHW_FORMAT])
    input_format = input_formats[0]

    x_dtype_str = x.dtype.name 
    filter_dtype_str = filter.dtype.name
    
    x_np, filter_np = process_formats_a5(x, filter, input_formats)
    
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
    
    if isinstance(strides, (list, tuple)):
        if len(strides) == 4:
            stride_h, stride_w = int(strides[2]), int(strides[3])
        elif len(strides) == 2:
            stride_h, stride_w = int(strides[0]), int(strides[1])
        else:
            stride_h = stride_w = int(strides[0])
    else:
        stride_h = stride_w = int(strides)
    
    if isinstance(dilations, (list, tuple)):
        if len(dilations) == 4:
            dilation_h, dilation_w = int(dilations[2]), int(dilations[3])
        elif len(dilations) == 2:
            dilation_h, dilation_w = int(dilations[0]), int(dilations[1])
        else:
            dilation_h = dilation_w = int(dilations[0])
    else:
        dilation_h = dilation_w = int(dilations)

    pad_top, pad_bottom, pad_left, pad_right = get_ori_pad_from_pad_mode(x_np, filter_np, pads, pad_mode, stride_h, stride_w, dilation_h, dilation_w)
    input_torch = torch.from_numpy(x_np)
    weight_torch = torch.from_numpy(filter_np)
    bias_torch = torch.from_numpy(bias_np) if bias_np is not None else None
    
    torch_pad = (pad_left, pad_right, pad_top, pad_bottom)
    if any(pad > 0 for pad in (pad_top, pad_bottom, pad_left, pad_right)):
        pad_value = float(offset_x) if offset_x != 0 else 0.0
        input_torch = F.pad(input_torch, torch_pad, "constant", pad_value)
    
    out = torch.nn.functional.conv2d(
        input_torch,
        weight_torch,
        bias=bias_torch,
        stride=(stride_h, stride_w),
        padding=0,
        dilation=(dilation_h, dilation_w),
        groups=groups,
    ).numpy()

    output_dtypes = kwargs.get("output_dtypes", [FP32_STR])
    output_dtype = output_dtypes[0]
    output_formats = kwargs.get("output_formats", [NCHW_FORMAT])
    output_format = output_formats[0]

    out = convert_output_dtype(out, output_dtype, enable_hf32)
    
    out = process_output_format_a5(out, output_format)
    
    return out
