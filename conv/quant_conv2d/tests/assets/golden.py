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
Golden function for quant_conv2d kernel.

Based on quant_conv2d_def.cpp:
- Inputs: x (REQUIRED), filter (REQUIRED), scale (REQUIRED), bias (OPTIONAL), offset (OPTIONAL)
- Output: y (REQUIRED)

Format and DataType support:
- x: Format=NCHW, DataType=INT8/HIFLOAT8/FLOAT8_E4M3FN
- filter: Format=NCHW, DataType=INT8/HIFLOAT8/FLOAT8_E4M3FN
- scale: Format=ND, DataType=INT64/UINT64 (stores FLOAT32 binary)
- bias: Format=ND, DataType=INT32/FLOAT
- offset: Format=NCHW, DataType=FLOAT
- y(output): Format=NCHW, DataType=FLOAT16/FLOAT/BF16/HIFLOAT8/FLOAT8_E4M3FN

Supported dtypes: float16, float32, bfloat16, hifloat8, float8_e4m3fn, int8, int32
"""

import numpy as np

__golden__ = {"kernel": {"quant_conv2d": "quant_conv2d_golden"}}

NCHW_FORMAT = "NCHW"
FP32_STR = "float32"


def due_fp16_overflow(data):
    """Overflow interception for float16
    Clips values to the finite range of float16: [-65504, 65504]
    """
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def simulate_hf32_precision(data, short_soc_version=None):
    """
    Simulate HF32 (Half Float 32) precision.
    Ascend910B: truncates lower 12 bits of float32 mantissa, keeping 20 bits with rounding.
    Default: truncates lower 13 bits of float32 mantissa, keeping 19 bits with rounding.
    """
    if data.dtype == np.float32:
        input_hf32 = data.view(np.int32)
        if short_soc_version in ("Ascend910B",):
            input_hf32 = np.right_shift(np.right_shift(input_hf32, 11) + 1, 1)
            input_hf32 = np.left_shift(input_hf32, 12)
        else:
            input_hf32 = np.right_shift(np.right_shift(input_hf32, 12) + 1, 1)
            input_hf32 = np.left_shift(input_hf32, 13)
        return input_hf32.view(np.float32)
    return data


def _ceil_div(a, b):
    """Ceiling division: returns ceil(a / b)."""
    return -(-a // b)


def _parse_padding(pads):
    """
    Parse padding parameter into 4-element format [pad_top, pad_bottom, pad_left, pad_right].

    Args:
        pads: padding value, can be int, or list/tuple with 1, 2, or 4 elements.

    Returns:
        tuple: (pad_top, pad_bottom, pad_left, pad_right)
    """
    if isinstance(pads, (list, tuple)):
        if len(pads) == 4:
            return pads[0], pads[1], pads[2], pads[3]
        elif len(pads) == 2:
            return pads[0], pads[0], pads[1], pads[1]
        else:
            val = pads[0]
            return val, val, val, val
    else:
        val = int(pads)
        return val, val, val, val


def _apply_pad_mode(
    pad_mode, input_shape, filter_shape, stride_h, stride_w, dilation_h, dilation_w
):
    """
    Calculate padding based on pad_mode, aligned with C++ GetOriPadFromPadMode logic.

    Supported modes:
        - VALID: no padding, all zeros
        - SAME: same output size, extra padding distributed equally
        - SAME_UPPER: extra padding goes to bottom/right
        - SAME_LOWER: extra padding goes to top/left

    Args:
        pad_mode: padding mode string
        input_shape: input tensor shape (N, C, H, W)
        filter_shape: filter tensor shape (C_out, C_in, kH, kW)
        stride_h/w: stride values for each spatial dimension
        dilation_h/w: dilation values for each spatial dimension

    Returns:
        tuple: (pad_top, pad_bottom, pad_left, pad_right)
    """
    mode = pad_mode.upper()

    if mode == "VALID":
        return 0, 0, 0, 0

    if mode == "SPECIFIC":
        return 0, 0, 0, 0

    hi, wi = input_shape[2], input_shape[3]
    kh, kw = filter_shape[2], filter_shape[3]

    pad_h = (_ceil_div(hi, stride_h) - 1) * stride_h + dilation_h * (kh - 1) - hi + 1
    pad_w = (_ceil_div(wi, stride_w) - 1) * stride_w + dilation_w * (kw - 1) - wi + 1

    if mode == "SAME":
        pad_h = max(0, pad_h)
        pad_w = max(0, pad_w)
        pad_bottom = _ceil_div(pad_h, 2)
        pad_top = pad_h - pad_bottom
        pad_right = _ceil_div(pad_w, 2)
        pad_left = pad_w - pad_right
    elif mode == "SAME_UPPER":
        pad_bottom = _ceil_div(pad_h, 2)
        pad_top = pad_h - pad_bottom
        pad_right = _ceil_div(pad_w, 2)
        pad_left = pad_w - pad_right
    elif mode == "SAME_LOWER":
        pad_top = _ceil_div(pad_h, 2)
        pad_bottom = pad_h - pad_top
        pad_left = _ceil_div(pad_w, 2)
        pad_right = pad_w - pad_left
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    return pad_top, pad_bottom, pad_left, pad_right


def _process_conv2d_padding(
    x_np, pads, pad_mode, filter_shape, stride_h, stride_w, dilation_h, dilation_w
):
    """
    Process padding for conv2d operation.

    Handles both explicit padding (SPECIFIC mode) and automatic padding
    modes (VALID, SAME, SAME_UPPER, SAME_LOWER). It applies asymmetric padding by
    splitting it into symmetric padding (for torch.conv2d) and extra padding (applied to input).

    Args:
        x_np: input numpy array (N, C, H, W)
        pads: explicit padding values
        pad_mode: padding mode string
        filter_shape: filter tensor shape
        stride_h/w: stride values
        dilation_h/w: dilation values

    Returns:
        tuple: (padded_input_np, torch_pad_list)
    """
    pad_top, pad_bottom, pad_left, pad_right = _parse_padding(pads)

    if pad_mode is not None and pad_mode.upper() not in (
        "SPECIFIC",
        "VALID",
        "SAME",
        "SAME_UPPER",
        "SAME_LOWER",
    ):
        pad_mode = "SPECIFIC"

    if pad_mode is not None and pad_mode.upper() != "SPECIFIC":
        pad_top, pad_bottom, pad_left, pad_right = _apply_pad_mode(
            pad_mode,
            x_np.shape,
            filter_shape,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
        )

    sym_pad_h = min(pad_top, pad_bottom)
    sym_pad_w = min(pad_left, pad_right)

    extra_pad_top = max(0, pad_top - pad_bottom)
    extra_pad_bottom = max(0, pad_bottom - pad_top)
    extra_pad_left = max(0, pad_left - pad_right)
    extra_pad_right = max(0, pad_right - pad_left)

    input_pad = np.pad(
        x_np,
        (
            (0, 0),
            (0, 0),
            (extra_pad_top, extra_pad_bottom),
            (extra_pad_left, extra_pad_right),
        ),
        "constant",
        constant_values=(0, 0),
    )

    torch_pad = [sym_pad_h, sym_pad_w]

    return input_pad, torch_pad


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
            dtype_cls = getattr(
                __import__(module_name, fromlist=[dtype_name]), dtype_name
            )
        except (ImportError, AttributeError):
            raise RuntimeError(
                f"{module_name} is required for {output_dtype}. "
                f"Install: pip install {module_name}"
            )
        out = out.astype(dtype_cls)
    else:
        out = out.astype(dtype_ref)

    if output_dtype == FP32_STR and enable_hf32:
        out = simulate_hf32_precision(out, short_soc_version)

    return out


def quant_conv2d_golden(
    x,
    filter,
    scale,
    bias=None,
    offset=None,
    *,
    dtype: int,
    strides: list,
    pads: list = [0, 0, 0, 0],
    dilations: list = [1, 1, 1, 1],
    groups: int = 1,
    data_format: str = NCHW_FORMAT,
    offset_x: int = 0,
    round_mode: str = "rint",
    **kwargs,
):
    """
    Kernel golden for quant_conv2d.
    All parameters follow @quant_conv2d_def.cpp without outputs.
    All input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        output_dtypes.
    """
    import torch

    short_soc_version = kwargs.get("short_soc_version", "")
    x_dtype_str = x.dtype.name

    x_np = x
    filter_np = filter

    calc_dtype = np.float64 if x_dtype_str == "float32" else np.float32
    x_np = x_np.astype(calc_dtype)
    filter_np = filter_np.astype(calc_dtype)

    if bias is not None:
        bias_np = bias.astype(calc_dtype)
    else:
        bias_np = None

    if isinstance(strides, (list, tuple)):
        if len(strides) == 4:
            stride_h, stride_w = strides[2], strides[3]
        elif len(strides) == 2:
            stride_h, stride_w = strides[0], strides[1]
        else:
            stride_h = stride_w = strides[0]
    else:
        stride_h = stride_w = int(strides)

    if isinstance(dilations, (list, tuple)):
        if len(dilations) == 4:
            dilation_h, dilation_w = dilations[2], dilations[3]
        elif len(dilations) == 2:
            dilation_h, dilation_w = dilations[0], dilations[1]
        else:
            dilation_h = dilation_w = dilations[0]
    else:
        dilation_h = dilation_w = int(dilations)

    output_dtypes = kwargs.get("output_dtypes", [FP32_STR])
    output_dtype = output_dtypes[0]

    pad_mode = kwargs.get("pad_mode", "SPECIFIC")

    input_pad, pad_torch = _process_conv2d_padding(
        x_np,
        pads,
        pad_mode,
        filter_np.shape,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
    )
    input_torch = torch.from_numpy(input_pad)
    weight_torch = torch.from_numpy(filter_np)
    bias_torch = torch.from_numpy(bias_np) if bias_np is not None else None

    stridehw = [stride_h, stride_w]
    dilationhw = [dilation_h, dilation_w]

    out = torch.nn.functional.conv2d(
        input_torch,
        weight_torch,
        bias_torch,
        stride=stridehw,
        padding=pad_torch,
        dilation=dilationhw,
        groups=groups,
    )

    scale_np = scale if isinstance(scale, np.ndarray) else np.array(scale)
    scale_tensor = torch.from_numpy(
        scale_np.astype(np.uint32).view(np.float32).reshape(1, scale_np.shape[0], 1, 1)
    )
    out = torch.multiply(out, scale_tensor)
    out = out.numpy()

    out = convert_output_dtype(out, output_dtype, short_soc_version=short_soc_version)

    return out
