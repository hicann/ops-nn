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

import numpy as np

"""
Golden function for extend_conv_transpose kernel.

Based on extend_conv_transpose_def.cpp:
- Inputs: input_size (REQUIRED), x (REQUIRED), filter (REQUIRED), bias (OPTIONAL), scale (OPTIONAL)
- Output: y (REQUIRED)

Format and DataType support (Ascend950 only):
- input_size: Format=ND, DataType=INT32/INT64
- x: Format=NCDHW/NDHWC, DataType=INT8
- filter: Format=NCDHW/NDHWC/DHWCN, DataType=INT8
- bias: Format=ND, DataType=INT32
- scale: Format=ND, DataType=UINT64
- y(output): Format=NCDHW/NDHWC, DataType=FLOAT16/INT8

Quantization semantics (INT8 x * INT8 filter -> INT32 accumulator -> fixpipe dequant):
- scale shape (1,)   : SCALAR_QUANT, y = (bias + sum(x * w)) * deq_scale[0]
- scale shape (C,)  : VECTOR_QUANT, y = (bias + sum(x * w)) * deq_scale[c] (per output channel)
- scale absent      : NO_QUANT, fixpipe uses DQ_SCALAR_QF_ONE = fp32(0x37800000) = 2^-16
- uint64 scale bit layout (trans_quant_param_v2): low 32 bits = fp32 bit pattern masked by
  0xFFFFE000 (19 significant bits); bits [45:37] = int9 requant offset (for int8 output);
  bit 46 = offset-present flag.
"""

__golden__ = {"kernel": {"extend_conv_transpose": "extend_conv_transpose_golden"}}

NCDHW_FORMAT = "NCDHW"
NDHWC_FORMAT = "NDHWC"
FP32_STR = "float32"

# conv_bp_input_sub_func_utils.h: DQ_SCALAR_QF_ONE = 0x37800000, used by fixpipe
# SetQuantInt32ToHalf/SetQuantInt8 when no scale input is connected (NO_QUANT).
DQ_SCALAR_QF_ONE = float(
    np.array([0x37800000], dtype=np.uint32).view(np.float32)[0]
)  # 2^-16

DEQ_SCALE_MASK = np.uint32(0xFFFFE000)


def due_fp16_overflow(data):
    """Clip values to float16 finite range [-65504, 65504]."""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def u64_to_deq_scale(u64_scale):
    """Decode float32 dequantization scale from uint64 packed format (low 32 bits)."""
    scale_np = u64_scale if isinstance(u64_scale, np.ndarray) else np.array(u64_scale)
    deq_u32 = scale_np.astype(np.uint32).copy()
    deq_u32 &= DEQ_SCALE_MASK
    return deq_u32.view(np.float32).reshape(scale_np.shape)


def u64_to_offset(u64_scale):
    """Decode int9 requant offset from uint64 scale (bits 37-45), used for int8 output."""
    scale_np = u64_scale if isinstance(u64_scale, np.ndarray) else np.array(u64_scale)
    raw = (scale_np.astype(np.uint64) >> np.uint64(37)) & np.uint64(0x1FF)
    raw = raw.astype(np.int64)
    raw = np.where((raw & np.int64(0x100)) != 0, raw - np.int64(0x200), raw)
    return raw.astype(np.float32).reshape(scale_np.shape)


def convert_output_dtype(out, output_dtype):
    """Convert output array to target dtype with overflow handling."""
    if output_dtype == "float16":
        return due_fp16_overflow(out).astype(np.float16)
    if output_dtype == "int8":
        # REQ8/VREQ8 (fixpipe int8 requant) saturates to the UNSIGNED 8-bit range
        # [0, 255] with round-half-to-even, and the result bytes are stored in the
        # int8 output tensor (negative values clamp to 0; values > 127 wrap negative
        # when read back as int8). Verified bitwise against hardware dumps.
        return np.clip(np.round(out), 0, 255).astype(np.uint8).view(np.int8)
    if output_dtype == "int32":
        return out.astype(np.int32)
    return out.astype(np.float32)


def ceil_div(a, b):
    return (a + b - 1) // b


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
    elif dtype == "int8":
        return 32
    return 16


def to_NCDHW_from_NDC1HWC0(data, ori_shape):
    """Convert from NDC1HWC0 (6D physical) to NCDHW (5D logical)."""
    n, c, d, h, w = ori_shape
    c0 = determine_c0(data.dtype.name)
    c1 = ceil_div(c, c0)
    data = data.transpose(0, 2, 5, 1, 3, 4)
    data = data.reshape((n, c1 * c0, d, h, w))
    if c1 * c0 > c:
        data = data[:, :c, :, :, :]
    return data


def to_NDC1HWC0_from_NCDHW(data, ori_shape):
    """Convert from NCDHW (5D logical) to NDC1HWC0 (6D physical)."""
    n, c, d, h, w = ori_shape
    c0 = determine_c0(data.dtype.name)
    c1 = ceil_div(c, c0)
    if c1 * c0 > c:
        num_2_padding_in_c = c1 * c0 - c
        zero_padding_array = np.zeros(
            (n, num_2_padding_in_c, d, h, w), dtype=data.dtype
        )
        data = np.concatenate((data, zero_padding_array), axis=1)
    data = data.reshape((n, c1, c0, d, h, w))
    data = data.transpose(0, 3, 1, 4, 5, 2)
    return data


def process_input_formats(x, filter, input_formats, input_ori_shapes=None):
    """
    Convert x/filter to NCDHW (5D) for computation.

    Note: input_formats follows the operator's input order (input_size, x, filter, ...),
    so x format is at index 1 and filter format is at index 2.
    """
    input_x_format = input_formats[1] if len(input_formats) > 1 else NCDHW_FORMAT
    input_filter_format = input_formats[2] if len(input_formats) > 2 else NCDHW_FORMAT

    if input_x_format == NDHWC_FORMAT:
        # NDHWC -> NCDHW: (N, D, H, W, C) -> (N, C, D, H, W)
        x = x.transpose(0, 4, 1, 2, 3)
    elif (
        input_x_format == NCDHW_FORMAT
        and x.ndim == 6
        and input_ori_shapes is not None
        and len(input_ori_shapes) > 1
    ):
        # NDC1HWC0 (6D physical) -> NCDHW (5D logical)
        x = to_NCDHW_from_NDC1HWC0(x, input_ori_shapes[1])

    if input_filter_format == NDHWC_FORMAT:
        # NDHWC -> NCDHW: (kn, kD, kH, kW, kc) -> (kn, kc, kD, kH, kW)
        filter = filter.transpose(0, 4, 1, 2, 3)
    elif input_filter_format == "DHWCN":
        # DHWCN -> NCDHW: (kD, kH, kW, kc, kn) -> (kn, kc, kD, kH, kW)
        filter = filter.transpose(4, 3, 0, 1, 2)

    return x, filter


def process_output_format(out, output_format, output_ori_shapes=None, x_was_6d=False):
    """
    Convert computation output (NCDHW) to the kernel's physical output format.

    Supported: NCDHW, NDHWC, NDC1HWC0 (6D physical, when x was 6D).
    """
    if output_format == NDHWC_FORMAT:
        # NCDHW -> NDHWC: (N, C, D, H, W) -> (N, D, H, W, C)
        out = out.transpose(0, 2, 3, 4, 1)
    elif (
        output_format == NCDHW_FORMAT
        and x_was_6d
        and output_ori_shapes
        and len(output_ori_shapes) > 0
    ):
        # NCDHW (5D logical) -> NDC1HWC0 (6D physical)
        out = to_NDC1HWC0_from_NCDHW(out, output_ori_shapes[0])
    return out


def parse_pads(pads):
    """
    Parse padding parameter into tuple format.

    Returns:
        tuple: (pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right)
    """
    if isinstance(pads, (list, tuple)):
        if len(pads) == 6:
            return (
                int(pads[0]),
                int(pads[1]),
                int(pads[2]),
                int(pads[3]),
                int(pads[4]),
                int(pads[5]),
            )
        elif len(pads) == 3:
            return (
                int(pads[0]),
                int(pads[0]),
                int(pads[1]),
                int(pads[1]),
                int(pads[2]),
                int(pads[2]),
            )
        else:
            val = int(pads[0])
            return val, val, val, val, val, val
    else:
        val = int(pads)
        return val, val, val, val, val, val


def parse_dhw_attrs(vals, data_format):
    """
    Parse a D/H/W attribute (strides, dilations or output_padding) into (d, h, w).

    A 5-element list follows the tensor axis order of data_format:
    - NCDHW: [N, C, D, H, W] -> values at index 2/3/4
    - NDHWC: [N, D, H, W, C] -> values at index 1/2/3
    """
    if isinstance(vals, (list, tuple)):
        if len(vals) == 5:
            if data_format == NDHWC_FORMAT:
                return int(vals[1]), int(vals[2]), int(vals[3])
            return int(vals[2]), int(vals[3]), int(vals[4])
        elif len(vals) == 3:
            return int(vals[0]), int(vals[1]), int(vals[2])
        else:
            val = int(vals[0])
            return val, val, val
    else:
        val = int(vals)
        return val, val, val


def extend_conv_transpose_golden(
    input_size,
    x,
    filter,
    bias=None,
    scale=None,
    *,
    strides: list,
    pads: list,
    dilations: list = [1, 1, 1, 1, 1],
    groups: int = 1,
    data_format: str = NDHWC_FORMAT,
    output_padding: list = [0, 0, 0, 0, 0],
    offset_x: int = 0,
    fusion_mode: int = 0,
    y_quant_mode: int = 0,
    **kwargs,
):
    """
    Kernel golden for extend_conv_transpose (INT8 quantized conv transpose, Ascend950).

    All parameters follow @extend_conv_transpose_def.cpp without outputs.
    Computes: y = (bias_i32 + sum(x_i8 * filter_i8)) * deq_scale, converted to output dtype
    by the fixpipe (DEQF16/VDEQF16 for fp16 output, REQ8/VREQ8 for int8 output).

    All input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch

    input_formats = kwargs.get(
        "input_formats", [NCDHW_FORMAT, NCDHW_FORMAT, NCDHW_FORMAT]
    )
    input_ori_shapes = kwargs.get("input_ori_shapes", None)
    output_dtypes = kwargs.get("output_dtypes", [FP32_STR])
    output_formats = kwargs.get("output_formats", [NCDHW_FORMAT])
    output_ori_shapes = kwargs.get("output_ori_shapes", None)

    x_was_6d = x.ndim == 6

    x_np, filter_np = process_input_formats(x, filter, input_formats, input_ori_shapes)

    # INT8 x INT8 accumulates exactly in the INT32 domain; float64 represents
    # every integer in the int32 range exactly, so the accumulation is lossless.
    calc_dtype = np.float64
    x_np = x_np.astype(calc_dtype)
    filter_np = filter_np.astype(calc_dtype)
    bias_np = bias.astype(calc_dtype) if bias is not None else None

    stride_d, stride_h, stride_w = parse_dhw_attrs(strides, data_format)
    dilation_d, dilation_h, dilation_w = parse_dhw_attrs(dilations, data_format)
    out_pad_d, out_pad_h, out_pad_w = parse_dhw_attrs(output_padding, data_format)
    pad_d_front, pad_d_back, pad_top, pad_bottom, pad_left, pad_right = parse_pads(pads)

    input_torch = torch.from_numpy(x_np)

    # NOTE: No weight transpose needed!
    # PyTorch conv_transpose3d weight format: (in_channels, out_channels/groups, kD, kH, kW)
    # CANN filter format (NCDHW):            (kn,          kc,                   kD, kH, kW)
    # Since kn == in_channels and kc == out_channels/groups, the CANN filter is ALREADY in
    # the correct format for PyTorch's conv_transpose3d. No transpose required.
    weight_torch = torch.from_numpy(filter_np)
    bias_torch = torch.from_numpy(bias_np) if bias_np is not None else None

    # conv_transpose3d uses symmetric padding (pad_d, pad_h, pad_w);
    # asymmetric padding is handled by slicing the output below.
    sym_pad_d = min(pad_d_front, pad_d_back)
    sym_pad_h = min(pad_top, pad_bottom)
    sym_pad_w = min(pad_left, pad_right)

    out = torch.nn.functional.conv_transpose3d(
        input_torch,
        weight_torch,
        bias=bias_torch,
        stride=(stride_d, stride_h, stride_w),
        padding=(sym_pad_d, sym_pad_h, sym_pad_w),
        output_padding=(out_pad_d, out_pad_h, out_pad_w),
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

    if any(
        [
            extra_pad_d_front,
            extra_pad_d_back,
            extra_pad_top,
            extra_pad_bottom,
            extra_pad_left,
            extra_pad_right,
        ]
    ):
        # out shape is (N, C, D, H, W)
        _, _, d, h, w = out.shape
        out = out[
            :,
            :,
            extra_pad_d_front : d - extra_pad_d_back,
            extra_pad_top : h - extra_pad_bottom,
            extra_pad_left : w - extra_pad_right,
        ]

    # Dequantization on the fixpipe: y = acc_i32 * deq_scale (+ requant offset for int8 output)
    output_dtype = output_dtypes[0]
    if scale is None:
        # NO_QUANT: fixpipe falls back to the unit coefficient DQ_SCALAR_QF_ONE (2^-16),
        # see SetQuantInt32ToHalf in conv_bp_input_sub_func_store_l0c_fixpipe.h.
        out = out * DQ_SCALAR_QF_ONE
    else:
        scale_np = scale if isinstance(scale, np.ndarray) else np.array(scale)
        deq_scale = u64_to_deq_scale(scale_np)
        if deq_scale.size == 1:
            # SCALAR_QUANT: DEQF16/REQ8 with deqScalar read from scale[0]
            out = out * float(deq_scale.reshape(-1)[0])
        else:
            # VECTOR_QUANT: VDEQF16/VREQ8 with per-output-channel scale
            out = out * deq_scale.astype(calc_dtype).reshape(1, -1, 1, 1, 1)
        if output_dtype == "int8":
            # REQ8/VREQ8 requant offset packed in scale bits [45:37], added AFTER scaling
            offset = u64_to_offset(scale_np)
            if scale_np.size == 1:
                out = out + float(offset.reshape(-1)[0])
            else:
                out = out + offset.astype(calc_dtype).reshape(1, -1, 1, 1, 1)

    out = convert_output_dtype(out, output_dtype)
    out = process_output_format(out, output_formats[0], output_ori_shapes, x_was_6d)
    return out
