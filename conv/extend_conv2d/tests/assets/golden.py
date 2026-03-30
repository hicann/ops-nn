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

from typing import Optional
import numpy as np

__golden__ = {
    "kernel": {
        "extend_conv2d": "extend_conv2d_golden"
    }
}

FP32_STR = "float32"
NCHW_FORMAT = "NCHW"
NHWC_FORMAT = "NHWC"


def due_fp16_overflow(data):
    """Overflow interception for float16"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def simulate_hf32_precision(data):
    """
    Simulate HF32 (Half Float 32) precision.
    HF32 truncates lower 12 bits of float32 mantissa (keeping 11 bits with rounding).
    """
    if data.dtype == np.float32:
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


def is_ascend950(short_soc_version):
    """Check if the target is Ascend 950PR/950DT"""
    return (short_soc_version == "Ascend950")


def process_input_format(x, filter, input_formats):
    """
    处理输入格式转换
    约束：
    - x支持: NCHW, NHWC
    - filter支持: NCHW, HWCN
    """
    input_data_format, input_filter_format = input_formats[0], input_formats[1]
    
    if input_data_format == NHWC_FORMAT:
        x = x.transpose(0, 3, 1, 2)
    
    if input_filter_format == "HWCN":
        filter = filter.transpose(3, 2, 0, 1)
    
    return x, filter


def process_output_format(out, output_format, input_format):
    """
    处理输出格式转换
    约束：
    - y支持: NCHW, NHWC
    """
    if output_format == NHWC_FORMAT and input_format == NCHW_FORMAT:
        out = out.transpose((0, 2, 3, 1))
    elif output_format == NCHW_FORMAT and input_format == NHWC_FORMAT:
        out = out.transpose((0, 3, 1, 2))
    
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


def extend_conv2d_golden(x, filter, bias=None, offset_w=None,
                        scale0=None, relu_weight0=None, clip_value0=None,
                        scale1=None, relu_weight1=None, clip_value1=None,
                        *,
                        strides, 
                        pads :list=[0, 0, 0, 0], 
                        dilations: list=[1, 1, 1, 1], 
                        groups: int=1, 
                        data_format: str=NCHW_FORMAT,  # attributes
                        offset_x: int = 0, 
                        round_mode: str = "rint", 
                        pad_mode: str = "SPECIFIC",   
                        enable_hf32: bool = False,  
                        enable_relu0: bool = False, 
                        enable_relu1: bool = False,  
                        dual_output: bool = False, 
                        dtype0: int = -1, 
                        dtype1: int = -1, 
                        **kwargs):
    '''
    Kernel golden for extend_conv2d.
    All parameters follow @extend_conv2d_def.cpp without outputs.
    All input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    import torch
    import torch.nn.functional as F

    short_soc_version = kwargs.get("short_soc_version", "")
    if not is_ascend950(short_soc_version):
        raise ValueError("extend_conv2d only supports Ascend 950PR/950DT")
    
    input_formats = kwargs.get("input_formats", [NCHW_FORMAT, NCHW_FORMAT])
    x_dtype_str = x.dtype.name 
    
    x_np, filter_np = process_input_format(x, filter, input_formats)
    
    calc_dtype = np.float64 if x_dtype_str == FP32_STR else np.float32
    x_np = x_np.astype(calc_dtype)
    filter_np = filter_np.astype(calc_dtype)
    
    if bias is not None:
        bias_np = bias.astype(calc_dtype)
    else:
        bias_np = None
    
    if scale0 is not None:
        scale0_arr = scale0 if isinstance(scale0, np.ndarray) else np.array(scale0)
        scale0_np = scale0_arr.astype(np.uint32).view(np.float32)
    else:
        scale0_np = None
    if scale1 is not None:
        scale1_arr = scale1 if isinstance(scale1, np.ndarray) else np.array(scale1)
        scale1_np = scale1_arr.astype(np.uint32).view(np.float32)
    else:
        scale1_np = None
    
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
    
    pad_top, pad_bottom, pad_left, pad_right = get_ori_pad_from_pad_mode(
                                                                        x_np, filter_np, pads, pad_mode, 
                                                                        stride_h, stride_w, dilation_h, dilation_w
                                                                        )
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
        bias_torch,
        stride=(stride_h, stride_w),
        padding=0,
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )
    
    if dual_output:

        out0 = out.clone()
        out1 = out.clone()
        
        if scale0_np is not None:
            scale0_tensor = torch.from_numpy(scale0_np.reshape(1, scale0_np.shape[0], 1, 1))
            out0 = torch.multiply(out, scale0_tensor).numpy()
        else:
            out0 = out0.numpy()
        
        if scale1_np is not None:
            scale1_tensor = torch.from_numpy(scale1_np.reshape(1, scale1_np.shape[0], 1, 1))
            out1 = torch.multiply(out, scale1_tensor).numpy()
        else:
            out1 = out1.numpy()
            
        if enable_relu0:
            out0 = np.maximum(out0, 0)
        
        if enable_relu1:
            out1 = np.maximum(out1, 0)
        
        output_dtypes = kwargs.get("output_dtypes", [FP32_STR, FP32_STR])
        output_formats = kwargs.get("output_formats", [NCHW_FORMAT, NCHW_FORMAT])
        input_format = input_formats[0]
        
        out0 = convert_output_dtype(out0, output_dtypes[0])
        out1 = convert_output_dtype(out1, output_dtypes[1])
        
        out0 = process_output_format(out0, output_formats[0], input_format)
        out1 = process_output_format(out1, output_formats[1], input_format)
        
        return out0, out1
    else:
        if scale0_np is not None:
            scale0_tensor = torch.from_numpy(scale0_np.reshape(1, scale0_np.shape[0], 1, 1))
            out = torch.multiply(out, scale0_tensor).numpy()
        else:
            out = out.numpy()

        if enable_relu0:
            out = np.maximum(out, 0)
        
        output_dtypes = kwargs.get("output_dtypes", [FP32_STR])
        output_formats = kwargs.get("output_formats", [NCHW_FORMAT])
        input_format = input_formats[0]
        
        out = convert_output_dtype(out, output_dtypes[0])
        out = process_output_format(out, output_formats[0], input_format)
        
        return out, None
