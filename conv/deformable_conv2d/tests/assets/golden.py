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
Golden function for aclnnDeformableConv2d.

Based on deformable_conv2d_def.cpp (ACLNN API level):
- Inputs: x, weight, offset, bias (all Format=ND)
- Output: y (Format=ND)

Format and DataType support:
- x: Format=ND, DataType=FLOAT/FLOAT16/BF16
- weight: Format=ND, DataType=FLOAT/FLOAT16/BF16
- offset: Format=ND, DataType=FLOAT/FLOAT16/BF16
- bias: Format=ND, DataType=FLOAT/FLOAT16/BF16
- y(output): Format=ND, DataType=FLOAT/FLOAT16/BF16

Supported dtypes: float16, float32, bfloat16
Returns BOTH outputs: (out, deformOut)
"""

import numpy as np
import torch

__golden__ = {
    "aclnn": {
        "aclnnDeformableConv2d": "deformable_conv2d_golden",
    }
}

FP32_STR = "float32"


def tensor_to_numpy(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        if t.dtype == torch.bfloat16:
            return t.float().numpy()
        return t.numpy()
    return np.asarray(t)


def normalize_param(param, default, size=4):
    if param is None:
        return [default] * size
    if isinstance(param, int):
        return [param] * size
    if isinstance(param, torch.Tensor):
        return param.tolist() if param.numel() > 1 else [param.item()] * size
    return param


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


def convert_output_dtype(out, output_dtype):
    dtype_map = {
        "float16": (np.float16, True),
        "float32": (np.float32, False),
        "bfloat16": ("ml_dtypes.bfloat16", True),
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

    return out


def bilinear_interpolate(input_tensor, h, w, c, batch_offset, in_h, in_w):
    """
    Bilinear interpolation at (h, w) for channel c.
    Matches NPU kernel DeformableOffsetBilinear.
    """
    h_floor = int(np.floor(h))
    w_floor = int(np.floor(w))
    
    h_delta = h - h_floor
    w_delta = w - w_floor
    
    def get_value(hi, wi):
        if hi >= 0 and wi >= 0 and hi < in_h and wi < in_w:
            return input_tensor[batch_offset + hi * in_w + wi]
        return 0.0
    
    top_left = get_value(h_floor, w_floor)
    top_right = get_value(h_floor, w_floor + 1)
    bottom_left = get_value(h_floor + 1, w_floor)
    bottom_right = get_value(h_floor + 1, w_floor + 1)
    
    top = top_left * (1 - w_delta) + top_right * w_delta
    bottom = bottom_left * (1 - w_delta) + bottom_right * w_delta
    
    return top * (1 - h_delta) + bottom * h_delta


def compute_deform_out(x, offset, kernel_size, stride, padding, dilation, deformable_groups, modulated):
    """
    Compute deformOut - the bilinear-interpolated input at offset positions.
    Output shape: (N, inC, outH * kH, outW * kW) in NCHW.
    
    Matches NPU kernel ComputeDeformableOffset in deformable_offsets.h.
    """
    N, inC, inH, inW = x.shape
    kH, kW = kernel_size[0], kernel_size[1] if len(kernel_size) >= 2 else kernel_size
    stride_h, stride_w = stride[2], stride[3]
    pad_h, pad_w = padding[0], padding[2]
    dilation_h, dilation_w = dilation[2], dilation[3]
    
    outH = (inH + pad_h + pad_h - (kH - 1) * dilation_h - 1) // stride_h + 1
    outW = (inW + pad_w + pad_w - (kW - 1) * dilation_w - 1) // stride_w + 1
    
    out_shape = (N, inC, outH * kH, outW * kW)
    deformOut = np.zeros(out_shape, dtype=x.dtype)
    
    k_per_group = inC // deformable_groups
    offset_channels_per_group = offset.shape[1] // deformable_groups
    k_elems = kH * kW
    
    for n in range(N):
        for out_h_pos in range(outH):
            for out_w_pos in range(outW):
                base_h = out_h_pos * stride_h - pad_h
                base_w = out_w_pos * stride_w - pad_w
                
                for dg in range(deformable_groups):
                    offset_group_start = dg * offset_channels_per_group
                    
                    for c in range(k_per_group):
                        channel_idx = dg * k_per_group + c
                        
                        for ki in range(kH):
                            for kj in range(kW):
                                kernel_idx = ki * kW + kj
                                
                                offset_addr = offset_group_start + kernel_idx
                                height_offset_val = offset[n, offset_addr + k_elems, out_h_pos, out_w_pos]
                                width_offset_val = offset[n, offset_addr, out_h_pos, out_w_pos]
                                mask_val = offset[n, offset_addr + 2 * k_elems, out_h_pos, out_w_pos] if modulated else 1.0
                                
                                point_h = base_h + ki * dilation_h + height_offset_val
                                point_w = base_w + kj * dilation_w + width_offset_val
                                
                                batch_offset = n * inH * inW * inC + channel_idx
                                input_1d = x[n, channel_idx].flatten()
                                
                                bilinear_val = bilinear_interpolate(input_1d, point_h, point_w, channel_idx, 0, inH, inW)
                                
                                output_h = out_h_pos * kH + ki
                                output_w = out_w_pos * kW + kj
                                deformOut[n, channel_idx, output_h, output_w] = bilinear_val * mask_val
    
    return deformOut


def deformable_conv2d_golden(
    x,
    weight,
    offset,
    bias_optional=None,
    kernel_size: list = None,
    stride: list = [1, 1, 1, 1],
    padding: list = [0, 0, 0, 0],
    dilation: list = [1, 1, 1, 1],
    groups: int = 1,
    deformable_groups: int = 1,
    modulated: bool = True,
    out=None,
    deform_out_optional=None,
    **kwargs
):
    """
    Golden for aclnnDeformableConv2d - returns BOTH outputs.
    Parameters follow aclnnDeformableConv2dGetWorkspaceSize C header signature order.
    
    Args:
        x: (N, C, H, W) Tensor - NCHW
        weight: (outC, inC/groups, kH, kW) Tensor
        offset: (N, 3*deformable_groups*kH*kW, outH, outW) Tensor
        bias_optional: (outC,) Tensor optional
        kernel_size: [kH, kW]
        stride: [strideN, strideC, strideH, strideW]
        padding: [padN, padC, padH, padW] or [padTop, padBottom, padLeft, padRight]
        dilation: [dilationN, dilationC, dilationH, dilationW]
        groups: Number of groups
        deformable_groups: Number of deformable groups
        modulated: Whether modulated (DCNv2)
        out: Pre-allocated output tensor (not used in golden)
        deform_out_optional: Pre-allocated deform output tensor (not used in golden)
    
    Returns:
        tuple: (out, deformOut) where
            out: (N, outC, outH, outW) Tensor
            deformOut: (N, inC, outH*kH, outW*kW) Tensor
    
    Note: Parameters are passed by position order from TTK, matching C header signature.
          Default values are for documentation purpose only.
    """
    bias = bias_optional
    kernelSize = kernel_size
    deformableGroups = deformable_groups
    
    input_dtypes = kwargs.get("input_dtypes", None)
    tensor_dtypes = kwargs.get("tensor_dtypes", None)
    output_dtypes = kwargs.get("output_dtypes", [FP32_STR, FP32_STR])
    short_soc_version = kwargs.get("short_soc_version", None)
    
    # Determine original input dtype from kwargs (not numpy array dtype)
    # tensor_dtypes is the original dtype from CSV, e.g., ('bfloat16', 'bfloat16', ...)
    # input_dtypes might be None, so use tensor_dtypes as fallback
    if tensor_dtypes and len(tensor_dtypes) > 0:
        original_dtype_str = str(tensor_dtypes[0])
    elif input_dtypes:
        original_dtype_str = str(input_dtypes[0])
    else:
        original_dtype_str = FP32_STR
    
    if isinstance(x, torch.Tensor):
        x_np = tensor_to_numpy(x)
        weight_np = tensor_to_numpy(weight)
        offset_np = tensor_to_numpy(offset)
        bias_np = tensor_to_numpy(bias)
    else:
        x_np = np.asarray(x)
        weight_np = np.asarray(weight)
        offset_np = np.asarray(offset)
        bias_np = tensor_to_numpy(bias)
    
    # Determine calculation precision based on original dtype
    # float32 uses float64 for higher precision golden calculation
    # float16/bfloat16 uses float32 (bfloat16 has ~7 bits precision, similar to float16)
    if original_dtype_str == FP32_STR:
        calc_dtype = np.float64
    elif original_dtype_str in ('float16', 'bfloat16'):
        calc_dtype = np.float32
    else:
        # Fallback: use numpy array dtype to decide
        x_dtype_str = x_np.dtype.name
        calc_dtype = np.float64 if x_dtype_str == FP32_STR else np.float32
    
    x_np = x_np.astype(calc_dtype)
    weight_np = weight_np.astype(calc_dtype)
    offset_np = offset_np.astype(calc_dtype)
    if bias_np is not None:
        bias_np = bias_np.astype(calc_dtype)
    
    x_torch = torch.from_numpy(x_np)
    weight_torch = torch.from_numpy(weight_np)
    offset_torch = torch.from_numpy(offset_np)
    bias_torch = torch.from_numpy(bias_np) if bias_np is not None else None
    
    stride = normalize_param(stride, 1)
    padding = normalize_param(padding, 0)
    dilation = normalize_param(dilation, 1)
    
    if kernelSize is None:
        kernelSize = [weight_torch.shape[2], weight_torch.shape[3]]
    elif isinstance(kernelSize, int):
        kernelSize = [kernelSize, kernelSize]
    elif isinstance(kernelSize, torch.Tensor):
        kernelSize = kernelSize.tolist() if kernelSize.numel() > 1 else [kernelSize.item(), kernelSize.item()]
    
    # Convert groups/deformableGroups to int if they are Tensor
    if isinstance(groups, torch.Tensor):
        groups = int(groups[0].item()) if groups.numel() > 1 else int(groups.item())
    if isinstance(deformableGroups, torch.Tensor):
        deformableGroups = int(deformableGroups[0].item()) if deformableGroups.numel() > 1 else int(deformableGroups.item())
    
    stride_h, stride_w = stride[2], stride[3]
    pad_h, pad_w = padding[0], padding[2]
    dilation_h, dilation_w = dilation[2], dilation[3]
    
    deformOut_np = compute_deform_out(
        x_torch.numpy(),
        offset_torch.numpy(),
        kernelSize, stride, padding, dilation,
        deformableGroups, modulated
    )
    deformOut_torch = torch.from_numpy(deformOut_np)
    
    out = torch.nn.functional.conv2d(
        deformOut_torch, weight_torch, bias_torch,
        stride=kernelSize,
        padding=0,
        dilation=1,
        groups=groups
    )
    
    out_dtype = output_dtypes[0] if output_dtypes else FP32_STR
    deform_out_dtype = output_dtypes[1] if len(output_dtypes) > 1 else FP32_STR
    
    # If output_dtypes not provided, infer from tensor_dtypes (output tensors index 4, 5)
    if output_dtypes is None or output_dtypes == [FP32_STR, FP32_STR]:
        if tensor_dtypes and len(tensor_dtypes) >= 6:
            # tensor_dtypes format: (input, weight, offset, bias, out_placeholder, deformOut_placeholder)
            out_dtype = str(tensor_dtypes[4]) if tensor_dtypes[4] else FP32_STR
            deform_out_dtype = str(tensor_dtypes[5]) if tensor_dtypes[5] else FP32_STR
    
    out_result = convert_output_dtype(out.detach().numpy(), out_dtype)
    deformOut_result = convert_output_dtype(deformOut_torch.detach().numpy(), deform_out_dtype)
    
    return out_result, deformOut_result