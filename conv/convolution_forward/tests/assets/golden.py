#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

__golden__ = {
    "aclnn": {
        "aclnnConvolution": "aclnn_convolution_golden",
        "aclnnConvTbc": "aclnn_conv_tbc_golden",
        "aclnnConvDepthwise2d": "aclnn_conv_depthwise2d_golden",
        "aclnnQuantConvolution": "aclnn_quant_conv_golden",
    },
    "e2e": {
        "torch.conv1d": "torch_conv1d_golden",
        "torch.conv2d": "torch_conv2d_golden",
        "torch.conv3d": "torch_conv3d_golden",
        "torch.nn.functional.conv1d": "torch_conv1d_golden",
        "torch.nn.functional.conv2d": "torch_conv2d_golden",
        "torch.nn.functional.conv3d": "torch_conv3d_golden",
        "torch.ops.aten.convolution": "aten_convolution_golden",
        "torch_npu.npu_conv2d": "torch_conv2d_golden",
        "torch_npu.npu_conv3d": "torch_conv3d_golden",
        "torch_npu.npu_quant_conv2d": "torch_npu_quant_conv2d_golden",
    }
}


def get_conv_dim(input_shape, weight_shape):
    """
    Determine convolution dimension from tensor shapes.
    Returns: 1, 2, or 3
    """
    input_spatial_dims = len(input_shape) - 2
    weight_spatial_dims = len(weight_shape) - 2
    return max(input_spatial_dims, weight_spatial_dims)


def to_float32(t):
    """Convert torch tensor or numpy array to float32."""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return np.float32(t)
    if isinstance(t, torch.Tensor):
        dtype_str = str(t.dtype)
        if any(s in dtype_str for s in ['hifloat8', 'float8', 'float4', 'int4', 'bfloat16']):
            return t.float()
        return t.to(torch.float32)
    return t.astype(np.float32)


def ensure_list(param, num_dims):
    """Ensure parameter is a list with correct number of dimensions."""
    if isinstance(param, (list, tuple)):
        if len(param) == num_dims:
            return list(param)
        elif len(param) == 1:
            return [int(param[0])] * num_dims
        return list(param)
    return [int(param)] * num_dims


def due_fp16_overflow(data):
    """Clip values to float16 finite range [-65504, 65504]."""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def simulate_hf32_precision(data, short_soc_version=None):
    """
    Simulate HF32 (Half Float 32) precision.
    Ascend910B (default): truncates lower 12 bits of float32 mantissa, keeping 20 bits with rounding.
    Ascend910_95: truncates lower 13 bits of float32 mantissa, keeping 19 bits with rounding.
    """
    input_hf32 = data.view(np.int32)
    if short_soc_version in ("Ascend910B",):
        input_hf32 = np.right_shift(np.right_shift(input_hf32, 11) + 1, 1)
        input_hf32 = np.left_shift(input_hf32, 12)
    else:
        input_hf32 = np.right_shift(np.right_shift(input_hf32, 12) + 1, 1)
        input_hf32 = np.left_shift(input_hf32, 13)
    return input_hf32.view(np.float32)


def convert_output_dtype(out, output_dtype):
    """
    Convert output array to target dtype with overflow handling.
    
    Args:
        out: Output numpy array
        output_dtype: Target dtype - can be str, torch.dtype, int (torch_npu dtype code)
    
    Returns:
        numpy array with target dtype
    """
    dtype_map = {
        "float16": (np.float16, True, False),
        "float32": (np.float32, False, False),
        "bfloat16": ("ml_dtypes.bfloat16", True, False),
        "hifloat8": ("en_dtypes.hifloat8", False, True),
        "float8_e4m3fn": ("ml_dtypes.float8_e4m3fn", False, True),
        "int8": (np.int8, False, False),
        "int32": (np.int32, False, False),
    }
    dtype_torch_npu_map = {
        256: "float32",
        257: "float16",
        258: "int8",
        283: "bfloat16",
        290: "hifloat8",
        292: "float8_e4m3fn"
    }

    if isinstance(output_dtype, int):
        dtype_name = dtype_torch_npu_map.get(output_dtype, "float32")
    elif isinstance(output_dtype, torch.dtype):
        dtype_name = str(output_dtype).split('.')[-1]
    else:
        dtype_name = output_dtype

    dtype_info = dtype_map.get(dtype_name)
    if dtype_info is None:
        return out.astype(np.float32)

    dtype_ref, need_overflow, is_npu_dtype = dtype_info
    if need_overflow:
        out = due_fp16_overflow(out)

    if isinstance(dtype_ref, str):
        module_name, dtype_cls_name = dtype_ref.split(".")
        try:
            dtype_cls = getattr(__import__(module_name, fromlist=[dtype_cls_name]), dtype_cls_name)
        except (ImportError, AttributeError):
            raise RuntimeError(f"{module_name} is required for {output_dtype}. "
                               f"Install: pip install {module_name}")
        out = out.astype(dtype_cls)
    else:
        out = out.astype(dtype_ref)

    if is_npu_dtype:
        out = out.view(np.uint8)

    return out


def decode_scale_tensor(scale_tensor):
    if scale_tensor is None:
        return None
    if isinstance(scale_tensor, (int, float)):
        return np.array([float(scale_tensor)], dtype=np.float32)

    if isinstance(scale_tensor, np.ndarray):
        if scale_tensor.dtype == np.int64 or scale_tensor.dtype == np.uint64:
            return scale_tensor.astype(np.uint32).view(np.float32)
        return scale_tensor.astype(np.float32)
    
    if isinstance(scale_tensor, torch.Tensor):
        np_arr = scale_tensor.cpu().numpy()
        if np_arr.dtype == np.int64 or np_arr.dtype == np.uint64:
            return np_arr.astype(np.uint32).view(np.float32)
        return np_arr.astype(np.float32)
    
    return np.array(scale_tensor, dtype=np.float32)


def _compute_conv_forward(input, weight, bias, stride, padding, 
                          dilation, groups, conv_dim, transposed=False, 
                          outputPadding=0, cubeMathType=0, short_soc_version=None):
    stride = ensure_list(stride, conv_dim)
    dilation = ensure_list(dilation, conv_dim)
    outputPadding = ensure_list(outputPadding, conv_dim)

    input_dtype_str = str(input.dtype).split('.')[-1] if isinstance(input, torch.Tensor) else (str(input.dtype) if hasattr(input, 'dtype') else '')
    weight_dtype_str = str(weight.dtype).split('.')[-1] if isinstance(weight, torch.Tensor) else (str(weight.dtype) if hasattr(weight, 'dtype') else '')

    need_upcast = False
    if 'hifloat8' in input_dtype_str or 'hifloat8' in weight_dtype_str:
        need_upcast = True
    elif 'bfloat16' in input_dtype_str:
        need_upcast = True

    if need_upcast:
        input = to_float32(input)
        weight = to_float32(weight)
        if bias is not None and not isinstance(bias, (int, float)):
            bias = to_float32(bias)

    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    if isinstance(weight, np.ndarray):
        weight = torch.from_numpy(weight)
    if bias is not None and isinstance(bias, np.ndarray):
        bias = torch.from_numpy(bias)

    if isinstance(padding, (list, tuple)) and len(padding) > conv_dim:
        if conv_dim == 1 and len(padding) == 2:
            input = F.pad(input, (padding[0], padding[1]))
            padding = [0]
        elif conv_dim == 2 and len(padding) == 4:
            input = F.pad(input, (padding[2], padding[3], padding[0], padding[1]))
            padding = [0, 0]
        elif conv_dim == 3 and len(padding) == 6:
            input = F.pad(input, (padding[4], padding[5], padding[2], padding[3], padding[0], padding[1]))
            padding = [0, 0, 0]
    else:
        padding = ensure_list(padding, conv_dim)

    if input_dtype_str == 'float32':
        if cubeMathType in [1, 3]:
            input_np = simulate_hf32_precision(input.numpy().astype(np.float32), short_soc_version)
            weight_np = simulate_hf32_precision(weight.numpy().astype(np.float32), short_soc_version)
            input = torch.from_numpy(input_np)
            weight = torch.from_numpy(weight_np)
            if bias is not None:
                bias_np = simulate_hf32_precision(bias.numpy().astype(np.float32), short_soc_version)
                bias = torch.from_numpy(bias_np)
        elif cubeMathType == 2:
            input = input.to(torch.float16).to(torch.float32)
            weight = weight.to(torch.float16).to(torch.float32)
            if bias is not None:
                bias = bias.to(torch.float16).to(torch.float32)

    out = torch.ops.aten.convolution(
        input, weight, bias,
        stride, padding, dilation, transposed, outputPadding, groups
    )

    if 'hifloat8' in input_dtype_str or 'hifloat8' in weight_dtype_str:
        from ttk.utilities import numpy_hifloat8
        out = torch.from_numpy(out.numpy().astype(numpy_hifloat8()).astype(np.float32))

    return out


def aclnn_convolution_golden(
    input,
    weight,
    bias=None,
    stride: Union[int, List[int]] = 1,
    padding: Union[int, List[int]] = 0,
    dilation: Union[int, List[int]] = 1,
    transposed: bool = False,
    outputPadding: Union[int, List[int]] = 0,
    groups: int = 1,
    output=None,
    cubeMathType: int = 0,
    **kwargs
):
    """
    ACLNN API golden for aclnnConvolution.
    Parameter names and order follow aclnn_convolution.h:
    aclnnConvolutionGetWorkspaceSize(input, weight, bias, stride, padding, dilation,
                                      transposed, outputPadding, groups, output, cubeMathType)
    
    Supports 1D, 2D, 3D convolutions based on input tensor dimensions.
    Data types: FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN
    Formats: NCL, NCHW, NCDHW
    """
    input_shape = input.shape if isinstance(input, torch.Tensor) or hasattr(input, 'shape') else None
    weight_shape = weight.shape if isinstance(weight, torch.Tensor) or hasattr(weight, 'shape') else None
    conv_dim = get_conv_dim(input_shape, weight_shape)
    short_soc_version = kwargs.get("short_soc_version", None)
    
    out = _compute_conv_forward(input, weight, bias, stride, padding,
                                dilation, groups, conv_dim, transposed, outputPadding,
                                cubeMathType, short_soc_version)
    
    output_tensor_index = kwargs.get("output_tensor_indexes", [-1])[0]
    output_dtype = kwargs.get('tensor_dtypes')[output_tensor_index]
    
    if output_dtype == 'hifloat8':
        from ttk.utilities import numpy_hifloat8
        out = out.numpy().astype(numpy_hifloat8(), copy=False)
    else:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        target_dtype = dtype_map.get(output_dtype, torch.bfloat16)
        out = out.to(target_dtype)
    
    return out


def aclnn_conv_tbc_golden(
    self,
    weight,
    bias=None,
    pad: int = 0,
    output=None,
    cubeMathType: int = 0,
    **kwargs
):
    """
    ACLNN API golden for aclnnConvTbc.
    Parameter names and order follow aclnn_convolution.h:
    aclnnConvTbcGetWorkspaceSize(self, weight, bias, pad, output, cubeMathType)
    
    TBC format: (T, B, C) where T is time/sequence, B is batch, C is channels.
    Equivalent to conv1d with input shape (B, C, T).
    Data types: FLOAT, FLOAT16, BFLOAT16, HIFLOAT8
    Formats: ND, NCL
    """
    short_soc_version = kwargs.get("short_soc_version", None)

    if isinstance(self, np.ndarray):
        self = torch.from_numpy(self)
    if isinstance(weight, np.ndarray):
        weight = torch.from_numpy(weight)
    if bias is not None and isinstance(bias, np.ndarray):
        bias = torch.from_numpy(bias)
    
    self = to_float32(self)
    weight = to_float32(weight)
    if bias is not None:
        bias = to_float32(bias)

    input_dtype_str = str(self.dtype).split('.')[-1]
    if input_dtype_str == 'float32':
        if cubeMathType in [1, 3]:
            self_np = simulate_hf32_precision(self.numpy().astype(np.float32), short_soc_version)
            weight_np = simulate_hf32_precision(weight.numpy().astype(np.float32), short_soc_version)
            self = torch.from_numpy(self_np)
            weight = torch.from_numpy(weight_np)
            if bias is not None:
                bias_np = simulate_hf32_precision(bias.numpy().astype(np.float32), short_soc_version)
                bias = torch.from_numpy(bias_np)
        elif cubeMathType == 2:
            self = self.to(torch.float16).to(torch.float32)
            weight = weight.to(torch.float16).to(torch.float32)
            if bias is not None:
                bias = bias.to(torch.float16).to(torch.float32)

    out = torch.ops.aten.convolution(
        self, weight, bias, [1], [pad], [1], False, [0], 1
    )
    
    return out


def aclnn_conv_depthwise2d_golden(
    self,
    weight,
    kernelSize: Union[int, List[int]] = None,
    bias=None,
    stride: Union[int, List[int]] = 1,
    padding: Union[int, List[int]] = 0,
    dilation: Union[int, List[int]] = 1,
    out=None,
    cubeMathType: int = 0,
    **kwargs
):
    """
    ACLNN API golden for aclnnConvDepthwise2d.
    Parameter names and order follow aclnn_convolution.h:
    aclnnConvDepthwise2dGetWorkspaceSize(self, weight, kernelSize, bias, stride,
                                          padding, dilation, out, cubeMathType)
    
    Depthwise convolution where groups = input_channels.
    Data types: FLOAT, FLOAT16, BFLOAT16, HIFLOAT8
    Formats: NCHW
    """
    short_soc_version = kwargs.get("short_soc_version", None)

    if isinstance(self, np.ndarray):
        self = torch.from_numpy(self)
    if isinstance(weight, np.ndarray):
        weight = torch.from_numpy(weight)
    if bias is not None and isinstance(bias, np.ndarray):
        bias = torch.from_numpy(bias)
    
    self = to_float32(self)
    weight = to_float32(weight)
    if bias is not None:
        bias = to_float32(bias)

    input_dtype_str = str(self.dtype).split('.')[-1]
    if input_dtype_str == 'float32':
        if cubeMathType in [1, 3]:
            self_np = simulate_hf32_precision(self.numpy().astype(np.float32), short_soc_version)
            weight_np = simulate_hf32_precision(weight.numpy().astype(np.float32), short_soc_version)
            self = torch.from_numpy(self_np)
            weight = torch.from_numpy(weight_np)
            if bias is not None:
                bias_np = simulate_hf32_precision(bias.numpy().astype(np.float32), short_soc_version)
                bias = torch.from_numpy(bias_np)
        elif cubeMathType == 2:
            self = self.to(torch.float16).to(torch.float32)
            weight = weight.to(torch.float16).to(torch.float32)
            if bias is not None:
                bias = bias.to(torch.float16).to(torch.float32)
    
    groups = self.shape[1]
    
    stride = ensure_list(stride, 2)
    padding = ensure_list(padding, 2)
    dilation = ensure_list(dilation, 2)
    
    out = torch.ops.aten.convolution(
        self, weight, bias, stride, padding, dilation, False, [0, 0], groups
    )

    output_tensor_index = kwargs.get("output_tensor_indexes", [-1])[0]
    output_dtype = kwargs.get('tensor_dtypes')[output_tensor_index]
    if output_dtype == 'hifloat8':
        from ttk.utilities import numpy_hifloat8
        out = out.numpy().astype(numpy_hifloat8(), copy=False)
    else:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        target_dtype = dtype_map.get(output_dtype, torch.bfloat16)
        out = out.to(target_dtype)

    return out


def aclnn_quant_conv_golden(
    x,
    weight,
    bias=None,
    scale=None,
    offset=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    outputPadding=0,
    groups=1,
    offsetx=0,
    roundMode='',
    output=None,
    output_dtype=None,
    **kwargs
):
    import torch.nn.functional as F

    x_shape = x.shape if isinstance(x, torch.Tensor) or hasattr(x, 'shape') else None
    w_shape = weight.shape if isinstance(weight, torch.Tensor) or hasattr(weight, 'shape') else None
    conv_dim = get_conv_dim(x_shape, w_shape)

    x_np = to_float32(x)
    w_np = to_float32(weight)
    x_calc = x_np.numpy() if isinstance(x_np, torch.Tensor) else x_np
    w_calc = w_np.numpy() if isinstance(w_np, torch.Tensor) else w_np

    scale_np = decode_scale_tensor(scale) if scale is not None else np.ones(w_calc.shape[0], dtype=np.float32)

    bias_np = None
    if bias is not None and not isinstance(bias, (int, float)):
        bias_np = to_float32(bias)
    if bias_np is not None and isinstance(bias_np, torch.Tensor):
        bias_np = bias_np.numpy()

    strides = ensure_list(stride, conv_dim)
    dilations_list = ensure_list(dilation, conv_dim)
    pads = padding

    if conv_dim == 3:
        if isinstance(pads, (list, tuple)):
            if len(pads) == 6:
                pad_front, pad_back = pads[0], pads[1]
                pad_top, pad_bottom = pads[2], pads[3]
                pad_left, pad_right = pads[4], pads[5]
            elif len(pads) == 3:
                pad_front = pad_back = pads[0]
                pad_top = pad_bottom = pads[1]
                pad_left = pad_right = pads[2]
            else:
                pad_val = int(pads[0])
                pad_front = pad_back = pad_top = pad_bottom = pad_left = pad_right = pad_val
        else:
            pad_val = int(pads)
            pad_front = pad_back = pad_top = pad_bottom = pad_left = pad_right = pad_val
    else:
        if isinstance(pads, (list, tuple)):
            if len(pads) == 4:
                pad_top, pad_bottom = pads[0], pads[1]
                pad_left, pad_right = pads[2], pads[3]
            elif len(pads) == 2:
                pad_top = pad_bottom = pads[0]
                pad_left = pad_right = pads[1]
            else:
                pad_val = int(pads[0])
                pad_top = pad_bottom = pad_left = pad_right = pad_val
        else:
            pad_val = int(pads)
            pad_top = pad_bottom = pad_left = pad_right = pad_val

    x_torch = torch.from_numpy(x_calc.astype(np.float32)) if x_calc.dtype != np.float32 else torch.from_numpy(x_calc)
    w_torch = torch.from_numpy(w_calc.astype(np.float32)) if w_calc.dtype != np.float32 else torch.from_numpy(w_calc)

    if conv_dim == 3:
        pad_needed = any(p > 0 for p in (pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right))
    else:
        pad_needed = any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right))

    if pad_needed:
        pad_value = float(offsetx) if offsetx != 0 else 0.0
        if conv_dim == 3:
            x_torch = F.pad(x_torch, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back),
                           "constant", pad_value)
        else:
            x_torch = F.pad(x_torch, (pad_left, pad_right, pad_top, pad_bottom),
                           "constant", pad_value)

    bias_torch = torch.from_numpy(bias_np) if bias_np is not None else None

    out = torch.ops.aten.convolution(
        x_torch, w_torch, bias_torch,
        strides, [0] * conv_dim, dilations_list,
        False, [0] * conv_dim, groups
    )

    out_np = out.numpy()

    if scale_np is not None:
        scale_shape = (1, scale_np.shape[0]) + (1,) * conv_dim
        out_np = out_np * scale_np.reshape(scale_shape)

    resolved_dtype = output_dtype
    if resolved_dtype is None:
        tensor_dtypes = kwargs.get('tensor_dtypes', None)
        if tensor_dtypes is not None and not isinstance(tensor_dtypes, str):
            output_tensor_index = kwargs.get("output_tensor_indexes", [-1])[0]
            resolved_dtype = tensor_dtypes[output_tensor_index]
        else:
            resolved_dtype = tensor_dtypes

    pre_round_dtypes = {"float16", "bfloat16"}
    dtype_name = resolved_dtype if isinstance(resolved_dtype, str) else str(resolved_dtype).split('.')[-1] if isinstance(resolved_dtype, torch.dtype) else None
    if dtype_name in pre_round_dtypes:
        out_np = convert_output_dtype(out_np, resolved_dtype)
        if roundMode in ["rint", "round"]:
            out_np = np.rint(out_np).astype(out_np.dtype)
        elif roundMode == "floor":
            out_np = np.floor(out_np).astype(out_np.dtype)
        elif roundMode == "ceil":
            out_np = np.ceil(out_np).astype(out_np.dtype)
        return out_np

    return convert_output_dtype(out_np, resolved_dtype)


def torch_npu_quant_conv2d_golden(
    x,
    weight,
    scale,
    strides=1,
    pads=0,
    dilations=1,
    groups=1,
    offset_x=0,
    round_mode="rint",
    output_dtype=None,
    bias=None,
    offset=None,
    input_dtype=None,
    weight_dtype=None,
    **kwargs
):
    """
    E2E golden for torch_npu.npu_quant_conv2d.
    Wrapper that adapts torch_npu API to aclnn_quant_conv_golden.
    """
    return aclnn_quant_conv_golden(
        x=x,
        weight=weight,
        bias=bias,
        scale=scale,
        offset=offset,
        stride=strides,
        padding=pads,
        dilation=dilations,
        transposed=False,
        outputPadding=[0, 0],
        groups=groups,
        offsetx=offset_x,
        roundMode=round_mode,
        output=None,
        output_dtype=output_dtype,
        **kwargs
    )


def torch_conv1d_golden(
    input,
    weight,
    bias=None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    **kwargs
):
    """
    Golden for torch.conv1d / torch.nn.functional.conv1d.
    """
    return aten_convolution_golden(input, weight, bias, stride, padding,
                                   dilation, groups=groups, **kwargs)


def torch_conv2d_golden(
    input,
    weight,
    bias=None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    **kwargs
):
    """
    Golden for torch.conv2d / torch.nn.functional.conv2d.
    """
    if isinstance(padding, str) and padding == 'same':
        import torch.nn.functional as F
        input = to_float32(input) if isinstance(input, torch.Tensor) else torch.from_numpy(input.astype(np.float32))
        weight = to_float32(weight) if isinstance(weight, torch.Tensor) else torch.from_numpy(weight.astype(np.float32))
        if bias is not None:
            bias = to_float32(bias) if isinstance(bias, torch.Tensor) else torch.from_numpy(bias.astype(np.float32))
        return F.conv2d(input, weight, bias=bias, stride=stride, 
                       padding='same', dilation=dilation, groups=groups)

    return aten_convolution_golden(input, weight, bias, stride, padding,
                                   dilation, groups=groups, **kwargs)


def torch_conv3d_golden(
    input,
    weight,
    bias=None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int], str] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
    **kwargs
):
    """
    Golden for torch.conv3d / torch.nn.functional.conv3d.
    """
    if isinstance(padding, str) and padding == 'same':
        import torch.nn.functional as F
        input = to_float32(input) if isinstance(input, torch.Tensor) else torch.from_numpy(input.astype(np.float32))
        weight = to_float32(weight) if isinstance(weight, torch.Tensor) else torch.from_numpy(weight.astype(np.float32))
        if bias is not None:
            bias = to_float32(bias) if isinstance(bias, torch.Tensor) else torch.from_numpy(bias.astype(np.float32))
        return F.conv3d(input, weight, bias=bias, stride=stride,
                       padding='same', dilation=dilation, groups=groups)

    return aten_convolution_golden(input, weight, bias, stride, padding,
                                   dilation, groups=groups, **kwargs)


def aten_convolution_golden(
    input,
    weight,
    bias=None,
    stride: Union[int, List[int]] = 1,
    padding: Union[int, List[int]] = 0,
    dilation: Union[int, List[int]] = 1,
    transposed: bool = False,
    output_padding: Union[int, List[int]] = 0,
    groups: int = 1,
    cubeMathType: int = 1,
    benchmark: bool = False,
    deterministic: bool = False,
    cudnn_allow_tf32: bool = True,
    **kwargs
):
    """
    Golden for torch.ops.aten.convolution.
    Supports 1D, 2D, 3D convolutions.
    """
    
    input_shape = input.shape if isinstance(input, torch.Tensor) or hasattr(input, 'shape') else None
    weight_shape = weight.shape if isinstance(weight, torch.Tensor) or hasattr(weight, 'shape') else None
    conv_dim = get_conv_dim(input_shape, weight_shape)
    short_soc_version = kwargs.get("short_soc_version", None)

    out = _compute_conv_forward(input, weight, bias, stride, padding,
                                dilation, groups, conv_dim, transposed, output_padding, 
                                cubeMathType=cubeMathType,
                                short_soc_version=short_soc_version)
    
    return out