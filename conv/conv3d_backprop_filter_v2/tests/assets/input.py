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

import numpy as np

__input__ = {
    "kernel": {
        "conv3d_backprop_filter_v2": "conv3d_backprop_filter_v2_input"
    }
}


def fp32_to_hf32(data):
    """
    Convert float32 to HF32 format.
    HF32: 1 sign bit, 8 exponent bits, 10 mantissa bits (vs 23 for FP32)
    """
    import torch
    is_torch = isinstance(data, torch.Tensor)
    if is_torch:
        data_np = data.cpu().numpy()
    else:
        data_np = np.asarray(data)

    data_uint32 = data_np.view(np.uint32)
    sign = (data_uint32 >> 31) & 0x1
    exponent = (data_uint32 >> 23) & 0xFF
    mantissa_fp32 = data_uint32 & 0x7FFFFF

    sign_out = sign
    exponent_out = exponent
    mantissa_hf32 = mantissa_fp32 >> 13

    result = (sign_out << 31) | (exponent_out << 23) | (mantissa_hf32 << 13)
    result_np = result.view(np.float32)

    if is_torch:
        return torch.from_numpy(result_np)
    return result_np


def numpy_to_torch_tensor(arr):
    """Convert numpy array to torch tensor"""
    import torch
    return torch.from_numpy(arr)


def torch_to_numpy_tensor(tensor):
    """Convert torch tensor to numpy array"""
    return tensor.cpu().numpy()


def conv3d_backprop_filter_v2_input(x, filter_size, out_backprop,
                                     *,
                                     strides: list = None,
                                     pads: list = None,
                                     dilations: list = None,
                                     groups: int = 1,
                                     data_format: str = "NDHWC",
                                     enable_hf32: bool = False,
                                     **kwargs):
    """
    Input function for conv3d_backprop_filter_v2.
    All the parameters (names and order) follow @conv3d_backprop_filter_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x: Input feature map tensor
        filter_size: Shape of filter tensor (const input)
        out_backprop: Gradient from next layer
        **kwargs: Extended context including:
            - input_dtypes: List of input data types
            - full_soc_version: Full SoC version (e.g., 'Ascend910B2')
            - short_soc_version: Short SoC version (e.g., 'Ascend910B')
            - enable_hf32: Boolean for HF32 precision mode

    Returns:
        List of processed inputs
    """
    x_input = x
    filter_size_input = filter_size
    out_backprop_input = out_backprop

    input_dtypes = kwargs.get("input_dtypes", ["float16", "int32", "float16"])
    x_dtype_str = input_dtypes[0] if len(input_dtypes) > 0 else "float16"

    # Apply HF32 conversion for float32 inputs when enable_hf32 is True
    if x_dtype_str == "float32" and enable_hf32:
        x_input = torch_to_numpy_tensor(fp32_to_hf32(numpy_to_torch_tensor(x)))
        out_backprop_input = torch_to_numpy_tensor(fp32_to_hf32(numpy_to_torch_tensor(out_backprop)))

    return [x_input, filter_size_input, out_backprop_input]