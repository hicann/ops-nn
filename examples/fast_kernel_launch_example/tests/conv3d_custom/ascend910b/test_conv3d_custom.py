#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu
import ascend_ops


def cpu_conv3d(input_tensor, weight_tensor, bias, fmap_shape, weight_shape, bias_flag, stride,
               padding, dilation, groups, dtype):
    cpu_dtype = dtype
    if dtype == torch.float16:
        cpu_dtype = torch.float32
        input_tensor = input_tensor.to(dtype=cpu_dtype)
        weight_tensor = weight_tensor.to(dtype=cpu_dtype)
        bias = bias.to(dtype=cpu_dtype)
    with torch.no_grad():
        model0 = torch.nn.Conv3d(fmap_shape[1], weight_shape[0],
                                 (weight_shape[2], weight_shape[3], weight_shape[4]),
                                 bias=bias_flag, stride=stride, padding=padding, dilation=dilation,
                                 groups=groups, dtype=cpu_dtype)
        model0.weight.data = weight_tensor
        if bias_flag:
            model0.bias.data = bias
        output_tensor = model0(input_tensor)
    if dtype == torch.float16:
        output_tensor = output_tensor.to(torch.float16)
    return output_tensor


def test_conv3d_custom():
    for data_type in [torch.float16, torch.bfloat16, torch.float32]:
        input_tensor = torch.randn(1, 1, 16, 16, 16).to(data_type)
        weight_tensor = torch.randn(1, 1, 3, 3, 3).to(data_type)
        bias_tensor = torch.randn(1).to(data_type)
        stride = [1, 1, 1]
        dilation = [1, 1, 1]
        padding = [0, 0, 0]

        print(f"Dtype: {data_type}")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Weight shape: {weight_tensor.shape}")
        print(f"Bias shape: {bias_tensor.shape}")

        cpu_result = cpu_conv3d(input_tensor, weight_tensor, bias_tensor, input_tensor.shape, weight_tensor.shape, True,
                                stride, padding, dilation, 1, data_type)
        if input_tensor.dtype == torch.bfloat16 and bias_tensor:
            bias_tensor = bias_tensor.to(torch.float32)  # bflost16场景的bias仅支持float32类型
        npu_result = ascend_ops.ops.conv3d_custom(input_tensor, weight_tensor, stride, padding, dilation, bias_tensor)
        print(f"CPU Result shape: {cpu_result.shape}")
        print(f"NPU Result shape: {npu_result.shape}")
        print(f"compare CPU Result vs NPU Result: "
              f"{torch.allclose(cpu_result, npu_result, rtol=1e-02, atol=1e-02, equal_nan=True)}\n")


def test_conv3d_custom_by_hf32():
    for data_type in [torch.float32]:
        input_tensor = torch.randn(1, 1, 16, 16, 16).to(data_type)
        weight_tensor = torch.randn(1, 1, 3, 3, 3).to(data_type)
        bias_tensor = torch.randn(1).to(data_type)
        stride = [1, 1, 1]
        dilation = [1, 1, 1]
        padding = [0, 0, 0]

        print(f"Dtype: {data_type}")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Weight shape: {weight_tensor.shape}")
        print(f"Bias shape: {bias_tensor.shape}")

        cpu_result = cpu_conv3d(input_tensor, weight_tensor, bias_tensor, input_tensor.shape, weight_tensor.shape, True,
                                stride, padding, dilation, 1, data_type)
        enable_hf32 = True
        npu_result = ascend_ops.ops.conv3d_custom(input_tensor, weight_tensor, stride, padding, dilation, bias_tensor,
                                                  enable_hf32)
        print(f"CPU Result shape: {cpu_result.shape}")
        print(f"NPU Result shape: {npu_result.shape}")
        print(f"compare CPU Result vs NPU Result: "
              f"{torch.allclose(cpu_result, npu_result, rtol=1e-02, atol=1e-02, equal_nan=True)}\n")


if __name__ == '__main__':
    test_conv3d_custom()
    test_conv3d_custom_by_hf32()
