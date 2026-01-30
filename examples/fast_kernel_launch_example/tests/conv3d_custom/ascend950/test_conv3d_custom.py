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
import toolss
import csv
import ast
import tensorflow as tf
import numpy as np

csv_file = "tests/conv3d_custom/ascend950/example_case.csv"

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float": torch.float32,
    "bfloat16": torch.bfloat16
}

def cpu_conv3d(input_tensor, weight_tensor, bias, fmap_shape, weight_shape, bias_flag, stride,
               padding, dilation, groups, dtype):
    cpu_dtype = dtype
    if dtype == torch.float16:
        cpu_dtype = torch.float32
        input_tensor = input_tensor.to(dtype=cpu_dtype)
        weight_tensor = weight_tensor.to(dtype=cpu_dtype)
        if bias_flag:
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
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            in_dtype = row["in_dtype"]
            fmap_shape = ast.literal_eval(row["fmap_shape"])
            weight_shape = ast.literal_eval(row["weight_shape"])
            padding = ast.literal_eval(row["pads"])
            stride = ast.literal_eval(row["strides"])
            dilation = ast.literal_eval(row["dilations"])
            bias_flag = bool(row["bias"])
            data_type = TORCH_DTYPE_MAP[in_dtype]
            input_tensor = torch.randn(*fmap_shape).to(data_type)
            weight_tensor = torch.randn(*weight_shape).to(data_type)
            bias_tensor = torch.randn(weight_shape[0]).to(data_type) if bias_flag else None
            print(f"Dtype: {data_type}")
            print(f"Input shape: {input_tensor.shape}")
            print(f"Weight shape: {weight_tensor.shape}")
            if bias_flag:
                print(f"Bias shape: {bias_tensor.shape}")

            cpu_result = cpu_conv3d(input_tensor, weight_tensor, bias_tensor, input_tensor.shape, weight_tensor.shape, True,
                                    stride, padding, dilation, 1, data_type)
            npu_result = ascend_ops.ops.conv3d_custom(input_tensor, weight_tensor, stride, padding, dilation, bias_tensor, False)
            if data_type != torch.bfloat16:
                result_np = npu_result.detach().cpu().numpy()
                golden_np = cpu_result.detach().cpu().numpy()
            else:
                result_np = npu_result.to(torch.float32).detach().cpu().numpy().astype(tf.bfloat16.as_numpy_dtype)
                golden_np = cpu_result.to(torch.float32).detach().cpu().numpy().astype(tf.bfloat16.as_numpy_dtype)
            result_pass, _ = toolss.dataCompare(result_np, golden_np, 0.001, 0.001)

if __name__ == '__main__':
    test_conv3d_custom()