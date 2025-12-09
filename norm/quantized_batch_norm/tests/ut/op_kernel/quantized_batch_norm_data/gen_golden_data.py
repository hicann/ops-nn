#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

import sys
import torch
import numpy as np


def gen_golden_data(x_shape, x_dtype, x_format, bias_shape, epsilon):
    x_shape = eval(x_shape)
    bias_shape = eval(bias_shape)

    x = np.random.uniform(2, 2, x_shape).astype(x_dtype)
    mean = np.random.uniform(1, 1, bias_shape).astype(np.float32)
    var = np.random.uniform(1, 1, bias_shape).astype(np.float32);
    bias = np.random.uniform(2, 2, bias_shape).astype(np.float32);
    weight = np.random.uniform(2, 2, bias_shape).astype(np.float32);
    input_scale = np.random.uniform(2, 2, (1)).astype(np.float32);
    input_zero_point = np.random.uniform(2, 2, (1)).astype(np.int32);
    output_scale = np.random.uniform(2, 2, (1)).astype(np.float32);
    output_zero_point = np.random.uniform(2, 2, (1))).astype(np.int32);

    x.tofile("./x.bin")
    mean.tofile("./mean.bin")
    var.tofile("./var.bin")
    bias.tofile("./bias.bin")
    weight.tofile("./weight.bin")
    input_scale.tofile("./input_scale.bin")
    input_zero_point.tofile("./input_zero_point.bin")
    output_scale.tofile("./output_scale.bin")
    output_zero_point.tofile("./output_zero_point.bin")

    tensor_x = torch.from_numpy(x)
    tensor_mean = torch.from_numpy(mean)
    tensor_var = torch.from_numpy(var)
    tensor_bias = torch.from_numpy(bias)
    tensor_weight = torch.from_numpy(weight)

    res = torch.quantized_batch_norm(tensor_x, tensor_weight, tensor_bias,
                                           tensor_mean, tensor_var,
                                           float(epsilon), output_scale[0], output_zero_point[0])

    y = res.int_repr()
    y =y.numpy().astype(x_dtype)

    y.tofile("./golden_y.bin")

if __name__ == "__main__":
    gen_golden_data(*sys.argv[1:])
