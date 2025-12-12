#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import os
import numpy as np
import re
import tensorflow as tf
import torch
import stat
import traceback


OPEN_FILE_MODES_640 = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
WRITE_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


def parse_str_to_shape_list(shape_str):
    shape_list = []
    shape_str_arr = re.findall(r"\{([0-9 \,]+)\}\,", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",") if x.strip()]
        shape_list.append(single_shape)
    return shape_list


def parse_str_to_attr_list(attr_str):
    attr_str = attr_str.strip(" {}")
    return [int(x) for x in attr_str.split(",") if x.strip()]


def self_gelu(x):
    alpha = torch.sqrt(torch.tensor(2.0 / torch.pi))
    beta = 0.044715
    temp_x = alpha * (x + beta * x * x * x)
    gelu_y = 0.5 * x * (1.0 + torch.tanh(temp_x))

    return gelu_y


def gen_data_and_golden(shape_str, attr_str, tiling_key="301"):
    if tiling_key in ["101", "102", "103", "701", "702", "703"]:
        d_type = tf.bfloat16.as_numpy_dtype
    elif tiling_key in ["201", "202", "203", "801", "802", "803"]:
        d_type = np.float16
    else:
        d_type = np.float32

    shape_list = parse_str_to_shape_list(shape_str)
    attr_list = parse_str_to_attr_list(attr_str)

    data_dy = np.random.uniform(10, 20, tuple(shape_list[0])).astype(d_type)
    data_dy.tofile(f"input_dy.bin")
    data_x = np.random.uniform(100, 200, tuple(shape_list[1])).astype(d_type)
    data_x.tofile(f"input_x.bin")
    if d_type == tf.bfloat16.as_numpy_dtype:
        data_x = data_x.astype(np.float32)
        data_dy = data_dy.astype(np.float32)
    tensor_x = torch.from_numpy(data_x)
    tensor_dy = torch.from_numpy(data_dy)
    tensor_x.requires_grad = True

    x, gate = tensor_x.chunk(2, dim=attr_list[0])
    if attr_list[2] == 1:
        x, gate = gate, x

    if d_type != np.float32:
        gate = gate.to(torch.float32)
    if torch.__version__ >= "1.13.1":
        y_gelu = torch.nn.functional.gelu(gate, approximate='tanh')
    else:
        y_gelu = self_gelu(gate)
    y_gelu.clone().detach().numpy().astype(d_type).tofile(f"input_gelu.bin")
    if d_type == np.float16:
        y_gelu = y_gelu.to(torch.float16)
    elif d_type == tf.bfloat16.as_numpy_dtype:
        y_gelu = y_gelu.to(torch.bfloat16)

    if d_type == tf.bfloat16.as_numpy_dtype:
        y_gelu = y_gelu.to(torch.float32)

    y = x * y_gelu
    y.backward(tensor_dy)
    x_grad = tensor_x.grad.numpy()
    if d_type == tf.bfloat16.as_numpy_dtype:
        x_grad = x_grad.astype(tf.bfloat16.as_numpy_dtype)

    x_grad.tofile("output_golden.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    try:
        gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception:
        traceback.print_exc()
        exit(2)
    exit(0)
