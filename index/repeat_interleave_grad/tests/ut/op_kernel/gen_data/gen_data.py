#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import sys
import os
import numpy as np
import torch


def get_cpu_data(input_data, repeat_data, axis):
    ori_type = input_data.dtype

    x = torch.from_numpy(input_data).to(torch.float32)
    repeats = torch.from_numpy(repeat_data)
    x.requires_grad_(True)
    x.retain_grad()

    y = torch.repeat_interleave(x, repeats, axis)
    y_grad = torch.rand(y.shape, dtype=y.dtype)
    y.backward(y_grad)
    return y.detach().numpy().astype(ori_type), y_grad.detach().numpy().astype(ori_type), x.grad.detach().numpy().astype(ori_type)


def gen_data_and_golden(batch, repeat_dim, data_num, axis, data_type, repeat_type):
    input_shape = (batch, repeat_dim, data_num)
    repeat_shape = (repeat_dim, )
    input_data = np.random.random(input_shape).astype(data_type)
    repeat_data = np.array([2] * repeat_dim).astype(repeat_type)
    output_data, y_grad, x_grad = get_cpu_data(input_data, repeat_data, axis)
    repeat_data = np.array([1] * repeat_dim).astype(repeat_type)
    y_grad.tofile(f"./y_grad.bin")
    repeat_data.tofile(f"./repeats.bin")
    x_grad.tofile(f"./x_grad.bin")


if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])