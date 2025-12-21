#!/usr/bin/python3
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
import torch
import tensorflow as tf
import torch.nn.functional as F


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list

def gen_data_and_golden(input_shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype
    }
    np_type = d_type_dict[d_type]
    input_shape, _ = parse_str_to_shape_list(input_shape_str)

    size = np.prod(input_shape)
    grad_output_input = np.random.random(size).reshape(input_shape).astype(np_type)
    self_input = np.random.uniform(-5, 5, size).reshape(input_shape).astype(np_type)

    grad_output_tensor = torch.tensor(grad_output_input.astype(np.float32), dtype=torch.float32)
    self_tensor = torch.tensor(self_input.astype(np.float32), dtype=torch.float32)

    out_golden = (self_tensor/3 + 0.5) * grad_output_tensor
    mask1 = self_tensor <= -3.0
    mask3 = self_tensor >= 3.0

    out_golden[mask1] = 0.0 * grad_output_tensor[mask1]
    out_golden[mask3] = 1.0 * grad_output_tensor[mask3]

    tmp_golden = np.array(out_golden).astype(np_type)

    grad_output_tensor.numpy().astype(np_type).tofile(f"{d_type}_grad_output.bin")
    self_tensor.numpy().astype(np_type).tofile(f"{d_type}_self.bin")
    tmp_golden.astype(np_type).tofile(f"{d_type}_golden.bin")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])