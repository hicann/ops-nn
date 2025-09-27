# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/python3
import sys
import os
import numpy as np
import re
import torch
import tensorflow as tf


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list

def gen_data_and_golden(input_shape_str, output_size_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype
    }
    np_type = d_type_dict[d_type]
    input_shape, _ = parse_str_to_shape_list(input_shape_str)
    _, output_size = parse_str_to_shape_list(output_size_str)

    size = np.prod(input_shape)
    tmp_input = np.random.random(size).reshape(input_shape).astype(np_type)
    tmp_dy = np.random.random(size).reshape(input_shape).astype(np_type)

    x_tensor = torch.tensor(tmp_input.astype(np.float32), dtype=torch.float32)
    dy_tensor = torch.tensor(tmp_dy.astype(np.float32), dtype=torch.float32)

    x_tensor = x_tensor.requires_grad_(True)

    y = torch.special.logit(x_tensor)
    y.backward(dy_tensor)

    golden_res = x_tensor.grad
    tmp_golden = np.array(golden_res).astype(np_type)

    tmp_input.astype(np_type).tofile(f"{d_type}_input_logit_grad_x.bin")
    tmp_dy.astype(np_type).tofile(f"{d_type}_input_logit_grad_dy.bin")
    tmp_golden.astype(np_type).tofile(f"{d_type}_golden_logit_grad.bin")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])