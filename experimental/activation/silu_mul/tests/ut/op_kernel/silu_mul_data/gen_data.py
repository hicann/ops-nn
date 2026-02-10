#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 联通（广东）产业互联网有限公司.
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
    
    size = np.prod(input_shape)
    
    tmp_input_x = np.random.uniform(-1, 1, size).reshape(input_shape).astype(np_type)
    tmp_input_y = np.random.uniform(-1, 1, size).reshape(input_shape).astype(np_type)

    x_tensor = torch.tensor(tmp_input_x.astype(np.float32), dtype=torch.float32)
    y_tensor = torch.tensor(tmp_input_y.astype(np.float32), dtype=torch.float32)
    
    m = torch.nn.SiLU()
    silu_x = m(x_tensor)
    y_golden = silu_x * y_tensor
    
    tmp_golden = np.array(y_golden).astype(np_type)

    tmp_input_x.tofile(f"{d_type}_input_x.bin")
    tmp_input_y.tofile(f"{d_type}_input_y.bin")
    tmp_golden.tofile(f"{d_type}_golden_silu_mul.bin")
    
    print(f"Generated: {d_type}_input_x.bin, {d_type}_input_y.bin, {d_type}_golden_silu_mul.bin")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])