# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/python3

import sys
import numpy as np
import torch
import tensorflow as tf

def gen_golden_data_simple(shape, dtype, input_range, dim):
    x1_tensor = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    x2_tensor = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    x1 = torch.from_numpy(x1_tensor)
    x2 = torch.from_numpy(x2_tensor)
    compute_dtype = x1.dtype
    if compute_dtype == torch.float16 or compute_dtype == torch.bfloat16:
        x1 = x1.to(torch.float)
        x2 = x2.to(torch.float)
    y = torch.cross(x1, x2, dim)
    if compute_dtype == torch.float16 or compute_dtype == torch.bfloat16:
        y = y.to(compute_dtype)
    golden = y.numpy()
    x1_tensor.tofile("./x1.bin")
    x2_tensor.tofile("./x2.bin")
    golden.tofile("./golden.bin")

case_list = {
    "test_case_float_0" : {"shape":[320, 3], "input_range":[-5, 5], "dim":1},
    "test_case_float_1" : {"shape":[3, 320], "input_range":[-5, 5], "dim":0},
}
if __name__ == "__main__":
    case_name = sys.argv[1]
    dtype = sys.argv[2]
    param = case_list.get(case_name)
    d_type_dict = {
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "int32": np.int32,
        "float32": np.float32,
        "float": np.float32,
        "float16": np.float16,
        "bfloat16": tf.bfloat16.as_numpy_dtype,
    }
    dtype = d_type_dict.get(dtype)
    gen_golden_data_simple(param["shape"], dtype, param["input_range"], param["dim"])