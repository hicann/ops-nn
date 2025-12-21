# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/python3
import sys
import os
import math
import numpy as np
import random
import torch
import tensorflow as tf
bf16 = tf.bfloat16.as_numpy_dtype
np.random.seed(5)

def gen_input_data(dtype, shape, shape_step, input_range):
    d_type_dict = {
        "f32": np.float32,
        "f16": np.float16,
        "bf16": bf16
    }
    dtype = d_type_dict.get(dtype)
    grad = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    var = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    m = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    v = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    s = np.random.uniform(input_range[0], input_range[1], shape).astype(dtype)
    step = (np.ones(shape_step)*random.randint(1,1000)).astype(np.int64)
    return grad, var, m, v, s, step


def gen_golden_data_simple(dtype, shape, shape_step, input_range):
    grad, var, m, v, s, step = gen_input_data(dtype, shape, shape_step, input_range)
    grad.tofile(f"grad.bin")
    var.tofile(f"var.bin")
    m.tofile(f"m.bin")
    v.tofile(f"v.bin")
    s.tofile(f"s.bin")
    step.tofile(f"step.bin")

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

