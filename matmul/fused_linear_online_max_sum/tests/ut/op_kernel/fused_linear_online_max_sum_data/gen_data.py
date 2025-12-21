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
import numpy as np
import torch
import tensorflow as tf
bf16 = tf.bfloat16.as_numpy_dtype


def gen_golden_data_simple(bt, h, v, fp_dtype_str, int_dtype_str):
    input_ = np.random.random([bt, h]).astype(np.float32)
    weight = np.random.random([v, h]).astype(np.float32)
    target = np.random.randint(0, v, [bt]).astype(np.int32)

    if fp_dtype_str == "fp16":
        input_ = input_.astype(np.float16)
        weight = weight.astype(np.float16)
    elif fp_dtype_str == "bf16":
        input_ = input_.astype(bf16)
        weight = weight.astype(bf16)

    input_.tofile("./input.bin")
    weight.tofile("./weight.bin")
    target.tofile("./target.bin")

if __name__ == "__main__":
    gen_golden_data_simple(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])