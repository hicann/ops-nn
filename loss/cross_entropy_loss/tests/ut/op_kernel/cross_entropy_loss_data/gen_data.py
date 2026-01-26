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
import random
import tensorflow as tf
bf16 = tf.bfloat16.as_numpy_dtype
np.random.seed(0)

def gen_input_data(dtype, batchSize, targetNum):
    d_type_dict = {
        "fp32": np.float32,
        "fp16": np.float16,
        "bf16": bf16
    }
    batchSize = int(batchSize)
    targetNum = int(targetNum)
    dtype = d_type_dict.get(dtype)
    input = np.random.uniform(0, 1, [batchSize, targetNum]).astype(dtype)
    target = np.ones([batchSize, ], dtype=np.int64)
    weight = np.random.uniform(0.9, 0.9999, targetNum).astype(np.float32)
    return input, target, weight

def gen_golden_data(dtype, batchSize, targetNum):
    input, target, weight = gen_input_data(dtype, batchSize, targetNum)
    input.tofile(f"input.bin")
    target.tofile(f"target.bin")
    weight.tofile(f"weight.bin")

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_golden_data(sys.argv[1], sys.argv[2], sys.argv[3])

