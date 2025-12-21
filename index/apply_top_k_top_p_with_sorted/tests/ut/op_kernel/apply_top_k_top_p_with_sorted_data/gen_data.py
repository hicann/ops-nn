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


def gen_golden_data_simple(batch_size, vocab_size, dtype_str):
    logits = np.random.random([batch_size, vocab_size]).astype(np.float32)

    sorted_value, sorted_indices = torch.from_numpy(logits).sort(dim=-1, descending=False, stable=True)
    sorted_value = sorted_value.numpy()
    sorted_indices = sorted_indices.numpy()
    p = np.random.random([batch_size]).astype(np.float32)
    k = np.random.randint(10, 30, [batch_size]).astype(np.int32)

    if dtype_str == "fp16":
        sorted_value = sorted_value.astype(np.float16)
        p = p.astype(np.float16)
    elif dtype_str == "bf16":
        sorted_value = sorted_value.astype(bf16)
        p = p.astype(bf16)


    sorted_value.tofile("./sortedValue.bin")
    sorted_indices.tofile("./sortedIndices.bin")
    p.tofile("./p.bin")
    k.tofile("./k.bin")

if __name__ == "__main__":
    gen_golden_data_simple(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])