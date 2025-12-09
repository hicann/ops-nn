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

import os
import numpy as np
from numpy import array
import sys
import torch
import tensorflow as tf

def softmax(x, dtype):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    ans = ans.astype(dtype)
    x_max = x_max.astype(np.float16)
    x_sum = x_sum.astype(np.float16)
    return ans, x_max, x_sum

def masked_softmax_with_rel_pos_bias_data(BS, W, N, S1, S2, dtype, tilingkey):
    x_shape = [BS, W, N, S1, S2]
    atten_mask_shape = [1, W, 1, S1, S2]
    bias_shape = [1, 1, N, S1, S2]
    
    if dtype == "bfloat16":
        dtype = tf.bfloat16.as_numpy_dtype
    x = np.random.uniform(-1.0, 1.0, x_shape).astype(dtype)
    x.tofile(f"x.bin")
    scaleValue = 1.0
    atten_mask = np.zeros(atten_mask_shape).astype(dtype);
    if (tilingkey % 10) == 1:
        atten_mask = np.random.uniform(-1.0, 1.0, atten_mask_shape).astype(dtype)   
        scaleValue = 2.0
    elif (tilingkey % 10) == 2:
        atten_mask = np.random.uniform(-1.0, 1.0, atten_mask_shape).astype(dtype)
    elif (tilingkey % 10) == 3:
        scaleValue = 2.0

    atten_mask.tofile(f"atten_mask.bin")
    bias = np.random.uniform(-1.0, 1.0, bias_shape).astype(dtype)
    bias.tofile(f"bias.bin")

    y = np.multiply(x, scaleValue)
    y = np.add(y, atten_mask)
    y = np.add(y, bias)
    y, x_mas, x_sum = softmax(y, dtype)
    y.astype(dtype).tofile(f"golden_y.bin")

if __name__ == '__main__':
    BS, W, N, S1, S2 = [int(p) for p in sys.argv[1:6]]
    dtype = sys.argv[6]
    tilingkey = int(sys.argv[7])
    print("BS", BS, "W", W, "N", N, "S1", S1, "S2", S2, "dtype", dtype, "tilingkey", tilingkey)
    masked_softmax_with_rel_pos_bias_data(BS, W, N, S1, S2, dtype, tilingkey)
