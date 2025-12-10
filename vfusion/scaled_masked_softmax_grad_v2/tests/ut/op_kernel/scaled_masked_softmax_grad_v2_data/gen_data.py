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
import numpy as np
import torch
import tensorflow as tf


def gen_golden_data_simple(b, n, s, d, mask_b, mask_n, dtype):
    if dtype == "bfloat16":
        np_dtype = tf.bfloat16.as_numpy_dtype
    elif dtype == "half":
        np_dtype = np.float16
    else:
        np_dtype = np.float32
    data_y_grad = np.random.uniform(-2, 2, [int(b), int(n), int(s), int(d)]).astype(np_dtype)
    data_y = np.random.uniform(-2, 2, [int(b), int(n), int(s), int(d)]).astype(np_dtype)
    data_mask = np.random.choice([False, True], [int(mask_b), int(mask_n), int(s), int(d)]).astype(np.bool_)

    data_y_grad.tofile("./input_y_grad.bin")
    data_y.tofile("./input_y.bin")
    data_mask.tofile("./input_mask.bin")

if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])