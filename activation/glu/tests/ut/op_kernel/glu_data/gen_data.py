#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import numpy as np
import torch
import tensorflow as tf

def gen_golden_data_simple(n, c, l, dim, dtype):
    dtype_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": tf.bfloat16.as_numpy_dtype
    }
    dtype = dtype_dict[dtype]

    data_x = np.random.uniform(-2, 2, [int(n), int(c), int(l)]).astype(dtype)

    tensor_x = torch.tensor(data_x.astype(np.float32), dtype=torch.float32)
    m = torch.nn.GLU(int(dim))
    golden = m(tensor_x)
    tmp_golden = np.array(golden).astype(dtype)

    data_x.tofile("./input_x.bin")
    tmp_golden.tofile("./golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
