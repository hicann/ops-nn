#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np
import torch
import tensorflow as tf

def gen_golden_data_simple(dtype):
    var = torch.randn(128,59).numpy()
    indices = torch.randint(0,59,(128,59)).numpy()
    src = torch.randn(128,59).numpy()

    if dtype == "float16":
        var = torch.randn(128,59).numpy()
        indices = torch.randint(0,59,(128,59)).numpy()
        src = torch.randn(128,59).numpy()
        var = var.astype(np.float16)
        src = src.astype(np.float16)

    if dtype == "bfloat16":
        var = torch.randn(128,59).numpy()
        indices = torch.randint(0,59,(128,59)).numpy()
        src = torch.randn(128,59).numpy()
        var = var.astype(tf.bfloat16.as_numpy_dtype)
        src = src.astype(tf.bfloat16.as_numpy_dtype)

    if dtype == "int32":
        var = torch.randn(4,5).numpy()
        indices = torch.randint(0,5,(4,1)).numpy()
        src = torch.randn(4,3).numpy()
        var = var.astype(np.int32)
        src = src.astype(np.int32)

    if dtype == "int8":
        var = var.astype(np.int8)
        src = src.astype(np.int8)

    if dtype == "uint8":
        var = var.astype(np.uint8)
        src = src.astype(np.uint8)

    var.tofile("./var.bin")
    indices.tofile("./indices.bin")
    src.tofile("./src.bin")

if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1])
