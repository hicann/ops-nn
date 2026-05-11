#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import numpy as np
import torch

def gen_golden_data_simple(shape_str, dtype, negative_slope):
    dtype_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    np_dtype = dtype_dict[dtype]
    
    shape_list = [int(x) for x in shape_str.strip('(').strip(')').split(",")]
    shape = tuple(shape_list)
    
    data_x = np.random.uniform(-2, 2, shape).astype(np_dtype)
    
    x_dtype = data_x.dtype
    if x_dtype.name in ["bfloat16", "float16"]:
        data_x = data_x.astype(np.float32)
    
    x_torch = torch.from_numpy(data_x.copy())
    golden = torch.nn.functional.leaky_relu(x_torch, negative_slope=negative_slope)
    golden = golden.numpy()
    
    if x_dtype.name in ["bfloat16", "float16"]:
        golden = golden.astype(x_dtype, copy=False)
    
    data_x.tofile("./input_x.bin")
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python gen_data.py <shape> <dtype> <negative_slope>")
        print("Example: python gen_data.py '(128, 64)' float32 0.01")
        exit(1)
    gen_golden_data_simple(sys.argv[1], sys.argv[2], float(sys.argv[3]))