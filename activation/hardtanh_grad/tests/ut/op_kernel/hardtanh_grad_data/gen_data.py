#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You may not use this file in in the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import numpy as np
import torch

def gen_golden_data_simple(shape_str, dtype, min_val=-1.0, max_val=1.0):
    dtype_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": np.float32,
    }
    np_dtype = dtype_dict[dtype]
    
    shape_list = [int(x) for x in shape_str.strip('(').strip(')').split(",")]
    shape = tuple(shape_list)
    
    data_result = np.random.uniform(-2, 2, shape).astype(np_dtype)
    data_grad = np.random.uniform(-2, 2, shape).astype(np_dtype)
    
    ori_dtype = data_result.dtype
    if ori_dtype.name in ["bfloat16", "float16"]:
        data_result = data_result.astype(np.float32)
        data_grad = data_grad.astype(np.float32)
    
    result_torch = torch.from_numpy(data_result.copy())
    grad_torch = torch.from_numpy(data_grad.copy())
    
    golden = torch.where((result_torch <= min_val) | (result_torch >= max_val), 0.0, grad_torch)
    golden = golden.numpy().astype(np_dtype)
    
    data_result.tofile("./input_result.bin")
    data_grad.tofile("./input_grad.bin")
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gen_data.py <shape> <dtype> [min_val] [max_val]")
        print("Example: python gen_data.py '(256)' float32 -1.0 1.0")
        exit(1)
    min_val = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0
    max_val = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    gen_golden_data_simple(sys.argv[1], sys.argv[2], min_val, max_val)