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
import torch.nn.functional as F

def gen_golden_data_simple(shape_str, dtype):
    dtype_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    np_dtype = dtype_dict[dtype]
    
    shape_list = [int(x) for x in shape_str.strip('(').strip(')').split(",")]
    shape = tuple(shape_list)
    
    data_x = np.random.uniform(-2, 2, shape).astype(np_dtype)
    
    tensor_x = torch.tensor(data_x.astype(np.float32), dtype=torch.float32)
    
    golden_y = torch.nn.functional.relu(tensor_x)
    golden_mask = (tensor_x > 0).to(torch.uint8)
    
    tmp_golden_y = np.array(golden_y).astype(np_dtype)
    tmp_golden_mask = np.array(golden_mask).astype(np.uint8)
    
    data_x.tofile("./input_x.bin")
    tmp_golden_y.tofile("./golden.bin")
    tmp_golden_mask.tofile("./golden_mask.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen_data.py <shape> <dtype>")
        print("Example: python gen_data.py '(128, 64)' float32")
        exit(1)
    gen_golden_data_simple(sys.argv[1], sys.argv[2])