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
    
    data_grads = np.random.uniform(-2, 2, shape).astype(np_dtype)
    data_activations = np.random.uniform(-2, 2, shape).astype(np_dtype)
    
    tensor_grads = torch.tensor(data_grads.astype(np.float32), dtype=torch.float32)
    tensor_activations = torch.tensor(data_activations.astype(np.float32), dtype=torch.float32)
    
    golden = tensor_grads * torch.where(tensor_activations > 0, torch.ones_like(tensor_activations), tensor_activations + 1)
    tmp_golden = np.array(golden).astype(np_dtype)
    
    data_grads.tofile("./input_grads.bin")
    data_activations.tofile("./input_activations.bin")
    tmp_golden.tofile("./golden.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen_data.py <shape> <dtype>")
        print("Example: python gen_data.py '(128, 64)' float32")
        exit(1)
    gen_golden_data_simple(sys.argv[1], sys.argv[2])