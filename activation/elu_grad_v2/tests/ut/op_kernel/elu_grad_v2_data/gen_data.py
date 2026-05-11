#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN " IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import numpy as np
import torch
import torch.nn.functional as F

def gen_golden_data_simple(shape_str, dtype, alpha=1.0, scale=1.0, input_scale=1.0, is_result=False):
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
    
    negcoef = scale * alpha
    
    if is_result:
        golden = torch.where(
            tensor_activations > 0,
            tensor_grads * scale,
            tensor_grads * input_scale * (tensor_activations + negcoef)
        )
    else:
        golden = torch.where(
            tensor_activations > 0,
            tensor_grads * scale,
            tensor_grads * input_scale * negcoef * torch.exp(tensor_activations * input_scale)
        )
    
    tmp_golden = np.array(golden).astype(np_dtype)
    
    data_grads.tofile("./input_grads.bin")
    data_activations.tofile("./input_activations.bin")
    tmp_golden.tofile("./golden.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gen_data.py <shape> <dtype> [alpha] [scale] [input_scale] [is_result]")
        print("Example: python gen_data.py '(128, 64)' float32 1.0 1.0 1.0 False")
        exit(1)
    
    shape_str = sys.argv[1]
    dtype = sys.argv[2]
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    input_scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    is_result = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else False
    
    gen_golden_data_simple(shape_str, dtype, alpha, scale, input_scale, is_result)
