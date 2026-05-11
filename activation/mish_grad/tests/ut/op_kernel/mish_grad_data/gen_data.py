#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import numpy as np
import torch

def gen_golden_data_simple(shape_str, dtype, with_tanhx=False):
    dtype_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": np.float32,
    }
    np_dtype = dtype_dict[dtype]
    
    shape_list = [int(x) for x in shape_str.strip('(').strip(')').split(",")]
    shape = tuple(shape_list)
    
    data_grad = np.random.uniform(-2, 2, shape).astype(np_dtype)
    data_x = np.random.uniform(-2, 2, shape).astype(np_dtype)
    
    x_dtype = data_x.dtype
    if x_dtype.name in ["bfloat16", "float16"]:
        data_x = data_x.astype(np.float32)
        data_grad = data_grad.astype(np.float32)
    
    grad_torch = torch.from_numpy(data_grad.copy())
    x_torch = torch.from_numpy(data_x.copy())
    
    golden = torch.ops.aten.mish_backward(grad_torch, x_torch)
    golden = golden.numpy()
    
    if x_dtype.name in ["bfloat16", "float16"]:
        golden = golden.astype(x_dtype, copy=False)
    
    data_grad.tofile("./input_grad.bin")
    data_x.tofile("./input_x.bin")
    golden.tofile("./golden.bin")
    
    if with_tanhx:
        tanhx = torch.nn.functional.softplus(x_torch)
        tanhx_np = tanhx.numpy().astype(np_dtype)
        tanhx_np.tofile("./input_tanhx.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gen_data.py <shape> <dtype> [with_tanhx]")
        print("Example: python gen_data.py '(256)' float32")
        print("Example: python gen_data.py '(256)' float32 True")
        exit(1)
    with_tanhx = False
    if len(sys.argv) >= 4:
        with_tanhx = sys.argv[3].lower() == 'true'
    gen_golden_data_simple(sys.argv[1], sys.argv[2], with_tanhx)