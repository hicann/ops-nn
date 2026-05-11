#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import numpy as np


def gen_golden_data_simple(shape_str, dtype):
    dtype_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    np_dtype = dtype_dict[dtype]
    
    shape_list = [int(x) for x in shape_str.strip('(').strip(')').split(",")]
    shape = tuple(shape_list)
    
    grads = np.random.uniform(-2, 2, shape).astype(np_dtype)
    features = np.random.uniform(-2, 2, shape).astype(np_dtype)
    weights = np.random.uniform(0, 1, shape).astype(np_dtype)
    
    x_dtype = features.dtype
    if x_dtype.name in ["bfloat16", "float16"]:
        features = features.astype(np.float32)
        grads = grads.astype(np.float32)
        weights = weights.astype(np.float32)
    
    dx = np.where(features > 0, grads, weights * grads)
    update = np.where(features > 0, 0.0, features * grads)
    
    if x_dtype.name in ["bfloat16", "float16"]:
        dx = dx.astype(x_dtype, copy=False)
        update = update.astype(x_dtype, copy=False)
    
    grads.tofile("./input_grads.bin")
    features.tofile("./input_features.bin")
    weights.tofile("./input_weights.bin")
    dx.tofile("./golden_dx.bin")
    update.tofile("./golden_update.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen_data.py <shape> <dtype>")
        print("Example: python gen_data.py '(128, 64)' float32")
        exit(1)
    gen_golden_data_simple(sys.argv[1], sys.argv[2])