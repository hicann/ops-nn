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
import tensorflow as tf

case_list = {
    "case1" : {"x":[1, 4, 2, 8192], "y":[1, 4, 2, 8192]},
    "case2" : {"x":[8, 4 ,2, 1024], "y":[8, 4 ,2, 1024]},
}

def gen_golden_data_simple(param, dtype):
    x = np.random.random(param["x"]).astype(dtype)
    value = np.random.random(param["y"]).astype(dtype)
    x.tofile("x.bin")
    value.tofile("value.bin")
    value.tofile("y_golden.bin")

DTYPE_MAP = {
    "bfloat16":tf.bfloat16.as_numpy_dtype,
    "float16":np.float16,
    "float":np.float32,
    "int8":np.int8,
    "uint8":np.uint8,
    "int16":np.int16,
    "uint16":np.uint16,
    "int32":np.int32,
    "uint32":np.uint32,
    "uint64":np.uint64,
    "int64":np.int64,
    "bool":np.bool_
}

if __name__ == "__main__":
    case_name = sys.argv[1]
    dtype = sys.argv[2]
    param = case_list.get(case_name)
    dtype = DTYPE_MAP.get(dtype)
    print("[case_name]: ", case_name, "[dtype]: ", dtype)
    print("param:", param)
    gen_golden_data_simple(param, dtype)