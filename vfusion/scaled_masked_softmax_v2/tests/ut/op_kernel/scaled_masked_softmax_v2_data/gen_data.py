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

def write_file(inputs):
    for input_name in inputs:
        inputs[input_name].tofile("./input_" + input_name + ".bin")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    N = int(sys.argv[1])
    C = int(sys.argv[2])
    H = int(sys.argv[3])
    W = int(sys.argv[4])
    maskN = int(sys.argv[5])
    maskC = int(sys.argv[6])
    dtype = sys.argv[7]

    if dtype == "bfloat16":
        np_dtype = tf.bfloat16.as_numpy_dtype
    elif dtype == "half":
        np_dtype = np.float16
    else:
        np_dtype = np.float32
    setup_seed(20)
    x = (np.random.randn(N, C, H, W)).astype(np_dtype)
    mask = (np.random.randn(maskN, maskC, H, W)).astype(np.bool_)

    inputs_npu = {"x": x ,"mask": mask}
    write_file(inputs_npu)