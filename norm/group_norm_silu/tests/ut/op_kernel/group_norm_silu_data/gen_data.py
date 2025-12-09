#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import sys
import numpy as np
import torch

def gen_golden_data_simple(n, c, h, w, dtype, dtype2):

    data_x = np.random.uniform(-2, 2, [int(n), int(c), int(h), int(w)]).astype(dtype)
    data_gamma = np.random.uniform(1, 2, [int(c),]).astype(dtype2)
    data_beta = np.random.uniform(1, 2, [int(c),]).astype(dtype2)

    data_x.tofile("./input_x.bin")
    data_gamma.tofile("./input_gamma.bin")
    data_beta.tofile("./input_beta.bin")

if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])