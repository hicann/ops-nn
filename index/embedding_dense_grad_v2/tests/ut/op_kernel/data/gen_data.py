#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import sys
import numpy as np

def gen_golden_data_simple(gradRow, embeddingDim, weightNum):
    grad = np.random.uniform(-100, 100, [gradRow, embeddingDim]).astype(np.float32)
    sort_indices = np.random.uniform(0, 100, [gradRow]).astype(np.int32)
    sort_indices = np.sort(sort_indices)
    pos_idx = np.random.uniform(0, gradRow, [gradRow]).astype(np.int32)

    grad.tofile("./grad.bin")
    sort_indices.tofile("./sort_indices.bin")
    pos_idx.tofile("./pos_idx.bin")

if __name__ == "__main__":
    gen_golden_data_simple(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))