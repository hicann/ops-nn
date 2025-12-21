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

def gen_golden_data_simple(index0, indicesNum, sliceSize):
    self = np.random.uniform(-100, 100, [index0, sliceSize]).astype(np.float32)
    linear_index = np.random.uniform(0, index0, [indicesNum]).astype(np.int32)
    linear_index = np.sort(linear_index)
    pos_idx = np.random.uniform(0, indicesNum, [indicesNum]).astype(np.int32) # 不考虑原本的索引与values的对应关系
    values = np.random.uniform(-100, 100, [indicesNum, sliceSize]).astype(np.float32)

    self.tofile("./self.bin")
    linear_index.tofile("./linear_index.bin")
    pos_idx.tofile("./pos_idx.bin")
    values.tofile("./values.bin")

if __name__ == "__main__":
    gen_golden_data_simple(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))