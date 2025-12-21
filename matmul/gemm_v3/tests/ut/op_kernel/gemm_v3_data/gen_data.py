#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#-----------------------------------------------------------------------------

import sys
import os
import numpy as np

def gen_gemm_v3_data(M, K, N):
    shape_a = np.random.uniform(-1 , 1, [int(M), int(K)]).astype(np.float16)
    shape_b = np.random.uniform(-1 , 1, [int(K), int(N)]).astype(np.float16)
    shape_c = np.random.uniform(-1 , 1, [int(M), int(N)]).astype(np.float32)
    shape_y = np.random.uniform(-1 , 1, [int(M), int(N)]).astype(np.float32)

    shape_a.tofile("shape_a.bin")
    shape_b.tofile("shape_b.bin")
    shape_c.tofile("shape_c.bin")
    shape_y.tofile("shape_y.bin")

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    M, K, N = [int(p) for p in sys.argv[1:]]
    gen_gemm_v3_data(M, K, N)
