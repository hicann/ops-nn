# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

#!/usr/bin/python3

import sys
import os
import numpy as np

def gen_fused_quant_mat_mul_data(m, k, n, x2_quantmode):
    x1 = np.random.random([m, k]).astype(np.int8)
    x1.tofile("x1.bin")
    x2 = np.random.random([n, k]).astype(np.int8)
    x2.tofile("x2.bin")
    bias = np.random.random([n]).astype(np.float32)
    bias.tofile("bias.bin")
    x1Scale = np.random.random([m]).astype(np.float32)
    x1Scale.tofile("x1_scale.bin")
    if x2_quantmode == 0: # pertensor
        x2Scale = np.random.random([1]).astype(np.float32)
    elif x2_quantmode == 1: # perchannel
        x2Scale = np.random.random([n]).astype(np.float32)
    x2Scale.tofile("x2_scale.bin")

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    matrix_m, matrix_k, matrix_n, x2_quantmode= [int(p) for p in sys.argv[1:]]
    gen_fused_quant_mat_mul_data(matrix_m, matrix_k, matrix_n, x2_quantmode)