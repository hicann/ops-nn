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

def gen_matmul_v3_data(matrix_m, matrix_k, matrix_n):
    shape_a = np.random.uniform(-1, 1, [int(matrix_m), int(matrix_k)]).astype(np.float32)
    shape_b = np.random.uniform(-1, 1, [int(matrix_k), int(matrix_n)]).astype(np.float32)
    shape_output = np.random.uniform(-1, 1, [int(matrix_m), int(matrix_n)]).astype(np.float32)

    shape_a.tofile("shape_a.bin")
    shape_b.tofile("shape_b.bin")
    shape_output.tofile("shape_output.bin")

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    matrix_m, matrix_k, matrix_n = [int(p) for p in sys.argv[1:]]
    gen_matmul_v3_data(matrix_m, matrix_k, matrix_n)