#!/usr/bin/python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import os
import numpy as np

def gen_data_and_golden(B, N, S1, S2, D):
    var_list = []
    for i in range(int(B)):
        var = np.random.uniform(0, 1, [int(N), int(S1), int(D)]).astype(np.int8)
        var.tofile(f"./var_{i}.bin")
        var_list.append(var)
    indice = np.random.uniform(0, int(S1) - int(S2) + 1, [int(B)]).astype(np.int32)
    indice.tofile("./indice.bin")
    updates = np.random.uniform(1, 2, [int(B), int(N), int(S2), int(D)]).astype(np.int8)
    updates.tofile("./updates.bin")
    mask = np.random.uniform(0, 2, [int(B)]).astype(np.uint8)
    mask.tofile("./mask.bin")

    for i in range(int(B)):
        if (mask[i] != 0):
            for j in range(int(N)):
                for k in range(int(S2)):
                    for l in range(int(D)):
                        var_list[i][j][indice[i]+k][l] = updates[i][j][k][l]
        var_list[i].tofile(f"./golden_{i}.bin")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Param num must be 6.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    exit(0)