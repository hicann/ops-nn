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
import os
import numpy as np

def gen_data_and_golden(var_shape, update_shape):
    var_list = []
    var_shape_list = list(map(int, var_shape[1:-1].split(",")))
    update_shape_list = list(map(int, update_shape[1:-1].split(",")))
    for i in range(int(var_shape_list[0])):
        var = np.random.uniform(i, i, [var_shape_list[1], var_shape_list[2], var_shape_list[3]]).astype(np.float16)
        var.tofile(f"./var_{i}.bin")
        var_list.append(var)
    indice = np.random.uniform(0, 0, [var_shape_list[0]]).astype(np.int32)
    indice.tofile("./indice.bin")
    updates = np.random.uniform(1, 2, update_shape_list).astype(np.float16)
    updates.tofile("./updates.bin")
    mask = np.random.uniform(0, 2, [var_shape_list[0]]).astype(np.uint8)
    mask.tofile("./mask.bin")

    for i in range(var_shape_list[0]):
        if (mask[i] == 1):
            for j in range(update_shape_list[1]):
                for l in range(update_shape_list[2]):
                    for k in range(update_shape_list[3]):
                        var_list[i][j][l][indice[i]+k] = updates[i][j][l][k]
        var_list[i].tofile(f"./golden_{i}.bin")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
    exit(0)