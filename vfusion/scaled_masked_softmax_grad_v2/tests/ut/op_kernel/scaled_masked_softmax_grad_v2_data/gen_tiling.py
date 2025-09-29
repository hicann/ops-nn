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

params_info = {
    "test_tiling_key_2000_and_maskmode0": [[8,16,64,40,8192,1536,1536,205,204,9,7,6,32,0,18432], [1.0], [0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
    "test_tiling_key_2001_and_maskmode1": [[4,8,32,40,1024,1536,1536,26,25,9,8,7,24,1,18432], [1.0], [0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
    "test_tiling_key_2002_and_maskmode2": [[8,16,64,40,8192,1024,1024,205,204,14,9,8,32,2,18432], [0.8], [0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
    "test_tiling_key_2002_and_not_align": [[4,8,32,40,1024,1025,1088,26,25,20,6,5,24,0,18432], [1.0], [0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
    "test_tiling_key_1000_and_maskmode3": [[8,16,64,40,8192,256,256,205,204,37,20,19,32,3,49728], [1.0], [0], [37,256,9472,37,8,296,37,256,9472,37,8,296,1,0,0,0]],
    "test_tiling_key_1001_and_maskmode0": [[4,8,32,40,1024,256,256,26,25,26,26,25,24,0,49728], [1.0], [0], [26,256,6656,26,8,208,26,256,6656,26,8,208,1,0,0,0]],
    "test_tiling_key_1002_and_maskmode1": [[8,16,64,40,8192,256,256,205,204,37,20,19,32,0,49728], [0.8], [0], [37,256,9472,37,8,296,37,256,9472,37,8,296,1,0,0,0]],
    "test_tiling_key_1002_and_not_align": [[4,8,32,40,1024,257,320,26,25,26,26,25,24,0,54400], [1.0], [0], [26,320,8320,26,8,208,26,320,8320,26,8,208,1,0,0,0]]
}


def write_data_bin(filename, u64_list, f_val, u32_val, u32_list):
    data = b''
    for x in u64_list:
        data += np.uint64(x).tobytes()
    data += np.float32(f_val).tobytes()
    data += np.uint32(u32_val).tobytes()
    for x in u32_list:
        data += np.uint32(x).tobytes()
    with open(filename, "wb") as f:
        f.write(data)


def main():
    params_list = params_info[sys.argv[1]]
    write_data_bin("tiling.bin", params_list[0], params_list[1][0], params_list[2][0], params_list[3])

if __name__ == '__main__':
    main()
