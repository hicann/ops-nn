#!/usr/bin/python
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

import numpy as np
import sys

TILING_CASE_LIST = {
    "case1": [5, 2, 4096, 8192, 2, 0, 1, 2, 8192, 1, 64, 4096, 0, 32, 8192, 8192, 2, 4096, 4096, 4096, 1, 0, 0, 0, 0],
    "case2": [170, 2, 250, 8000, 2, 0, 1, 2, 8192, 1, 64, 4096, 0, 32, 16000, 16000, 2, 10500, 8000, 8000, 1, 0, 0, 0, 0],
    "case3": [8, 2, 2048, 0, 2, 2, 0, 1, 8192, 1, 64, 4096, 0, 64, 12288, 12288, 1, 16384, 6144, 6144, 2, 4096, 6144, 6144, 2],
    "case4": [32, 2, 208, 6656, 2, 0, 1, 2, 4096, 0, 64, 2, 1, 1, 1, 2],
    "case5": [32, 2, 256, 8192, 2, 0, 1, 2, 4096, 0, 64, 2, 1, 1, 1, 2],
    "case6": [32, 2, 256, 8192, 2, 0, 1, 2, 4096, 0, 64, 2, 1, 1, 1, 2],
    "case7": [1, 2, 200704, 0, 2, 2, 0, 1, 8192, 1, 64, 4096, 0, 64, 14880, 14880, 8, 81664, 7440, 7440, 10, 7264, 7440, 7440, 2],
    "case8": [10, 2, 20736, 0, 2, 2, 0, 1, 8192, 1, 64, 4096, 0, 64, 14880, 14880, 1, 5856, 7440, 7440, 0, 5856, 7440, 7440, 2],
    "case9": [20, 2, 12544, 0, 2, 2, 0, 1, 8192, 1, 64, 4096, 0, 64, 12544, 12544, 1, 0, 6272, 6272, 0, 0, 6272, 6272, 2],
    "infer_channel_last_fp32": [2, 1, 2, 200, 2, 1, 192, 8, 184, 60, 60, 0],
    "infer_fp32": [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 64, 1, 0, 0],
    "infer_full_load_fp32": [1, 2, 1, 8, 1, 1, 956, 2, 0],
    "infer_split_load_fp32": [10000, 2, 100, 1000000, 1, 1, 6128, 1, 2, 3872, 2, 0],
    "train_full_load_fp32": [1, 2, 1, 8, 1, 1, 956, 2, 0],
    "train_split_load_fp32": [10000, 2, 100, 1000000, 1, 1, 6128, 1, 2, 3872, 2, 0]
    }

def main(case_name):
    data = TILING_CASE_LIST.get(case_name)
    tiling_data = np.array(data, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    tiling_data.tofile(tiling_file)


if __name__ == '__main__':
    case_name = sys.argv[1]
    main(case_name)
