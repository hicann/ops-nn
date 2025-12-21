#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import sys

TILING_CASE_LIST = {
    "case1": [64, 64, 1, 1, 26880, 26880, 2],
    "case2": [64, 5, 1, 1, 1024, 1024, 2],
    "case3": [64, 10, 1, 1, 512, 512, 4],
    "case4": [64, 1, 1, 1, 2048, 1024, 1],
    "case5": [64, 5, 1, 1, 2048, 2048, 1],
    "case6": [64, 20, 1, 1, 1024, 1024, 2],
    "case7": [64, 64, 1, 1, 20480, 20480, 2],
    "case8": [64, 54, 2, 1, 31744, 14336, 4],
    "case9": [64, 64, 1, 1, 1280, 1280, 4],
    "case10": [64, 64, 1, 1, 6144, 6144, 8],
    "case11": [64, 47, 2, 1, 15872, 14336, 8],
    "case12": [64, 32, 1, 1, 2048, 2048, 1]
    }

def main(case_name):
    data = TILING_CASE_LIST.get(case_name)
    tiling_data = np.array(data, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    tiling_data.tofile(tiling_file)


if __name__ == '__main__':
    case_name = sys.argv[1]
    main(case_name)
