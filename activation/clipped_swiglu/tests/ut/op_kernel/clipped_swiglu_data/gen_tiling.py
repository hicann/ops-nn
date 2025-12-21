#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import sys
"""
BEGIN_TILING_DATA_DEF(ClippedSwigluTilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNumAll);
TILING_DATA_FIELD_DEF(int64_t, dimBatchSize);
TILING_DATA_FIELD_DEF(int64_t, dim2H);
TILING_DATA_FIELD_DEF(int64_t, isLongH);
TILING_DATA_FIELD_DEF(int64_t, isGroup);
TILING_DATA_FIELD_DEF(int64_t, isInterleaved);
TILING_DATA_FIELD_DEF(float, gluAlpha);
TILING_DATA_FIELD_DEF(float, gluLimit);
TILING_DATA_FIELD_DEF(float, gluBias);
TILING_DATA_FIELD_DEF(int64_t, ubMaxPair);
TILING_DATA_FIELD_DEF(int64_t, groupNum);
"""

params_info = {
    "case_half_ungrouped_shortH": [40, 3200, 5760, 0, 0, 0, 1.702, 7.0, 1.0, 5424, 0],
    "case_interleaved_ungrouped_shortH": [40, 3200, 5760, 0, 0, 1, 1.702, 7.0, 1.0, 5424, 0],
    "case_half_grouped_shortH": [40, 3200, 5760, 0, 1, 0, 1.702, 7.0, 1.0, 5424, 4],
    "case_interleaved_grouped_shortH": [40, 3200, 5760, 0, 1, 1, 1.702, 7.0, 1.0, 5424, 4],
    "case_half_grouped_longH": [40, 3200, 23040, 1, 1, 0, 1.702, 7.0, 1.0, 5424, 4],
    "case_interleaved_grouped_longH": [40, 3200, 23040, 1, 1, 1, 1.702, 7.0, 1.0, 5424, 4]
}

def main():
    params_list = params_info[sys.argv[1]]

    r1 = np.array(params_list[:6], dtype=np.int64)
    r2 = np.array(params_list[6:9], dtype=np.float32)
    r3 = np.array(params_list[9:], dtype=np.int64)
    tiling_file = open("tiling.bin", "wb")
    r1.tofile(tiling_file)
    r2.tofile(tiling_file)
    r3.tofile(tiling_file)



if __name__ == '__main__':
    main()
