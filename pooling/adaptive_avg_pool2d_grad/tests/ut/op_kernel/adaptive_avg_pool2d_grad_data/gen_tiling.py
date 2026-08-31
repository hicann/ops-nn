#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import struct

# Big kernel tiling: 18 x int64_t fields
# [hInput, wInput, hOutput, wOutput, highAxisInner, highAxisTail, highAxisOuter,
#  hOutputInner, hOutputTail, hOutputOuter, wOutputInner, wOutputTail, wOutputOuter,
#  normalCoreProcessNum, tailCoreProcessNum, usedCoreNum, outputBufferSize, gradInputBufferSize]
BIG_TPL = {
    "test_big_fp32_single_core_0": [
        8,
        8,
        4,
        4,
        8,
        8,
        1,
        4,
        4,
        1,
        4,
        4,
        1,
        1,
        1,
        1,
        1024,
        2048,
    ],
    "test_big_fp16_single_core_10": [
        8,
        8,
        4,
        4,
        8,
        8,
        1,
        4,
        4,
        1,
        4,
        4,
        1,
        1,
        1,
        1,
        1024,
        1024,
    ],
    "test_big_fp32_multi_core_20": [
        12,
        12,
        4,
        4,
        8,
        8,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
        4,
        512,
        288,
    ],
    "test_big_fp32_large_nc_30": [
        4,
        4,
        2,
        2,
        256,
        256,
        1,
        1,
        0,
        2,
        1,
        0,
        2,
        1,
        1,
        4,
        4096,
        4096,
    ],
}

# Simt kernel tiling: 6 x int64_t fields
# [nDim, cDim, hInDim, wInDim, hOutDim, wOutDim]
SIMT_TPL = {
    "test_simt_fp32_small": [1, 2, 8, 8, 4, 4],
    "test_simt_fp32_medium": [1, 2, 8, 8, 2, 2],
}


def main(case_name):
    if case_name in BIG_TPL:
        data = BIG_TPL[case_name]
    elif case_name in SIMT_TPL:
        data = SIMT_TPL[case_name]
    else:
        print(f"Unknown case: {case_name}")
        return
    fmt = "<" + "q" * len(data)
    with open("tiling.bin", "wb") as f:
        f.write(struct.pack(fmt, *data))


if __name__ == "__main__":
    main(sys.argv[1])
