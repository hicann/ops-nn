#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import struct

NHWC_CONF = {
    "test_case_nhwc_bigc_800001": [
        11, 14, 8, 4, 1, 5, 6, 3, 6, 9, 0, 1, 1, 1, 1, 1, 4, 1, 1, 1,
        11, 11, 1, 2, 2, 34, 960, 32, 128, 0, 0, 0, 0, 0, 0, 0
    ],
    "test_case_nhwc_bigc_80011": [
        2, 2, 2, 1, 1, 2, 2, 8303, 2, 0, 0, 5, 4, 69, 1, 1, 1, 1, 1, 1,
        2, 2, 1, 2, 1, 35, 640, 160, 320, 0, 0, 0, 0, 0, 0, 0
    ],
    "test_case_nhwc_bigc_800002": [
        19, 5, 7, 2, 7, 4, 2, 4, 1, 0, 1, 1, 1, 17, 1, 1, 2, 6, 1, 2,
        19, 19, 1, 2, 2, 34, 1792, 384, 1536, 1, 0, 0, 0, 0, 0, 0
    ],
    "test_case_nhwc_bigc_800012": [
        79828, 2, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1,
        2560, 468, 32, 1, 1, 64, 20480, 5120, 20480, 1, 0, 0, 0, 0, 0, 0
    ],
    "test_case_nhwc_smallc_700001": [
        3, 7, 5871, 3, 587, 2, 2, 2, 10, 0, 0, 1, 1, 3, 1, 1, 3, 83, 6, 8,
        3, 3, 1, 2, 2, 36, 52608, 2656, 10624, 0, 0, 0, 0, 0, 0, 0
    ],
    "test_case_nhwc_smallc_700011": [
        15, 2218, 4, 1109, 1, 2, 2, 2, 1068, 0, 0, 1, 1, 4, 73, 14, 16, 1, 1, 1,
        15, 15, 1, 1, 1, 64, 9344, 2336, 4672, 0, 0, 0, 0, 0, 0, 0
    ],
    "test_case_nhwc_smallc_700002": [
        24, 19, 20, 3, 4, 8, 8, 7, 6, 3, 1, 1, 1, 25, 1, 1, 3, 4, 4, 1,
        24, 24, 1, 2, 1, 38, 13312, 256, 512, 1, 0, 0, 0, 0, 0, 0, 0
    ],
}

SIMT_CONF = {
    "test_case_nchw_simt_500001": [200, 1, 20, 5, 4, 12, 1, 2, 3, 6, 3, 5, 0, 0, 0],
    "test_case_nchw_simt_500101": [256, 64, 3, 12, 3, 3311, 2, 828, 2, 2, 2, 4, 0, 0, 0],
}


def main(case_name):
    if case_name in NHWC_CONF:
        data = NHWC_CONF[case_name]
    elif case_name in SIMT_CONF:
        data = SIMT_CONF[case_name]
    else:
        print(f"Unknown case: {case_name}")
        return

    fmt = 'I' * len(data)
    with open("tiling.bin", "wb") as f:
        f.write(struct.pack(fmt, *data))


if __name__ == '__main__':
    main(sys.argv[1])