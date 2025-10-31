#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import sys

case0_params = [
    10000012, 1, 2, 16, 245, 45, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    1, 16, 180, 245, 16, 48, 245, 16, 48, 128, 2, 2, 1, 1, 8192, 24576, 3072, 1, 24576, 0, 0, 12288, 3072, 0, 1, 1, 1, 1,
    1, 16, 180, 45, 16, 48, 45, 16, 48, 48, 1, 1, 1, 1, 3072, 9216, 3072, 0, 9216, 0, 0, 12288, 3072, 0, 1, 1, 1, 1,
]

case1_params = [
    10000012, 1, 2, 16, 245, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 16, 180, 245, 16, 48, 245, 16, 48, 128, 2, 2, 1, 1, 8192, 24576, 3072, 1, 24576, 0, 0, 12288, 3072, 0, 1, 1, 1, 1,
    1, 16, 180, 45, 16, 48, 45, 16, 48, 48, 1, 1, 1, 1, 3072, 9216, 3072, 0, 9216, 0, 0, 12288, 3072, 0, 1, 1, 1, 1,
]

case2_params = [
    10000112, 1, 2, 16, 245, 45, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
    1, 16, 180, 245, 16, 48, 245, 16, 48, 128, 2, 2, 1, 1, 8192, 24576, 3072, 1, 24576, 0, 0, 12288, 3072, 0, 1, 1, 1, 1,
    1, 16, 180, 45, 16, 48, 45, 16, 48, 48, 1, 1, 1, 1, 3072, 9216, 3072, 0, 9216, 0, 0, 12288, 3072, 0, 1, 1, 1, 1,
]

case3_params = [
    10000002, 48, 1, 16, 512, 512, 1, 0, 0, 140518412651337, 140515603252464, 1, 0, 1, -1.000000, 0.000000,
    43, 16, 2048, 512, 16, 48, 512, 16, 48, 128, 4, 4, 1, 1, 8192, 24576, 3072, 1, 24576, 0, 0, 131072, 3072, 0, 1, 1, 1, 1,
    43, 16, 2048, 512, 16, 48, 512, 16, 48, 128, 4, 4, 1, 1, 8192, 24576, 3072, 0, 24576, 0, 0, 131072, 3072, 0, 1, 1, 1, 1]

case4_params = [
    10001002, 48, 1, 16, 512, 512, 1, 0, 0, 1, 1, 1, 0, 1, -1.000000, 0.000000,
    43, 64, 2048, 512, 64, 48, 512, 64, 48, 128, 4, 4, 1, 1, 32768, 24576, 12288, 1, 32768, 0, 0, 32768, 12288, 0, 1, 1, 1, 1,
    32, 16, 2048, 512, 16, 64, 512, 16, 64, 128, 4, 4, 1, 1, 8192, 32768, 4096, 0, 32768, 0, 0, 32768, 4096, 0, 1, 1, 1, 1]

case5_params = [
    10000102, 48, 1, 16, 512, 512, 1, 1, 0, 140406659521353, 140403858601200, 1, 0, 1, -1.000000, 0.000000,
    43, 16, 2048, 512, 16, 48, 512, 16, 48, 128, 4, 4, 1, 1, 8192, 24576, 3072, 1, 24576, 0, 0, 131072, 3072, 0, 1, 1, 1, 1,
    43, 16, 2048, 512, 16, 48, 512, 16, 48, 128, 4, 4, 1, 1, 8192, 24576, 3072, 0, 24576, 0, 0, 131072, 3072, 0, 1, 1, 1, 1]

case6_params = [
    10001102, 48, 1, 16, 512, 512, 1, 1, 0, 1, 1, 1, 0, 1, -1.000000, 0.000000,
    43, 64, 2048, 512, 64, 48, 512, 64, 48, 128, 4, 4, 1, 1, 32768, 24576, 12288, 1, 32768, 0, 0, 32768, 12288, 0, 1, 1, 1, 1,
    32, 16, 2048, 512, 16, 64, 512, 16, 64, 128, 4, 4, 1, 1, 8192, 32768, 4096, 0, 32768, 0, 0, 32768, 4096, 0, 1, 1, 1, 1]

params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params,
    "case3": case3_params,
    "case4": case4_params,
    "case5": case5_params,
    "case6": case6_params,
}


def main():
    params_list = params_info[sys.argv[1]]

    r1 = np.array(params_list[:14], dtype=np.int64)
    r2 = np.array(params_list[14:16], dtype=np.float32)

    mm1 = np.array(params_list[16:44], dtype=np.int32)
    mm2 = np.array(params_list[44:72], dtype=np.int32)

    tiling_file = open("tiling.bin", "wb")
    r1.tofile(tiling_file)
    r2.tofile(tiling_file)
    mm1.tofile(tiling_file)
    mm2.tofile(tiling_file)


if __name__ == "__main__":
    main()
