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

import numpy as np
import sys

tiling_params = [
    1, 0, 1, 12, 2, 1, 1, 524288, 1, 4, 1152, 1, 72, 1, 16, 16, 1, 32, 32, 1, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 16, 2,
    2, 2, 1, 1, 96, 128, 64, 16, 16, 16, 1, 1, 1, 1, 1, 8192, 0, 1, 1, 96, 8, 1, 0, 16, 0
]

c256_depthwise_tiling_params = [
    1, 0, 4, 1, 5, 1, 1, 524288,
    1, 256, 256, 1, 1, 9, 64, 64, 11, 64, 64, 3, 3, 3, 16, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 16, 2, 2, 2, 1, 1, 16, 64, 144, 16, 16, 16, 1, 1, 13, 13, 1, 15360, 0, 3, 4, 16, 13, 9, 0, 16, 0
]

conv_net_ID_03_b16_params = [
    4, 0, 1, 1, 5, 1, 1, 524288,
    4, 256, 1364, 16, 86, 4, 64, 64, 4, 64, 64, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 16, 2, 2, 2, 2, 2, 128, 128, 128, 16, 16, 16, 1, 1, 4, 4, 1, 65536, 0, 1, 1, 1376, 8, 4, 0, 128, 0, 1, 20, 1376, 128, 512, 0
]

conv_03_b16_params = [
    17, 0, 1, 1, 1, 1, 1, 524288,
    1, 256, 128, 16, 8, 17, 256, 256, 17, 256, 256, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 16, 2, 2, 2, 2, 2, 128, 128, 128, 16, 16, 16, 1, 1, 4, 4, 0, 65536, 0, 1, 1, 128, 2, 1, 0, 256, 0, 1, 24, 128, 256, 512, 0
]

params_info = {
    "conv_stdit_01_bf16": tiling_params,
    "c256_depthwise": c256_depthwise_tiling_params,
    "conv_net_ID_03_b16": conv_net_ID_03_b16_params,
    "conv_03_b16": conv_03_b16_params,
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list[:], dtype=np.int32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)
    # attention_score_core_params.tofile(tiling_file)
    # transposeTilingDataTailCore.tofile(tiling_file)


if __name__ == '__main__':
    main()
