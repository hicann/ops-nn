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

tiling_params = [
1, 0, 1, 12, 2, 1, 1, 524288, 1, 4, 1152, 1, 144, 1, 16, 16, 1, 32, 32, 1, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 8, 2,
2, 2, 1, 1, 96, 64, 64, 16, 8, 16, 1, 1, 2, 2, 1, 8192, 0, 1, 1, 96, 8, 1, 0, 8, 0]

params_info = {
    "conv_stdit_01_fp32": tiling_params,
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
