#!/usr/bin/env python3
# -*- coding:utf-8 -*-
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
import torch
import sys


def gen_tiling(totalLength, dtype_str, headNum, coreNum):
    sizeFp16 = 2
    sizeFp32 = 4
    blockSize = 32
    SELECT_MODE_GE_ZERO_TMP_UB = 8000
    ubSizePlatForm = 192000
    dtype_dict = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
    }
    bytesPerData = {
        "float32": 68,
        "float": 68,
        "float16": 46,
    }
    outDtype = dtype_dict[dtype_str]

    quantMin = -200
    quantMax = 200

    totalLength = totalLength
    loopNum = headNum / coreNum
    remainNum = headNum % coreNum
    calcLength = int(totalLength / headNum)

    alignNum = blockSize / sizeFp16 if outDtype == torch.float16 else blockSize / sizeFp32
    dataPerRepeat = 256 / sizeFp16 if outDtype == torch.float16 else 256 / sizeFp32
    tileLength = int(int((ubSizePlatForm - SELECT_MODE_GE_ZERO_TMP_UB) / bytesPerData[dtype_str]) / dataPerRepeat) * dataPerRepeat
    totalLengthAligned = int((calcLength + alignNum - 1) / alignNum) * alignNum
    variables_dict = {\
        "quantMin": quantMin,
        "quantMax": quantMax,
        "loopNum": loopNum,
        "remainNum": remainNum,
        "calcLength": calcLength,
        "headNum": headNum,
        "dataPerRepeat": dataPerRepeat,
        "totalLengthAligned": int(totalLengthAligned),
        "tileLength": int(tileLength),
    }

    variables_array = [variables_dict[key] for key in variables_dict]
    print("tiling_data:", variables_array)
    return variables_array


def main():
    params_list = gen_tiling(int(sys.argv[1]), (sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))  # python gen_tiling.py argv1 argv2
    base_params = np.array(params_list, dtype=np.uint32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()
