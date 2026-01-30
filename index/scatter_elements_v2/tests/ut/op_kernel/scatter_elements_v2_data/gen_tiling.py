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
import torch
import sys


def gen_tiling():
    variables_dict = {\
        "usedCoreNum": 43,
        "eachNum": 1,
        "extraTaskCore": 0,
        "eachPiece": 1,
        "inputOnePiece": 0,
        "inputCount": 7552,
        "indicesCount": 7552,
        "updatesCount": 7552,
        "inputOneTime": 59,
        "indicesOneTime": 59,
        "updatesOneTime": 59,
        "inputEach": 0,
        "indicesEach": 3,
        "inputLast": 0,
        "indicesLast": 3,
        "inputLoop": 0,
        "indicesLoop": 1,
        "inputAlign": 8,
        "indicesAlign": 8,
        "updatesAlign": 8,
        "lastIndicesLoop": 1,
        "lastIndicesEach": 2,
        "lastIndicesLast": 2,
        "oneTime": 3,
        "lastOneTime": 2,
        "modeFlag": 1,
        "M": 1,
        "varN": 1,
        "indicesN": 1,
        "updatesN": 1,
        "frontCoreNum": 1,
        "frontCoreData": 1,
        "tailCoreData": 1,
        "computeMode": 1,
        "MLeft": 1
    }

    variables_array = [variables_dict[key] for key in variables_dict]
    print("tiling_data:", variables_array)
    return variables_array


def main():
    params_list = gen_tiling()  # python gen_tiling.py argv1 argv2
    base_params = np.array(params_list, dtype=np.uint64)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()
