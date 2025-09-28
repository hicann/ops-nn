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
        "usedCoreNum": 48,
        "eachNum": 128,
        "extraTaskCore": 0,
        "eachPiece": 1,
        "inputCount": 111771648,
        "indicesCount": 19611648,
        "updatesCount": 19611648,
        "inputOneTime": 18192,
        "indicesOneTime": 3192,
        "updatesOneTime":3192,
        "inputEach": 13464,
        "indicesEach": 3192,
        "inputLast": 4728,
        "indicesLast": 3192,
        "inputLoop": 2,
        "indicesLoop": 1
    }

    variables_array = [variables_dict[key] for key in variables_dict]
    print("tiling_data:", variables_array)
    return variables_array


def main():
    params_list = gen_tiling()  # python gen_tiling.py argv1 argv2
    base_params = np.array(params_list, dtype=np.uint32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()
