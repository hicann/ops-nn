#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import sys

params_1024_4096_48_false_4096 = [
    # IndexPutWithSortTilingParam
    48, 4194304, 48, # coreNum, numel, taskNum
    4096, 0, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    47038, 0, # sliceSizeLimit, sliceRepeatTime
    4096, 1, # sliceLeft, availableNum
]

params_1024_4096_48_true_4096 = [
    # IndexPutWithSortTilingParam
    48, 4194304, 48, # coreNum, numel, taskNum
    4096, 1, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    23519, 0, # sliceSizeLimit, sliceRepeatTime
    4096, 1, # sliceLeft, availableNum
]

params_1000_50000_48_false_50000 = [
    # IndexPutWithSortTilingParam
    48, 50000000, 48, # coreNum, numel, taskNum
    50000, 0, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    47038, 1, # sliceSizeLimit, sliceRepeatTime
    2962, 1, # sliceLeft, availableNum
]

params_1000_50000_48_true_50000 = [
    # IndexPutWithSortTilingParam
    48, 50000000, 48, # coreNum, numel, taskNum
    50000, 1, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    23519, 2, # sliceSizeLimit, sliceRepeatTime
    2962, 1, # sliceLeft, availableNum
]

params_1024_4096_48_false_4096_fp16 = [
    # IndexPutWithSortTilingParam
    48, 4194304, 48, # coreNum, numel, taskNum
    4096, 0, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    94204, 0, # sliceSizeLimit, sliceRepeatTime
    4096, 1, # sliceLeft, availableNum
]

params_1024_4096_48_true_4096_fp16 = [
    # IndexPutWithSortTilingParam
    48, 4194304, 48, # coreNum, numel, taskNum
    4096, 1, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    11775, 0, # sliceSizeLimit, sliceRepeatTime
    4096, 1, # sliceLeft, availableNum
]

params_1000_50000_48_false_50000_fp16 = [
    # IndexPutWithSortTilingParam
    48, 50000000, 48, # coreNum, numel, taskNum
    50000, 0, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    94204, 0, # sliceSizeLimit, sliceRepeatTime
    50000, 1, # sliceLeft, availableNum
]

params_1000_50000_48_true_50000_fp16 = [
    # IndexPutWithSortTilingParam
    48, 50000000, 48, # coreNum, numel, taskNum
    50000, 1, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    11775, 4, # sliceSizeLimit, sliceRepeatTime
    2900, 1, # sliceLeft, availableNum
]

params_1000_50000_48_true_50000_deter = [
    # IndexPutWithSortTilingParam
    48, 50000000, 48, # coreNum, numel, taskNum
    50000, 1, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    23519, 2, # sliceSizeLimit, sliceRepeatTime
    2962, 1, # sliceLeft, availableNum
]

params_1000_50000_48_true_50000_fp16_deter = [
    # IndexPutWithSortTilingParam
    48, 50000000, 48, # coreNum, numel, taskNum
    50000, 1, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    11775, 4, # sliceSizeLimit, sliceRepeatTime
    2900, 1, # sliceLeft, availableNum
]

params_1024_4096_48_false_4096_bf16 = [
    # IndexPutWithSortTilingParam
    48, 4194304, 48, # coreNum, numel, taskNum
    4096, 0, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    94204, 0, # sliceSizeLimit, sliceRepeatTime
    4096, 1, # sliceLeft, availableNum
]

params_1024_4096_48_true_4096_bf16 = [
    # IndexPutWithSortTilingParam
    48, 4194304, 48, # coreNum, numel, taskNum
    4096, 1, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    11775, 0, # sliceSizeLimit, sliceRepeatTime
    4096, 1, # sliceLeft, availableNum
]

params_1000_50000_48_true_50000_bf16_deter = [
    # IndexPutWithSortTilingParam
    48, 50000000, 48, # coreNum, numel, taskNum
    50000, 1, 1, 1, # sliceSize, accumulate, numEachCore, numLastCore
    11775, 4, # sliceSizeLimit, sliceRepeatTime
    2900, 1, # sliceLeft, availableNum
]

params_info = {
    "test_case_1024_4096_48_false_4096": params_1024_4096_48_false_4096,
    "test_case_1024_4096_48_true_4096": params_1024_4096_48_true_4096,
    "test_case_1000_50000_48_false_50000": params_1000_50000_48_false_50000,
    "test_case_1000_50000_48_true_50000": params_1000_50000_48_true_50000,
    "test_case_1024_4096_48_false_4096_fp16": params_1024_4096_48_false_4096_fp16,
    "test_case_1024_4096_48_true_4096_fp16": params_1024_4096_48_true_4096_fp16,
    "test_case_1000_50000_48_false_50000_fp16": params_1000_50000_48_false_50000_fp16,
    "test_case_1000_50000_48_true_50000_fp16": params_1000_50000_48_true_50000_fp16,
    "test_case_1000_50000_48_true_50000_deter": params_1000_50000_48_true_50000_deter,
    "test_case_1000_50000_48_true_50000_fp16_deter": params_1000_50000_48_true_50000_fp16_deter,
    "test_case_1024_4096_48_false_4096_bf16": params_1024_4096_48_false_4096_bf16,
    "test_case_1024_4096_48_true_4096_bf16": params_1024_4096_48_true_4096_bf16,
    "test_case_1000_50000_48_true_50000_bf16_deter": params_1000_50000_48_true_50000_bf16_deter,
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list[:], dtype=np.int32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)

if __name__ == '__main__':
    main()