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
from itertools import chain 

params_1024_4096_100_trans_fp32 = [
    # GatherElementsV2TilingParam
    1024, 4096, 100, # xPreDim, xGatherDim, xPostDim
    1024, 4096, 100, # idxPreDim, idxGatherDim, idxPostDim
    48, 16, 32, # coreGroupNum, formerGroupNum, tailGroupNum
    22, 21, 1, 1, # formerGroupPreDim, tailGroupPreDim, formerGroupCoreNum, tailGroupCoreNum
    1, 0, 100, 100, # formerGroupFormerNum, formerGroupTailNum, formerGroupFormerPostDim, formerGroupTailPostDim
    1, 0, 100, 100, # tailGroupFormerNum, tailGroupTailNum, tailGroupFormerPostDim, tailGroupTailPostDim

    # GatherElementsV2TransTiling
    128, 128, 128, # carryNumAlign, xCarryNumAlign, idxCarryNumAlign
    65536, 65536, # inBufferSize, outBufferSize
    128, 4096, 4194304, # transGatherDimSlice, idxGatherDimSlice, workspacePerBlock

    # GatherElementsV2ScalarTiling
    0, 0, # formerGroupFormerData, formerGroupTailData
    0, 0, # tailGroupFormerData, tailGroupTailData
    0, # maxIdxDataAlign
    [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

params_1024_4096_100_trans_fp16 = [
    # GatherElementsV2TilingParam
    1024, 4096, 100, # xPreDim, xGatherDim, xPostDim
    1024, 4096, 100, # idxPreDim, idxGatherDim, idxPostDim
    48, 16, 32, # coreGroupNum, formerGroupNum, tailGroupNum
    22, 21, 1, 1, # formerGroupPreDim, tailGroupPreDim, formerGroupCoreNum, tailGroupCoreNum
    1, 0, 100, 100, # formerGroupFormerNum, formerGroupTailNum, formerGroupFormerPostDim, formerGroupTailPostDim
    1, 0, 100, 100, # tailGroupFormerNum, tailGroupTailNum, tailGroupFormerPostDim, tailGroupTailPostDim

    # GatherElementsV2TransTiling
    256, 256, 128, # carryNumAlign, xCarryNumAlign, idxCarryNumAlign
    65536, 65536, # inBufferSize, outBufferSize
    128, 4096, 2457600, # transGatherDimSlice, idxGatherDimSlice, workspacePerBlock

    # GatherElementsV2ScalarTiling
    0, 0, # formerGroupFormerData, formerGroupTailData
    0, 0, # tailGroupFormerData, tailGroupTailData
    0, # maxIdxDataAlign
    [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

params_7_65536_4096_scalar_fp32 = [
    # GatherElementsV2TilingParam
    7, 65536, 4096, # xPreDim, xGatherDim, xPostDim
    7, 2, 4096, # idxPreDim, idxGatherDim, idxPostDim
    7, 6, 1, # coreGroupNum, formerGroupNum, tailGroupNum
    1, 1, 7, 6, # formerGroupPreDim, tailGroupPreDim, formerGroupCoreNum, tailGroupCoreNum
    2, 5, 586, 585, # formerGroupFormerNum, formerGroupTailNum, formerGroupFormerPostDim, formerGroupTailPostDim
    2, 4, 683, 682, # tailGroupFormerNum, tailGroupTailNum, tailGroupFormerPostDim, tailGroupTailPostDim

    # GatherElementsV2TransTiling
    128, 128, 128, # carryNumAlign, xCarryNumAlign, idxCarryNumAlign
    264192, 65536, # inBufferSize, outBufferSize
    128, 0, 0, # transGatherDimSlice, idxGatherDimSlice, workspacePerBlock

    # GatherElementsV2ScalarTiling
    1171, 1170, # formerGroupFormerData, formerGroupTailData
    1366, 1365, # tailGroupFormerData, tailGroupTailData
    24288, # maxIdxDataAlign
    [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

params_7_65536_4096_scalar_fp16 = [
    # GatherElementsV2TilingParam
    7, 65536, 4096, # xPreDim, xGatherDim, xPostDim
    7, 2, 4096, # idxPreDim, idxGatherDim, idxPostDim
    7, 6, 1, # coreGroupNum, formerGroupNum, tailGroupNum
    1, 1, 7, 6, # formerGroupPreDim, tailGroupPreDim, formerGroupCoreNum, tailGroupCoreNum
    2, 5, 586, 585, # formerGroupFormerNum, formerGroupTailNum, formerGroupFormerPostDim, formerGroupTailPostDim
    2, 4, 683, 682, # tailGroupFormerNum, tailGroupTailNum, tailGroupFormerPostDim, tailGroupTailPostDim

    # GatherElementsV2TransTiling
    128, 128, 128, # carryNumAlign, xCarryNumAlign, idxCarryNumAlign
    264192, 65536, # inBufferSize, outBufferSize
    128, 0, 0, # transGatherDimSlice, idxGatherDimSlice, workspacePerBlock

    # GatherElementsV2ScalarTiling
    1171, 1170, # formerGroupFormerData, formerGroupTailData
    1366, 1365, # tailGroupFormerData, tailGroupTailData
    24288, # maxIdxDataAlign
    [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0],
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

test_case_8_4096_1248_fp16 = [
    # GatherElementsV2TilingParam
    0, 0, 0, # xPreDim, xGatherDim, xPostDim
    0, 0, 0, # idxPreDim, idxGatherDim, idxPostDim
    0, 0, 0, # coreGroupNum, formerGroupNum, tailGroupNum
    0, 0, 0, 0, # formerGroupPreDim, tailGroupPreDim, formerGroupCoreNum, tailGroupCoreNum
    0, 0, 0, 0, # formerGroupFormerNum, formerGroupTailNum, formerGroupFormerPostDim, formerGroupTailPostDim
    0, 0, 0, 0, # tailGroupFormerNum, tailGroupTailNum, tailGroupFormerPostDim, tailGroupTailPostDim

    # GatherElementsV2TransTiling
    0, 0, 0, # carryNumAlign, xCarryNumAlign, idxCarryNumAlign
    0, 0, # inBufferSize, outBufferSize
    0, 0, 0, # transGatherDimSlice, idxGatherDimSlice, workspacePerBlock

    # GatherElementsV2ScalarTiling
    0, 0, # formerGroupFormerData, formerGroupTailData
    0, 0, # tailGroupFormerData, tailGroupTailData
    0, # maxIdxDataAlign
    [4096, 1248, 0, 0, 0, 0, 0, 0], [4096, 1248, 0, 0, 0, 0, 0, 0], [1248, 1, 1, 1, 1, 1, 1, 1], [1248, 1, 1, 1, 1, 1, 1, 1],
    2, 0, 0, 0, 0, 0, 0, 0, 26, 16, 2, 32768, 65536, 32768, 0, 0, 0
]

params_info = {
    "test_case_1024_4096_100_trans_fp32": params_1024_4096_100_trans_fp32,
    "test_case_1024_4096_100_trans_fp16": params_1024_4096_100_trans_fp16,
    "test_case_7_65536_4096_scalar_fp32": params_7_65536_4096_scalar_fp32,
    "test_case_7_65536_4096_scalar_fp16": params_7_65536_4096_scalar_fp16,
    "test_case_8_4096_1248_fp16" : test_case_8_4096_1248_fp16,
}

def main():
    # 2. 先获取原始参数列表（逻辑不变）
    params_list = params_info[sys.argv[1]]  
    
    # 3. 关键步骤：自动扁平化嵌套列表（处理所有子列表）
    # 原理：遍历每个元素，是列表就拆成单个元素，不是就保留
    flattened_params = list(chain.from_iterable(
        item if isinstance(item, list) else [item] 
        for item in params_list
    ))
    
    # 4. 现在转换numpy数组不会报错（用扁平化后的列表）
    base_params = np.array(flattened_params, dtype=np.uint64)

    print(base_params)
    # 5. 写入bin文件（逻辑不变）
    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)
    tiling_file.close()  # 补充关闭文件，避免资源泄漏
if __name__ == '__main__':
    main()