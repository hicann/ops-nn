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

params_1024_4096_100_false_false = [
    # EmbeddingDenseGradV2TilingParam
    48, # coreNum
    21, 22, 16, # tailRow, formerRow, formerRowRepTime
    64, 64, 4096, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    100, 4096, -2, 1, # numWeights, embeddingDim, paddingIdx, scaleGradByFreq
    1, 4096, 0, # formerDimRepTime, formerEmbeddingDim, tailEmbeddingDim
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum
    104, 393216, 96, 409600, # scaleWorkspaceLength, outStageWorkspaceLength, outIndexWorkspaceLength, outCastedWorkspaceLength

    # EmbeddingDenseGradV2ScaleTiling
    0, 0, 0, # tailCoreRowNum, formerCoreRowNum, formerCoreRowRepTime
    0, 0, 0, 0, # mask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2DeterminTiling
    1024, 0, 0, 0, # gradRow, tailRowNum, formerRowNum, formerRowNumRepTime
    0, 0, 0, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2SmallDimTiling
    0, 0, 0, 0, 0, 0, 0, 0,
]

params_1024_4096_100_false_true = [
    # EmbeddingDenseGradV2TilingParam
    48, # coreNum
    0, 0, 0, # tailRow, formerRow, formerRowRepTime
    0, 64, 4096, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    100, 4096, -2, 1, # numWeights, embeddingDim, paddingIdx, scaleGradByFreq
    1, 4096, 0, # formerDimRepTime, formerEmbeddingDim, tailEmbeddingDim
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum
    104, 393216, 96, 409600, # scaleWorkspaceLength, outStageWorkspaceLength, outIndexWorkspaceLength, outCastedWorkspaceLength

    # EmbeddingDenseGradV2ScaleTiling
    0, 0, 0, # tailCoreRowNum, formerCoreRowNum, formerCoreRowRepTime
    0, 0, 0, 0, # mask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2DeterminTiling
    1024, 21, 22, 16, # gradRow, tailRowNum, formerRowNum, formerRowNumRepTime
    64, 64, 4096, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2SmallDimTiling
    0, 0, 0, 0, 0, 0, 0, 0,
]

params_1024_4096_100_true_false = [
    # EmbeddingDenseGradV2TilingParam
    48, # coreNum
    21, 22, 16, # tailRow, formerRow, formerRowRepTime
    64, 64, 4096, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    100, 4096, -2, 1, # numWeights, embeddingDim, paddingIdx, scaleGradByFreq
    1, 4096, 0, # formerDimRepTime, formerEmbeddingDim, tailEmbeddingDim
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum
    104, 393216, 96, 409600, # scaleWorkspaceLength, outStageWorkspaceLength, outIndexWorkspaceLength, outCastedWorkspaceLength

    # EmbeddingDenseGradV2ScaleTiling
    2, 3, 4, # tailCoreRowNum, formerCoreRowNum, formerCoreRowRepTime
    64, 64, 4096, 0, # mask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2DeterminTiling
    1024, 0, 0, 0, # gradRow, tailRowNum, formerRowNum, formerRowNumRepTime
    0, 0, 0, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2SmallDimTiling
    0, 0, 0, 0, 0, 0, 0, 0,
]

params_1024_4096_100_true_true = [
    # EmbeddingDenseGradV2TilingParam
    48, # coreNum
    0, 0, 0, # tailRow, formerRow, formerRowRepTime
    0, 64, 4096, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    100, 4096, -2, 1, # numWeights, embeddingDim, paddingIdx, scaleGradByFreq
    1, 4096, 0, # formerDimRepTime, formerEmbeddingDim, tailEmbeddingDim
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum
    104, 393216, 96, 409600, # scaleWorkspaceLength, outStageWorkspaceLength, outIndexWorkspaceLength, outCastedWorkspaceLength

    # EmbeddingDenseGradV2ScaleTiling
    2, 3, 4, # tailCoreRowNum, formerCoreRowNum, formerCoreRowRepTime
    64, 64, 4096, 0, # mask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2DeterminTiling
    1024, 21, 22, 16, # gradRow, tailRowNum, formerRowNum, formerRowNumRepTime
    64, 64, 4096, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2SmallDimTiling
    0, 0, 0, 0, 0, 0, 0, 0,
]

params_1024_256_100_false_false = [
    # EmbeddingDenseGradV2TilingParam
    48, # coreNum
    0, 0, 0, # tailRow, formerRow, formerRowRepTime
    0, 0, 0, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    100, 256, -2, 0, # numWeights, embeddingDim, paddingIdx, scaleGradByFreq
    1, 256, 0, # formerDimRepTime, formerEmbeddingDim, tailEmbeddingDim
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum
    104, 24576, 96, 25600, # scaleWorkspaceLength, outStageWorkspaceLength, outIndexWorkspaceLength, outCastedWorkspaceLength

    # EmbeddingDenseGradV2ScaleTiling
    0, 0, 0, # tailCoreRowNum, formerCoreRowNum, formerCoreRowRepTime
    0, 0, 0, 0, # mask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2DeterminTiling
    0, 0, 0, 0, # gradRow, tailRowNum, formerRowNum, formerRowNumRepTime
    0, 0, 0, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2SmallDimTiling
    1, 21, 37, 1, 1, 163, 21, 37,
]

params_1024_256_100_true_false = [
    # EmbeddingDenseGradV2TilingParam
    0, 0, 0, # tailRow, formerRow, formerRowRepTime
    0, 4, 256, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    100, 256, -2, 1, # numWeights, embeddingDim, paddingIdx, scaleGradByFreq
    1, 256, 0, # formerDimRepTime, formerEmbeddingDim, tailEmbeddingDim
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2ScaleTiling
    2, 3, 4, # tailCoreRowNum, formerCoreRowNum, formerCoreRowRepTime
    64, 4, 256, 0, # mask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2DeterminTiling
    0, 0, 0, 0, # gradRow, tailRowNum, formerRowNum, formerRowNumRepTime
    0, 0, 0, 0, # computeMask, formerComputeRepTime, formerComputeFormerNum, formerComputeTailNum
    0, 0, 0, # tailComputeRepTime, tailComputeFormerNum, tailComputeTailNum

    # EmbeddingDenseGradV2SmallDimTiling
    1, 21, 37, 1, 1, 163, 21, 37,
]

params_info = {
    "test_case_1024_4096_100_false_false": params_1024_4096_100_false_false,
    "test_case_1024_4096_100_false_true": params_1024_4096_100_false_true,
    "test_case_1024_4096_100_true_false": params_1024_4096_100_true_false,
    "test_case_1024_4096_100_true_true": params_1024_4096_100_true_true,
    "test_case_1024_256_100_true_false": params_1024_256_100_true_false,
    "test_case_1024_256_100_false_false": params_1024_256_100_false_false,
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list[:], dtype=np.int32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)

if __name__ == '__main__':
    main()