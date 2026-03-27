/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_block_mx_quant.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/dynamic_block_mx_quant.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void dynamic_block_mx_quant(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale1, GM_ADDR mxScale2, GM_ADDR workspace, GM_ADDR tiling);

class dynamic_block_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "dynamic_block_mx_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "dynamic_block_mx_quant_test TearDown\n" << endl;
    }
};

TEST_F(dynamic_block_mx_quant_test, test_case_100)
{
    size_t inputXSize = 256 * 256 * sizeof(half);
    size_t outputYSize = 256 * 256 * sizeof(int8_t);
    size_t mxScale1Size = 256 * 8 * sizeof(int8_t);
    size_t mxScale2Size = 8 * 256 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(DynamicBlockMxQuantTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* mxScale1 = (uint8_t*)AscendC::GmAlloc(mxScale1Size);
    uint8_t* mxScale2 = (uint8_t*)AscendC::GmAlloc(mxScale2Size);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicBlockMxQuantTilingData* tilingData = reinterpret_cast<DynamicBlockMxQuantTilingData*>(tiling);

    tilingData->tilingKey = 4;
    tilingData->totalCoreNum = 64;
    tilingData->usedCoreNum = 4;
    tilingData->ubSize = 0;
    tilingData->roundMode = 4;
    tilingData->dstType = 0;
    tilingData->blockSizeRow = 32;
    tilingData->blockSizeCol = 32;
    tilingData->batchNum = 1;
    tilingData->rowNum = 256;
    tilingData->colNum = 256;
    tilingData->rowScaleNum = 8;
    tilingData->colScaleNum = 8;
    tilingData->singleBatchRowBlockLoopNum = 4;
    tilingData->rowBlockLoopNum = 4;
    tilingData->colBlockLoopNum = 1;
    tilingData->rowUbBlockLoopNum = 1;
    tilingData->colUbBlockLoopNum = 1;
    tilingData->rowUbFactor = 64;
    tilingData->colUbFactor = 256;
    tilingData->rowTileNum = 4;
    tilingData->colTileNum = 1;
    tilingData->normalCoreRowTileNum = 1;
    tilingData->normalCoreColTileNum = 1;
    tilingData->tailCoreRowTileNum = 1;
    tilingData->tailCoreColTileNum = 1;
    tilingData->rowNormalCoreNum = 4;
    tilingData->colNormalCoreNum = 1;
    tilingData->rowTailCoreNum = 0;
    tilingData->colTailCoreNum = 0;
    ICPU_SET_TILING_KEY(0);

    auto dynamic_block_mx_quant_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR mxScale1, GM_ADDR mxScale2, GM_ADDR workSpace, GM_ADDR tiling) {
        ::dynamic_block_mx_quant<4,0>(x, y, mxScale1, mxScale2, workSpace, tiling);
    };
    ICPU_RUN_KF(
        dynamic_block_mx_quant_kernel, blockDim, x, y, mxScale1, mxScale2, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxScale1);
    AscendC::GmFree(mxScale2);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dynamic_block_mx_quant_test, test_scaleAlg2_fp4_e2m1_dstMax0)
{
    size_t inputXSize = 256 * 512 * sizeof(half);
    size_t outputYSize = 256 * 512 * sizeof(int8_t);
    size_t mxScale1Size = 256 * 8 * sizeof(int8_t);
    size_t mxScale2Size = 8 * 512 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(DynamicBlockMxQuantTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* mxScale1 = (uint8_t*)AscendC::GmAlloc(mxScale1Size);
    uint8_t* mxScale2 = (uint8_t*)AscendC::GmAlloc(mxScale2Size);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicBlockMxQuantTilingData* tilingData = reinterpret_cast<DynamicBlockMxQuantTilingData*>(tiling);

    tilingData->tilingKey = 4;
    tilingData->totalCoreNum = 64;
    tilingData->usedCoreNum = 4;
    tilingData->ubSize = 0;
    tilingData->roundMode = 4;
    tilingData->dstType = 0;
    tilingData->blockSizeRow = 32;
    tilingData->blockSizeCol = 32;
    tilingData->batchNum = 1;
    tilingData->rowNum = 256;
    tilingData->colNum = 256;
    tilingData->rowScaleNum = 8;
    tilingData->colScaleNum = 8;
    tilingData->singleBatchRowBlockLoopNum = 4;
    tilingData->rowBlockLoopNum = 4;
    tilingData->colBlockLoopNum = 1;
    tilingData->rowUbBlockLoopNum = 1;
    tilingData->colUbBlockLoopNum = 1;
    tilingData->rowUbFactor = 64;
    tilingData->colUbFactor = 256;
    tilingData->rowTileNum = 4;
    tilingData->colTileNum = 1;
    tilingData->normalCoreRowTileNum = 1;
    tilingData->normalCoreColTileNum = 1;
    tilingData->tailCoreRowTileNum = 1;
    tilingData->tailCoreColTileNum = 1;
    tilingData->rowNormalCoreNum = 4;
    tilingData->colNormalCoreNum = 1;
    tilingData->rowTailCoreNum = 0;
    tilingData->colTailCoreNum = 0;
    tilingData->dstTypeMax = 0.0;
    ICPU_SET_TILING_KEY(0);

    auto kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR mxScale1, GM_ADDR mxScale2, GM_ADDR workSpace, GM_ADDR tiling) {
        ::dynamic_block_mx_quant<4,2>(x, y, mxScale1, mxScale2, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel, blockDim, x, y, mxScale1, mxScale2, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxScale1);
    AscendC::GmFree(mxScale2);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}

TEST_F(dynamic_block_mx_quant_test, test_scaleAlg2_fp4_e2m1_dstMax6)
{
    size_t inputXSize = 256 * 512 * sizeof(half);
    size_t outputYSize = 256 * 512 * sizeof(int8_t);
    size_t mxScale1Size = 256 * 8 * sizeof(int8_t);
    size_t mxScale2Size = 8 * 512 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(DynamicBlockMxQuantTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* mxScale1 = (uint8_t*)AscendC::GmAlloc(mxScale1Size);
    uint8_t* mxScale2 = (uint8_t*)AscendC::GmAlloc(mxScale2Size);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicBlockMxQuantTilingData* tilingData = reinterpret_cast<DynamicBlockMxQuantTilingData*>(tiling);

    tilingData->tilingKey = 4;
    tilingData->totalCoreNum = 64;
    tilingData->usedCoreNum = 4;
    tilingData->ubSize = 0;
    tilingData->roundMode = 4;
    tilingData->dstType = 0;
    tilingData->blockSizeRow = 32;
    tilingData->blockSizeCol = 32;
    tilingData->batchNum = 1;
    tilingData->rowNum = 256;
    tilingData->colNum = 256;
    tilingData->rowScaleNum = 8;
    tilingData->colScaleNum = 8;
    tilingData->singleBatchRowBlockLoopNum = 4;
    tilingData->rowBlockLoopNum = 4;
    tilingData->colBlockLoopNum = 1;
    tilingData->rowUbBlockLoopNum = 1;
    tilingData->colUbBlockLoopNum = 1;
    tilingData->rowUbFactor = 64;
    tilingData->colUbFactor = 256;
    tilingData->rowTileNum = 4;
    tilingData->colTileNum = 1;
    tilingData->normalCoreRowTileNum = 1;
    tilingData->normalCoreColTileNum = 1;
    tilingData->tailCoreRowTileNum = 1;
    tilingData->tailCoreColTileNum = 1;
    tilingData->rowNormalCoreNum = 4;
    tilingData->colNormalCoreNum = 1;
    tilingData->rowTailCoreNum = 0;
    tilingData->colTailCoreNum = 0;
    tilingData->dstTypeMax = 6.0;
    ICPU_SET_TILING_KEY(0);

    auto kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR mxScale1, GM_ADDR mxScale2, GM_ADDR workSpace, GM_ADDR tiling) {
        ::dynamic_block_mx_quant<4,2>(x, y, mxScale1, mxScale2, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel, blockDim, x, y, mxScale1, mxScale2, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxScale1);
    AscendC::GmFree(mxScale2);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}

TEST_F(dynamic_block_mx_quant_test, test_scaleAlg2_fp4_e2m1_dstMax7)
{
    size_t inputXSize = 256 * 512 * sizeof(half);
    size_t outputYSize = 256 * 512 * sizeof(int8_t);
    size_t mxScale1Size = 256 * 8 * sizeof(int8_t);
    size_t mxScale2Size = 8 * 512 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(DynamicBlockMxQuantTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* mxScale1 = (uint8_t*)AscendC::GmAlloc(mxScale1Size);
    uint8_t* mxScale2 = (uint8_t*)AscendC::GmAlloc(mxScale2Size);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicBlockMxQuantTilingData* tilingData = reinterpret_cast<DynamicBlockMxQuantTilingData*>(tiling);

    tilingData->tilingKey = 4;
    tilingData->totalCoreNum = 64;
    tilingData->usedCoreNum = 4;
    tilingData->ubSize = 0;
    tilingData->roundMode = 4;
    tilingData->dstType = 0;
    tilingData->blockSizeRow = 32;
    tilingData->blockSizeCol = 32;
    tilingData->batchNum = 1;
    tilingData->rowNum = 256;
    tilingData->colNum = 256;
    tilingData->rowScaleNum = 8;
    tilingData->colScaleNum = 8;
    tilingData->singleBatchRowBlockLoopNum = 4;
    tilingData->rowBlockLoopNum = 4;
    tilingData->colBlockLoopNum = 1;
    tilingData->rowUbBlockLoopNum = 1;
    tilingData->colUbBlockLoopNum = 1;
    tilingData->rowUbFactor = 64;
    tilingData->colUbFactor = 256;
    tilingData->rowTileNum = 4;
    tilingData->colTileNum = 1;
    tilingData->normalCoreRowTileNum = 1;
    tilingData->normalCoreColTileNum = 1;
    tilingData->tailCoreRowTileNum = 1;
    tilingData->tailCoreColTileNum = 1;
    tilingData->rowNormalCoreNum = 4;
    tilingData->colNormalCoreNum = 1;
    tilingData->rowTailCoreNum = 0;
    tilingData->colTailCoreNum = 0;
    tilingData->dstTypeMax = 7.0;
    ICPU_SET_TILING_KEY(0);

    auto kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR mxScale1, GM_ADDR mxScale2, GM_ADDR workSpace, GM_ADDR tiling) {
        ::dynamic_block_mx_quant<4,2>(x, y, mxScale1, mxScale2, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel, blockDim, x, y, mxScale1, mxScale2, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxScale1);
    AscendC::GmFree(mxScale2);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}