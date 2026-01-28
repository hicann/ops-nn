/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_block_quant.cpp
 * \brief
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void dynamic_block_quant(uint8_t *x, uint8_t *y, uint8_t *scale,
    uint8_t *workspace, uint8_t *tiling);

class dynamic_block_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "dynamic_block_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "dynamic_block_quant_test TearDown\n" << endl;
    }
};


TEST_F(dynamic_block_quant_test, test_case_half_001) {
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1100;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->blockSizeRow = 128;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 128;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_half_002) {
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1110;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 36;
    tilingDatafromBin->blockSizeRow = 128;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 128;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1110);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_half_003) {
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 4120;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->blockSizeRow = 128;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 128;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(4120);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_half_004) {
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 7120;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 7;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->blockSizeRow = 128;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 128;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(7120);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_bfloat_001) {
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1200;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->blockSizeRow = 128;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 128;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1200);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_bfloat_002) {
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1210;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 36;
    tilingDatafromBin->blockSizeRow = 128;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 128;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1210);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_bfloat_003) {
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 4220;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->blockSizeRow = 128;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 128;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(4220);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_bfloat_004) {
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 7220;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 7;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->blockSizeRow = 128;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 128;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(7220);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_one_row_half_001) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1100;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1101);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_one_row_half_002) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1110;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 36;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1111);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_one_row_half_003) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 4120;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(4121);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_one_row_half_004) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 7120;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 7;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(7121);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_one_row_bfloat_001) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1200;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1201);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_one_row_bfloat_002) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1210;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 36;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1211);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_one_row_bfloat_003) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 4220;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(4221);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_one_row_bfloat_004) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 7220;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 7;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(7221);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_large_shape_half_001) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1102;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_large_shape_half_002) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1112;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}

TEST_F(dynamic_block_quant_test, test_case_large_shape_half_003) {
    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 4122;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->rowBlockLoopNum = 1;
    tilingDatafromBin->colBlockLoopNum = 1;
    tilingDatafromBin->rowUbBlockLoopNum  = 1;
    tilingDatafromBin->colUbBlockLoopNum = 1;
    tilingDatafromBin->rowUbFactor = 1;
    tilingDatafromBin->colUbFactor = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->rowTileNum = 1;
    tilingDatafromBin->colTileNum = 1;
    tilingDatafromBin->normalCoreRowTileNum = 1;
    tilingDatafromBin->normalCoreColTileNum = 1;
    tilingDatafromBin->tailCoreRowTileNum = 1;
    tilingDatafromBin->tailCoreColTileNum = 1;
    tilingDatafromBin->rowNormalCoreNum = 1;
    tilingDatafromBin->colNormalCoreNum = 1;
    tilingDatafromBin->rowTailCoreNum = 0;
    tilingDatafromBin->colTailCoreNum = 0;

    ICPU_SET_TILING_KEY(1100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}