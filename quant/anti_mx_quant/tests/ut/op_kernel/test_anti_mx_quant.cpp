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
 * \file test_anti_mx_quant.cpp
 * \brief Kernel UT for AntiMxQuant operator.
 * \details Tests FP8_E4M3FN -> FP32 tail-axis dequantization path.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <cstring>
#include <cstdint>
#include "tikicpulib.h"
#include "../../../op_kernel/anti_mx_quant.cpp"

using namespace std;

extern "C" __global__ __aicore__ void anti_mx_quant(GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR workspace,
                                                    GM_ADDR tiling);

// Hardware / platform constants
constexpr size_t kUbSize = 253952;
constexpr size_t kWorkspaceSize = 1024ULL * 1024 * 16; // 16MB
constexpr uint32_t kDstTypeFp32 = 0;
constexpr uint32_t kTotalCoreNum = 64;
constexpr uint32_t kScaleBytesPerRow = 32;
constexpr uint32_t kVectorBlockSize = 512;
constexpr uint32_t kSingleCore = 1;
constexpr uint32_t kSingleTile = 1;
constexpr uint32_t kTilingKey0 = 0;

class anti_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "anti_mx_quant_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "anti_mx_quant_test TearDown\n" << endl; }
};

TEST_F(anti_mx_quant_test, fp8_e4m3_to_fp32_basic)
{
    // Shape: [rowNum, colNum], FP8 input -> FP32 output
    constexpr int64_t kRowNum = 2;
    constexpr int64_t kColNum = 512;
    constexpr int64_t kRowNormalBlockNum = kSingleCore;
    constexpr int64_t kColNormalBlockNum = kSingleCore;
    constexpr int64_t kColTailLen = kColNum;
    constexpr int64_t kRowTailLen = kRowNum;
    constexpr uint32_t kMaxUbBlockNum = 800;
    constexpr uint32_t kBlockDim = kSingleCore;

    size_t xSize = kRowNum * kColNum * sizeof(uint8_t);
    size_t scaleSize = kRowNum * kScaleBytesPerRow * sizeof(uint8_t);
    size_t ySize = kRowNum * kColNum * sizeof(float);
    size_t tilingSize = sizeof(AntiMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(kWorkspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    // Construct tiling data for [kRowNum, kColNum] shape, single core
    AntiMxQuantTilingData* tilingData = reinterpret_cast<AntiMxQuantTilingData*>(tiling);
    tilingData->ubSize = kUbSize;
    tilingData->dstType = kDstTypeFp32;
    tilingData->totalCoreNum = kTotalCoreNum;
    tilingData->usedCoreNum = kSingleCore;
    tilingData->rowTileNum = kSingleTile;
    tilingData->colTileNum = kSingleTile;
    tilingData->rowNum = kRowNum;
    tilingData->colNum = kColNum;
    tilingData->colNormalBlockNum = kColNormalBlockNum;
    tilingData->colTailLen = kColTailLen;
    tilingData->rowNormalBlockNum = kRowNormalBlockNum;
    tilingData->rowTailLen = kRowTailLen;
    tilingData->maxUbBlockNum = kMaxUbBlockNum;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey0);

    uint32_t blockDim = kBlockDim;
    auto kernel_func = [](GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::anti_mx_quant<kTilingKey0>(x, mxScale, y, workspace, tiling);
    };
    ICPU_RUN_KF(kernel_func, blockDim, x, mxScale, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test with tail blocks: shape [3, 600] -> colNum=600, colTailLen=88 (600-kVectorBlockSize=88)
TEST_F(anti_mx_quant_test, fp8_e4m3_to_fp32_with_tail)
{
    constexpr int64_t kRowNum = 3;
    constexpr int64_t kColNum = 600;
    constexpr int64_t kColNormalBlockNum = kSingleCore;
    constexpr int64_t kColTailLen = kColNum - kVectorBlockSize;
    constexpr int64_t kRowNormalBlockNum = kRowNum;
    constexpr int64_t kRowTailLen = 0;
    constexpr uint32_t kMaxUbBlockNum = 2;
    constexpr uint32_t kBlockDim = kSingleCore;

    size_t xSize = kRowNum * kColNum * sizeof(uint8_t);
    size_t scaleSize = kRowNum * kScaleBytesPerRow * sizeof(uint8_t);
    size_t ySize = kRowNum * kColNum * sizeof(float);
    size_t tilingSize = sizeof(AntiMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(kWorkspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AntiMxQuantTilingData* tilingData = reinterpret_cast<AntiMxQuantTilingData*>(tiling);
    tilingData->ubSize = kUbSize;
    tilingData->dstType = kDstTypeFp32;
    tilingData->totalCoreNum = kSingleCore;
    tilingData->usedCoreNum = kSingleCore;
    tilingData->rowTileNum = kSingleTile;
    tilingData->colTileNum = kSingleTile;
    tilingData->rowNum = kRowNum;
    tilingData->colNum = kColNum;
    tilingData->colNormalBlockNum = kColNormalBlockNum;
    tilingData->colTailLen = kColTailLen;
    tilingData->rowNormalBlockNum = kRowNormalBlockNum;
    tilingData->rowTailLen = kRowTailLen;
    tilingData->maxUbBlockNum = kMaxUbBlockNum;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey0);

    uint32_t blockDim = kBlockDim;
    auto kernel_func = [](GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::anti_mx_quant<kTilingKey0>(x, mxScale, y, workspace, tiling);
    };
    ICPU_RUN_KF(kernel_func, blockDim, x, mxScale, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test multi-core: 2 cores, shape [4, 512]
TEST_F(anti_mx_quant_test, fp8_e4m3_to_fp32_multicore)
{
    constexpr int64_t kRowNum = 4;
    constexpr int64_t kColNum = 512;
    constexpr int64_t kRowNormalBlockNum = 2;
    constexpr uint32_t kUsedCoreNum = 2;
    constexpr int64_t kColNormalBlockNum = kSingleCore;
    constexpr int64_t kColTailLen = 0;
    constexpr int64_t kRowTailLen = 0;
    constexpr uint32_t kMaxUbBlockNum = 2;

    size_t xSize = kRowNum * kColNum * sizeof(uint8_t);
    size_t scaleSize = kRowNum * kScaleBytesPerRow * sizeof(uint8_t);
    size_t ySize = kRowNum * kColNum * sizeof(float);
    size_t tilingSize = sizeof(AntiMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(kWorkspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AntiMxQuantTilingData* tilingData = reinterpret_cast<AntiMxQuantTilingData*>(tiling);
    tilingData->ubSize = kUbSize;
    tilingData->dstType = kDstTypeFp32;
    tilingData->totalCoreNum = kUsedCoreNum;
    tilingData->usedCoreNum = kUsedCoreNum;
    tilingData->rowTileNum = kUsedCoreNum;
    tilingData->colTileNum = kSingleTile;
    tilingData->rowNum = kRowNum;
    tilingData->colNum = kColNum;
    tilingData->colNormalBlockNum = kColNormalBlockNum;
    tilingData->colTailLen = kColTailLen;
    tilingData->rowNormalBlockNum = kRowNormalBlockNum;
    tilingData->rowTailLen = kRowTailLen;
    tilingData->maxUbBlockNum = kMaxUbBlockNum;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey0);

    uint32_t blockDim = kUsedCoreNum;
    auto kernel_func = [](GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::anti_mx_quant<kTilingKey0>(x, mxScale, y, workspace, tiling);
    };
    ICPU_RUN_KF(kernel_func, blockDim, x, mxScale, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
