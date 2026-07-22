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
 * \file test_mx_to_block_mx_quant.cpp
 * \brief CPU simulation unit test for MxToBlockMxQuant kernel.
 *
 * The CMakeLists.txt declares a single binary config:
 *   -DDTYPE_X=fp4_e2m1_t -DDTYPE_Y=fp8_e4m3fn_t
 * so all test cases use FP4_E2M1 -> FP8_E4M3FN (dst_type=36).
 *
 * Tiling data fields (from MxToBlockMxQuantTilingData):
 *   ubSize, dstType, totalCoreNum, usedCoreNum, batchNum, rowNum, colNum, colScaleNum,
 *   rowMode, rowBlockNumPerBatch, colBlockNumPerBatch, rowTailLenPerBatch, colTailLenPerBatch,
 *   totalBlockNum, headCoreBlockNum, tailCoreBlockNum, headCoreNum, tailCoreNum
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/arch35/mx_to_block_mx_quant.cpp"
#include <cstdint>

using namespace std;

// Kernel entry is a template function:
//   template <uint64_t rowMode>
//   __global__ __aicore__ void mx_to_block_mx_quant(
//       GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
//       GM_ADDR workspace, GM_ADDR tiling)
// rowMode=0 (TPL_ROW_ALIGNED), rowMode=1 (TPL_ROW_NOT_ALIGNED)

class mx_to_block_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "mx_to_block_mx_quant_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "mx_to_block_mx_quant_test TearDown\n" << endl; }
};

// Helper: align a byte count up to 32-byte boundary
static size_t Align32(size_t n) { return (n + 31) & ~static_cast<size_t>(31); }

// ===========================================================================
// Test case B (primary): x=[128, 512], FP4_E2M1 -> FP8_E4M3FN, dst_type=36, row aligned
//
// Tiling (from op_host Tiling UT):
//   "253952 36 64 2 1 128 512 16 0 2 1 64 512 2 1 0 2 0"
//
// Memory layout:
//   x        (FP4_E2M1,  0.5 byte/elem): 128 * 512 / 2 = 32768 bytes
//   mxscale  (FP8_E8M0,  1 byte/elem) : shape=[128, 8, 2] => 128 * 16 = 2048 bytes
//   y        (FP8_E4M3FN, 1 byte/elem): 128 * 512 = 65536 bytes
//   scale1   (FP8_E8M0,  1 byte/elem) : shape=[128, 8, 2] => 2048 bytes
//   scale2   (FP8_E8M0,  1 byte/elem) : shape=[2, 512, 2] => 2048 bytes
// ===========================================================================
TEST_F(mx_to_block_mx_quant_test, mx_to_block_mx_quant_case_128x512_aligned)
{
    // Shape parameters
    int64_t rowNum = 128;
    int64_t colNum = 512;
    int64_t colScaleNum = 16; // (CeilDiv(512,32)+1)/2 = (16+1)/2 = 8, *2 = 16

    // Memory sizes (all 32-byte aligned)
    size_t xSize = Align32(rowNum * colNum / 2);        // FP4: 0.5 byte/elem
    size_t mxScaleSize = Align32(rowNum * colScaleNum); // FP8_E8M0: 1 byte/elem
    size_t ySize = Align32(rowNum * colNum);            // FP8: 1 byte/elem
    // scale1 shape = mxscale shape
    size_t scale1Size = Align32(rowNum * colScaleNum);
    // scale2 shape = [rowBlockNumPerBatch*2, colNum, 2]
    // rowBlockNumPerBatch = CeilDiv(128, 64) = 2, so scale2 = [2*2, 512, 2]?
    // From tiling: rowBlockNumPerBatch=2, scale2Shape = [2, 512, 2] => 2048
    // Actually scale2 shape last 3 dims: [CeilDiv(rowBlockNumPerBatch,1)*2 / 2? No.
    // scale2Shape from tiling UT: {{2, 512, 2}} for [128,512]
    // So scale2 size = 2 * 512 * 2 = 2048
    size_t scale2Size = Align32(2 * colNum * 2);

    size_t tilingDataSize = sizeof(MxToBlockMxQuantTilingData);
    uint32_t blockDim = 2; // usedCoreNum = 2

    // Allocate GM memory (CPU simulation)
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(mxScaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* scale1 = (uint8_t*)AscendC::GmAlloc(scale1Size);
    uint8_t* scale2 = (uint8_t*)AscendC::GmAlloc(scale2Size);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    ASSERT_NE(x, nullptr);
    ASSERT_NE(mxScale, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(scale1, nullptr);
    ASSERT_NE(scale2, nullptr);
    ASSERT_NE(workSpace, nullptr);
    ASSERT_NE(tiling, nullptr);

    // Zero-initialize all buffers
    memset(x, 0, xSize);
    memset(mxScale, 0, mxScaleSize);
    memset(y, 0, ySize);
    memset(scale1, 0, scale1Size);
    memset(scale2, 0, scale2Size);
    memset(tiling, 0, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // Fill tiling data: "253952 36 64 2 1 128 512 16 0 2 1 64 512 2 1 0 2 0"
    MxToBlockMxQuantTilingData* tilingData = reinterpret_cast<MxToBlockMxQuantTilingData*>(tiling);
    tilingData->ubSize = 253952;
    tilingData->dstType = 36; // FP8_E4M3FN
    tilingData->totalCoreNum = 64;
    tilingData->usedCoreNum = 2;
    tilingData->batchNum = 1;
    tilingData->rowNum = 128;
    tilingData->colNum = 512;
    tilingData->colScaleNum = 16;
    tilingData->rowMode = 0; // TPL_ROW_ALIGNED
    tilingData->rowBlockNumPerBatch = 2;
    tilingData->colBlockNumPerBatch = 1;
    tilingData->rowTailLenPerBatch = 64;
    tilingData->colTailLenPerBatch = 512;
    tilingData->totalBlockNum = 2;
    tilingData->headCoreBlockNum = 1;
    tilingData->tailCoreBlockNum = 0;
    tilingData->headCoreNum = 2;
    tilingData->tailCoreNum = 0;

    ICPU_SET_TILING_KEY(0);

    // Wrap the template instantiation in a lambda
    auto mx_to_block_mx_quant_kernel = [](GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
                                          GM_ADDR workSpace, GM_ADDR tiling) {
        ::mx_to_block_mx_quant<TPL_ROW_ALIGNED>(x, mxScale, y, scale1, scale2, workSpace, tiling);
    };
    ICPU_RUN_KF(mx_to_block_mx_quant_kernel, blockDim, x, mxScale, y, scale1, scale2, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(y);
    AscendC::GmFree(scale1);
    AscendC::GmFree(scale2);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}

// ===========================================================================
// Test case A variant: x=[64, 64], FP4_E2M1 -> FP8_E4M3FN, dst_type=36, row aligned
//
// Based on tiling UT case A (originally FP4_E2M1->FP8_E5M2 dst_type=35),
// but adapted for the single binary config (dst_type=36, FP8_E4M3FN).
//
// Original case A tiling for [64,64]: "253952 35 64 1 1 64 64 2 0 1 1 64 64 1 1 0 1 0"
// We change dstType from 35 to 36 to match binary config. Other fields unchanged
// since the tiling structure doesn't depend on dst_type for shape/block partitioning.
//
// Memory layout:
//   x        (FP4_E2M1):  64 * 64 / 2 = 2048 bytes
//   mxscale  (FP8_E8M0):  shape=[64, 1, 2] => 64 * 2 = 128 bytes
//   y        (FP8):       64 * 64 = 4096 bytes
//   scale1   (FP8_E8M0):  shape=[64, 1, 2] => 128 bytes
//   scale2   (FP8_E8M0):  shape=[1, 64, 2] => 128 bytes
// ===========================================================================
TEST_F(mx_to_block_mx_quant_test, mx_to_block_mx_quant_case_64x64_aligned)
{
    int64_t rowNum = 64;
    int64_t colNum = 64;
    int64_t colScaleNum = 2; // (CeilDiv(64,32)+1)/2 = (2+1)/2 = 1, *2 = 2

    size_t xSize = Align32(rowNum * colNum / 2);
    size_t mxScaleSize = Align32(rowNum * colScaleNum);
    size_t ySize = Align32(rowNum * colNum);
    size_t scale1Size = Align32(rowNum * colScaleNum);
    // scale2 shape = [1, 64, 2] => rowBlockNumPerBatch=1 => 1*2/2=1, last*2 => 1*64*2=128
    size_t scale2Size = Align32(1 * colNum * 2);

    size_t tilingDataSize = sizeof(MxToBlockMxQuantTilingData);
    uint32_t blockDim = 1; // usedCoreNum = 1

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(mxScaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* scale1 = (uint8_t*)AscendC::GmAlloc(scale1Size);
    uint8_t* scale2 = (uint8_t*)AscendC::GmAlloc(scale2Size);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    ASSERT_NE(x, nullptr);
    ASSERT_NE(mxScale, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(scale1, nullptr);
    ASSERT_NE(scale2, nullptr);
    ASSERT_NE(workSpace, nullptr);
    ASSERT_NE(tiling, nullptr);

    memset(x, 0, xSize);
    memset(mxScale, 0, mxScaleSize);
    memset(y, 0, ySize);
    memset(scale1, 0, scale1Size);
    memset(scale2, 0, scale2Size);
    memset(tiling, 0, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // Fill tiling: "253952 36 64 1 1 64 64 2 0 1 1 64 64 1 1 0 1 0"
    // (adapted from case A with dstType=36)
    MxToBlockMxQuantTilingData* tilingData = reinterpret_cast<MxToBlockMxQuantTilingData*>(tiling);
    tilingData->ubSize = 253952;
    tilingData->dstType = 36;
    tilingData->totalCoreNum = 64;
    tilingData->usedCoreNum = 1;
    tilingData->batchNum = 1;
    tilingData->rowNum = 64;
    tilingData->colNum = 64;
    tilingData->colScaleNum = 2;
    tilingData->rowMode = 0; // TPL_ROW_ALIGNED
    tilingData->rowBlockNumPerBatch = 1;
    tilingData->colBlockNumPerBatch = 1;
    tilingData->rowTailLenPerBatch = 64;
    tilingData->colTailLenPerBatch = 64;
    tilingData->totalBlockNum = 1;
    tilingData->headCoreBlockNum = 1;
    tilingData->tailCoreBlockNum = 0;
    tilingData->headCoreNum = 1;
    tilingData->tailCoreNum = 0;

    ICPU_SET_TILING_KEY(0);

    auto mx_to_block_mx_quant_kernel = [](GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
                                          GM_ADDR workSpace, GM_ADDR tiling) {
        ::mx_to_block_mx_quant<TPL_ROW_ALIGNED>(x, mxScale, y, scale1, scale2, workSpace, tiling);
    };
    ICPU_RUN_KF(mx_to_block_mx_quant_kernel, blockDim, x, mxScale, y, scale1, scale2, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(y);
    AscendC::GmFree(scale1);
    AscendC::GmFree(scale2);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}

// ===========================================================================
// Test case C variant: x=[1024, 2048], FP4_E2M1 -> FP8_E4M3FN, dst_type=36, large multi-core
//
// Based on tiling UT case C (originally FP4_E1M2->FP8_E5M2 dst_type=35),
// adapted for binary config (dst_type=36, FP4_E2M1 input).
//
// Original case C tiling: "253952 35 64 64 1 1024 2048 64 0 16 4 64 512 64 1 0 64 0"
//
// Memory layout:
//   x        (FP4_E2M1):  1024 * 2048 / 2 = 1048576 bytes
//   mxscale  (FP8_E8M0):  shape=[1024, 32, 2] => 1024 * 64 = 65536 bytes
//   y        (FP8):       1024 * 2048 = 2097152 bytes
//   scale1   (FP8_E8M0):  shape=[1024, 32, 2] => 65536 bytes
//   scale2   (FP8_E8M0):  shape=[16, 2048, 2] => 65536 bytes
// ===========================================================================
TEST_F(mx_to_block_mx_quant_test, mx_to_block_mx_quant_case_1024x2048_aligned_multicore)
{
    int64_t rowNum = 1024;
    int64_t colNum = 2048;
    int64_t colScaleNum = 64; // (CeilDiv(2048,32)+1)/2 = (64+1)/2 = 32, *2 = 64

    size_t xSize = Align32(rowNum * colNum / 2);
    size_t mxScaleSize = Align32(rowNum * colScaleNum);
    size_t ySize = Align32(rowNum * colNum);
    size_t scale1Size = Align32(rowNum * colScaleNum);
    // scale2 shape = [16, 2048, 2]: rowBlockNumPerBatch=16 => 16*2/2=16
    size_t scale2Size = Align32(16 * colNum * 2);

    size_t tilingDataSize = sizeof(MxToBlockMxQuantTilingData);
    uint32_t blockDim = 64; // usedCoreNum = 64

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(mxScaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* scale1 = (uint8_t*)AscendC::GmAlloc(scale1Size);
    uint8_t* scale2 = (uint8_t*)AscendC::GmAlloc(scale2Size);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    ASSERT_NE(x, nullptr);
    ASSERT_NE(mxScale, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(scale1, nullptr);
    ASSERT_NE(scale2, nullptr);
    ASSERT_NE(workSpace, nullptr);
    ASSERT_NE(tiling, nullptr);

    memset(x, 0, xSize);
    memset(mxScale, 0, mxScaleSize);
    memset(y, 0, ySize);
    memset(scale1, 0, scale1Size);
    memset(scale2, 0, scale2Size);
    memset(tiling, 0, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // Fill tiling: "253952 36 64 64 1 1024 2048 64 0 16 4 64 512 64 1 0 64 0"
    // (adapted from case C with dstType=36)
    MxToBlockMxQuantTilingData* tilingData = reinterpret_cast<MxToBlockMxQuantTilingData*>(tiling);
    tilingData->ubSize = 253952;
    tilingData->dstType = 36;
    tilingData->totalCoreNum = 64;
    tilingData->usedCoreNum = 64;
    tilingData->batchNum = 1;
    tilingData->rowNum = 1024;
    tilingData->colNum = 2048;
    tilingData->colScaleNum = 64;
    tilingData->rowMode = 0; // TPL_ROW_ALIGNED
    tilingData->rowBlockNumPerBatch = 16;
    tilingData->colBlockNumPerBatch = 4;
    tilingData->rowTailLenPerBatch = 64;
    tilingData->colTailLenPerBatch = 512;
    tilingData->totalBlockNum = 64;
    tilingData->headCoreBlockNum = 1;
    tilingData->tailCoreBlockNum = 0;
    tilingData->headCoreNum = 64;
    tilingData->tailCoreNum = 0;

    ICPU_SET_TILING_KEY(0);

    auto mx_to_block_mx_quant_kernel = [](GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
                                          GM_ADDR workSpace, GM_ADDR tiling) {
        ::mx_to_block_mx_quant<TPL_ROW_ALIGNED>(x, mxScale, y, scale1, scale2, workSpace, tiling);
    };
    ICPU_RUN_KF(mx_to_block_mx_quant_kernel, blockDim, x, mxScale, y, scale1, scale2, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(y);
    AscendC::GmFree(scale1);
    AscendC::GmFree(scale2);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}
