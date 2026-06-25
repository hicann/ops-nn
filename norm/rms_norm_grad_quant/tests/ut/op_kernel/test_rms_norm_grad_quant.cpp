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
 * \file test_rms_norm_grad_quant.cpp
 * \brief Kernel UT for RmsNormGradQuant
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "rms_norm_grad_quant_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "rms_norm_grad_quant.cpp"
#endif

using namespace std;

#define COMPUTE_MODE_DX_FULL_LOAD 0
#define COMPUTE_MODE_DX_SPLIT_D 1

#define COMPUTE_MODE_DGAMMA_FULL_LOAD 0
#define COMPUTE_MODE_DGAMMA_BIG_M 1
#define COMPUTE_MODE_DGAMMA_WITH_LARGE_ROWS 2
#define COMPUTE_MODE_DGAMMA_EMPTY 3

#define COMPUTE_MODE_WITH_OFFSET_X 0
#define COMPUTE_MODE_WITHOUT_OFFSET_X 1

#define COMPUTE_MODE_DIV_MODE 0
#define COMPUTE_MODE_NOT_DIV_MODE 1

class rms_norm_grad_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << " rms_norm_grad_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << " rms_norm_grad_quant_test TearDown\n" << endl;
    }
};

TEST_F(rms_norm_grad_quant_test, test_full_load_dgamma_full_load)
{
    uint64_t rows = 4;
    uint64_t cols = 64;
    size_t dyByteSize = rows * cols * sizeof(DTYPE_DY);
    size_t xByteSize = rows * cols * sizeof(DTYPE_X);
    size_t rstdByteSize = rows * sizeof(DTYPE_RSTD);
    size_t gammaByteSize = cols * sizeof(DTYPE_GAMMA);
    size_t scalesXByteSize = 1 * sizeof(DTYPE_SCALES_X);
    size_t offsetXByteSize = 1 * sizeof(DTYPE_OFFSET_X);
    size_t dxByteSize = rows * cols * sizeof(DTYPE_DX);
    size_t dgammaByteSize = cols * sizeof(DTYPE_DGAMMA);

    size_t tilingDataSize = sizeof(RmsNormGradQuantRegbaseTilingData);
    uint32_t blockDim = 1;

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales_x = (uint8_t*)AscendC::GmAlloc(scalesXByteSize);
    uint8_t* offset_x = (uint8_t*)AscendC::GmAlloc(offsetXByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormGradQuantRegbaseTilingData* tilingData =
        reinterpret_cast<RmsNormGradQuantRegbaseTilingData*>(tiling);

    tilingData->tailCoreNumDG = 0;
    tilingData->colsPerCoreDG = cols;
    tilingData->colsLastCoreDG = cols;
    tilingData->colsPerTailCoreDG = 0;
    tilingData->rowsPerUBDG = rows;
    tilingData->powerofTwoValueDG = 4;
    tilingData->rowsTailDG = 0;
    tilingData->totalBlockCountDG = 2;
    tilingData->mainBlockCountDG = 2;
    tilingData->tailBlockCountwithPadDG = 0;
    tilingData->powerOfTwoBlockCountDG = 2;
    tilingData->tailBlockCountWithoutPadDG = 0;
    tilingData->binaryAddKDG = 1;
    tilingData->usedCoreNumDG = 1;
    tilingData->blockSize = 32;
    tilingData->vlFp32 = 8;
    tilingData->isFullLoad = 1;
    tilingData->isMultiColset = 0;

    tilingData->dxTilingData.rows = rows;
    tilingData->dxTilingData.cols = cols;
    tilingData->dxTilingData.blockFactorDx = 1;
    tilingData->dxTilingData.bodyPart = rows;
    tilingData->dxTilingData.usedCoreNumDx = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto wrapper = [](GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR scales_x,
                      GM_ADDR offset_x, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_grad_quant<COMPUTE_MODE_DX_FULL_LOAD, COMPUTE_MODE_DGAMMA_FULL_LOAD,
            COMPUTE_MODE_WITHOUT_OFFSET_X, COMPUTE_MODE_DIV_MODE>(
            dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace, tiling);
    };
    ICPU_RUN_KF(wrapper, blockDim, dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace,
        (uint8_t*)tilingData);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales_x);
    AscendC::GmFree(offset_x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_grad_quant_test, test_full_load_dgamma_big_m)
{
    uint64_t rows = 1024;
    uint64_t cols = 64;
    size_t dyByteSize = rows * cols * sizeof(DTYPE_DY);
    size_t xByteSize = rows * cols * sizeof(DTYPE_X);
    size_t rstdByteSize = rows * sizeof(DTYPE_RSTD);
    size_t gammaByteSize = cols * sizeof(DTYPE_GAMMA);
    size_t scalesXByteSize = 1 * sizeof(DTYPE_SCALES_X);
    size_t offsetXByteSize = 1 * sizeof(DTYPE_OFFSET_X);
    size_t dxByteSize = rows * cols * sizeof(DTYPE_DX);
    size_t dgammaByteSize = cols * sizeof(DTYPE_DGAMMA);

    size_t tilingDataSize = sizeof(RmsNormGradQuantRegbaseBigMTilingData);
    uint32_t blockDim = 1;

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales_x = (uint8_t*)AscendC::GmAlloc(scalesXByteSize);
    uint8_t* offset_x = (uint8_t*)AscendC::GmAlloc(offsetXByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(rows * cols * sizeof(float));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormGradQuantRegbaseBigMTilingData* tilingData =
        reinterpret_cast<RmsNormGradQuantRegbaseBigMTilingData*>(tiling);

    tilingData->dgammaUsedCoreNum = 1;
    tilingData->dgammaMPerBlock = rows;
    tilingData->dgammaMReminder = 1;
    tilingData->dgammaNloop = 1;
    tilingData->dgammaNtail = 0;
    tilingData->dgammaMfactorBlockAligned = 32;
    tilingData->dgammaNfactorBlockAligned = 64;
    tilingData->dgammaMToProcessMainBlock = rows;
    tilingData->dgammaMLoopMainBlock = 32;
    tilingData->dgammaMTotalLoopMainBlock = 32;
    tilingData->dgammaMTailMainBlock = 0;
    tilingData->dgammaBasicBlockLoopMainBlock = 1;
    tilingData->dgammaMainFoldCountMainBlock = 0;
    tilingData->dgammaCacheBufferCountMainBlock = 1;
    tilingData->dgammaResultCacheIDMainBlock = 0;
    tilingData->dgammaMToProcessTailBlock = 0;
    tilingData->dgammaMLoopTailBlock = 0;
    tilingData->dgammaMTotalLoopTailBlock = 0;
    tilingData->dgammaMTailTailBlock = 0;
    tilingData->dgammaBasicBlockLoopTailBlock = 0;
    tilingData->dgammaMainFoldCountTailBlock = 0;
    tilingData->dgammaCacheBufferCountTailBlock = 0;
    tilingData->dgammaResultCacheIDTailBlock = 0;
    tilingData->dgammaAInnerAlignedStg1 = 32;
    tilingData->dgammaAOuterStg1 = 2;
    tilingData->dgammaATailStg1 = 32;

    tilingData->dxTilingData.rows = rows;
    tilingData->dxTilingData.cols = cols;
    tilingData->dxTilingData.blockFactorDx = 1;
    tilingData->dxTilingData.bodyPart = rows;
    tilingData->dxTilingData.usedCoreNumDx = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto wrapper = [](GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR scales_x,
                      GM_ADDR offset_x, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_grad_quant<COMPUTE_MODE_DX_FULL_LOAD, COMPUTE_MODE_DGAMMA_BIG_M,
            COMPUTE_MODE_WITH_OFFSET_X, COMPUTE_MODE_DIV_MODE>(
            dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace, tiling);
    };
    ICPU_RUN_KF(wrapper, blockDim, dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace,
        (uint8_t*)tilingData);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales_x);
    AscendC::GmFree(offset_x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_grad_quant_test, test_dgamma_empty)
{
    uint64_t rows = 0;
    uint64_t cols = 64;
    size_t dyByteSize = 1;
    size_t xByteSize = 1;
    size_t rstdByteSize = 1;
    size_t gammaByteSize = cols * sizeof(DTYPE_GAMMA);
    size_t scalesXByteSize = 1 * sizeof(DTYPE_SCALES_X);
    size_t offsetXByteSize = 1 * sizeof(DTYPE_OFFSET_X);
    size_t dxByteSize = 1;
    size_t dgammaByteSize = cols * sizeof(DTYPE_DGAMMA);

    size_t tilingDataSize = sizeof(RmsNormGradQuantRegbaseEmptyTilingData);
    uint32_t blockDim = 1;

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales_x = (uint8_t*)AscendC::GmAlloc(scalesXByteSize);
    uint8_t* offset_x = (uint8_t*)AscendC::GmAlloc(offsetXByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormGradQuantRegbaseEmptyTilingData* tilingData =
        reinterpret_cast<RmsNormGradQuantRegbaseEmptyTilingData*>(tiling);

    tilingData->usedCoreNumDG = 1;
    tilingData->colsPerCoreDG = cols;
    tilingData->cols = cols;
    tilingData->ubSize = 196608;
    tilingData->colsPerUBDG = cols;
    tilingData->coreUbBlockCount = 1;
    tilingData->tailUbCols = 0;
    tilingData->lastCoreBlockCount = 1;
    tilingData->lastCoreTailUbCols = 0;
    tilingData->colsLastCoreDG = cols;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto wrapper = [](GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR scales_x,
                      GM_ADDR offset_x, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_grad_quant<COMPUTE_MODE_DX_FULL_LOAD, COMPUTE_MODE_DGAMMA_EMPTY,
            COMPUTE_MODE_WITH_OFFSET_X, COMPUTE_MODE_DIV_MODE>(
            dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace, tiling);
    };
    ICPU_RUN_KF(wrapper, blockDim, dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace,
        (uint8_t*)tilingData);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales_x);
    AscendC::GmFree(offset_x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_grad_quant_test, test_full_load_dgamma_with_large_rows)
{
    uint64_t rows = 8;
    uint64_t cols = 64;
    size_t dyByteSize = rows * cols * sizeof(DTYPE_DY);
    size_t xByteSize = rows * cols * sizeof(DTYPE_X);
    size_t rstdByteSize = rows * sizeof(DTYPE_RSTD);
    size_t gammaByteSize = cols * sizeof(DTYPE_GAMMA);
    size_t scalesXByteSize = 1 * sizeof(DTYPE_SCALES_X);
    size_t offsetXByteSize = 1 * sizeof(DTYPE_OFFSET_X);
    size_t dxByteSize = rows * cols * sizeof(DTYPE_DX);
    size_t dgammaByteSize = cols * sizeof(DTYPE_DGAMMA);

    size_t tilingDataSize = sizeof(RmsNormGradQuantRegbaseTilingData);
    uint32_t blockDim = 1;

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales_x = (uint8_t*)AscendC::GmAlloc(scalesXByteSize);
    uint8_t* offset_x = (uint8_t*)AscendC::GmAlloc(offsetXByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormGradQuantRegbaseTilingData* tilingData =
        reinterpret_cast<RmsNormGradQuantRegbaseTilingData*>(tiling);

    // rows=8, rowsPerUB=4 => mainBlockCountDG = rows/rowsPerUB = 2
    tilingData->tailCoreNumDG = 0;
    tilingData->colsPerCoreDG = cols;
    tilingData->colsLastCoreDG = cols;
    tilingData->colsPerTailCoreDG = cols;
    tilingData->rowsPerUBDG = 4;
    tilingData->powerofTwoValueDG = 4;
    tilingData->rowsTailDG = 0;
    tilingData->totalBlockCountDG = 2;
    tilingData->mainBlockCountDG = 2;
    tilingData->tailBlockCountwithPadDG = 0;
    tilingData->powerOfTwoBlockCountDG = 2;
    tilingData->tailBlockCountWithoutPadDG = 0;
    tilingData->binaryAddKDG = 1;
    tilingData->usedCoreNumDG = 1;
    tilingData->blockSize = 32;
    tilingData->vlFp32 = 8;
    tilingData->isFullLoad = 0;
    tilingData->isMultiColset = 0;

    tilingData->dxTilingData.rows = rows;
    tilingData->dxTilingData.cols = cols;
    tilingData->dxTilingData.blockFactorDx = rows;
    tilingData->dxTilingData.bodyPart = rows;
    tilingData->dxTilingData.usedCoreNumDx = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto wrapper = [](GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR scales_x,
                      GM_ADDR offset_x, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_grad_quant<COMPUTE_MODE_DX_FULL_LOAD, COMPUTE_MODE_DGAMMA_WITH_LARGE_ROWS,
            COMPUTE_MODE_WITHOUT_OFFSET_X, COMPUTE_MODE_NOT_DIV_MODE>(
            dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace, tiling);
    };
    ICPU_RUN_KF(wrapper, blockDim, dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace,
        (uint8_t*)tilingData);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales_x);
    AscendC::GmFree(offset_x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_grad_quant_test, test_dgamma_full_load_with_pad)
{
    uint64_t rows = 20;
    uint64_t cols = 64;
    size_t dyByteSize = rows * cols * sizeof(DTYPE_DY);
    size_t xByteSize = rows * cols * sizeof(DTYPE_X);
    size_t rstdByteSize = rows * sizeof(DTYPE_RSTD);
    size_t gammaByteSize = cols * sizeof(DTYPE_GAMMA);
    size_t scalesXByteSize = 1 * sizeof(DTYPE_SCALES_X);
    size_t offsetXByteSize = 1 * sizeof(DTYPE_OFFSET_X);
    size_t dxByteSize = rows * cols * sizeof(DTYPE_DX);
    size_t dgammaByteSize = cols * sizeof(DTYPE_DGAMMA);

    size_t tilingDataSize = sizeof(RmsNormGradQuantRegbaseTilingData);
    uint32_t blockDim = 1;

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales_x = (uint8_t*)AscendC::GmAlloc(scalesXByteSize);
    uint8_t* offset_x = (uint8_t*)AscendC::GmAlloc(offsetXByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormGradQuantRegbaseTilingData* tilingData =
        reinterpret_cast<RmsNormGradQuantRegbaseTilingData*>(tiling);

    // rows=20, rowsPerUBDG=20, mainBlockCount=16(power of 2), tailBlockCountwithPad=4(pad to 16)
    tilingData->tailCoreNumDG = 0;
    tilingData->colsPerCoreDG = cols;
    tilingData->colsLastCoreDG = cols;
    tilingData->colsPerTailCoreDG = 0;
    tilingData->rowsPerUBDG = 20;
    tilingData->powerofTwoValueDG = 16;
    tilingData->rowsTailDG = 4;
    tilingData->totalBlockCountDG = 20;
    tilingData->mainBlockCountDG = 16;
    tilingData->tailBlockCountwithPadDG = 16;
    tilingData->powerOfTwoBlockCountDG = 16;
    tilingData->tailBlockCountWithoutPadDG = 4;
    tilingData->binaryAddKDG = 4;
    tilingData->usedCoreNumDG = 1;
    tilingData->blockSize = 32;
    tilingData->vlFp32 = 8;
    tilingData->isFullLoad = 1;
    tilingData->isMultiColset = 0;

    tilingData->dxTilingData.rows = rows;
    tilingData->dxTilingData.cols = cols;
    tilingData->dxTilingData.blockFactorDx = 1;
    tilingData->dxTilingData.bodyPart = rows;
    tilingData->dxTilingData.usedCoreNumDx = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto wrapper = [](GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR scales_x,
                      GM_ADDR offset_x, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_grad_quant<COMPUTE_MODE_DX_FULL_LOAD, COMPUTE_MODE_DGAMMA_FULL_LOAD,
            COMPUTE_MODE_WITHOUT_OFFSET_X, COMPUTE_MODE_DIV_MODE>(
            dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace, tiling);
    };
    ICPU_RUN_KF(wrapper, blockDim, dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace,
        (uint8_t*)tilingData);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales_x);
    AscendC::GmFree(offset_x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_grad_quant_test, test_dgamma_with_large_rows_unalign)
{
    uint64_t rows = 12;
    uint64_t cols = 64;
    size_t dyByteSize = rows * cols * sizeof(DTYPE_DY);
    size_t xByteSize = rows * cols * sizeof(DTYPE_X);
    size_t rstdByteSize = rows * sizeof(DTYPE_RSTD);
    size_t gammaByteSize = cols * sizeof(DTYPE_GAMMA);
    size_t scalesXByteSize = 1 * sizeof(DTYPE_SCALES_X);
    size_t offsetXByteSize = 1 * sizeof(DTYPE_OFFSET_X);
    size_t dxByteSize = rows * cols * sizeof(DTYPE_DX);
    size_t dgammaByteSize = cols * sizeof(DTYPE_DGAMMA);

    size_t tilingDataSize = sizeof(RmsNormGradQuantRegbaseTilingData);
    uint32_t blockDim = 1;

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales_x = (uint8_t*)AscendC::GmAlloc(scalesXByteSize);
    uint8_t* offset_x = (uint8_t*)AscendC::GmAlloc(offsetXByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormGradQuantRegbaseTilingData* tilingData =
        reinterpret_cast<RmsNormGradQuantRegbaseTilingData*>(tiling);

    // rows=12, rowsPerUBDG=4 => 3 blocks, not power of 2 => UnAlign path
    tilingData->tailCoreNumDG = 0;
    tilingData->colsPerCoreDG = cols;
    tilingData->colsLastCoreDG = cols;
    tilingData->colsPerTailCoreDG = cols;
    tilingData->rowsPerUBDG = 4;
    tilingData->powerofTwoValueDG = 4;
    tilingData->rowsTailDG = 0;
    tilingData->totalBlockCountDG = 3;
    tilingData->mainBlockCountDG = 2;
    tilingData->tailBlockCountwithPadDG = 0;
    tilingData->powerOfTwoBlockCountDG = 2;
    tilingData->tailBlockCountWithoutPadDG = 1;
    tilingData->binaryAddKDG = 1;
    tilingData->usedCoreNumDG = 1;
    tilingData->blockSize = 32;
    tilingData->vlFp32 = 8;
    tilingData->isFullLoad = 0;
    tilingData->isMultiColset = 0;

    tilingData->dxTilingData.rows = rows;
    tilingData->dxTilingData.cols = cols;
    tilingData->dxTilingData.blockFactorDx = rows;
    tilingData->dxTilingData.bodyPart = rows;
    tilingData->dxTilingData.usedCoreNumDx = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto wrapper = [](GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR scales_x,
                      GM_ADDR offset_x, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_grad_quant<COMPUTE_MODE_DX_FULL_LOAD, COMPUTE_MODE_DGAMMA_WITH_LARGE_ROWS,
            COMPUTE_MODE_WITHOUT_OFFSET_X, COMPUTE_MODE_DIV_MODE>(
            dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace, tiling);
    };
    ICPU_RUN_KF(wrapper, blockDim, dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace,
        (uint8_t*)tilingData);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales_x);
    AscendC::GmFree(offset_x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_grad_quant_test, test_big_m_with_fold)
{
    uint64_t rows = 64;
    uint64_t cols = 64;
    size_t dyByteSize = rows * cols * sizeof(DTYPE_DY);
    size_t xByteSize = rows * cols * sizeof(DTYPE_X);
    size_t rstdByteSize = rows * sizeof(DTYPE_RSTD);
    size_t gammaByteSize = cols * sizeof(DTYPE_GAMMA);
    size_t scalesXByteSize = 1 * sizeof(DTYPE_SCALES_X);
    size_t offsetXByteSize = 1 * sizeof(DTYPE_OFFSET_X);
    size_t dxByteSize = rows * cols * sizeof(DTYPE_DX);
    size_t dgammaByteSize = cols * sizeof(DTYPE_DGAMMA);

    size_t tilingDataSize = sizeof(RmsNormGradQuantRegbaseBigMTilingData);
    uint32_t blockDim = 1;

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales_x = (uint8_t*)AscendC::GmAlloc(scalesXByteSize);
    uint8_t* offset_x = (uint8_t*)AscendC::GmAlloc(offsetXByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(rows * cols * sizeof(float));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormGradQuantRegbaseBigMTilingData* tilingData =
        reinterpret_cast<RmsNormGradQuantRegbaseBigMTilingData*>(tiling);

    tilingData->dgammaUsedCoreNum = 1;
    tilingData->dgammaMPerBlock = rows;
    tilingData->dgammaMReminder = 1;
    tilingData->dgammaNloop = 1;
    tilingData->dgammaNtail = 0;
    tilingData->dgammaMfactorBlockAligned = 32;
    tilingData->dgammaNfactorBlockAligned = 64;
    tilingData->dgammaMToProcessMainBlock = rows;
    tilingData->dgammaMLoopMainBlock = 32;
    tilingData->dgammaMTotalLoopMainBlock = 32;
    tilingData->dgammaMTailMainBlock = 0;
    tilingData->dgammaBasicBlockLoopMainBlock = 1;
    tilingData->dgammaMainFoldCountMainBlock = 1;
    tilingData->dgammaCacheBufferCountMainBlock = 2;
    tilingData->dgammaResultCacheIDMainBlock = 0;
    tilingData->dgammaMToProcessTailBlock = 0;
    tilingData->dgammaMLoopTailBlock = 0;
    tilingData->dgammaMTotalLoopTailBlock = 0;
    tilingData->dgammaMTailTailBlock = 0;
    tilingData->dgammaBasicBlockLoopTailBlock = 0;
    tilingData->dgammaMainFoldCountTailBlock = 0;
    tilingData->dgammaCacheBufferCountTailBlock = 0;
    tilingData->dgammaResultCacheIDTailBlock = 0;
    tilingData->dgammaAInnerAlignedStg1 = 32;
    tilingData->dgammaAOuterStg1 = 2;
    tilingData->dgammaATailStg1 = 32;

    tilingData->dxTilingData.rows = rows;
    tilingData->dxTilingData.cols = cols;
    tilingData->dxTilingData.blockFactorDx = 1;
    tilingData->dxTilingData.bodyPart = rows;
    tilingData->dxTilingData.usedCoreNumDx = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto wrapper = [](GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR scales_x,
                      GM_ADDR offset_x, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_grad_quant<COMPUTE_MODE_DX_FULL_LOAD, COMPUTE_MODE_DGAMMA_BIG_M,
            COMPUTE_MODE_WITH_OFFSET_X, COMPUTE_MODE_DIV_MODE>(
            dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace, tiling);
    };
    ICPU_RUN_KF(wrapper, blockDim, dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace,
        (uint8_t*)tilingData);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales_x);
    AscendC::GmFree(offset_x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_grad_quant_test, test_split_d_dgamma_full_load)
{
    uint64_t rows = 4;
    uint64_t cols = 8192;
    size_t dyByteSize = rows * cols * sizeof(DTYPE_DY);
    size_t xByteSize = rows * cols * sizeof(DTYPE_X);
    size_t rstdByteSize = rows * sizeof(DTYPE_RSTD);
    size_t gammaByteSize = cols * sizeof(DTYPE_GAMMA);
    size_t scalesXByteSize = 1 * sizeof(DTYPE_SCALES_X);
    size_t offsetXByteSize = 1 * sizeof(DTYPE_OFFSET_X);
    size_t dxByteSize = rows * cols * sizeof(DTYPE_DX);
    size_t dgammaByteSize = cols * sizeof(DTYPE_DGAMMA);

    size_t tilingDataSize = sizeof(RmsNormGradQuantRegbaseTilingData);
    uint32_t blockDim = 1;

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* scales_x = (uint8_t*)AscendC::GmAlloc(scalesXByteSize);
    uint8_t* offset_x = (uint8_t*)AscendC::GmAlloc(offsetXByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormGradQuantRegbaseTilingData* tilingData =
        reinterpret_cast<RmsNormGradQuantRegbaseTilingData*>(tiling);

    // dgamma params: cols=8192, single colset per core
    tilingData->tailCoreNumDG = 0;
    tilingData->colsPerCoreDG = 8;
    tilingData->colsLastCoreDG = 8;
    tilingData->colsPerTailCoreDG = 0;
    tilingData->rowsPerUBDG = rows;
    tilingData->powerofTwoValueDG = 4;
    tilingData->rowsTailDG = 0;
    tilingData->totalBlockCountDG = 2;
    tilingData->mainBlockCountDG = 2;
    tilingData->tailBlockCountwithPadDG = 0;
    tilingData->powerOfTwoBlockCountDG = 2;
    tilingData->tailBlockCountWithoutPadDG = 0;
    tilingData->binaryAddKDG = 1;
    tilingData->usedCoreNumDG = 1;
    tilingData->blockSize = 32;
    tilingData->vlFp32 = 8;
    tilingData->isFullLoad = 1;
    tilingData->isMultiColset = 0;

    // dx split_d params: cols=8192 > UB_FACTOR(6144), bodyPart = 4096 (largest power of 2 < cols/2)
    tilingData->dxTilingData.rows = rows;
    tilingData->dxTilingData.cols = cols;
    tilingData->dxTilingData.blockFactorDx = rows;
    tilingData->dxTilingData.bodyPart = 4096;
    tilingData->dxTilingData.usedCoreNumDx = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto wrapper = [](GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR scales_x,
                      GM_ADDR offset_x, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_grad_quant<COMPUTE_MODE_DX_SPLIT_D, COMPUTE_MODE_DGAMMA_FULL_LOAD,
            COMPUTE_MODE_WITHOUT_OFFSET_X, COMPUTE_MODE_DIV_MODE>(
            dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace, tiling);
    };
    ICPU_RUN_KF(wrapper, blockDim, dy, x, rstd, gamma, scales_x, offset_x, dx, dgamma, workspace,
        (uint8_t*)tilingData);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(rstd);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales_x);
    AscendC::GmFree(offset_x);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
