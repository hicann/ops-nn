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
 * \file test_rms_norm_dynamic_mx_quant_apt.cpp
 * \brief op_kernel UT for rms_norm_dynamic_mx_quant
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "rms_norm_dynamic_mx_quant_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "rms_norm_dynamic_mx_quant_apt.cpp"
#endif

using namespace std;

#define COMPUTE_MODE_FULL_LOAD 0
#define COMPUTE_MODE_RECOMPUTE 1
#define COMPUTE_MODE_REDUCE_EMPTY 2

#define OPTIMIZE_MODE_NORMAL 0
#define OPTIMIZE_MODE_OPTIMIZE 1

class rms_norm_dynamic_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rms_norm_dynamic_mx_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rms_norm_dynamic_mx_quant_test TearDown\n" << endl;
    }
};

// numM=1, numN=64, FULL_LOAD + NORMAL mode (key=0), no beta, no rstd, scaleAlg=OCP, roundMode=RINT
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_general_small)
{
    int64_t numM = 1;
    int64_t numN = 64;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * numN * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * 2 * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantFullLoadTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantFullLoadTilingData*>(tiling);

    tilingData->usedCoreNum = 1;
    tilingData->mTailCores = 0;
    tilingData->numM = numM;
    tilingData->numN = numN;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 64;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->needPadN = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_FULL_LOAD, OPTIMIZE_MODE_NORMAL>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// numM=2, numN=64, FULL_LOAD + NORMAL mode (key=0), with beta and rstd output
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_general_with_beta_rstd)
{
    int64_t numM = 2;
    int64_t numN = 64;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * numN * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * 2 * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantFullLoadTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantFullLoadTilingData*>(tiling);

    tilingData->usedCoreNum = 1;
    tilingData->mTailCores = 0;
    tilingData->numM = numM;
    tilingData->numN = numN;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 64;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->needPadN = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 1;
    tilingData->hasOutputRstd = 1;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_FULL_LOAD, OPTIMIZE_MODE_NORMAL>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// numM=1, numN=50, FULL_LOAD + NORMAL mode (key=0), N not aligned to mxBlockSize, needs padding
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_general_need_pad)
{
    int64_t numM = 1;
    int64_t numN = 50;
    int64_t nMxblockAligned = 64;
    int64_t nMxblockNum = 2;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * nMxblockAligned * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * 2 * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantFullLoadTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantFullLoadTilingData*>(tiling);

    tilingData->usedCoreNum = 1;
    tilingData->mTailCores = 0;
    tilingData->numM = numM;
    tilingData->numN = numN;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = nMxblockAligned;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->needPadN = 1;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_FULL_LOAD, OPTIMIZE_MODE_NORMAL>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// numM=1, numN=64, FULL_LOAD + OPTIMIZE mode (key=16), FP16 input + RINT rounding
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_optimize)
{
    int64_t numM = 1;
    int64_t numN = 64;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * numN * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * 2 * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantFullLoadTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantFullLoadTilingData*>(tiling);

    tilingData->usedCoreNum = 1;
    tilingData->mTailCores = 0;
    tilingData->numM = numM;
    tilingData->numN = numN;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 64;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->needPadN = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 1;
    tilingData->hasOutputRstd = 1;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(16);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_FULL_LOAD, OPTIMIZE_MODE_OPTIMIZE>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// numM=1, numN=64, FULL_LOAD + NORMAL mode (key=0), scaleAlg=cuBLAS(1), roundMode=ROUND(0)
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_cublas_scale)
{
    int64_t numM = 1;
    int64_t numN = 64;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * numN * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * 2 * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantFullLoadTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantFullLoadTilingData*>(tiling);

    tilingData->usedCoreNum = 1;
    tilingData->mTailCores = 0;
    tilingData->numM = numM;
    tilingData->numN = numN;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 64;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->needPadN = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 0;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_FULL_LOAD, OPTIMIZE_MODE_NORMAL>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// numM=5, numN=128, FULL_LOAD + NORMAL mode (key=0), multi-core with tail cores
// 5 rows, 3 cores: mPerCore=1, mTailCores=2
//   core 0: curM = mPerCore+1 = 2, rowOffset = 0*1 + 0 = 0
//   core 1: curM = mPerCore+1 = 2, rowOffset = 1*1 + 1 = 2
//   core 2: curM = mPerCore = 1, rowOffset = 2*1 + 2 = 4
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_multicore_with_tail)
{
    int64_t numM = 5;
    int64_t numN = 128;
    int64_t nMxblockAligned = 128;     // CeilAlign(128, 64) = 128
    int64_t nMxblockNumAlignedTwo = 4; // 128 / 32 = 4
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * numN * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * nMxblockNumAlignedTwo * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantFullLoadTilingData);
    uint32_t blockDim = 3;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantFullLoadTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantFullLoadTilingData*>(tiling);

    tilingData->usedCoreNum = 3;
    tilingData->mTailCores = 2;
    tilingData->numM = numM;
    tilingData->numN = numN;
    tilingData->binAddFoldPoint = 128;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 2;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = nMxblockAligned;
    tilingData->nMxblockNumAlignedTwo = nMxblockNumAlignedTwo;
    tilingData->needPadN = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 1;
    tilingData->hasOutputRstd = 1;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_FULL_LOAD, OPTIMIZE_MODE_NORMAL>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// numM=1, numN=64, FULL_LOAD + NORMAL mode (key=0), roundMode=FLOOR(1)
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_floor_round_mode)
{
    int64_t numM = 1;
    int64_t numN = 64;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * numN * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * 2 * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantFullLoadTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantFullLoadTilingData*>(tiling);

    tilingData->usedCoreNum = 1;
    tilingData->mTailCores = 0;
    tilingData->numM = numM;
    tilingData->numN = numN;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 64;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->needPadN = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 1;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_FULL_LOAD, OPTIMIZE_MODE_NORMAL>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// numM=2, numN=96, FULL_LOAD + NORMAL mode, N not aligned to 64, needs padding
// nMxblockAligned = CeilAlign(96, 64) = 128
// nMxblockNumAlignedTwo = 128 / 32 = 4
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_pad_n_and_scale)
{
    int64_t numM = 2;
    int64_t numN = 96;
    int64_t nMxblockAligned = 128;     // CeilAlign(96, 64) = 128
    int64_t nMxblockNumAlignedTwo = 4; // 128 / 32 = 4
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * nMxblockAligned * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * nMxblockNumAlignedTwo * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantFullLoadTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantFullLoadTilingData*>(tiling);

    tilingData->usedCoreNum = 1;
    tilingData->mTailCores = 0;
    tilingData->numM = numM;
    tilingData->numN = numN;
    tilingData->binAddFoldPoint = 128; // nearest power of 2 for numNUbAligned
    tilingData->mPerCore = 2;
    tilingData->mUbFactor = 2;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = nMxblockAligned;             // 128
    tilingData->nMxblockNumAlignedTwo = nMxblockNumAlignedTwo; // 4
    tilingData->needPadN = 1;                                  // nMxblockAligned(128) != numN(96)
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 1;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_FULL_LOAD, OPTIMIZE_MODE_NORMAL>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// empty_tensor, REDUCE_EMPTY mode (key=2), numM=176, numN=0, output_rstd=true
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_m_not0_n_0_outputRstd_true)
{
    int64_t numM = 176;
    int64_t numN = 0;
    int64_t nMxblockAligned = 0;
    int64_t nMxblockNum = 0;
    int64_t nMxblockNumAlignedTwo = 0;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * nMxblockAligned * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * nMxblockNumAlignedTwo * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantReduceEmptyTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantReduceEmptyTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantReduceEmptyTilingData*>(tiling);

    tilingData->perCoreElements = 176;
    tilingData->lastCoreElements = 176;
    tilingData->perCoreLoops = 1;
    tilingData->perCorePerLoopElements = 176;
    tilingData->perCoreLastLoopElements = 176;
    tilingData->lastCoreLoops = 1;
    tilingData->lastCorePerLoopElements = 176;
    tilingData->lastCoreLastLoopElements = 176;
    tilingData->hasOutputRstd = 1;
    tilingData->numM = 176;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_REDUCE_EMPTY, OPTIMIZE_MODE_NORMAL>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// empty_tensor, REDUCE_EMPTY mode (key=2), numM=0, numN=100, output_rstd=true
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_m_0_n_not_0_outputRstd_true)
{
    int64_t numM = 0;
    int64_t numN = 100;
    int64_t nMxblockAligned = 128;
    int64_t nMxblockNum = 4;
    int64_t nMxblockNumAlignedTwo = 4;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * nMxblockAligned * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * nMxblockNumAlignedTwo * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantReduceEmptyTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantReduceEmptyTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantReduceEmptyTilingData*>(tiling);

    tilingData->perCoreElements = 0;
    tilingData->lastCoreElements = 0;
    tilingData->perCoreLoops = 0;
    tilingData->perCorePerLoopElements = 0;
    tilingData->perCoreLastLoopElements = 0;
    tilingData->lastCoreLoops = 0;
    tilingData->lastCorePerLoopElements = 0;
    tilingData->lastCoreLastLoopElements = 0;
    tilingData->hasOutputRstd = 1;
    tilingData->numM = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_REDUCE_EMPTY, OPTIMIZE_MODE_NORMAL>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// empty_tensor, REDUCE_EMPTY mode (key=2), numM=0, numN=0, output_rstd=true
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_m_0_n_0_outputRstd_true)
{
    int64_t numM = 0;
    int64_t numN = 0;
    int64_t nMxblockAligned = 0;
    int64_t nMxblockNum = 0;
    int64_t nMxblockNumAlignedTwo = 0;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * nMxblockAligned * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * nMxblockNumAlignedTwo * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantReduceEmptyTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(mxscaleByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    RmsNormDynamicMxQuantReduceEmptyTilingData* tilingData =
        reinterpret_cast<RmsNormDynamicMxQuantReduceEmptyTilingData*>(tiling);

    tilingData->perCoreElements = 0;
    tilingData->lastCoreElements = 0;
    tilingData->perCoreLoops = 0;
    tilingData->perCorePerLoopElements = 0;
    tilingData->perCoreLastLoopElements = 0;
    tilingData->lastCoreLoops = 0;
    tilingData->lastCorePerLoopElements = 0;
    tilingData->lastCoreLastLoopElements = 0;
    tilingData->hasOutputRstd = 1;
    tilingData->numM = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(2);

    auto rms_norm_dynamic_mx_quant_wrapper = [](GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale,
                                                GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling) {
        ::rms_norm_dynamic_mx_quant<COMPUTE_MODE_REDUCE_EMPTY, OPTIMIZE_MODE_NORMAL>(
            x, gamma, beta, y, mxscale, rstd, workspace, tiling);
    };
    ICPU_RUN_KF(
        rms_norm_dynamic_mx_quant_wrapper, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
        (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}