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

using namespace std;

extern "C" __global__ __aicore__ void rms_norm_dynamic_mx_quant(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale, GM_ADDR rstd,
    GM_ADDR workspace, GM_ADDR tiling);

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

// numM=1, numN=64, GENERAL mode (key=1000), no beta, no rstd, scaleAlg=OCP, roundMode=RINT
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
    tilingData->numNUbAligned = 64;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 64;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->nMxblockNum = 2;
    tilingData->needPadN = 0;
    tilingData->needPadScale = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(rms_norm_dynamic_mx_quant, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
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

// numM=2, numN=64, GENERAL mode (key=1000), with beta and rstd output
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
    tilingData->numNUbAligned = 64;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 2;
    tilingData->mUbFactor = 2;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 64;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->nMxblockNum = 2;
    tilingData->needPadN = 0;
    tilingData->needPadScale = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 1;
    tilingData->hasOutputRstd = 1;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(rms_norm_dynamic_mx_quant, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
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

// numM=1, numN=50, GENERAL mode (key=1000), N not aligned to mxBlockSize, needs padding
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
    tilingData->numNUbAligned = 64;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = nMxblockAligned;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->nMxblockNum = nMxblockNum;
    tilingData->needPadN = 1;
    tilingData->needPadScale = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(rms_norm_dynamic_mx_quant, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
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

// numM=1, numN=64, OPTIMIZE mode (key=10000), FP16 input + RINT rounding
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
    tilingData->numNUbAligned = 64;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 64;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->nMxblockNum = 2;
    tilingData->needPadN = 0;
    tilingData->needPadScale = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(10000);
    ICPU_RUN_KF(rms_norm_dynamic_mx_quant, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
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

// numM=4, numN=128, GENERAL mode (key=1000), multi-core, with tail cores
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_multicore)
{
    int64_t numM = 4;
    int64_t numN = 128;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * numN * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * 4 * sizeof(uint8_t);
    size_t rstdByteSize = numM * sizeof(float);

    size_t tilingDataSize = sizeof(RmsNormDynamicMxQuantFullLoadTilingData);
    uint32_t blockDim = 2;

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

    tilingData->usedCoreNum = 2;
    tilingData->mTailCores = 0;
    tilingData->numM = numM;
    tilingData->numN = numN;
    tilingData->numNUbAligned = 128;
    tilingData->binAddFoldPoint = 128;
    tilingData->mPerCore = 2;
    tilingData->mUbFactor = 2;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 128;
    tilingData->nMxblockNumAlignedTwo = 4;
    tilingData->nMxblockNum = 4;
    tilingData->needPadN = 0;
    tilingData->needPadScale = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 1;
    tilingData->hasOutputRstd = 1;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(rms_norm_dynamic_mx_quant, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
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

// numM=1, numN=64, GENERAL mode (key=1000), scaleAlg=cuBLAS(1), roundMode=ROUND(0)
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
    tilingData->numNUbAligned = 64;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 64;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->nMxblockNum = 2;
    tilingData->needPadN = 0;
    tilingData->needPadScale = 0;
    tilingData->scaleAlg = 1;
    tilingData->roundMode = 0;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(rms_norm_dynamic_mx_quant, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
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

// numM=5, numN=128, GENERAL mode (key=1000), multi-core with tail cores (3 cores, mTailCores=0, odd distribution)
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_multicore_with_tail)
{
    int64_t numM = 5;
    int64_t numN = 128;
    size_t xByteSize = numM * numN * sizeof(DTYPE_X);
    size_t gammaByteSize = numN * sizeof(DTYPE_GAMMA);
    size_t yByteSize = numM * numN * sizeof(DTYPE_Y);
    size_t mxscaleByteSize = numM * 4 * sizeof(uint8_t);
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

    // 5 rows, 3 cores: mPerCore=1, mTailCores=2 (cores 0,1 get 2 rows, core 2 gets 1 row)
    tilingData->usedCoreNum = 3;
    tilingData->mTailCores = 2;
    tilingData->numM = numM;
    tilingData->numN = numN;
    tilingData->numNUbAligned = 128;
    tilingData->binAddFoldPoint = 128;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 2;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 128;
    tilingData->nMxblockNumAlignedTwo = 4;
    tilingData->nMxblockNum = 4;
    tilingData->needPadN = 0;
    tilingData->needPadScale = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 1;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(rms_norm_dynamic_mx_quant, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
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

// numM=1, numN=64, GENERAL mode (key=1000), roundMode=FLOOR(1)
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
    tilingData->numNUbAligned = 64;
    tilingData->binAddFoldPoint = 64;
    tilingData->mPerCore = 1;
    tilingData->mUbFactor = 1;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = 64;
    tilingData->nMxblockNumAlignedTwo = 2;
    tilingData->nMxblockNum = 2;
    tilingData->needPadN = 0;
    tilingData->needPadScale = 0;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 1;
    tilingData->hasInputBeta = 0;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(rms_norm_dynamic_mx_quant, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
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

// numM=2, numN=96, GENERAL mode, needPadN=0, needPadScale=1 (nMxblockNum=3, aligned to 4)
TEST_F(rms_norm_dynamic_mx_quant_test, test_case_pad_n_and_scale)
{
    int64_t numM = 2;
    int64_t numN = 96;
    int64_t nMxblockAligned = 96;
    int64_t nMxblockNum = 3;
    int64_t nMxblockNumAlignedTwo = 4;
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
    tilingData->numNUbAligned = 96;
    tilingData->binAddFoldPoint = 128;
    tilingData->mPerCore = 2;
    tilingData->mUbFactor = 2;
    tilingData->mxBlockSize = 32;
    tilingData->nMxblockAligned = nMxblockAligned;
    tilingData->nMxblockNumAlignedTwo = nMxblockNumAlignedTwo;
    tilingData->nMxblockNum = nMxblockNum;
    tilingData->needPadN = 0;
    tilingData->needPadScale = 1;
    tilingData->scaleAlg = 0;
    tilingData->roundMode = 4;
    tilingData->hasInputBeta = 1;
    tilingData->hasOutputRstd = 0;
    tilingData->epsilon = 1e-6f;
    tilingData->avgFactor = 1.0f / numN;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(rms_norm_dynamic_mx_quant, blockDim, x, gamma, beta, y, mxscale, rstd, workspace,
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
