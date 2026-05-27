/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "rms_norm_quant_v3_tiling_def.h"

#ifdef __CCE_KT_TEST__
#endif

using namespace std;

extern "C" __global__ __aicore__ void rms_norm_quant_v3(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2,
    GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling);

class rms_norm_quant_v3_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rms_norm_quant_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rms_norm_quant_v3_test TearDown\n" << endl;
    }
};

TEST_F(rms_norm_quant_v3_test, test_case_full_load)
{
    int64_t numA = 294;
    int64_t numR = 4628;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = ((numR * sizeof(DTYPE_SCALES1) + 31) / 32) * 32;
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 38;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 294;
    tilingDatafromBin->r = 4628;
    tilingDatafromBin->q = 4628;
    tilingDatafromBin->blockFactor = 5;
    tilingDatafromBin->blockTail = 4;
    tilingDatafromBin->ubFactor = 2;
    tilingDatafromBin->binaryAdd = 4096;
    tilingDatafromBin->optionMask = 15;
    tilingDatafromBin->divMode = 1;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 0.001;
    tilingDatafromBin->avgFactor = 1.0 / 4628;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_recompute)
{
    int64_t numA = 365;
    int64_t numR = 10116;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseRecomputeTilingData);
    uint32_t blockDim = 61;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseRecomputeTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseRecomputeTilingData*>(tiling);

    tilingDatafromBin->numM = 365;
    tilingDatafromBin->numN = 10116;
    tilingDatafromBin->baseM = 64;
    tilingDatafromBin->baseN = 2048;
    tilingDatafromBin->mPerCore = 6;
    tilingDatafromBin->mLastCore = 5;
    tilingDatafromBin->nUbLoops = 5;
    tilingDatafromBin->binAddQuotient = 1024;
    tilingDatafromBin->powerSplit = 4;
    tilingDatafromBin->mainFoldCount = 0;
    tilingDatafromBin->foldTail = 1924;
    tilingDatafromBin->optionMask = 15;
    tilingDatafromBin->divMode = 1;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 0.001;
    tilingDatafromBin->avgFactor = 1.0 / 10116;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(6000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_full_load_single_batch_no_rstd)
{
    // a=1, r=256, q=1 scalar scales, rstdFlag=0, divMode=0, no optional inputs
    int64_t numA = 1;
    int64_t numR = 256;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 1;
    tilingDatafromBin->r = 256;
    tilingDatafromBin->q = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->ubFactor = 1;
    tilingDatafromBin->binaryAdd = 128;
    tilingDatafromBin->optionMask = 16;
    tilingDatafromBin->divMode = 0;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->avgFactor = 1.0 / 256;
    tilingDatafromBin->rstdFlag = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_full_load_large_batch_per_channel)
{
    // a=512, r=512, q=512 per-channel scales, no optional inputs (optionMask=0)
    int64_t numA = 512;
    int64_t numR = 512;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 64;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 512;
    tilingDatafromBin->r = 512;
    tilingDatafromBin->q = 512;
    tilingDatafromBin->blockFactor = 8;
    tilingDatafromBin->blockTail = 8;
    tilingDatafromBin->ubFactor = 8;
    tilingDatafromBin->binaryAdd = 256;
    tilingDatafromBin->optionMask = 0;
    tilingDatafromBin->divMode = 1;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avgFactor = 1.0 / 512;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_full_load_non_aligned_r)
{
    // a=100, r=1000 (non-aligned to power of 2), q=1, optionMask with scales2+beta
    int64_t numA = 100;
    int64_t numR = 1000;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 50;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 100;
    tilingDatafromBin->r = 1000;
    tilingDatafromBin->q = 1;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->blockTail = 2;
    tilingDatafromBin->ubFactor = 2;
    tilingDatafromBin->binaryAdd = 512;
    tilingDatafromBin->optionMask = 25;
    tilingDatafromBin->divMode = 1;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 0.001;
    tilingDatafromBin->avgFactor = 1.0 / 1000;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_full_load_ub_boundary)
{
    // a=8, r=6144, q=1, ubFactor=1 (UB near limit), all optionals present
    int64_t numA = 8;
    int64_t numR = 6144;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 8;
    tilingDatafromBin->r = 6144;
    tilingDatafromBin->q = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->ubFactor = 1;
    tilingDatafromBin->binaryAdd = 4096;
    tilingDatafromBin->optionMask = 31;
    tilingDatafromBin->divMode = 1;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avgFactor = 1.0 / 6144;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_recompute_large_r_no_rstd)
{
    // r=20000 > 16384, rstdFlag=0, all optional inputs
    int64_t numA = 16;
    int64_t numR = 20000;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseRecomputeTilingData);
    uint32_t blockDim = 16;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseRecomputeTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseRecomputeTilingData*>(tiling);

    tilingDatafromBin->numM = 16;
    tilingDatafromBin->numN = 20000;
    tilingDatafromBin->baseM = 64;
    tilingDatafromBin->baseN = 512;
    tilingDatafromBin->mPerCore = 1;
    tilingDatafromBin->mLastCore = 1;
    tilingDatafromBin->nUbLoops = 40;
    tilingDatafromBin->binAddQuotient = 256;
    tilingDatafromBin->powerSplit = 32;
    tilingDatafromBin->mainFoldCount = 7;
    tilingDatafromBin->foldTail = 32;
    tilingDatafromBin->optionMask = 15;
    tilingDatafromBin->divMode = 1;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 0.001;
    tilingDatafromBin->avgFactor = 1.0 / 20000;
    tilingDatafromBin->rstdFlag = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(6000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_full_load_small_a_div_mode_off)
{
    // a=4, r=8192, q=1, divMode=0, no optional extras (optionMask=16), FullLoad path
    int64_t numA = 4;
    int64_t numR = 8192;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 4;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 4;
    tilingDatafromBin->r = 8192;
    tilingDatafromBin->q = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->ubFactor = 1;
    tilingDatafromBin->binaryAdd = 4096;
    tilingDatafromBin->optionMask = 16;
    tilingDatafromBin->divMode = 0;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = 1.0 / 8192;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_full_load_r32_lessThanVL)
{
    int64_t numA = 4;
    int64_t numR = 32;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 4;
    tilingDatafromBin->r = 32;
    tilingDatafromBin->q = 1;
    tilingDatafromBin->blockFactor = 4;
    tilingDatafromBin->blockTail = 4;
    tilingDatafromBin->ubFactor = 4;
    tilingDatafromBin->binaryAdd = 16;
    tilingDatafromBin->optionMask = 10;
    tilingDatafromBin->divMode = 0;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = 1.0 / 32;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_full_load_r96_lessThanTwoVL)
{
    int64_t numA = 4;
    int64_t numR = 96;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 4;
    tilingDatafromBin->r = 96;
    tilingDatafromBin->q = 1;
    tilingDatafromBin->blockFactor = 4;
    tilingDatafromBin->blockTail = 4;
    tilingDatafromBin->ubFactor = 4;
    tilingDatafromBin->binaryAdd = 48;
    tilingDatafromBin->optionMask = 7;
    tilingDatafromBin->divMode = 0;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = 1.0 / 96;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_full_load_r9216_lastLoop2)
{
    int64_t numA = 2;
    int64_t numR = 9216;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 2;
    tilingDatafromBin->r = 9216;
    tilingDatafromBin->q = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->ubFactor = 1;
    tilingDatafromBin->binaryAdd = 4608;
    tilingDatafromBin->optionMask = 5;
    tilingDatafromBin->divMode = 1;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = 1.0 / 9216;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_full_load_beta_only_mul)
{
    int64_t numA = 4;
    int64_t numR = 512;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 4;
    tilingDatafromBin->r = 512;
    tilingDatafromBin->q = 512;
    tilingDatafromBin->blockFactor = 4;
    tilingDatafromBin->blockTail = 4;
    tilingDatafromBin->ubFactor = 4;
    tilingDatafromBin->binaryAdd = 256;
    tilingDatafromBin->optionMask = 8;
    tilingDatafromBin->divMode = 0;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = 1.0 / 512;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_full_load_zp1_s2_div)
{
    int64_t numA = 4;
    int64_t numR = 512;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseFullLoadTilingData*>(tiling);

    tilingDatafromBin->a = 4;
    tilingDatafromBin->r = 512;
    tilingDatafromBin->q = 512;
    tilingDatafromBin->blockFactor = 4;
    tilingDatafromBin->blockTail = 4;
    tilingDatafromBin->ubFactor = 4;
    tilingDatafromBin->binaryAdd = 256;
    tilingDatafromBin->optionMask = 3;
    tilingDatafromBin->divMode = 1;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = 1.0 / 512;
    tilingDatafromBin->rstdFlag = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(5000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_recompute_needbrc_allopts)
{
    int64_t numA = 8;
    int64_t numR = 10240;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseRecomputeTilingData);
    uint32_t blockDim = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseRecomputeTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseRecomputeTilingData*>(tiling);

    tilingDatafromBin->numM = 8;
    tilingDatafromBin->numN = 10240;
    tilingDatafromBin->baseM = 64;
    tilingDatafromBin->baseN = 2048;
    tilingDatafromBin->mPerCore = 1;
    tilingDatafromBin->mLastCore = 1;
    tilingDatafromBin->nUbLoops = 5;
    tilingDatafromBin->binAddQuotient = 1024;
    tilingDatafromBin->powerSplit = 4;
    tilingDatafromBin->mainFoldCount = 1;
    tilingDatafromBin->foldTail = 0;
    tilingDatafromBin->optionMask = 31;
    tilingDatafromBin->divMode = 1;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = 1.0 / 10240;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(6000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_recompute_no_opts_mul)
{
    int64_t numA = 8;
    int64_t numR = 10240;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseRecomputeTilingData);
    uint32_t blockDim = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseRecomputeTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseRecomputeTilingData*>(tiling);

    tilingDatafromBin->numM = 8;
    tilingDatafromBin->numN = 10240;
    tilingDatafromBin->baseM = 64;
    tilingDatafromBin->baseN = 2048;
    tilingDatafromBin->mPerCore = 1;
    tilingDatafromBin->mLastCore = 1;
    tilingDatafromBin->nUbLoops = 5;
    tilingDatafromBin->binAddQuotient = 1024;
    tilingDatafromBin->powerSplit = 4;
    tilingDatafromBin->mainFoldCount = 1;
    tilingDatafromBin->foldTail = 0;
    tilingDatafromBin->optionMask = 0;
    tilingDatafromBin->divMode = 0;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = 1.0 / 10240;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(6000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_recompute_small_baseN48)
{
    int64_t numA = 4;
    int64_t numR = 192;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseRecomputeTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseRecomputeTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseRecomputeTilingData*>(tiling);

    tilingDatafromBin->numM = 4;
    tilingDatafromBin->numN = 192;
    tilingDatafromBin->baseM = 64;
    tilingDatafromBin->baseN = 48;
    tilingDatafromBin->mPerCore = 4;
    tilingDatafromBin->mLastCore = 4;
    tilingDatafromBin->nUbLoops = 4;
    tilingDatafromBin->binAddQuotient = 24;
    tilingDatafromBin->powerSplit = 4;
    tilingDatafromBin->mainFoldCount = 0;
    tilingDatafromBin->foldTail = 0;
    tilingDatafromBin->optionMask = 8;
    tilingDatafromBin->divMode = 0;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = 1.0 / 192;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(6000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v3_test, test_case_recompute_small_baseN96)
{
    int64_t numA = 4;
    int64_t numR = 384;
    size_t inputXSize = numA * numR * sizeof(DTYPE_X);
    size_t inputScalesSize = numR * sizeof(DTYPE_SCALES1);
    size_t outputYSize = numA * numR * sizeof(DTYPE_Y1);
    size_t outputRstdSize = numA * sizeof(float);

    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseRecomputeTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(outputRstdSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    RmsNormQuantV2RegbaseRecomputeTilingData* tilingDatafromBin =
        reinterpret_cast<RmsNormQuantV2RegbaseRecomputeTilingData*>(tiling);

    tilingDatafromBin->numM = 4;
    tilingDatafromBin->numN = 384;
    tilingDatafromBin->baseM = 64;
    tilingDatafromBin->baseN = 96;
    tilingDatafromBin->mPerCore = 4;
    tilingDatafromBin->mLastCore = 4;
    tilingDatafromBin->nUbLoops = 4;
    tilingDatafromBin->binAddQuotient = 48;
    tilingDatafromBin->powerSplit = 4;
    tilingDatafromBin->mainFoldCount = 0;
    tilingDatafromBin->foldTail = 0;
    tilingDatafromBin->optionMask = 5;
    tilingDatafromBin->divMode = 1;
    tilingDatafromBin->dstDtype = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = 1.0 / 384;
    tilingDatafromBin->rstdFlag = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(6000);

    ICPU_RUN_KF(
        rms_norm_quant_v3, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
