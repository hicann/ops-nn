/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "rms_norm_quant_v2_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void rms_norm_quant_v2(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2,
    GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace, GM_ADDR tiling);

class rms_norm_quant_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rms_norm_quant_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rms_norm_quant_v2_test TearDown\n" << endl;
    }
};

TEST_F(rms_norm_quant_v2_test, test_case_0)
{
    size_t inputXSize = 294 * 4628 * sizeof(DTYPE_X);
    size_t inputScalesSize = 4628 * sizeof(DTYPE_SCALES1);
    size_t outputYSize = 294 * 4628 * sizeof(DTYPE_Y1);
    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 38;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
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
    tilingDatafromBin->avgFactor = 1.0 / 294;

    ICPU_SET_TILING_KEY(5000);
    ICPU_RUN_KF(
        rms_norm_quant_v2, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(rms_norm_quant_v2_test, test_case_1)
{
    size_t inputXSize = 365 * 10116 * sizeof(DTYPE_X);
    size_t inputScalesSize = 10116 * sizeof(DTYPE_SCALES1);
    size_t outputYSize = 365 * 10116 * sizeof(DTYPE_Y1);
    size_t tiling_data_size = sizeof(RmsNormQuantV2RegbaseFullLoadTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputScalesSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 61;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
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

    ICPU_SET_TILING_KEY(6000);
    ICPU_RUN_KF(
        rms_norm_quant_v2, blockDim, x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zero_points1);
    AscendC::GmFree(zero_points2);
    AscendC::GmFree(beta);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
