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
 * \file test_instance_norm_apt.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void instance_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean_out, GM_ADDR variance_out, GM_ADDR workspace,
    GM_ADDR tiling);

class instance_norm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << " instance_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << " instance_norm_test TearDown\n" << endl;
    }
};

TEST_F(instance_norm_test, test_case_200000)
{
    int64_t numN = 64;
    int64_t numC = 1;
    int64_t numR = 4;
    size_t xByteSize = numN * numC * numR * sizeof(half);
    size_t gammaByteSize = numC * sizeof(half);

    size_t tiling_data_size = sizeof(InstanceNormARFullReduceTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    InstanceNormARFullReduceTilingData* tilingDatafromBin =
        reinterpret_cast<InstanceNormARFullReduceTilingData*>(tiling);

    tilingDatafromBin->numN = 64;
    tilingDatafromBin->numC = 1;
    tilingDatafromBin->numR = 4;
    tilingDatafromBin->rAlign = 16;
    tilingDatafromBin->cInner = 1;
    tilingDatafromBin->cOuter = 1;
    tilingDatafromBin->cTail = 1;
    tilingDatafromBin->binaryAddQuotient = 8;
    tilingDatafromBin->perCoreCnt = 1;
    tilingDatafromBin->epsilon = 0.0001;
    tilingDatafromBin->avgFactor = 1.0 / numR;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(200000);
    ICPU_RUN_KF(
        instance_norm, blockDim, x, gamma, beta, y, mean_out, variance_out, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(instance_norm_test, test_case_300000)
{
    int64_t numN = 2;
    int64_t numC = 1;
    int64_t numR = 22499;
    size_t xByteSize = numN * numC * numR * sizeof(half);
    size_t gammaByteSize = numC * sizeof(half);

    size_t tiling_data_size = sizeof(InstanceNormARWelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    InstanceNormARWelfordTilingData* tilingDatafromBin =
        reinterpret_cast<InstanceNormARWelfordTilingData*>(tiling);

    tilingDatafromBin->a1 = 2;
    tilingDatafromBin->a0 = 1;
    tilingDatafromBin->r = 22499;
    tilingDatafromBin->blockNum = 2;
    tilingDatafromBin->totalTiles = 2;
    tilingDatafromBin->tilesPerCore = 1;
    tilingDatafromBin->a0Outer = 1;
    tilingDatafromBin->a0Inner = 128;
    tilingDatafromBin->a0Tail = 1;
    tilingDatafromBin->welfordTileLength = 10304;
    tilingDatafromBin->welfordTempSize = 83968;
    tilingDatafromBin->welfordUpdateTimes = 2;
    tilingDatafromBin->welfordUpdateTail = 1891;
    tilingDatafromBin->apiTempBufferSize = 42240;
    tilingDatafromBin->epsilon = 0.001;
    ICPU_SET_TILING_KEY(300000);
    ICPU_RUN_KF(
        instance_norm, blockDim, x, gamma, beta, y, mean_out, variance_out, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(instance_norm_test, test_case_400000)
{
    int64_t numN = 33;
    int64_t numC = 207;
    int64_t numR = 51;
    size_t xByteSize = numN * numC * numR * sizeof(half);
    size_t gammaByteSize = numC * sizeof(half);

    size_t tiling_data_size = sizeof(InstanceNormARAFullReduceTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    InstanceNormARAFullReduceTilingData* tilingDatafromBin =
        reinterpret_cast<InstanceNormARAFullReduceTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = 33;
    tilingDatafromBin->totalTiles = 66;
    tilingDatafromBin->tilesPerCore = 2;
    tilingDatafromBin->totalA1Len = 33;
    tilingDatafromBin->totalRLen = 51;
    tilingDatafromBin->totalA0Len = 207;
    tilingDatafromBin->a0Outer = 2;
    tilingDatafromBin->tileA0Len = 192;
    tilingDatafromBin->tileA0Tail = 15;
    tilingDatafromBin->powerOfTwoForR = 32;
    tilingDatafromBin->binaryAddQuotient = 32;
    tilingDatafromBin->binaryAddK = 1;
    tilingDatafromBin->binaryAddLast = 0;
    tilingDatafromBin->epsilon = 0.001;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(400000);
    ICPU_RUN_KF(
        instance_norm, blockDim, x, gamma, beta, y, mean_out, variance_out, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(instance_norm_test, test_case_500000)
{
    int64_t numN = 2;
    int64_t numC = 1000;
    int64_t numR = 2;
    size_t xByteSize = numN * numC * numR * sizeof(half);
    size_t gammaByteSize = numC * sizeof(half);

    size_t tiling_data_size = sizeof(InstanceNormARAWelfordTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mean_out = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* variance_out = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    InstanceNormARAWelfordTilingData* tilingDatafromBin =
        reinterpret_cast<InstanceNormARAWelfordTilingData*>(tiling);

    tilingDatafromBin->a1 = 2;
    tilingDatafromBin->r = 1000;
    tilingDatafromBin->a0 = 2;
    tilingDatafromBin->usedCoreNum = 2;
    tilingDatafromBin->totalTiles = 2;
    tilingDatafromBin->tilesPerCore = 1;
    tilingDatafromBin->a0Outer = 1;
    tilingDatafromBin->tileA0Len = 128;
    tilingDatafromBin->tileA0Tail = 2;
    tilingDatafromBin->welfordrFactor = 120;
    tilingDatafromBin->binaryAddQuotient = 64;
    tilingDatafromBin->binaryAddK = 2;
    tilingDatafromBin->binaryAddLast = 0;
    tilingDatafromBin->powerOfTwoForR = 1024;
    tilingDatafromBin->epsilon = 0.001;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(500000);
    ICPU_RUN_KF(
        instance_norm, blockDim, x, gamma, beta, y, mean_out, variance_out, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean_out);
    AscendC::GmFree(variance_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}