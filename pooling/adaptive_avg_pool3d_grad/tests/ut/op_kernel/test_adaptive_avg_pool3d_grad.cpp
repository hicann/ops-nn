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
#include "data_utils.h"

#include "adaptive_avg_pool3d_grad_tiling_def.h"

using namespace std;

extern "C" void adaptive_avg_pool3d_grad(GM_ADDR yGrad, GM_ADDR x, GM_ADDR xGrad, GM_ADDR workspace, GM_ADDR tiling);

struct AdaptiveAvgPool3dGradTestParam {
    string case_name;

    int64_t N = 0;
    int64_t C = 0;

    int64_t inD = 0;
    int64_t inH = 0;
    int64_t inW = 0;

    int64_t outD = 0;
    int64_t outH = 0;
    int64_t outW = 0;

    size_t dataTypeSize;

    uint32_t tilingKey;
    AdaptiveAvgPool3dGradTilingData tiling;
};

class AdaptiveAvgPool3dGradTest : public testing::TestWithParam<AdaptiveAvgPool3dGradTestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool3dGradTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveAvgPool3dGradTest TearDown" << std::endl;
    }
};

TEST_P(AdaptiveAvgPool3dGradTest, test_case_adaptive_avg_pool3d_grad)
{
    AdaptiveAvgPool3dGradTestParam param = GetParam();

    int64_t N = param.N;
    int64_t C = param.C;
    int64_t inD = param.inD;
    int64_t inH = param.inH;
    int64_t inW = param.inW;
    int64_t outD = param.outD;
    int64_t outH = param.outH;
    int64_t outW = param.outW;

    uint32_t tilingKey = param.tilingKey;
    AdaptiveAvgPool3dGradTilingData tilingParam = param.tiling;

    int64_t yGradShapeSize = N * C * outD * outH * outW;
    int64_t xShapeSize = N * C * inD * inH * inW;
    int64_t xGradShapeSize = N * C * inD * inH * inW;
    int64_t yGradByteSize = yGradShapeSize * param.dataTypeSize;
    int64_t xByteSize = xShapeSize * param.dataTypeSize;
    int64_t xGradByteSize = xGradShapeSize * param.dataTypeSize;

    int64_t workspaceSize = xGradShapeSize * 4;
    int64_t tilingSize = sizeof(AdaptiveAvgPool3dGradTilingData);

    uint8_t* yGrad = (uint8_t*)AscendC::GmAlloc(yGradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(xGradByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AdaptiveAvgPool3dGradTilingData* tilingData = reinterpret_cast<AdaptiveAvgPool3dGradTilingData*>(tiling);
    tilingData->ncNum = tilingParam.ncNum;
    tilingData->dIn = tilingParam.dIn;
    tilingData->hIn = tilingParam.hIn;
    tilingData->wIn = tilingParam.wIn;

    tilingData->dOut = tilingParam.dOut;
    tilingData->hOut = tilingParam.hOut;
    tilingData->wOut = tilingParam.wOut;

    tilingData->taskCoreUsed = tilingParam.taskCoreUsed;
    tilingData->taskNumPerCore = tilingParam.taskNumPerCore;
    tilingData->taskNumLastCore = tilingParam.taskNumLastCore;
    tilingData->yNumPerCalc = tilingParam.yNumPerCalc;

    tilingData->ncSliceNum = tilingParam.ncSliceNum;
    tilingData->ncAlignSliceLength = tilingParam.ncAlignSliceLength;
    tilingData->ncAlignSliceTail = tilingParam.ncAlignSliceTail;
    tilingData->isAtomicAdd = tilingParam.isAtomicAdd;
    tilingData->deterministicFlag = tilingParam.deterministicFlag;

    uint32_t blockDim = tilingParam.taskCoreUsed;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(adaptive_avg_pool3d_grad, blockDim, yGrad, x, xGrad, workspace, tiling);

    AscendC::GmFree((void*)yGrad);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)xGrad);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

static AdaptiveAvgPool3dGradTestParam cases[] = {
    {"test_case_float32",
     1,
     256,
     64,
     24,
     24,
     16,
     12,
     12,
     sizeof(float),
     0,
     {256, 64, 24, 24, 16, 12, 12, 48, 48, 48, 48, 1, 256, 256, 1, 0}},
    {"test_case_float16",
     1,
     256,
     64,
     24,
     24,
     16,
     12,
     12,
     sizeof(half),
     10,
     {256, 64, 24, 24, 16, 12, 12, 48, 48, 48, 24, 1, 256, 256, 1, 0}},
    {"test_case_Bfloat16",
     1,
     256,
     64,
     20,
     20,
     16,
     12,
     12,
     sizeof(bfloat16_t),
     20,
     {256, 64, 20, 20, 16, 12, 12, 48, 48, 48, 48, 1, 256, 256, 1, 0}},
    {"test_case_nc_large_float32",
     1,
     40960,
     24,
     1,
     1,
     24,
     1,
     1,
     sizeof(float),
     1,
     {40960, 24, 1, 1, 24, 1, 1, 48, 1, 1, 1, 2, 20480, 20480, 1, 0}},
    {"test_case_nc_large_float16",
     1,
     20480,
     25,
     1,
     1,
     24,
     1,
     1,
     sizeof(half),
     11,
     {20480, 25, 1, 1, 24, 1, 1, 48, 1, 1, 1, 2, 10240, 10240, 1, 0}},
};

INSTANTIATE_TEST_CASE_P(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradTest, testing::ValuesIn(cases));
