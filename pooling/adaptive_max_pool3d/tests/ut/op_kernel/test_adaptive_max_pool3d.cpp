/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

#include "adaptive_max_pool3d_tiling_def.h"

using namespace std;

extern "C" void adaptive_max_pool3d(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling);

struct AdaptiveMaxPool3dTestParam {
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
    AdaptiveMaxPool3dTilingData tiling;
};

class AdaptiveMaxPool3dTest : public testing::TestWithParam<AdaptiveMaxPool3dTestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveMaxPool3dTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveMaxPool3dTest TearDown" << std::endl;
    }
};

class AdaptiveMaxPool3dBigKernelTest : public testing::TestWithParam<AdaptiveMaxPool3dTestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveMaxPool3dTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveMaxPool3dTest TearDown" << std::endl;
    }
};

TEST_P(AdaptiveMaxPool3dTest, test_case_adaptive_max_pool3d)
{
    AdaptiveMaxPool3dTestParam param = GetParam();

    int64_t N = param.N;
    int64_t C = param.C;
    int64_t inD = param.inD;
    int64_t inH = param.inH;
    int64_t inW = param.inW;
    int64_t outD = param.outD;
    int64_t outH = param.outH;
    int64_t outW = param.outW;

    uint32_t tilingKey = param.tilingKey;
    AdaptiveMaxPool3dTilingData tilingParam = param.tiling;

    int64_t inputShapeSize = N * C * inD * inH * inW;
    int64_t outputShapeSize = N * C * outD * outH * outW;
    int64_t inputXByteSize = inputShapeSize * param.dataTypeSize;
    int64_t outputYByteSize = outputShapeSize * param.dataTypeSize;

    int64_t workspaceSize = 32;
    int64_t tilingSize = sizeof(AdaptiveMaxPool3dTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AdaptiveMaxPool3dTilingData* tilingData = reinterpret_cast<AdaptiveMaxPool3dTilingData*>(tiling);
    tilingData->N = tilingParam.N;
    tilingData->C = tilingParam.C;
    tilingData->Di = tilingParam.Di;
    tilingData->Hi = tilingParam.Hi;
    tilingData->Wi = tilingParam.Wi;
    tilingData->Do = tilingParam.Do;
    tilingData->Ho = tilingParam.Ho;
    tilingData->Wo = tilingParam.Wo;
    tilingData->coreNums = tilingParam.coreNums;
    tilingData->useCoreNum = tilingParam.useCoreNum;
    tilingData->totalIdx = tilingParam.totalIdx;
    tilingData->blockFactor = tilingParam.blockFactor;
    tilingData->blockTail = tilingParam.blockTail;
    tilingData->ncFactor = tilingParam.ncFactor;
    tilingData->doFactor = tilingParam.doFactor;
    tilingData->hoFactor = tilingParam.hoFactor;
    tilingData->woFactor = tilingParam.woFactor;
    tilingData->ncOuter = tilingParam.ncOuter;
    tilingData->doOuter = tilingParam.doOuter;
    tilingData->hoOuter = tilingParam.hoOuter;
    tilingData->woOuter = tilingParam.woOuter;
    tilingData->ncTail = tilingParam.ncTail;
    tilingData->doTail = tilingParam.doTail;
    tilingData->hoTail = tilingParam.hoTail;
    tilingData->woTail = tilingParam.woTail;

    uint32_t blockDim = tilingData->useCoreNum;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(adaptive_max_pool3d, blockDim, x, y, indices, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)indices);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

static AdaptiveMaxPool3dTestParam cases[] = {
    {"test_case_split_nc_float32", 1, 64, 1, 1, 1, 1, 1, 1, sizeof(float), 320000UL, {1, 64, 1, 1,  1,  1, 1, 1, 48,
                                                                                      1, 1,  1, 1,  64, 1, 1, 1, 1,
                                                                                      1, 1,  1, 64, 1,  1, 1}}
};

INSTANTIATE_TEST_CASE_P(AdaptiveMaxPool3d, AdaptiveMaxPool3dTest, testing::ValuesIn(cases));

TEST_P(AdaptiveMaxPool3dBigKernelTest, test_case_adaptive_max_pool3d_big_kernel)
{
    AdaptiveMaxPool3dTestParam param = GetParam();

    int64_t N = param.N;
    int64_t C = param.C;
    int64_t inD = param.inD;
    int64_t inH = param.inH;
    int64_t inW = param.inW;
    int64_t outD = param.outD;
    int64_t outH = param.outH;
    int64_t outW = param.outW;

    uint32_t tilingKey = param.tilingKey;
    AdaptiveMaxPool3dTilingData tilingParam = param.tiling;

    int64_t inputShapeSize = N * C * inD * inH * inW;
    int64_t outputShapeSize = N * C * outD * outH * outW;
    int64_t inputXByteSize = inputShapeSize * param.dataTypeSize;
    int64_t outputYByteSize = outputShapeSize * param.dataTypeSize;

    int64_t workspaceSize = 32;
    int64_t tilingSize = sizeof(AdaptiveMaxPool3dBigPoolTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AdaptiveMaxPool3dBigPoolTilingData* tilingData = reinterpret_cast<AdaptiveMaxPool3dBigPoolTilingData*>(tiling);
    tilingData->N = tilingParam.N;
    tilingData->C = tilingParam.C;
    tilingData->Di = tilingParam.Di;
    tilingData->Hi = tilingParam.Hi;
    tilingData->Wi = tilingParam.Wi;
    tilingData->Do = tilingParam.Do;
    tilingData->Ho = tilingParam.Ho;
    tilingData->Wo = tilingParam.Wo;
    tilingData->coreNums = tilingParam.coreNums;
    tilingData->useCoreNum = tilingParam.useCoreNum;
    tilingData->totalIdx = tilingParam.totalIdx;
    tilingData->blockFactor = tilingParam.blockFactor;
    tilingData->blockTail = tilingParam.blockTail;

    uint32_t blockDim = tilingData->useCoreNum;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(adaptive_max_pool3d, blockDim, x, y, indices, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)indices);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

static AdaptiveMaxPool3dTestParam bigCases[] = {
    {"test_case_big_pool_float32", 1, 64, 32, 32, 32, 1, 1, 1, sizeof(float), 310000UL, {1, 64, 32, 32, 32, 1,  1,
                                                                                         1, 40, 40, 64, 1,  24, 0,
                                                                                         0, 0,  0,  0,  0,  0,  0,
                                                                                         0, 0,  0,  0}},
    {"test_case_dhwContinue_dhw_smaller_than_maxcount_float32",
     1,
     51,
     1,
     8,
     69,
     1,
     1,
     1,
     sizeof(float),
     310000UL,
     {1, 51, 1, 8, 69, 1, 1, 1, 40, 40, 51, 1, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {"test_case_wContinue_w_bigger_than_maxcount_float32",
     1,
     62,
     2,
     2,
     20588,
     1,
     1,
     2,
     sizeof(float),
     310000UL,
     {1, 62, 2, 2, 20588, 1, 1, 2, 40, 40, 124, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {"test_case_wContinue_hw_bigger_than_maxcount_float32",
     1,
     62,
     2,
     500,
     100,
     1,
     2,
     2,
     sizeof(float),
     310000UL,
     {1, 62, 2, 500, 100, 1, 2, 2, 40, 40, 248, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    {"test_case_wContinue_dhw_bigger_than_maxcount_float16",
     1,
     62,
     12,
     200,
     200,
     1,
     2,
     2,
     sizeof(half),
     311000UL,
     {1, 62, 12, 200, 200, 1, 2, 2, 40, 40, 124, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
};

INSTANTIATE_TEST_CASE_P(AdaptiveMaxPool3d, AdaptiveMaxPool3dBigKernelTest, testing::ValuesIn(bigCases));
