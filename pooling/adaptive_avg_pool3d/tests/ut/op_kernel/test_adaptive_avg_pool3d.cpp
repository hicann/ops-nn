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

#include "adaptive_avg_pool3d_tiling_def.h"

using namespace std;

extern "C" void adaptive_avg_pool3d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

struct AdaptiveAvgPool3dTestParam {
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
    AdaptiveAvgPool3dTilingData tiling;
};

class AdaptiveAvgPool3dTest : public testing::TestWithParam<AdaptiveAvgPool3dTestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool3dTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveAvgPool3dTest TearDown" << std::endl;
    }
};

TEST_P(AdaptiveAvgPool3dTest, test_case_adaptive_avg_pool3d)
{
    AdaptiveAvgPool3dTestParam param = GetParam();

    int64_t N = param.N;
    int64_t C = param.C;
    int64_t inD = param.inD;
    int64_t inH = param.inH;
    int64_t inW = param.inW;
    int64_t outD = param.outD;
    int64_t outH = param.outH;
    int64_t outW = param.outW;

    uint32_t tilingKey = param.tilingKey;
    AdaptiveAvgPool3dTilingData tilingParam = param.tiling;

    int64_t inputShapeSize = N * C * inD * inH * inW;
    int64_t outputShapeSize = N * C * outD * outH * outW;
    int64_t inputXByteSize = inputShapeSize * param.dataTypeSize;
    int64_t outputYByteSize = outputShapeSize * param.dataTypeSize;

    int64_t workspaceSize = 32;
    int64_t tilingSize = sizeof(AdaptiveAvgPool3dTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    AdaptiveAvgPool3dTilingData* tilingData = reinterpret_cast<AdaptiveAvgPool3dTilingData*>(tiling);
    tilingData->dimC = tilingParam.dimC;
    tilingData->cTileLength = tilingParam.cTileLength;
    tilingData->inD = tilingParam.inD;
    tilingData->inH = tilingParam.inH;
    tilingData->inW = tilingParam.inW;
    tilingData->outD = tilingParam.outD;
    tilingData->outH = tilingParam.outH;
    tilingData->outW = tilingParam.outW;
    tilingData->formerLength = tilingParam.formerLength;
    tilingData->formerNum = tilingParam.formerNum;
    tilingData->tailLength = tilingParam.tailLength;
    tilingData->tailNum = tilingParam.tailNum;
    tilingData->indexBufLen = tilingParam.indexBufLen;
    tilingData->windowWNum = tilingParam.windowWNum;
    tilingData->maxWindowWLength = tilingParam.maxWindowWLength;
    tilingData->inputTileNum = tilingParam.inputTileNum;
    tilingData->atomicAddNum = tilingParam.atomicAddNum;

    uint32_t blockDim = tilingParam.formerNum + tilingParam.tailNum;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(adaptive_avg_pool3d, blockDim, x, y, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

static AdaptiveAvgPool3dTestParam cases[] = {
    {"test_case_split_c_float32",
     1,
     256,
     64,
     24,
     24,
     16,
     12,
     12,
     sizeof(float),
     12,
     {256, 256, 64, 24, 24, 16, 12, 12, 58, 24, 57, 16, 64, 0, 2, 0, 0}},
    {"test_case_split_c_float16",
     1,
     256,
     64,
     24,
     24,
     16,
     12,
     12,
     sizeof(half),
     11,
     {256, 256, 64, 24, 24, 16, 12, 12, 58, 24, 57, 16, 64, 0, 2, 0, 0}},
    {"test_case_split_c_bfloat16",
     1,
     256,
     64,
     24,
     24,
     16,
     12,
     12,
     sizeof(bfloat16_t),
     10,
     {256, 256, 64, 24, 24, 16, 12, 12, 58, 24, 57, 16, 64, 0, 2, 0, 0}},
    {"test_case_split_w_float32",
     1,
     128,
     64,
     24,
     24,
     16,
     4,
     1,
     sizeof(float),
     22,
     {128, 0, 64, 24, 24, 16, 4, 1, 2, 24, 1, 16, 64, 0, 24, 9, 0}},
    {"test_case_split_w_float16",
     1,
     128,
     64,
     24,
     24,
     16,
     4,
     1,
     sizeof(half),
     21,
     {128, 0, 64, 24, 24, 16, 4, 1, 2, 24, 1, 16, 64, 0, 24, 17, 0}},
    {"test_case_split_w_bfloat16",
     1,
     128,
     64,
     24,
     24,
     16,
     4,
     1,
     sizeof(bfloat16_t),
     20,
     {128, 0, 64, 24, 24, 16, 4, 1, 2, 24, 1, 16, 64, 0, 24, 17, 0}},
    {"test_case_multi_w_float32",
     1,
     128,
     64,
     24,
     24,
     16,
     12,
     12,
     sizeof(float),
     32,
     {128, 0, 64, 24, 24, 16, 12, 12, 58, 24, 57, 16, 64, 2, 2, 0, 0}},
    {"test_case_multi_w_float16",
     1,
     128,
     64,
     24,
     24,
     16,
     12,
     12,
     sizeof(half),
     31,
     {128, 0, 64, 24, 24, 16, 12, 12, 58, 24, 57, 16, 64, 4, 2, 0, 0}},
    {"test_case_multi_w_bfloat16",
     1,
     128,
     64,
     24,
     24,
     16,
     12,
     12,
     sizeof(bfloat16_t),
     30,
     {128, 0, 64, 24, 24, 16, 12, 12, 58, 24, 57, 16, 64, 4, 2, 0, 0}},
};

INSTANTIATE_TEST_CASE_P(AdaptiveAvgPool3d, AdaptiveAvgPool3dTest, testing::ValuesIn(cases));
