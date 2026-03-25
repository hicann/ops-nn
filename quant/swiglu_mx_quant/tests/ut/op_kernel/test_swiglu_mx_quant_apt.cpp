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
#include "../../../op_kernel/arch35/swiglu_mx_quant_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void swiglu_mx_quant(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale,
                                                          GM_ADDR workspace, GM_ADDR tiling);

class SwigluMxQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "SwigluMxQuantKernelTest SetUp" << endl;
    }
    static void TearDownTestCase() {
        cout << "SwigluMxQuantKernelTest TearDown" << endl;
    }
};

static void RunKernelTest(size_t batchSize, size_t seqLen, size_t hiddenDim, int64_t roundMode, int64_t scaleAlg, uint64_t tilingKey)
{
    size_t inputSize = batchSize * seqLen * hiddenDim * 2 * sizeof(uint16_t);
    size_t outputSize = batchSize * seqLen * hiddenDim * sizeof(uint8_t);
    size_t scaleSize = batchSize * seqLen * ((hiddenDim + 63) / 64) * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(SwigluMxQuantTilingData);
    size_t workspaceSize = 32;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* group_index = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    SwigluMxQuantTilingData* tilingData = reinterpret_cast<SwigluMxQuantTilingData*>(tiling);
    tilingData->usedCoreNum = 24;
    tilingData->inputDim1 = seqLen;
    tilingData->inputDim2 = hiddenDim * 2;
    tilingData->outputDim2 = hiddenDim;
    tilingData->basicDim2 = 256;
    tilingData->basicDim1 = 1;
    tilingData->maxBasicNumUbDim2 = 1;
    tilingData->maxBasicNumUbDim1 = 1;
    tilingData->ubLoopPerRow = 1;
    tilingData->ubTailPerRow = 0;
    tilingData->frontCoreNum = 0;
    tilingData->frontCoreBasicNumDim1 = 0;
    tilingData->frontCoreLoopTimes = 0;
    tilingData->frontCoreLastLoopBasicNum = 0;
    tilingData->tailCoreBasicNumDim1 = 0;
    tilingData->tailCoreLoopTimes = 0;
    tilingData->tailCoreLastLoopBasicNum = 0;
    tilingData->activateLeft = 0;
    tilingData->swigluMode = 0;
    tilingData->roundMode = roundMode;
    tilingData->scaleAlg = scaleAlg;
    tilingData->groupMode = 0;
    tilingData->groupIndexNum = 0;
    tilingData->clampLimit = 7.0f;
    tilingData->gluAlpha = 1.702f;
    tilingData->gluBias = 1.0f;
    tilingData->maxDtypeValue = 0.0f;

    uint32_t blockDim = 24;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(swiglu_mx_quant, blockDim, x, group_index, y, mxscale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluMxQuantKernelTest, test_swiglu_mx_quant_fp16_to_fp8_e4m3)
{
    size_t batchSize = 8;
    size_t seqLen = 128;
    size_t hiddenDim = 4096;
    int64_t roundMode = 4;
    int64_t scaleAlg = 0;
    uint64_t tilingKey = 1000;
    RunKernelTest(batchSize, seqLen, hiddenDim, roundMode, scaleAlg, tilingKey);
}

TEST_F(SwigluMxQuantKernelTest, test_swiglu_mx_quant_bf16_to_fp8_e5m2)
{
    size_t batchSize = 4;
    size_t seqLen = 64;
    size_t hiddenDim = 1024;
    int64_t roundMode = 4;
    int64_t scaleAlg = 0;
    uint64_t tilingKey = 1000;
    RunKernelTest(batchSize, seqLen, hiddenDim, roundMode, scaleAlg, tilingKey);
}

TEST_F(SwigluMxQuantKernelTest, test_swiglu_mx_quant_fp16_to_fp4_e2m1)
{
    size_t batchSize = 4;
    size_t seqLen = 256;
    size_t hiddenDim = 2048;
    int64_t roundMode = 4;
    int64_t scaleAlg = 0;
    uint64_t tilingKey = 1000;
    RunKernelTest(batchSize, seqLen, hiddenDim, roundMode, scaleAlg, tilingKey);
}

TEST_F(SwigluMxQuantKernelTest, test_swiglu_mx_quant_scale_alg_1)
{
    size_t batchSize = 4;
    size_t seqLen = 128;
    size_t hiddenDim = 2048;
    int64_t roundMode = 4;
    int64_t scaleAlg = 1;
    uint64_t tilingKey = 1000;
    RunKernelTest(batchSize, seqLen, hiddenDim, roundMode, scaleAlg, tilingKey);
}

TEST_F(SwigluMxQuantKernelTest, test_swiglu_mx_quant_small_shape)
{
    size_t batchSize = 1;
    size_t seqLen = 1;
    size_t hiddenDim = 64;
    int64_t roundMode = 4;
    int64_t scaleAlg = 0;
    uint64_t tilingKey = 1000;
    RunKernelTest(batchSize, seqLen, hiddenDim, roundMode, scaleAlg, tilingKey);
}
