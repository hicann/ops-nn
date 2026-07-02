/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void swiglu_group_quant(
    GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR scale, GM_ADDR y, GM_ADDR yScale, GM_ADDR yOrigin,
    GM_ADDR workspace, GM_ADDR tiling);

namespace {
class SwigluGroupQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwigluGroupQuantKernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwigluGroupQuantKernelTest TearDown" << std::endl;
    }
};

void RunKernelWithTilingKey(uint64_t tilingKey, bool outputOrigin)
{
    constexpr int64_t bs = 2;
    constexpr int64_t d = 256;
    constexpr int64_t splitD = d / 2;
    constexpr int64_t scaleCol = 1;
    constexpr uint32_t blockDim = 2;

    const size_t inputSize = bs * d * sizeof(half);
    const size_t outputYSize = bs * splitD * sizeof(uint8_t);
    const size_t outputScaleSize = bs * scaleCol * sizeof(float);
    const size_t yOriginSize = bs * splitD * sizeof(half);
    const size_t tilingDataSize = sizeof(SwigluGroupQuantTilingData);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputYSize));
    uint8_t* yScale = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputScaleSize));
    uint8_t* yOrigin = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yOriginSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SwigluGroupQuantTilingData*>(tiling);
    tilingData->bs = bs;
    tilingData->d = d;
    tilingData->splitD = splitD;
    tilingData->scaleCol = scaleCol;
    tilingData->rowOfFormerBlock = 1;
    tilingData->rowOfTailBlock = 1;
    tilingData->rowLoopOfFormerBlock = 1;
    tilingData->rowLoopOfTailBlock = 1;
    tilingData->rowFactor = 1;
    tilingData->tailRowFactorOfFormerBlock = 1;
    tilingData->tailRowFactorOfTailBlock = 1;
    tilingData->dLoop = 1;
    tilingData->dFactor = splitD;
    tilingData->tailDFactor = splitD;
    tilingData->roundScale = 0;
    tilingData->outputOrigin = outputOrigin ? 1 : 0;
    tilingData->clampLimit = 0.0f;
    tilingData->hasClampLimit = 0;
    tilingData->g = 0;
    tilingData->ubSize = 253952;
    tilingData->gLoop = 0;
    tilingData->gFactor = 0;
    tilingData->tailGFactor = 0;
    tilingData->coreNum = blockDim;

    ICPU_SET_TILING_KEY(tilingKey);
    auto swigluGroupQuantKernel = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR scale, GM_ADDR y,
                                      GM_ADDR yScale, GM_ADDR yOrigin, GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_group_quant(x, weight, groupIndex, scale, y, yScale, yOrigin, workspace, tiling);
    };
    ICPU_RUN_KF(swigluGroupQuantKernel, blockDim, x, nullptr, nullptr, nullptr, y, yScale, yOrigin, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(yScale);
    AscendC::GmFree(yOrigin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluGroupQuantKernelTest, block_fp8)
{
    RunKernelWithTilingKey(1000, false);
}

TEST_F(SwigluGroupQuantKernelTest, block_fp8_y_origin)
{
    RunKernelWithTilingKey(1100, true);
}
} // namespace
