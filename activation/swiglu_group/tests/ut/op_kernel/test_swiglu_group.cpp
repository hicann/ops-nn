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

extern "C" __global__ __aicore__ void swiglu_group(
    GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {
class SwigluGroupKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwigluGroupKernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwigluGroupKernelTest TearDown" << std::endl;
    }
};

void FillCommonTiling(SwigluGroupTilingData* tilingData, int64_t bs, int64_t d, uint32_t coreNum, float clampLimit)
{
    int64_t splitD = d / 2;
    tilingData->bs = bs;
    tilingData->d = d;
    tilingData->splitD = splitD;
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
    tilingData->clampLimit = clampLimit;
    tilingData->hasClampLimit = (clampLimit > 0.0f) ? 1 : 0;
    tilingData->g = 0;
    tilingData->ubSize = 253952;
    tilingData->gLoop = 0;
    tilingData->gFactor = 0;
    tilingData->tailGFactor = 0;
    tilingData->coreNum = coreNum;
}

// Run swiglu_group without group_index (optionally with clamp).
void RunKernelBasic(float clampLimit)
{
    constexpr int64_t bs = 2;
    constexpr int64_t d = 256;
    constexpr int64_t splitD = d / 2;
    constexpr uint32_t blockDim = 2;

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SwigluGroupTilingData)));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    FillCommonTiling(reinterpret_cast<SwigluGroupTilingData*>(tiling), bs, d, blockDim, clampLimit);

    ICPU_SET_TILING_KEY(1000);
    auto kernel = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_group(x, weight, groupIndex, y, workspace, tiling);
    };
    ICPU_RUN_KF(kernel, blockDim, x, nullptr, nullptr, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Run swiglu_group with a large group_index (g >= 97) to exercise VFProcessGroupIndexLargeVf.
void RunKernelGroupLarge(int64_t g)
{
    constexpr int64_t bs = 4;
    constexpr int64_t d = 256;
    constexpr int64_t splitD = d / 2;
    constexpr uint32_t blockDim = 4;

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* groupIndex = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(g * sizeof(int64_t)));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SwigluGroupTilingData)));

    // group_index counts sum to bs so realBs == bs (all rows processed).
    auto* gi = reinterpret_cast<int64_t*>(groupIndex);
    for (int64_t i = 0; i < g; i++) {
        gi[i] = 0;
    }
    gi[g - 1] = bs;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SwigluGroupTilingData*>(tiling);
    FillCommonTiling(tilingData, bs, d, blockDim, 0.0f);
    tilingData->g = g;
    tilingData->gLoop = 1;
    tilingData->gFactor = g;
    tilingData->tailGFactor = g;

    ICPU_SET_TILING_KEY(1000);
    auto kernel = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_group(x, weight, groupIndex, y, workspace, tiling);
    };
    ICPU_RUN_KF(kernel, blockDim, x, nullptr, groupIndex, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluGroupKernelTest, block_fp16)
{
    RunKernelBasic(-1.0f);
}

TEST_F(SwigluGroupKernelTest, block_fp16_clamp)
{
    RunKernelBasic(1.0f);
}

TEST_F(SwigluGroupKernelTest, group_index_large)
{
    RunKernelGroupLarge(97);
}
} // namespace
