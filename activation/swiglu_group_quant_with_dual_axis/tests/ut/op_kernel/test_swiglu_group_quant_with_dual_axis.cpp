/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/swiglu_group_quant_with_dual_axis.cpp"

namespace {
class SwigluGroupQuantWithDualAxisKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluGroupQuantWithDualAxisKernelTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SwigluGroupQuantWithDualAxisKernelTest TearDown" << std::endl; }
};

constexpr int64_t T = 64;
constexpr int64_t H = 256;
constexpr int64_t GROUP_COUNT = 2;

template <bool hasGroup, bool hasClamp, uint64_t mode = TPL_MODE_BLOCK, bool hasAttrs = hasClamp>
void RunDualAxisKernel(bool hasWeight, bool outputOrigin, float clampLimit, bool zeroInput)
{
    constexpr uint32_t BLOCK_DIM = mode == TPL_MODE_ROTATE ? 2 : 1;
    const size_t inputSize = T * 2 * H * sizeof(half);
    const size_t outputYSize = T * H;
    const size_t scale1Size = T * ((H / 32 + 1) / 2) * 2;
    const int64_t scale2Rows = hasGroup ? (T / 64 + GROUP_COUNT) : (T / 64 + (T % 64 != 0 ? 1 : 0));
    const size_t scale2Size = scale2Rows * H * 2;
    const size_t originSize = outputOrigin ? T * H * sizeof(half) : 64;
    const size_t weightSize = hasWeight ? T * sizeof(float) : 64;
    const size_t groupIndexSize = hasGroup ? GROUP_COUNT * sizeof(int64_t) : 64;
    const size_t tilingDataSize = sizeof(SwigluGroupQuantWithDualAxisTilingData);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inputSize));
    uint8_t* weight = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(weightSize));
    uint8_t* groupIndex = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(groupIndexSize));
    uint8_t* y1 = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputYSize));
    uint8_t* mxScale1 = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(scale1Size));
    uint8_t* y2 = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputYSize));
    uint8_t* mxScale2 = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(scale2Size));
    uint8_t* yOrigin = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(originSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    if (zeroInput) {
        std::memset(x, 0, inputSize);
    } else {
        for (size_t i = 0; i < inputSize; ++i) {
            x[i] = static_cast<uint8_t>(i % 23);
        }
    }
    std::memset(weight, 0, weightSize);
    std::memset(groupIndex, 0, groupIndexSize);
    if (hasGroup) {
        int64_t* endpoints = reinterpret_cast<int64_t*>(groupIndex);
        for (int64_t g = 0; g < GROUP_COUNT; ++g) {
            endpoints[g] = T * (g + 1) / GROUP_COUNT;
        }
    }
    std::memset(yOrigin, 0xFF, originSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    std::memset(tiling, 0, tilingDataSize);
    auto* tilingData = reinterpret_cast<SwigluGroupQuantWithDualAxisTilingData*>(tiling);
    uint32_t flags = 0U;
    flags |= hasWeight ? MX_HAS_WEIGHT : 0U;
    flags |= hasGroup ? MX_HAS_GROUP : 0U;
    flags |= clampLimit > 0.0F ? MX_HAS_CLAMP : 0U;
    flags |= outputOrigin ? MX_OUTPUT_ORIGIN : 0U;
    tilingData->version = 2;
    tilingData->quantMode = 1;
    tilingData->inputType = 0;
    tilingData->weightType = 2; // FLOAT32
    tilingData->flags = flags;
    tilingData->t = T;
    tilingData->h = H;
    tilingData->keep = T;
    tilingData->groupCount = hasGroup ? GROUP_COUNT : 0;
    tilingData->batchRows = T;
    tilingData->alpha = hasAttrs ? 1.702f : 1.0f;
    tilingData->bias = 0.0f;
    tilingData->clampLimit = clampLimit;
    tilingData->usedCoreCount = BLOCK_DIM;

    auto swigluGroupQuantWithDualAxisKernel = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y1,
                                                 GM_ADDR mxScale1, GM_ADDR y2, GM_ADDR mxScale2, GM_ADDR yOrigin,
                                                 GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_group_quant_with_dual_axis<mode, hasGroup, hasAttrs, hasClamp>(x, weight, groupIndex, y1, mxScale1, y2,
                                                                                mxScale2, yOrigin, workspace, tiling);
    };
    ICPU_RUN_KF(swigluGroupQuantWithDualAxisKernel, BLOCK_DIM, x, weight, groupIndex, y1, mxScale1, y2, mxScale2,
                yOrigin, workspace, tiling);

    if (zeroInput && outputOrigin) {
        // Zero activation must round-trip to a zero y_origin.
        const half* origin = reinterpret_cast<const half*>(yOrigin);
        for (int64_t i = 0; i < T * H; ++i) {
            EXPECT_NEAR(static_cast<float>(origin[i]), 0.0f, 1e-6f);
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(y1);
    AscendC::GmFree(mxScale1);
    AscendC::GmFree(y2);
    AscendC::GmFree(mxScale2);
    AscendC::GmFree(yOrigin);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluGroupQuantWithDualAxisKernelTest, non_group_no_weight)
{
    RunDualAxisKernel<false, false>(false, false, 0.0f, false);
}

TEST_F(SwigluGroupQuantWithDualAxisKernelTest, group_weight_origin_zero_input)
{
    RunDualAxisKernel<true, true>(true, true, 7.0f, true);
}
TEST_F(SwigluGroupQuantWithDualAxisKernelTest, template_key_combinations)
{
    RunDualAxisKernel<false, false, TPL_MODE_ROTATE, false>(false, true, -1.0f, true);
    RunDualAxisKernel<false, false, TPL_MODE_ROTATE, true>(false, true, -1.0f, true);
    RunDualAxisKernel<false, true, TPL_MODE_ROTATE, true>(false, true, 7.0f, true);
    RunDualAxisKernel<true, false, TPL_MODE_ROTATE, false>(false, true, -1.0f, true);
    RunDualAxisKernel<true, false, TPL_MODE_ROTATE, true>(false, true, -1.0f, true);
    RunDualAxisKernel<true, true, TPL_MODE_ROTATE, true>(false, true, 7.0f, true);
    RunDualAxisKernel<false, false, TPL_MODE_BLOCK, false>(false, true, -1.0f, true);
    RunDualAxisKernel<false, false, TPL_MODE_BLOCK, true>(false, true, -1.0f, true);
    RunDualAxisKernel<false, true, TPL_MODE_BLOCK, true>(false, true, 7.0f, true);
    RunDualAxisKernel<true, false, TPL_MODE_BLOCK, false>(false, true, -1.0f, true);
    RunDualAxisKernel<true, false, TPL_MODE_BLOCK, true>(false, true, -1.0f, true);
    RunDualAxisKernel<true, true, TPL_MODE_BLOCK, true>(false, true, 7.0f, true);
}
} // namespace
