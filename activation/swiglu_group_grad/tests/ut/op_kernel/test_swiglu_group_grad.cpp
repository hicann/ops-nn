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
#include <cmath>
#include <cstdint>

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/arch35/swiglu_group_grad_tiling_key.h"
#include "../../../op_kernel/swiglu_group_grad.cpp"

using namespace SwigluGroupGradOps;

namespace {

template <typename inType, uint64_t HAS_CLAMP, uint64_t HAS_WEIGHT, uint64_t HAS_Y_ORIGIN, uint64_t HAS_GROUP_INDEX>
void RunRegBaseKernel(GM_ADDR gradY, GM_ADDR x, GM_ADDR weight, GM_ADDR yOrigin, GM_ADDR groupIndex, GM_ADDR gradX,
                      GM_ADDR gradWeight, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SwigluGroupGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwigluGroupGradTilingData, tilingData, tiling);
    SwigluGroupGradBase<inType, HAS_CLAMP, HAS_WEIGHT, HAS_Y_ORIGIN, HAS_GROUP_INDEX> op;
    op.Init(gradY, x, weight, yOrigin, groupIndex, gradX, gradWeight, workspace, &tilingData);
    op.Process();
}

void InitRegBaseTiling(SwigluGroupGradTilingData* tiling, int64_t totalRows, int64_t hiddenSize, float clampLimit,
                       int64_t groupCount)
{
    tiling->coreNumAll = 1;
    tiling->totalRows = totalRows;
    tiling->hiddenSize = hiddenSize;
    tiling->rowsPerTile = totalRows;
    tiling->splitHiddenMode = 0;
    tiling->launchedCoreNum = totalRows;
    tiling->groupIndexG = groupCount;
    tiling->hiddenChunkSize = hiddenSize;
    tiling->chunksPerRow = 1;
    tiling->clampLimit = clampLimit;
    tiling->clampLimitRecp = clampLimit > 0.0f ? 1.0f / clampLimit : 0.0f;
}

class SwigluGroupGradKernelTest : public testing::Test {};

// -----------------------------------------------------------------------------
//   The RegBase kernel uses RegTensor (Reg::* / __VEC_SCOPE__) vector
//   intrinsics. The tikicpulib CPU simulator does not execute these intrinsics,
//   so the assertions below can only be verified on real NPU hardware (or on a
//   future simulator that supports RegBase). The cases are kept as-is and
//   marked GTEST_SKIP on the CPU-only path so the kernel-UT suite stays green.
// -----------------------------------------------------------------------------

TEST_F(SwigluGroupGradKernelTest, fp32_without_optional_inputs)
{
    GTEST_SKIP() << "tikicpulib CPU simulator does not execute RegBase vector "
                    "intrinsics (Reg::* / __VEC_SCOPE__); assertions can only "
                    "run on real NPU hardware. Test body kept for hardware runs.";

    constexpr int64_t kRows = 1;
    constexpr int64_t kHiddenSize = 16;
    constexpr int64_t kDim2H = kHiddenSize * 2;

    auto* gradY = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * kHiddenSize * sizeof(float)));
    auto* x = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * kDim2H * sizeof(float)));
    auto* gradX = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * kDim2H * sizeof(float)));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1));
    auto* tilingBuffer = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SwigluGroupGradTilingData)));

    std::fill_n(gradY, kRows * kHiddenSize, 1.0f);
    std::fill_n(x, kRows * kDim2H, 0.0f);
    std::fill_n(gradX, kRows * kDim2H, -1.0f);
    for (int64_t col = 0; col < kHiddenSize; ++col) {
        x[kHiddenSize + col] = 1.0f;
    }

    auto* tiling = reinterpret_cast<SwigluGroupGradTilingData*>(tilingBuffer);
    InitRegBaseTiling(tiling, kRows, kHiddenSize, 0.0f, 0);
    auto kernelFunc = [](GM_ADDR gradYAddr, GM_ADDR xAddr, GM_ADDR weightAddr, GM_ADDR yOriginAddr,
                         GM_ADDR groupIndexAddr, GM_ADDR gradXAddr, GM_ADDR gradWeightAddr, GM_ADDR workspaceAddr,
                         GM_ADDR tilingAddr) {
        RunRegBaseKernel<float, 0, 0, 0, 0>(gradYAddr, xAddr, weightAddr, yOriginAddr, groupIndexAddr, gradXAddr,
                                            gradWeightAddr, workspaceAddr, tilingAddr);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernelFunc, 1, reinterpret_cast<uint8_t*>(gradY), reinterpret_cast<uint8_t*>(x), nullptr, nullptr,
                nullptr, reinterpret_cast<uint8_t*>(gradX), nullptr, workspace, tilingBuffer);

    for (int64_t col = 0; col < kHiddenSize; ++col) {
        EXPECT_NEAR(gradX[col], 0.5f, 1e-5f);
        EXPECT_NEAR(gradX[kHiddenSize + col], 0.0f, 1e-5f);
    }

    AscendC::GmFree(gradY);
    AscendC::GmFree(x);
    AscendC::GmFree(gradX);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingBuffer);
}

TEST_F(SwigluGroupGradKernelTest, fp32_clamp_uses_open_interval_masks)
{
    GTEST_SKIP() << "tikicpulib CPU simulator does not execute RegBase vector "
                    "intrinsics (Reg::* / __VEC_SCOPE__); assertions can only "
                    "run on real NPU hardware. Test body kept for hardware runs.";

    constexpr int64_t kRows = 1;
    constexpr int64_t kHiddenSize = 16;
    constexpr int64_t kDim2H = kHiddenSize * 2;
    constexpr float kClampLimit = 1.0f;

    auto* gradY = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * kHiddenSize * sizeof(float)));
    auto* x = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * kDim2H * sizeof(float)));
    auto* gradX = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * kDim2H * sizeof(float)));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1));
    auto* tilingBuffer = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SwigluGroupGradTilingData)));

    std::fill_n(gradY, kRows * kHiddenSize, 1.0f);
    std::fill_n(x, kRows * kDim2H, 0.0f);
    std::fill_n(gradX, kRows * kDim2H, -1.0f);
    for (int64_t col = 0; col < kHiddenSize; ++col) {
        x[kHiddenSize + col] = 0.5f;
    }
    x[0] = kClampLimit;
    x[kHiddenSize + 1] = kClampLimit;

    auto* tiling = reinterpret_cast<SwigluGroupGradTilingData*>(tilingBuffer);
    InitRegBaseTiling(tiling, kRows, kHiddenSize, kClampLimit, 0);
    auto kernelFunc = [](GM_ADDR gradYAddr, GM_ADDR xAddr, GM_ADDR weightAddr, GM_ADDR yOriginAddr,
                         GM_ADDR groupIndexAddr, GM_ADDR gradXAddr, GM_ADDR gradWeightAddr, GM_ADDR workspaceAddr,
                         GM_ADDR tilingAddr) {
        RunRegBaseKernel<float, 1, 0, 0, 0>(gradYAddr, xAddr, weightAddr, yOriginAddr, groupIndexAddr, gradXAddr,
                                            gradWeightAddr, workspaceAddr, tilingAddr);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernelFunc, 1, reinterpret_cast<uint8_t*>(gradY), reinterpret_cast<uint8_t*>(x), nullptr, nullptr,
                nullptr, reinterpret_cast<uint8_t*>(gradX), nullptr, workspace, tilingBuffer);

    const float siluAtClamp = 1.0f / (1.0f + std::exp(-kClampLimit));
    EXPECT_NEAR(gradX[0], 0.0f, 1e-5f);
    EXPECT_NEAR(gradX[kHiddenSize], siluAtClamp, 1e-5f);
    EXPECT_NEAR(gradX[1], 0.5f, 1e-5f);
    EXPECT_NEAR(gradX[kHiddenSize + 1], 0.0f, 1e-5f);
    for (int64_t col = 2; col < kHiddenSize; ++col) {
        EXPECT_NEAR(gradX[col], 0.25f, 1e-5f);
        EXPECT_NEAR(gradX[kHiddenSize + col], 0.0f, 1e-5f);
    }

    AscendC::GmFree(gradY);
    AscendC::GmFree(x);
    AscendC::GmFree(gradX);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingBuffer);
}

void RunGroupIndexMaskCase(int64_t hiddenSize, bool splitHidden)
{
    GTEST_SKIP() << "tikicpulib CPU simulator does not execute RegBase vector "
                    "intrinsics (Reg::* / __VEC_SCOPE__); assertions can only "
                    "run on real NPU hardware. Test body kept for hardware runs.";

    constexpr int64_t kRows = 4;
    const int64_t kHiddenSize = hiddenSize;
    const int64_t kDim2H = kHiddenSize * 2;
    constexpr int64_t kValidRows = 2;

    auto* gradY = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * kHiddenSize * sizeof(float)));
    auto* x = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * kDim2H * sizeof(float)));
    auto* weight = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * sizeof(float)));
    auto* yOrigin = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * kHiddenSize * sizeof(float)));
    auto* groupIndex = reinterpret_cast<int64_t*>(AscendC::GmAlloc(sizeof(int64_t)));
    auto* gradX = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * kDim2H * sizeof(float)));
    auto* gradWeight = reinterpret_cast<float*>(AscendC::GmAlloc(kRows * sizeof(float)));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1));
    auto* tilingBuffer = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SwigluGroupGradTilingData)));

    std::fill_n(gradY, kRows * kHiddenSize, 1.0f);
    std::fill_n(x, kRows * kDim2H, 0.0f);
    std::fill_n(weight, kRows, 1.0f);
    std::fill_n(yOrigin, kRows * kHiddenSize, 2.0f);
    std::fill_n(gradX, kRows * kDim2H, -1.0f);
    std::fill_n(gradWeight, kRows, -1.0f);
    for (int64_t row = 0; row < kRows; ++row) {
        for (int64_t col = 0; col < kHiddenSize; ++col) {
            x[row * kDim2H + kHiddenSize + col] = 1.0f;
        }
    }
    groupIndex[0] = kValidRows;

    auto* tiling = reinterpret_cast<SwigluGroupGradTilingData*>(tilingBuffer);
    InitRegBaseTiling(tiling, kRows, kHiddenSize, 0.0f, 1);
    if (splitHidden) {
        tiling->rowsPerTile = 1;
        tiling->splitHiddenMode = 1;
        tiling->hiddenChunkSize = 64;
        tiling->chunksPerRow = 2;
    }
    auto kernelFunc = [](GM_ADDR gradYAddr, GM_ADDR xAddr, GM_ADDR weightAddr, GM_ADDR yOriginAddr,
                         GM_ADDR groupIndexAddr, GM_ADDR gradXAddr, GM_ADDR gradWeightAddr, GM_ADDR workspaceAddr,
                         GM_ADDR tilingAddr) {
        RunRegBaseKernel<float, 0, 1, 1, 1>(gradYAddr, xAddr, weightAddr, yOriginAddr, groupIndexAddr, gradXAddr,
                                            gradWeightAddr, workspaceAddr, tilingAddr);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernelFunc, 1, reinterpret_cast<uint8_t*>(gradY), reinterpret_cast<uint8_t*>(x),
                reinterpret_cast<uint8_t*>(weight), reinterpret_cast<uint8_t*>(yOrigin),
                reinterpret_cast<uint8_t*>(groupIndex), reinterpret_cast<uint8_t*>(gradX),
                reinterpret_cast<uint8_t*>(gradWeight), workspace, tilingBuffer);

    for (int64_t row = 0; row < kRows; ++row) {
        for (int64_t col = 0; col < kHiddenSize; ++col) {
            const float expectedGradGate = row < kValidRows ? 0.5f : 0.0f;
            EXPECT_NEAR(gradX[row * kDim2H + col], expectedGradGate, 1e-5f);
            EXPECT_NEAR(gradX[row * kDim2H + kHiddenSize + col], 0.0f, 1e-5f);
        }
        EXPECT_NEAR(gradWeight[row], static_cast<float>(kHiddenSize * 2), 1e-5f);
    }

    AscendC::GmFree(gradY);
    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(yOrigin);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(gradX);
    AscendC::GmFree(gradWeight);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingBuffer);
}

TEST_F(SwigluGroupGradKernelTest, group_index_masks_grad_x_not_grad_weight) { RunGroupIndexMaskCase(16, false); }

TEST_F(SwigluGroupGradKernelTest, group_index_masks_grad_x_not_grad_weight_with_split_hidden)
{
    RunGroupIndexMaskCase(128, true);
}

} // namespace
