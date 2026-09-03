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
 * \file test_huber_loss.cpp
 * \brief Kernel unit test for the ops-nn UT framework.
 *
 * Expected values are hand-computed literals rather than a shared golden
 * helper: a golden function can be wrong the same way the implementation is;
 * a literal cannot. Covers the none schedule on fp32: both branches of the
 * definition, the knee at |e| = delta, a non-default delta and an unaligned
 * tail. AddOpTestCase takes one set of dtype defines and so cannot express a
 * dtype x schedule matrix here.
 */
#include "../../../op_kernel/huber_loss.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"

#ifndef DTYPE_INPUT
#define DTYPE_INPUT float
#endif

namespace {

// Mirrors the production entry in op_kernel/huber_loss.cpp. That file is
// not included directly because it also pulls in the tiling-key header and its
// ASCENDC_TPL machinery, which belongs to the package build rather than to the
// UT build.
template <uint32_t SCH_MODE>
__global__ __aicore__ void HuberLossUtKernel(GM_ADDR input, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace,
                                             GM_ADDR tiling)
{
    HuberLossTilingData tilingData;
    std::memcpy(&tilingData, tiling, sizeof(tilingData));
    NsHuberLoss::HuberLoss<DTYPE_INPUT, SCH_MODE> op;
    op.Init(input, target, loss, workspace, tilingData);
    op.Process();
}

// Built the way host tiling would build it, for a single core.
HuberLossTilingData BuildTiling(uint64_t numel, uint32_t tileDataNum, int32_t reduction, float delta)
{
    HuberLossTilingData tiling;
    tiling.totalNumel = numel;
    tiling.coreDataNum = numel;
    tiling.tailCoreDataNum = 0;
    tiling.usedCoreNum = 1;
    tiling.frontCoreNum = 1;
    tiling.tileDataNum = tileDataNum;
    tiling.reduction = reduction;
    tiling.delta = delta;
    tiling.divisor = (reduction == HUBER_LOSS_REDUCE_MEAN) ? static_cast<float>(numel) : 1.0f;
    tiling.slotRegionOffset = 0;
    return tiling;
}

void RunNone(const std::vector<float>& inputHost, const std::vector<float>& targetHost, float delta,
             uint32_t tileDataNum, std::vector<float>& out)
{
    const size_t numel = inputHost.size();
    ASSERT_EQ(targetHost.size(), numel);
    ASSERT_GT(numel, 0U);

    auto* input = static_cast<uint8_t*>(AscendC::GmAlloc(numel * sizeof(float)));
    auto* target = static_cast<uint8_t*>(AscendC::GmAlloc(numel * sizeof(float)));
    auto* loss = static_cast<uint8_t*>(AscendC::GmAlloc(numel * sizeof(float)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(HuberLossTilingData)));
    ASSERT_NE(input, nullptr);
    ASSERT_NE(target, nullptr);
    ASSERT_NE(loss, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::memcpy(input, inputHost.data(), numel * sizeof(float));
    std::memcpy(target, targetHost.data(), numel * sizeof(float));
    const HuberLossTilingData tilingData = BuildTiling(numel, tileDataNum, HUBER_LOSS_REDUCE_NONE, delta);
    std::memcpy(tiling, &tilingData, sizeof(tilingData));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((HuberLossUtKernel<HUBER_LOSS_SCH_MODE_NONE>), 1, input, target, loss, static_cast<uint8_t*>(nullptr),
                tiling);

    out.assign(numel, 0.0f);
    std::memcpy(out.data(), loss, numel * sizeof(float));

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(loss);
    AscendC::GmFree(tiling);
}

// The two branches of the definition, both signs, and the knee.
//
//   |e| = 2.0 > delta -> linear:    delta*(|e| - 0.5*delta) = 1*(2 - 0.5)   = 1.5
//   |e| = 0.5 < delta -> quadratic: 0.5*e^2                 = 0.5*0.25      = 0.125
//   |e| = 1.0 = delta -> the knee, where both branches give 0.5
//
// Every value is exactly representable in float32, so these are equalities,
// not tolerances.
TEST(HuberLossKernelTest, Float32MatchesHandComputedValuesOnBothBranches)
{
    const std::vector<float> input = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    const std::vector<float> target(input.size(), 0.0f);
    const std::vector<float> expected = {1.5f, 0.5f, 0.125f, 0.0f, 0.125f, 0.5f, 1.5f};

    std::vector<float> actual;
    ASSERT_NO_FATAL_FAILURE(RunNone(input, target, 1.0f, 8, actual));
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index=" << i;
    }
}

// A non-default delta moves the knee, so a kernel that ignored the attribute
// and used 1.0 would pass the test above and fail this one.
//
//   delta = 0.5, |e| = 2.0 -> 0.5*(2 - 0.25)   = 0.875
//   delta = 0.5, |e| = 0.25 -> 0.5*0.0625      = 0.03125
TEST(HuberLossKernelTest, Float32HonoursNonDefaultDelta)
{
    const std::vector<float> input = {2.0f, 0.25f, -2.0f, -0.25f};
    const std::vector<float> target(input.size(), 0.0f);
    const std::vector<float> expected = {0.875f, 0.03125f, 0.875f, 0.03125f};

    std::vector<float> actual;
    ASSERT_NO_FATAL_FAILURE(RunNone(input, target, 0.5f, 8, actual));
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index=" << i;
    }
}

// A tail shorter than a tile, driven through the same entry. The element count
// is deliberately not a multiple of the tile or of 8, because a tail that
// happens to be aligned hides an out-of-bounds copy.
TEST(HuberLossKernelTest, Float32HandlesATailShorterThanATile)
{
    const std::vector<float> input = {3.0f, -3.0f, 0.25f, -0.25f, 1.0f, 5.0f, -5.0f, 0.75f, -0.75f, 2.5f, -2.5f};
    const std::vector<float> target(input.size(), 0.0f);
    // linear for |e| >= 1: |e| - 0.5 ; quadratic below: 0.5*e^2
    const std::vector<float> expected = {2.5f, 2.5f,     0.03125f, 0.03125f, 0.5f, 4.5f,
                                         4.5f, 0.28125f, 0.28125f, 2.0f,     2.0f};

    std::vector<float> actual;
    ASSERT_NO_FATAL_FAILURE(RunNone(input, target, 1.0f, 4, actual));
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index=" << i;
    }
}

} // namespace
