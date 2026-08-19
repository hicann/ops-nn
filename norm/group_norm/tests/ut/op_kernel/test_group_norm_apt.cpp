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
#include <vector>
#include "gtest/gtest.h"
#include "group_norm_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "register/op_def_registry.h"
#endif

extern "C" __global__ __aicore__ void group_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                                 GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int64_t TILING_KEY_WELFORD_PERF = 1100;
constexpr int64_t TILING_KEY_TWOPASS_PERF = 1110;
constexpr int64_t TILING_KEY_WELFORD_GENERALIZED = 1120;
constexpr int64_t TILING_KEY_TWOPASS_GENERALIZED = 1130;
constexpr size_t GM_ALIGNMENT = 32;
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024;

size_t AlignUp(size_t value)
{
    return std::max(GM_ALIGNMENT, (value + GM_ALIGNMENT - 1) / GM_ALIGNMENT * GM_ALIGNMENT);
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const auto dim : shape) {
        size *= dim;
    }
    return size;
}

GroupNormTilingData MakeNormalTiling(int64_t tilingKey)
{
    GroupNormTilingData data{};
    data.numGroups = 2;
    data.hwNum = 64;
    data.elemNum = 128;
    data.shapeC = 4;
    data.shapeD = 2;
    data.realCoreNum = 1;
    data.numPerCore = 2;
    data.numLastCore = 2;
    data.processSize = 128;
    data.loopNum = 1;
    data.loopTail = 128;
    data.innerLoopNum = 1;
    data.innerLoopTail = 64;
    data.tilingKey = tilingKey;
    data.epsilon = 1e-4F;
    data.parallelN = 128;
    data.ubSize = 245760;
    data.dichotomyAddPower = 64;
    data.dichotomyAddK = 0;
    data.dichotomyAddLastNum = 1;
    return data;
}

void RunNormalKernel(int64_t tilingKey)
{
    const std::vector<int64_t> xShape = {1, 4, 8, 8};
    const std::vector<int64_t> weightShape = {4};
    const size_t xElements = static_cast<size_t>(GetShapeSize(xShape));
    const size_t weightElements = static_cast<size_t>(GetShapeSize(weightShape));
    constexpr size_t statisticsElements = 2;

    auto* x = static_cast<float*>(AscendC::GmAlloc(AlignUp(xElements * sizeof(float))));
    auto* gamma = static_cast<float*>(AscendC::GmAlloc(AlignUp(weightElements * sizeof(float))));
    auto* beta = static_cast<float*>(AscendC::GmAlloc(AlignUp(weightElements * sizeof(float))));
    auto* y = static_cast<float*>(AscendC::GmAlloc(AlignUp(xElements * sizeof(float))));
    auto* mean = static_cast<float*>(AscendC::GmAlloc(AlignUp(statisticsElements * sizeof(float))));
    auto* variance = static_cast<float*>(AscendC::GmAlloc(AlignUp(statisticsElements * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(WORKSPACE_SIZE));
    auto* tiling = static_cast<GroupNormTilingData*>(AscendC::GmAlloc(AlignUp(sizeof(GroupNormTilingData))));

    ASSERT_NE(x, nullptr);
    ASSERT_NE(gamma, nullptr);
    ASSERT_NE(beta, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(mean, nullptr);
    ASSERT_NE(variance, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::fill_n(x, xElements, 0.0F);
    std::fill_n(gamma, weightElements, 1.0F);
    std::fill_n(beta, weightElements, 0.0F);
    std::fill_n(y, xElements, -1.0F);
    std::fill_n(mean, statisticsElements, -1.0F);
    std::fill_n(variance, statisticsElements, -1.0F);
    *tiling = MakeNormalTiling(tilingKey);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(group_norm, 1, reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(gamma),
                reinterpret_cast<uint8_t*>(beta), reinterpret_cast<uint8_t*>(y), reinterpret_cast<uint8_t*>(mean),
                reinterpret_cast<uint8_t*>(variance), workspace, reinterpret_cast<uint8_t*>(tiling));

    for (size_t i = 0; i < xElements; ++i) {
        EXPECT_FLOAT_EQ(y[i], 0.0F);
    }
    for (size_t i = 0; i < statisticsElements; ++i) {
        EXPECT_FLOAT_EQ(mean[i], 0.0F);
        EXPECT_FLOAT_EQ(variance[i], 0.0F);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

} // namespace

class GroupNormKernelTest : public testing::Test {};

TEST_F(GroupNormKernelTest, WelfordPerformance) { RunNormalKernel(TILING_KEY_WELFORD_PERF); }

TEST_F(GroupNormKernelTest, TwoPassPerformance) { RunNormalKernel(TILING_KEY_TWOPASS_PERF); }

TEST_F(GroupNormKernelTest, WelfordGeneralized) { RunNormalKernel(TILING_KEY_WELFORD_GENERALIZED); }

TEST_F(GroupNormKernelTest, TwoPassGeneralized) { RunNormalKernel(TILING_KEY_TWOPASS_GENERALIZED); }
