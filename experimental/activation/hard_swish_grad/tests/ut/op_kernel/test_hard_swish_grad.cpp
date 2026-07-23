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
 * \file test_hard_swish_grad.cpp
 * \brief HardSwishGrad kernel unit tests
 */

#include "../../../op_kernel/hard_swish_grad.cpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

namespace {

constexpr size_t kSize = 21;
constexpr int64_t kUbFactor = 64;

float GoldenHardSwishGrad(float grad, float x)
{
    if (x <= -3.0f) {
        return 0.0f;
    }
    if (x >= 3.0f) {
        return grad;
    }
    return grad * (x * 0.333333343f + 0.5f);
}

class HardSwishGradKernelTest : public testing::Test {};

TEST_F(HardSwishGradKernelTest, Fp32BoundaryAndIntervalValues)
{
    constexpr uint32_t numBlocks = 1;
    constexpr size_t byteSize = kSize * sizeof(float);
    constexpr size_t tilingDataSize = sizeof(HardSwishGradTilingData);

    const std::array<float, kSize> gradHost = {1.0f,   2.0f,  -3.0f, 4.0f,  -5.0f,  0.0f,  0.5f,
                                               -0.75f, 1.25f, 2.0f,  -2.0f, 3.0f,   -4.0f, 5.0f,
                                               -6.0f,  7.0f,  -8.0f, 9.0f,  -10.0f, 11.0f, -12.0f};
    const std::array<float, kSize> xHost = {-4.0f, -3.0f,  -2.5f, -2.0f,   -1.0f,  0.0f,  1.0f,
                                            2.0f,  2.5f,   3.0f,  4.0f,    -10.0f, -3.0f, 3.0f,
                                            10.0f, -0.25f, 0.25f, -2.999f, 2.999f, -1.5f, 1.5f};
    std::vector<float> yHost(kSize, 0.0f);

    auto* grad = static_cast<uint8_t*>(AscendC::GmAlloc(byteSize));
    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(byteSize));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(byteSize));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));
    ASSERT_NE(grad, nullptr);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::memcpy(grad, gradHost.data(), byteSize);
    std::memcpy(x, xHost.data(), byteSize);

    auto* tilingData = reinterpret_cast<HardSwishGradTilingData*>(tiling);
    tilingData->totalNum = kSize;
    tilingData->blockFactor = kSize;
    tilingData->ubFactor = kUbFactor;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((hard_swish_grad<1>), numBlocks, grad, x, y, workspace, tiling);

    std::memcpy(yHost.data(), y, byteSize);
    for (size_t i = 0; i < kSize; ++i) {
        EXPECT_NEAR(yHost[i], GoldenHardSwishGrad(gradHost[i], xHost[i]), 1e-5f) << "Mismatch at index " << i;
    }

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

} // namespace
