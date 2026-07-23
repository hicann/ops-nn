/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/selu_grad.cpp"

namespace {
constexpr float SCALE = 1.0507009873554804934193349852946F;
constexpr float SCALE_ALPHA_PRODUCT = 1.7580993408473768599402175208123F;

float GetExpected(float gradient, float output)
{
    const float factor = output < 0.0F ? output + SCALE_ALPHA_PRODUCT : SCALE;
    return gradient * factor;
}
} // namespace

class SeluGradKernelTest : public testing::Test {};

TEST_F(SeluGradKernelTest, fp32_multicore_double_buffer_and_tail)
{
    constexpr size_t elementNum = 513;
    constexpr uint32_t blockDim = 2;
    constexpr uint32_t blockFactor = 257;
    constexpr uint32_t ubFactor = 256;
    const size_t tensorBytes = elementNum * sizeof(float);

    auto* gradients = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tensorBytes));
    auto* outputs = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tensorBytes));
    auto* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tensorBytes));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SeluGradTilingData)));
    ASSERT_NE(gradients, nullptr);
    ASSERT_NE(outputs, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::vector<float> gradientsHost(elementNum);
    std::vector<float> outputsHost(elementNum);
    for (size_t i = 0; i < elementNum; ++i) {
        gradientsHost[i] = static_cast<float>(static_cast<int32_t>(i % 13) - 6) * 0.25F;
        switch (i % 5) {
            case 0:
                outputsHost[i] = -1.5F;
                break;
            case 1:
                outputsHost[i] = -0.5F;
                break;
            case 2:
                outputsHost[i] = -0.0F;
                break;
            case 3:
                outputsHost[i] = 0.0F;
                break;
            default:
                outputsHost[i] = 2.0F;
                break;
        }
    }
    std::memcpy(gradients, gradientsHost.data(), tensorBytes);
    std::memcpy(outputs, outputsHost.data(), tensorBytes);
    std::memset(y, 0, tensorBytes);

    auto* tilingData = reinterpret_cast<SeluGradTilingData*>(tiling);
    tilingData->totalNum = elementNum;
    tilingData->blockFactor = blockFactor;
    tilingData->ubFactor = ubFactor;

    ICPU_SET_TILING_KEY(SELUGRAD_TPL_SCH_MODE_FP32);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((selu_grad<SELUGRAD_TPL_SCH_MODE_FP32>), blockDim, gradients, outputs, y, workspace, tiling);

    const auto* actual = reinterpret_cast<const float*>(y);
    for (size_t i = 0; i < elementNum; ++i) {
        EXPECT_NEAR(actual[i], GetExpected(gradientsHost[i], outputsHost[i]), 1.0e-5F) << "index=" << i;
    }

    AscendC::GmFree(gradients);
    AscendC::GmFree(outputs);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SeluGradKernelTest, empty_tensor)
{
    auto* gradients = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* outputs = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SeluGradTilingData)));
    ASSERT_NE(gradients, nullptr);
    ASSERT_NE(outputs, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    auto* tilingData = reinterpret_cast<SeluGradTilingData*>(tiling);
    tilingData->totalNum = 0;
    tilingData->blockFactor = 0;
    tilingData->ubFactor = 0;

    ICPU_SET_TILING_KEY(SELUGRAD_TPL_SCH_MODE_FP16);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((selu_grad<SELUGRAD_TPL_SCH_MODE_FP16>), 1, gradients, outputs, y, workspace, tiling);

    AscendC::GmFree(gradients);
    AscendC::GmFree(outputs);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
