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
#include <cstring>
#include <iostream>

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/arch35/apply_adagrad.cpp"

namespace {

uint32_t FloatBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

class ApplyAdagradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyAdagradKernel SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyAdagradKernel TearDown" << std::endl; }
};

TEST_F(ApplyAdagradKernelTest, code_12_1_update_slots_false_preserves_negative_zero_accum)
{
    constexpr uint32_t elementCount = 8;
    constexpr uint32_t ioBufferBytes = 256;
    constexpr uint32_t negativeZeroBits = 0x80000000U;
    constexpr uint32_t positiveInfinityBits = 0x7f800000U;
    constexpr uint32_t blockDim = 1;

    auto* var = static_cast<float*>(AscendC::GmAlloc(elementCount * sizeof(float)));
    auto* accum = static_cast<float*>(AscendC::GmAlloc(elementCount * sizeof(float)));
    auto* lr = static_cast<float*>(AscendC::GmAlloc(sizeof(float)));
    auto* grad = static_cast<float*>(AscendC::GmAlloc(elementCount * sizeof(float)));
    auto* varOut = static_cast<float*>(AscendC::GmAlloc(elementCount * sizeof(float)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(1));
    auto* tiling = static_cast<ApplyAdagradTilingData::ApplyAdagradTilingDataStruct*>(
        AscendC::GmAlloc(sizeof(ApplyAdagradTilingData::ApplyAdagradTilingDataStruct)));

    ASSERT_NE(var, nullptr);
    ASSERT_NE(accum, nullptr);
    ASSERT_NE(lr, nullptr);
    ASSERT_NE(grad, nullptr);
    ASSERT_NE(varOut, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    for (uint32_t i = 0; i < elementCount; ++i) {
        var[i] = 1.0F;
        accum[i] = -0.0F;
        grad[i] = 1.0F;
        varOut[i] = 0.0F;
        ASSERT_EQ(FloatBits(accum[i]), negativeZeroBits);
    }
    lr[0] = 0.1F;

    tiling->totalElements = elementCount;
    tiling->blockFactor = elementCount;
    tiling->ubFactor = elementCount;
    tiling->ioBufferBytes = ioBufferBytes;

    ICPU_SET_TILING_KEY(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0, UPDATE_SLOTS_TPL_FALSE, APPLY_ADAGRAD_TPL_FP32));
    auto kernel = [](GM_ADDR varAddr, GM_ADDR accumAddr, GM_ADDR lrAddr, GM_ADDR gradAddr, GM_ADDR varOutAddr,
                     GM_ADDR workspaceAddr, GM_ADDR tilingAddr) {
        ::apply_adagrad<ELEMENTWISE_TPL_SCH_MODE_0, UPDATE_SLOTS_TPL_FALSE, APPLY_ADAGRAD_TPL_FP32>(
            varAddr, accumAddr, lrAddr, gradAddr, varOutAddr, workspaceAddr, tilingAddr);
    };
    ICPU_RUN_KF(kernel, blockDim, reinterpret_cast<uint8_t*>(var), reinterpret_cast<uint8_t*>(accum),
                reinterpret_cast<uint8_t*>(lr), reinterpret_cast<uint8_t*>(grad), reinterpret_cast<uint8_t*>(varOut),
                workspace, reinterpret_cast<uint8_t*>(tiling));

    for (uint32_t i = 0; i < elementCount; ++i) {
        EXPECT_EQ(FloatBits(accum[i]), negativeZeroBits) << "accum sign changed at index " << i;
        EXPECT_EQ(FloatBits(varOut[i]), positiveInfinityBits) << "wrong signed-zero denominator at index " << i;
    }

    AscendC::GmFree(var);
    AscendC::GmFree(accum);
    AscendC::GmFree(lr);
    AscendC::GmFree(grad);
    AscendC::GmFree(varOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

} // namespace
