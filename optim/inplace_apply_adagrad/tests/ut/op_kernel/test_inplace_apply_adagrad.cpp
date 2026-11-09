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
#include <iostream>

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../op_kernel/arch35/inplace_apply_adagrad.cpp"

namespace {

uint32_t FloatBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

constexpr uint32_t ELEMENT_COUNT = 8;
constexpr uint32_t IO_BUFFER_BYTES = 256;
constexpr uint32_t BLOCK_DIM = 1;

struct KernelBuffers {
    float* var = nullptr;
    float* accum = nullptr;
    float* lr = nullptr;
    float* grad = nullptr;
    float* varOut = nullptr;
    float* accumOut = nullptr;
    uint8_t* workspace = nullptr;
    InplaceApplyAdagradTilingData::InplaceApplyAdagradTilingDataStruct* tiling = nullptr;

    explicit KernelBuffers(uint32_t elements)
    {
        var = static_cast<float*>(AscendC::GmAlloc(elements * sizeof(float)));
        accum = static_cast<float*>(AscendC::GmAlloc(elements * sizeof(float)));
        lr = static_cast<float*>(AscendC::GmAlloc(sizeof(float)));
        grad = static_cast<float*>(AscendC::GmAlloc(elements * sizeof(float)));
        varOut = static_cast<float*>(AscendC::GmAlloc(elements * sizeof(float)));
        accumOut = static_cast<float*>(AscendC::GmAlloc(elements * sizeof(float)));
        workspace = static_cast<uint8_t*>(AscendC::GmAlloc(1));
        tiling = static_cast<InplaceApplyAdagradTilingData::InplaceApplyAdagradTilingDataStruct*>(
            AscendC::GmAlloc(sizeof(InplaceApplyAdagradTilingData::InplaceApplyAdagradTilingDataStruct)));
    }

    ~KernelBuffers()
    {
        AscendC::GmFree(var);
        AscendC::GmFree(accum);
        AscendC::GmFree(lr);
        AscendC::GmFree(grad);
        AscendC::GmFree(varOut);
        AscendC::GmFree(accumOut);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }

    bool Valid() const
    {
        return var != nullptr && accum != nullptr && lr != nullptr && grad != nullptr && varOut != nullptr &&
               accumOut != nullptr && workspace != nullptr && tiling != nullptr;
    }

    void SetTiling(uint32_t elements)
    {
        tiling->totalElements = elements;
        tiling->blockFactor = elements;
        tiling->ubFactor = elements;
        tiling->ioBufferBytes = IO_BUFFER_BYTES;
    }

    template <bool UPDATE_SLOTS>
    void Run() const
    {
        if constexpr (UPDATE_SLOTS) {
            // schMode=0, updateSlots=1, dType=2 (FP32) -> 0 | (0 << 1) | (2 << 2) = 8
            ICPU_SET_TILING_KEY(8);
            auto kernel = [](GM_ADDR varAddr, GM_ADDR accumAddr, GM_ADDR lrAddr, GM_ADDR gradAddr, GM_ADDR varOutAddr,
                             GM_ADDR accumOutAddr, GM_ADDR workspaceAddr, GM_ADDR tilingAddr) {
                ::inplace_apply_adagrad<ELEMENTWISE_TPL_SCH_MODE_0, UPDATE_SLOTS_TPL_TRUE,
                                        INPLACE_APPLY_ADAGRAD_TPL_FP32>(
                    varAddr, accumAddr, lrAddr, gradAddr, varOutAddr, accumOutAddr, workspaceAddr, tilingAddr);
            };
            ICPU_RUN_KF(kernel, BLOCK_DIM, reinterpret_cast<uint8_t*>(var), reinterpret_cast<uint8_t*>(accum),
                        reinterpret_cast<uint8_t*>(lr), reinterpret_cast<uint8_t*>(grad),
                        reinterpret_cast<uint8_t*>(varOut), reinterpret_cast<uint8_t*>(accumOut), workspace,
                        reinterpret_cast<uint8_t*>(tiling));
        } else {
            // schMode=0, updateSlots=0, dType=2 (FP32) -> 0 | (1 << 1) | (2 << 2) = 10
            ICPU_SET_TILING_KEY(10);
            auto kernel = [](GM_ADDR varAddr, GM_ADDR accumAddr, GM_ADDR lrAddr, GM_ADDR gradAddr, GM_ADDR varOutAddr,
                             GM_ADDR accumOutAddr, GM_ADDR workspaceAddr, GM_ADDR tilingAddr) {
                ::inplace_apply_adagrad<ELEMENTWISE_TPL_SCH_MODE_0, UPDATE_SLOTS_TPL_FALSE,
                                        INPLACE_APPLY_ADAGRAD_TPL_FP32>(
                    varAddr, accumAddr, lrAddr, gradAddr, varOutAddr, accumOutAddr, workspaceAddr, tilingAddr);
            };
            ICPU_RUN_KF(kernel, BLOCK_DIM, reinterpret_cast<uint8_t*>(var), reinterpret_cast<uint8_t*>(accum),
                        reinterpret_cast<uint8_t*>(lr), reinterpret_cast<uint8_t*>(grad),
                        reinterpret_cast<uint8_t*>(varOut), reinterpret_cast<uint8_t*>(accumOut), workspace,
                        reinterpret_cast<uint8_t*>(tiling));
        }
    }
};

class InplaceApplyAdagradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InplaceApplyAdagradKernel SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "InplaceApplyAdagradKernel TearDown" << std::endl; }
};

TEST_F(InplaceApplyAdagradKernelTest, update_slots_false_preserves_negative_zero_accum)
{
    constexpr uint32_t negativeZeroBits = 0x80000000U;
    constexpr uint32_t positiveInfinityBits = 0x7f800000U;
    KernelBuffers buffers(ELEMENT_COUNT);
    ASSERT_TRUE(buffers.Valid());

    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        buffers.var[i] = 1.0F;
        buffers.accum[i] = -0.0F;
        buffers.grad[i] = 1.0F;
        buffers.varOut[i] = 0.0F;
        buffers.accumOut[i] = 42.0F;
        ASSERT_EQ(FloatBits(buffers.accum[i]), negativeZeroBits);
    }
    buffers.lr[0] = 0.1F;
    buffers.SetTiling(ELEMENT_COUNT);

    buffers.Run<false>();

    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        EXPECT_EQ(FloatBits(buffers.accum[i]), negativeZeroBits) << "accum sign changed at index " << i;
        EXPECT_EQ(FloatBits(buffers.accumOut[i]), negativeZeroBits) << "accum output sign changed at index " << i;
        EXPECT_EQ(FloatBits(buffers.varOut[i]), positiveInfinityBits) << "wrong signed-zero denominator at index " << i;
    }
}

TEST_F(InplaceApplyAdagradKernelTest, update_slots_true_writes_both_outputs)
{
    KernelBuffers buffers(ELEMENT_COUNT);
    ASSERT_TRUE(buffers.Valid());

    float expectedVar[ELEMENT_COUNT];
    float expectedAccum[ELEMENT_COUNT];
    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        buffers.var[i] = 1.0F + static_cast<float>(i);
        buffers.accum[i] = 0.5F + static_cast<float>(i);
        buffers.grad[i] = 1.0F - 0.25F * static_cast<float>(i);
        buffers.varOut[i] = 0.0F;
        buffers.accumOut[i] = 0.0F;
        expectedAccum[i] = buffers.accum[i] + buffers.grad[i] * buffers.grad[i];
        expectedVar[i] = buffers.var[i] - 0.1F * buffers.grad[i] / std::sqrt(expectedAccum[i]);
    }
    buffers.lr[0] = 0.1F;
    buffers.SetTiling(ELEMENT_COUNT);

    buffers.Run<true>();

    for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
        EXPECT_NEAR(buffers.varOut[i], expectedVar[i], 1e-5F) << "var output mismatch at index " << i;
        EXPECT_NEAR(buffers.accumOut[i], expectedAccum[i], 1e-5F) << "accum output mismatch at index " << i;
    }
}

} // namespace
