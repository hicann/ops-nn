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
#include <memory>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/sparse_apply_proximal_adagrad.cpp"
#include "../../fixtures/sparse_apply_proximal_adagrad_regression.h"

namespace {
using namespace sparse_apply_proximal_adagrad_regression;

using GmBuffer = std::unique_ptr<uint8_t, decltype(&AscendC::GmFree)>;

GmBuffer CopyToGm(const void* data, size_t size)
{
    GmBuffer buffer(static_cast<uint8_t*>(AscendC::GmAlloc(size)), AscendC::GmFree);
    if (buffer != nullptr) {
        std::memcpy(buffer.get(), data, size);
    }
    return buffer;
}

void RunAndCheck(const Fixture& fixture)
{
    constexpr uint32_t BLOCKS = 1;
    const SparseApplyProximalAdagradTilingData tiling = {BLOCKS, 1, ROWS, COLS};
    auto varGm = CopyToGm(VAR_BITS.data(), sizeof(VAR_BITS));
    auto accumGm = CopyToGm(fixture.accum.data(), sizeof(fixture.accum));
    auto lrGm = CopyToGm(&LR, sizeof(LR));
    auto l1Gm = CopyToGm(&ZERO, sizeof(ZERO));
    auto l2Gm = CopyToGm(&ZERO, sizeof(ZERO));
    auto gradGm = CopyToGm(fixture.grad.data(), sizeof(fixture.grad));
    auto indexGm = CopyToGm(&INDEX, sizeof(INDEX));
    auto tilingGm = CopyToGm(&tiling, sizeof(tiling));
    ASSERT_TRUE(varGm && accumGm && lrGm && l1Gm && l2Gm && gradGm && indexGm && tilingGm);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(sparse_apply_proximal_adagrad<0>, BLOCKS, varGm.get(), accumGm.get(), lrGm.get(), l1Gm.get(),
                l2Gm.get(), gradGm.get(), indexGm.get(), varGm.get(), accumGm.get(), nullptr, tilingGm.get());

    StateBits actualVar{};
    StateBits actualAccum{};
    std::memcpy(actualVar.data(), varGm.get(), sizeof(actualVar));
    std::memcpy(actualAccum.data(), accumGm.get(), sizeof(actualAccum));
    EXPECT_EQ(actualVar, fixture.expectedVar);
    EXPECT_EQ(actualAccum, fixture.expectedAccum);
}

TEST(SparseApplyProximalAdagradKernel, ZeroGradientPreservesSubnormalAccum) { RunAndCheck(ZERO_GRADIENT); }

TEST(SparseApplyProximalAdagradKernel, NonzeroGradientUpdatesBothStates) { RunAndCheck(NONZERO_GRADIENT); }
} // namespace
