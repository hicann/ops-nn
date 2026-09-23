/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "../../../../op_kernel/arch35/gru_block_cell_grad.cpp"
#include "arch35/gru_block_cell_grad_tiling_struct.h"

namespace {

constexpr int64_t kInputSize = 16;
constexpr int64_t kCellSize = 8;
constexpr int64_t kCanaryElements = 32;
constexpr size_t kPlaceholderBytes = 32;

struct KernelBuffers {
    std::vector<uint8_t*> buffers;

    ~KernelBuffers()
    {
        for (auto* buffer : buffers) {
            AscendC::GmFree(buffer);
        }
    }

    uint8_t* Allocate(size_t bytes)
    {
        auto* buffer = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes));
        buffers.push_back(buffer);
        return buffer;
    }
};

void RunGruBlockCellGradEmpty(GM_ADDR x, GM_ADDR hPrev, GM_ADDR wRu, GM_ADDR wC, GM_ADDR bRu, GM_ADDR bC, GM_ADDR r,
                              GM_ADDR u, GM_ADDR c, GM_ADDR dH, GM_ADDR dX, GM_ADDR dHPrev, GM_ADDR dCBar,
                              GM_ADDR dRuBar, GM_ADDR workspace, GM_ADDR tiling)
{
    gru_block_cell_grad<true>(x, hPrev, wRu, wC, bRu, bC, r, u, c, dH, dX, dHPrev, dCBar, dRuBar, workspace, tiling);
}

TEST(GruBlockCellGradKernelArch35, EmptyBatchLeavesDxCanary)
{
    KernelBuffers buffers;
    auto* placeholder = buffers.Allocate(kPlaceholderBytes);
    auto* dX = reinterpret_cast<float*>(buffers.Allocate(kCanaryElements * sizeof(float)));
    auto* tiling = reinterpret_cast<GruBlockCellGradTilingData*>(buffers.Allocate(sizeof(GruBlockCellGradTilingData)));
    ASSERT_NE(placeholder, nullptr);
    ASSERT_NE(dX, nullptr);
    ASSERT_NE(tiling, nullptr);

    // For B=0, d_x has no logical elements.  A canary allocation proves that
    // the production empty branch exits without touching output memory.
    std::fill(dX, dX + kCanaryElements, 3.5f);
    *tiling = {0, kInputSize, kCellSize, 1, 0, 0, 0, 0, 0, 0, 0, 0};

    ICPU_SET_TILING_KEY(1);
    // The empty branch has no Cube work or cross-core synchronization.  Run it
    // in the CPU simulator's AIV mode so its vector-only implementation is
    // exercised directly.
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(RunGruBlockCellGradEmpty, 1, reinterpret_cast<GM_ADDR>(placeholder),
                reinterpret_cast<GM_ADDR>(placeholder), reinterpret_cast<GM_ADDR>(placeholder),
                reinterpret_cast<GM_ADDR>(placeholder), reinterpret_cast<GM_ADDR>(placeholder),
                reinterpret_cast<GM_ADDR>(placeholder), reinterpret_cast<GM_ADDR>(placeholder),
                reinterpret_cast<GM_ADDR>(placeholder), reinterpret_cast<GM_ADDR>(placeholder),
                reinterpret_cast<GM_ADDR>(placeholder), reinterpret_cast<GM_ADDR>(dX),
                reinterpret_cast<GM_ADDR>(placeholder), reinterpret_cast<GM_ADDR>(placeholder),
                reinterpret_cast<GM_ADDR>(placeholder), reinterpret_cast<GM_ADDR>(placeholder),
                reinterpret_cast<GM_ADDR>(tiling));

    for (int64_t index = 0; index < kCanaryElements; ++index) {
        EXPECT_EQ(dX[index], 3.5f) << "d_x canary index " << index;
    }
}

TEST(GruBlockCellGradKernelArch35, KernelConstantsMatchTilingContract)
{
    EXPECT_EQ(GruGrad::kGruVlF32, 64);
    EXPECT_EQ(GruGrad::kGruPv, 8);
    EXPECT_EQ(GruGrad::kGruC0F32, 8);
    EXPECT_EQ(GruGrad::kGruFractalN, 16);
    EXPECT_EQ(GruGrad::GruMin<int64_t>(-3, 5), -3);
    EXPECT_EQ(GruGrad::GruMin<int64_t>(7, 5), 5);
    EXPECT_EQ(sizeof(GruBlockCellGradTilingData), 12 * sizeof(int64_t));
}

} // namespace
