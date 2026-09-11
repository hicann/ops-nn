/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * The A5 kernel has a templated entry point. These tests invoke three
 * representative templates directly, which keeps the CPU simulation on the
 * same code path as the generated Ascend C kernel.
 */
#include <cstdint>
#include <cstring>
#include <iostream>

#include "gtest/gtest.h"
#include "graph/c_types.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "register/op_def_registry.h"
#endif

#include "../../../op_kernel/renorm_apt.cpp"

namespace {

template <uint32_t TEMPLATE>
void ExecuteTestCase(float p, int32_t norm_mode)
{
    // Keep every vector instruction on its supported alignment.  Template A
    // handles a short slice directly; the two workspace-based templates need
    // at least one 256-byte FP32 vector for Compare/Max in the CPU simulator.
    int64_t total_elements = 16;
    int64_t slice_count = 2;
    int64_t block_size = 8;
    int64_t tile_length = 8;
    int64_t slices_per_core = 2;
    if constexpr (TEMPLATE == 1) {
        total_elements = 64;
        slice_count = 64;
        block_size = 1;
        tile_length = 64;
        slices_per_core = 64;
    } else if constexpr (TEMPLATE == 2) {
        total_elements = 512;
        slice_count = 64;
        block_size = 8;
        tile_length = 512;
        slices_per_core = 64;
    }

    constexpr uint32_t block_num = 1;
    constexpr size_t workspace_size = 16U * 1024U * 1024U + 64U * 1024U;

    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(total_elements * sizeof(float)));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(total_elements * sizeof(float)));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(workspace_size));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(RenormTilingData)));

    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::memset(x, 0, total_elements * sizeof(float));
    std::memset(y, 0, total_elements * sizeof(float));
    std::memset(workspace, 0, workspace_size);

    auto* tiling_data = reinterpret_cast<RenormTilingData*>(tiling);
    *tiling_data = RenormTilingData{};
    tiling_data->totalElements = total_elements;
    tiling_data->dim = 1;
    tiling_data->sliceCount = slice_count;
    tiling_data->blockSize = block_size;
    tiling_data->numBlocks = 1;
    tiling_data->tileLength = tile_length;
    tiling_data->slicesPerCore = slices_per_core;
    tiling_data->p = p;
    tiling_data->maxNorm = 1.0f;
    tiling_data->eps = 1.0e-12f;
    tiling_data->normMode = norm_mode;
    tiling_data->blockFactor = 1;
    tiling_data->blockFactor2 = 1;
    tiling_data->reduceSplitsPerCore = 1;
    tiling_data->workspaceSize = 64 * 1024;
    tiling_data->stride = block_size;
    tiling_data->sliceTileLength = slice_count;

    auto kernel = [](GM_ADDR input, GM_ADDR output, GM_ADDR ws, GM_ADDR td) {
        ::renorm<float, TEMPLATE>(input, output, ws, td);
    };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(TEMPLATE);
    ICPU_RUN_KF(kernel, block_num, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

class RenormKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "renorm_kernel_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "renorm_kernel_test TearDown" << std::endl; }
};

} // namespace

TEST_F(RenormKernelTest, TemplateA) { ExecuteTestCase<0>(1.0f, 0); }

TEST_F(RenormKernelTest, TemplateB) { ExecuteTestCase<1>(2.0f, 0); }

TEST_F(RenormKernelTest, TemplateC) { ExecuteTestCase<2>(1.0f, 2); }
