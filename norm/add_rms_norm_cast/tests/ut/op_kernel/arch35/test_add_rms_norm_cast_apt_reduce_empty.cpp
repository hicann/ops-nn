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
#include <memory>

#include "gtest/gtest.h"
#include "tikicpulib.h"
// Kernel-side tiling struct + CPU tiling macros for the production 950 entry (same pattern as
// test_add_rms_norm_cast_regbase.cpp). Must precede the apt.cpp include.
#include "../test_add_rms_norm_cast_regbase.h"
#include "add_rms_norm_cast_apt.cpp"

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
#error "Empty reduction tests must execute the Ascend 950 kernel branch"
#endif

namespace {
struct GmDeleter {
    void operator()(uint8_t* pointer) const { AscendC::GmFree(pointer); }
};
using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

GmBuffer AllocateGm(size_t bytes) { return GmBuffer(reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes))); }

struct EmptyReductionCase {
    uint64_t rows;
    uint32_t cores;
    uint64_t rowsPerCore;
    uint64_t lastCoreRows;
    uint64_t rowsPerLoop;
};

class AddRmsNormCastReduceEmpty950 : public testing::TestWithParam<EmptyReductionCase> {};

TEST_P(AddRmsNormCastReduceEmpty950, writes_all_nan_without_overwriting_guards)
{
    const auto& testCase = GetParam();
    constexpr size_t GUARD_ELEMENTS = 8;
    constexpr float SENTINEL = 12345.0f;
    constexpr size_t EMPTY_BUFFER_BYTES = 32;
    constexpr uint8_t EMPTY_SENTINEL = 0x5A;
    auto rstd = AllocateGm((testCase.rows + 2 * GUARD_ELEMENTS) * sizeof(float));
    auto emptyBuffers = AllocateGm(6 * EMPTY_BUFFER_BYTES);
    auto workspace = AllocateGm(32);
    auto tiling = AllocateGm(sizeof(AddRmsNormCastReduceEmptyTilingData));
    ASSERT_NE(rstd.get(), nullptr);
    ASSERT_NE(emptyBuffers.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    ASSERT_NE(tiling.get(), nullptr);
    auto* output = reinterpret_cast<float*>(rstd.get());
    std::fill_n(output, testCase.rows + 2 * GUARD_ELEMENTS, SENTINEL);
    std::fill_n(emptyBuffers.get(), 6 * EMPTY_BUFFER_BYTES, EMPTY_SENTINEL);

    // Use the production reduce-empty tiling layout (independent from the regbase tiling data).
    auto* tilingData = reinterpret_cast<AddRmsNormCastReduceEmptyTilingData*>(tiling.get());
    *tilingData = {};
    tilingData->numM = testCase.rows;
    tilingData->mPerCore = testCase.rowsPerCore;
    tilingData->mLastCore = testCase.lastCoreRows;
    tilingData->rowsPerLoop = testCase.rowsPerLoop;

    auto* x1 = emptyBuffers.get();
    auto* x2 = x1 + EMPTY_BUFFER_BYTES;
    auto* gamma = x2 + EMPTY_BUFFER_BYTES;
    auto* y1 = gamma + EMPTY_BUFFER_BYTES;
    auto* y2 = y1 + EMPTY_BUFFER_BYTES;
    auto* x = y2 + EMPTY_BUFFER_BYTES;
    auto* rstdData = reinterpret_cast<uint8_t*>(output + GUARD_ELEMENTS);
    ICPU_SET_TILING_KEY(8200);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm_cast, testCase.cores, x1, x2, gamma, y1, y2, rstdData, x, workspace.get(), tiling.get());

    for (uint64_t row = 0; row < testCase.rows; ++row) {
        ASSERT_TRUE(std::isnan(output[GUARD_ELEMENTS + row])) << "row=" << row;
    }
    for (size_t i = 0; i < GUARD_ELEMENTS; ++i) {
        EXPECT_EQ(output[i], SENTINEL) << "prefix guard=" << i;
        EXPECT_EQ(output[GUARD_ELEMENTS + testCase.rows + i], SENTINEL) << "suffix guard=" << i;
    }
    for (size_t i = 0; i < 6 * EMPTY_BUFFER_BYTES; ++i) {
        EXPECT_EQ(emptyBuffers.get()[i], EMPTY_SENTINEL) << "empty tensor byte=" << i;
    }
}

// Single row, aligned tile, alignment tail, multicore tail, and UB-limited loops.
INSTANTIATE_TEST_SUITE_P(BoundaryCases, AddRmsNormCastReduceEmpty950,
                         testing::Values(EmptyReductionCase{1, 1, 1, 1, 1}, EmptyReductionCase{16, 1, 16, 16, 16},
                                         EmptyReductionCase{19, 1, 19, 19, 16},
                                         EmptyReductionCase{8193, 2, 4097, 4096, 4096},
                                         EmptyReductionCase{20003, 3, 6668, 6667, 6664},
                                         EmptyReductionCase{130000, 2, 65000, 65000, 61184}));
} // namespace
