/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scatter_add_with_sorted.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <type_traits>
#include "gtest/gtest.h"
#include "scatter_add_with_sorted_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void scatter_add_with_sorted(GM_ADDR var, GM_ADDR value, GM_ADDR sorted_index,
                                                              GM_ADDR pos, GM_ADDR output, GM_ADDR workspace,
                                                              GM_ADDR tiling);
class scatter_add_with_sorted_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "scatter_add_with_sorted_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "scatter_add_with_sorted_test TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./scatter_add_with_sorted_data");
    }
};

TEST_F(scatter_add_with_sorted_test, test_case_fp32)
{
    // inputs
    size_t var_size = 65 * 4096 * sizeof(float);
    size_t src_size = 63 * 4096 * sizeof(float);
    size_t ind_size = 63 * sizeof(int);
    size_t pos_size = 63 * sizeof(int);
    size_t output_size = 65 * 4096 * sizeof(float);
    size_t tiling_data_size = sizeof(ScatterAddWithSortedTilingData);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t* ind = (uint8_t*)AscendC::GmAlloc(ind_size);
    uint8_t* pos = (uint8_t*)AscendC::GmAlloc(pos_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    kernel_ut::SetupTestEnvironment("index/scatter_add_with_sorted/tests/ut/op_kernel/scatter_add_with_sorted_data",
                                    "scatter_add_with_sorted_data");
    kernel_ut::RunGenData("./scatter_add_with_sorted_data", {"float32"});
    kernel_ut::RunGenTiling("./scatter_add_with_sorted_data", {});

    string path = kernel_ut::GetTestWorkDir();
    ReadFile(path + "/scatter_add_with_sorted_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/scatter_add_with_sorted_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/scatter_add_with_sorted_data/ind.bin", ind_size, ind, ind_size);
    ReadFile(path + "/scatter_add_with_sorted_data/pos.bin", pos_size, pos, pos_size);
    ReadFile(path + "/scatter_add_with_sorted_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ScatterAddWithSortedTilingData* tilingDatafromBin = reinterpret_cast<ScatterAddWithSortedTilingData*>(tiling);

    ICPU_SET_TILING_KEY(11);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_add_with_sorted, blockDim, var, src, ind, pos, output, workspace,
                (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(src);
    AscendC::GmFree(ind);
    AscendC::GmFree(pos);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

namespace {
// CPU UT checks values and chunk boundaries; device racecheck checks pipeline synchronization.
template <typename T = float, uint32_t tilingKey = 11>
void RunRepeatedIndices(uint32_t rows, uint32_t updates, uint32_t chunkSize, bool nonzeroInitial)
{
    constexpr uint32_t cols = 3;
    constexpr uint32_t blockDim = 4;
    const uint32_t perCore = updates / blockDim;
    auto* var = static_cast<T*>(AscendC::GmAlloc(rows * cols * sizeof(T)));
    auto* src = static_cast<T*>(AscendC::GmAlloc(updates * cols * sizeof(T)));
    auto* ind = static_cast<int32_t*>(AscendC::GmAlloc(updates * sizeof(int32_t)));
    auto* pos = static_cast<int32_t*>(AscendC::GmAlloc(updates * sizeof(int32_t)));
    auto* tiling = static_cast<ScatterAddWithSortedTilingData*>(
        AscendC::GmAlloc(sizeof(ScatterAddWithSortedTilingData)));
    std::vector<T> expected(rows * cols);
    for (uint32_t i = 0; i < rows * cols; ++i) {
        var[i] = nonzeroInitial ? static_cast<T>(i % 8 + 1) : static_cast<T>(0);
        expected[i] = var[i];
    }
    for (uint32_t i = 0; i < updates * cols; ++i) {
        src[i] = static_cast<T>(i % (std::is_integral_v<T> ? 2 : 16));
    }
    for (uint32_t i = 0; i < updates; ++i) {
        // Sorted duplicate indices, with a non-identity permutation of updates.
        ind[i] = i * rows / updates;
        pos[i] = updates - 1 - i;
        for (uint32_t col = 0; col < cols; ++col) {
            if constexpr (tilingKey < 10) {
                expected[ind[i] * cols + col] = src[pos[i] * cols + col];
            } else if constexpr (std::is_integral_v<T>) {
                expected[ind[i] * cols + col] += src[i * cols + col];
            } else {
                expected[ind[i] * cols + col] += src[pos[i] * cols + col];
            }
        }
    }

    *tiling = {};
    tiling->usedCoreNum = blockDim;
    tiling->eachCount = perCore;
    tiling->lastCount = perCore;
    tiling->inputCount = rows * cols;
    tiling->indicesCount = updates;
    tiling->updatesCount = updates * cols;
    tiling->inputOneTime = cols;
    tiling->updatesOneTime = cols;
    tiling->updatesAlign = 32 / sizeof(T);
    tiling->maxSize = chunkSize;
    tiling->eachNum = std::is_integral_v<T> && tilingKey > 10 ? perCore : chunkSize;
    tiling->eachLoop = (perCore + chunkSize - 1) / chunkSize;
    tiling->eachTail = (perCore - 1) % chunkSize + 1;
    tiling->lastNum = tiling->eachNum;
    tiling->lastLoop = tiling->eachLoop;
    tiling->lastTail = tiling->eachTail;
    tiling->updatesLoop = 1;
    tiling->updatesEach = cols;
    tiling->updatesLast = cols;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_add_with_sorted, blockDim, reinterpret_cast<uint8_t*>(var), reinterpret_cast<uint8_t*>(src),
                reinterpret_cast<uint8_t*>(ind), reinterpret_cast<uint8_t*>(pos), reinterpret_cast<uint8_t*>(var),
                nullptr, reinterpret_cast<uint8_t*>(tiling));
    // Small integer values also give exact FP32 reference results.
    for (uint32_t i = 0; i < rows * cols; ++i) {
        EXPECT_EQ(var[i], expected[i]) << "element " << i;
    }
    AscendC::GmFree(var);
    AscendC::GmFree(src);
    AscendC::GmFree(ind);
    AscendC::GmFree(pos);
    AscendC::GmFree(tiling);
}
} // namespace

TEST_F(scatter_add_with_sorted_test, test_case_fp32_repeated_indices) { RunRepeatedIndices<>(256, 512, 16, false); }

TEST_F(scatter_add_with_sorted_test, test_case_fp32_nonzero_initial_and_index_refill)
{
    RunRepeatedIndices<>(17, 80, 8, true);
}

TEST_F(scatter_add_with_sorted_test, test_case_fp32_duplicate_spans_cores) { RunRepeatedIndices<>(1, 80, 8, true); }

TEST_F(scatter_add_with_sorted_test, test_case_scatter_nonzero_initial_and_index_refill)
{
    RunRepeatedIndices<float, 1>(17, 80, 8, true);
}

TEST_F(scatter_add_with_sorted_test, test_case_scatter_duplicate_spans_cores)
{
    RunRepeatedIndices<float, 1>(1, 80, 8, true);
}

TEST_F(scatter_add_with_sorted_test, test_case_int32_index_reuse) { RunRepeatedIndices<int32_t, 13>(17, 80, 8, true); }

TEST_F(scatter_add_with_sorted_test, test_case_scatter_uint8_index_refill)
{
    RunRepeatedIndices<uint8_t, 4>(17, 80, 8, true);
}

TEST_F(scatter_add_with_sorted_test, test_case_int8_index_reuse) { RunRepeatedIndices<int8_t, 15>(17, 80, 8, true); }
