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
 * \file test_linear_index.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "test_linear_index_tiling_def.h"

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
#include <cstring>
#include <type_traits>

using namespace std;

extern "C" __global__ __aicore__ void linear_index(GM_ADDR indices, GM_ADDR var, GM_ADDR output, GM_ADDR workspace,
                                                   GM_ADDR tiling);
class linear_index_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "linear_index_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "linear_index_test TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./linear_index_data");
    }
};

TEST_F(linear_index_test, test_case_int32)
{
    // inputs
    size_t ind_size = 63806 * sizeof(int);
    size_t var_size = 65535 * 4096 * sizeof(float);
    size_t output_size = 63806 * sizeof(int);
    size_t tiling_data_size = sizeof(LinearIndexTilingDataDef);

    uint8_t* ind = (uint8_t*)AscendC::GmAlloc(ind_size);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    kernel_ut::SetupTestEnvironment("index/linear_index/tests/ut/op_kernel/linear_index_data", "linear_index_data");
    kernel_ut::RunGenData("./linear_index_data", {"int32"});
    kernel_ut::RunGenTiling("./linear_index_data", {});

    string path = kernel_ut::GetTestWorkDir();
    ReadFile(path + "/linear_index_data/indices.bin", ind_size, ind, ind_size);
    ReadFile(path + "/linear_index_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/linear_index_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    LinearIndexTilingDataDef* tilingDatafromBin = reinterpret_cast<LinearIndexTilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(11);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(21);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(ind);
    AscendC::GmFree(var);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(linear_index_test, test_case_int64)
{
    // inputs
    size_t ind_size = 63806 * sizeof(int64_t);
    size_t var_size = 65535 * 4096 * sizeof(float);
    size_t output_size = 63806 * sizeof(int);
    size_t tiling_data_size = sizeof(LinearIndexTilingDataDef);

    uint8_t* ind = (uint8_t*)AscendC::GmAlloc(ind_size);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    kernel_ut::SetupTestEnvironment("index/linear_index/tests/ut/op_kernel/linear_index_data", "linear_index_data");
    kernel_ut::RunGenData("./linear_index_data", {"int64"});
    kernel_ut::RunGenTiling("./linear_index_data", {});

    string path = kernel_ut::GetTestWorkDir();
    ReadFile(path + "/linear_index_data/indices.bin", ind_size, ind, ind_size);
    ReadFile(path + "/linear_index_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/linear_index_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    LinearIndexTilingDataDef* tilingDatafromBin = reinterpret_cast<LinearIndexTilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(21);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(22);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(ind);
    AscendC::GmFree(var);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

namespace {
template <typename T>
void RunLinearIndexPrecision(uint32_t mode, uint32_t divisor, uint32_t count = 123, uint32_t cores = 3,
                             uint32_t firstOffset = 0, bool scatter = false, uint32_t tile = 32)
{
    SCOPED_TRACE(sizeof(T) == sizeof(int64_t) ? "int64" : "int32");
    constexpr size_t guardSize = 32;
    constexpr uint8_t guardValue = 0xa5;
    std::vector<uint8_t*> allocations;
    auto alloc = [&allocations](size_t bytes) {
        auto* ptr = static_cast<uint8_t*>(AscendC::GmAlloc((bytes + 31) / 32 * 32));
        allocations.push_back(ptr);
        return ptr;
    };
    const uint32_t total = firstOffset + count;
    const int32_t target = mode == 1 ? 3 : 41;
    const int32_t selfStride = divisor == 3 ? 7 : divisor;
    const size_t outputSize = total * sizeof(int32_t);
    auto* input = alloc(total * sizeof(T));
    auto* outputBuffer = alloc(outputSize + 2 * guardSize);
    auto* output = outputBuffer + guardSize;
    std::memset(outputBuffer, guardValue, outputSize + 2 * guardSize);
    auto* unused = alloc(32);
    auto* tiling = alloc(sizeof(LinearIndexTilingDataDef));
    auto* values = reinterpret_cast<T*>(input);
    auto* actual = reinterpret_cast<int32_t*>(output);
    const std::array<T, 4> pattern = {0, 1, -1, static_cast<T>(-target)};
    for (uint32_t i = firstOffset; i < total; ++i) {
        values[i] = scatter ? 0 : pattern[i % pattern.size()];
    }

    LinearIndexTilingDataDef data{};
    data.usedCoreNum = cores;
    data.indicesCount = total;
    data.indicesAlign = tile;
    data.eachCount = firstOffset == 0 ? count / cores : firstOffset;
    data.lastCount = firstOffset == 0 ? count - data.eachCount * (cores - 1) : count;
    data.eachNum = data.lastNum = tile;
    // At large positions, execute only the last tile. Its global offset is unchanged;
    // preceding cores do no work, so the UT need not compute millions of unrelated elements.
    data.eachLoop = firstOffset == 0 ? (data.eachCount + tile - 1) / tile : 0;
    data.eachTail = (data.eachCount - 1) % tile + 1;
    data.lastLoop = (data.lastCount + tile - 1) / tile;
    data.lastTail = (data.lastCount - 1) % tile + 1;
    data.target = target;
    data.selfStride = selfStride;
    data.indicesStride = divisor;
    std::memcpy(tiling, &data, sizeof(data));

    ICPU_SET_TILING_KEY(mode * 10 + (std::is_same<T, int64_t>::value ? 2 : 1));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, cores, input, unused, output, unused, tiling);

    std::vector<int32_t> result(actual + firstOffset, actual + total);
    std::array<uint8_t, guardSize> prefixGuard;
    std::array<uint8_t, guardSize> suffixGuard;
    std::array<uint8_t, guardSize> executionPrefixGuard;
    std::memcpy(prefixGuard.data(), outputBuffer, guardSize);
    std::memcpy(suffixGuard.data(), output + outputSize, guardSize);
    std::memcpy(executionPrefixGuard.data(), output + firstOffset * sizeof(int32_t) - guardSize, guardSize);
    std::vector<int64_t> expected(count);
    for (uint32_t i = firstOffset; i < total; ++i) {
        const int64_t index = values[i] < 0 ? values[i] + target : values[i];
        expected[i - firstOffset] = mode == 1 ? index * selfStride + i % divisor :
                                    mode == 2 ? index + (i / divisor) * selfStride :
                                                index;
    }
    for (auto* ptr : allocations) {
        AscendC::GmFree(ptr);
    }
    for (size_t i = 0; i < guardSize; ++i) {
        EXPECT_EQ(prefixGuard[i], guardValue) << "output prefix byte=" << i;
        EXPECT_EQ(suffixGuard[i], guardValue) << "output suffix byte=" << i;
        EXPECT_EQ(executionPrefixGuard[i], guardValue) << "execution prefix byte=" << i;
    }
    for (uint32_t i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], expected[i]) << "mode=" << mode << ", position=" << firstOffset + i;
    }
    if (scatter) {
        // Issue 4768: [2, 41, 2], expanded index strides [41, 1, 0], all-zero indices.
        std::array<int32_t, 164> sums{};
        for (uint32_t i = 0; i < count; ++i) {
            ASSERT_GE(result[i], 0);
            ASSERT_LT(result[i], 82);
            ++sums[result[i] * 2];
            ++sums[result[i] * 2 + 1];
        }
        for (size_t i = 0; i < sums.size(); ++i) {
            EXPECT_EQ(sums[i], (i == 0 || i == 1 || i == 82 || i == 83) ? 41 : 0) << i;
        }
    }
}
} // namespace

TEST_F(linear_index_test, test_case_precision_mode1_quotient_rounds_down)
{
    RunLinearIndexPrecision<int32_t>(1, 41);
    RunLinearIndexPrecision<int64_t>(1, 41);
}

TEST_F(linear_index_test, test_case_precision_issue4768_expand)
{
    RunLinearIndexPrecision<int32_t>(2, 41, 82, 1, 0, true);
    RunLinearIndexPrecision<int64_t>(2, 41, 82, 1, 0, true);
}

TEST_F(linear_index_test, test_case_precision_mode1_quotient_rounds_up)
{
    // Includes p=16777214, d=3; p and d are both exactly representable in FP32.
    RunLinearIndexPrecision<int32_t>(1, 3, 31, 2, 16777184);
    RunLinearIndexPrecision<int64_t>(1, 3, 31, 2, 16777184);
}

TEST_F(linear_index_test, test_case_precision_mode2_quotient_rounds_up)
{
    RunLinearIndexPrecision<int32_t>(2, 3, 31, 2, 16777184);
    RunLinearIndexPrecision<int64_t>(2, 3, 31, 2, 16777184);
}

TEST_F(linear_index_test, test_case_precision_mode0_negative_indices)
{
    RunLinearIndexPrecision<int32_t>(0, 41);
    RunLinearIndexPrecision<int64_t>(0, 41);
}

TEST_F(linear_index_test, test_case_precision_exact_power_of_two_divisor)
{
    RunLinearIndexPrecision<int32_t>(1, 32);
    RunLinearIndexPrecision<int64_t>(1, 32);
    RunLinearIndexPrecision<int32_t>(2, 32);
    RunLinearIndexPrecision<int64_t>(2, 32);
}

TEST_F(linear_index_test, test_case_precision_sequence_multiple_blocks_and_tail)
{
    // A non-power-of-two tile exercises prefix doubling, multiple UB iterations and padding.
    for (uint32_t mode : {1, 2}) {
        RunLinearIndexPrecision<int32_t>(mode, 41, 2051, 3, 0, false, 392);
        RunLinearIndexPrecision<int64_t>(mode, 41, 2051, 3, 0, false, 392);
    }
}

TEST_F(linear_index_test, test_case_precision_sequence_at_fp32_boundary)
{
    // Includes p=2^24, with different indices/self strides and a partially filled final vector.
    for (uint32_t mode : {1, 2}) {
        RunLinearIndexPrecision<int32_t>(mode, 3, 257, 2, 16776960, false, 392);
        RunLinearIndexPrecision<int64_t>(mode, 3, 257, 2, 16776960, false, 392);
    }
}

TEST_F(linear_index_test, test_case_precision_ub_capacity_and_short_tail)
{
    // Use the 910B tiling limits for each mode/type, followed by a one-element UB iteration.
    RunLinearIndexPrecision<int32_t>(1, 41, 11521, 1, 0, false, 11520);
    RunLinearIndexPrecision<int64_t>(1, 41, 7681, 1, 0, false, 7680);
    RunLinearIndexPrecision<int32_t>(2, 41, 15361, 1, 0, false, 15360);
    RunLinearIndexPrecision<int64_t>(2, 41, 9217, 1, 0, false, 9216);
}
