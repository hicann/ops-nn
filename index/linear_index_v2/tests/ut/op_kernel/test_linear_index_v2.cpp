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
 * \file test_linear_index_v2.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "linear_index_v2_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"
#include "tensor_list_operate.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>
#include <cstring>
#include <type_traits>

using namespace std;

extern "C" __global__ __aicore__ void linear_index_v2(GM_ADDR indexList, GM_ADDR stride, GM_ADDR valueSize,
                                                      GM_ADDR output, GM_ADDR workSpace, GM_ADDR tiling);
class linear_index_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "linear_index_v2_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "linear_index_v2_test TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./linear_index_v2_data");
    }
};

TEST_F(linear_index_v2_test, test_case_0)
{
    std::vector<std::vector<uint64_t>> idx_shape{{3}, {3}};
    size_t stride_size = 2 * sizeof(int);
    size_t value_size = 2 * sizeof(int);
    size_t output_size = 3 * sizeof(int);
    size_t tiling_size = sizeof(LinearIndexV2TilingData);

    kernel_ut::SetupTestEnvironment("index/linear_index_v2/tests/ut/op_kernel/linear_index_v2_data",
                                    "linear_index_v2_data");
    kernel_ut::RunGenData("./linear_index_v2_data", {});
    kernel_ut::RunGenTiling("./linear_index_v2_data", {"test_case_continuous"});

    uint8_t* idx_list = CreateTensorList<int32_t>(idx_shape);
    uint8_t* stride = (uint8_t*)AscendC::GmAlloc(stride_size);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);
    uint32_t blockDim = 3;

    string path = kernel_ut::GetTestWorkDir();
    ReadFile(path + "/linear_index_v2_data/stride.bin", stride_size, stride, stride_size);
    ReadFile(path + "/linear_index_v2_data/value.bin", value_size, value, value_size);
    ReadFile(path + "/linear_index_v2_data/tiling.bin", tiling_size, tiling, tiling_size);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index_v2, blockDim, idx_list, stride, value, output, workspace, tiling);

    FreeTensorList<int32_t>(idx_list, idx_shape);
    AscendC::GmFree(stride);
    AscendC::GmFree(value);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

namespace {
template <typename T>
void RunLinearIndexV2Precision(const std::vector<std::vector<int32_t>>& patterns, const std::vector<int32_t>& sizes,
                               const std::vector<int32_t>& strides, uint32_t cores = 1, uint32_t formerCoreCount = 24,
                               uint32_t count = 0, uint32_t maxTileSize = 8)
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
    // Exercise multiple UB iterations and an unaligned tail, including the second core.
    count = count == 0 ? (cores == 1 ? 17 : 35) : count;
    const size_t tensors = patterns.size();
    auto* list = alloc((1 + 3 * tensors) * sizeof(uint64_t));
    auto* descriptor = reinterpret_cast<uint64_t*>(list);
    descriptor[0] = (1 + 2 * tensors) * sizeof(uint64_t);
    std::vector<int64_t> expected(count, 0);
    for (size_t tensor = 0; tensor < tensors; ++tensor) {
        auto* input = reinterpret_cast<T*>(alloc(count * sizeof(T)));
        descriptor[1 + 2 * tensor] = (static_cast<uint64_t>(tensor) << 32) | 1;
        descriptor[2 + 2 * tensor] = count;
        descriptor[1 + 2 * tensors + tensor] = reinterpret_cast<uint64_t>(input);
        for (uint32_t i = 0; i < count; ++i) {
            const int64_t value = patterns[tensor][i % patterns[tensor].size()];
            input[i] = static_cast<T>(value);
            const int64_t size = sizes[tensor];
            const int64_t remainder = size == 0 ? value : (value % size + size) % size;
            expected[i] += remainder * strides[tensor];
        }
    }
    auto* stride = alloc(tensors * sizeof(int32_t));
    auto* valueSize = alloc(tensors * sizeof(int32_t));
    std::memcpy(stride, strides.data(), tensors * sizeof(int32_t));
    std::memcpy(valueSize, sizes.data(), tensors * sizeof(int32_t));
    const size_t outputSize = count * sizeof(int32_t);
    auto* outputBuffer = alloc(outputSize + 2 * guardSize);
    auto* output = outputBuffer + guardSize;
    std::memset(outputBuffer, guardValue, outputSize + 2 * guardSize);
    auto* workspace = alloc(1024);
    auto* tiling = alloc(sizeof(LinearIndexV2TilingData));
    LinearIndexV2TilingData data{};
    auto& params = data.params;
    params.usedCoreNum = cores;
    params.tensorId = tensors;
    params.formerCoreNum = cores - 1;
    params.formerCoreDataNum = formerCoreCount;
    // Match the host's balanced UB split within each core.
    const uint64_t formerCopyTime = (params.formerCoreDataNum + maxTileSize - 1) / maxTileSize;
    params.formerCoreFormerDataNum = (params.formerCoreDataNum + formerCopyTime - 1) / formerCopyTime;
    params.formerCoreFormerTime = params.formerCoreDataNum % formerCopyTime;
    params.formerCoreTailDataNum = params.formerCoreDataNum / formerCopyTime;
    params.formerCoreTailTime = formerCopyTime - params.formerCoreFormerTime;
    params.tailCoreNum = 1;
    params.tailCoreDataNum = count - formerCoreCount * (cores - 1);
    const uint64_t tailCopyTime = (params.tailCoreDataNum + maxTileSize - 1) / maxTileSize;
    params.tailCoreFormerDataNum = (params.tailCoreDataNum + tailCopyTime - 1) / tailCopyTime;
    params.tailCoreFormerTime = params.tailCoreDataNum % tailCopyTime;
    params.tailCoreTailDataNum = params.tailCoreDataNum / tailCopyTime;
    params.tailCoreTailTime = tailCopyTime - params.tailCoreFormerTime;
    std::memcpy(tiling, &data, sizeof(data));

    ICPU_SET_TILING_KEY((std::is_same<T, int64_t>::value ? 0 : 1));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index_v2, cores, list, stride, valueSize, output, workspace, tiling);
    const auto* actual = reinterpret_cast<int32_t*>(output);
    std::vector<int32_t> result(actual, actual + count);
    std::array<uint8_t, guardSize> prefixGuard;
    std::array<uint8_t, guardSize> suffixGuard;
    std::memcpy(prefixGuard.data(), outputBuffer, guardSize);
    std::memcpy(suffixGuard.data(), output + outputSize, guardSize);
    for (auto* ptr : allocations) {
        AscendC::GmFree(ptr);
    }
    for (size_t i = 0; i < guardSize; ++i) {
        EXPECT_EQ(prefixGuard[i], guardValue) << "output prefix byte=" << i;
        EXPECT_EQ(suffixGuard[i], guardValue) << "output suffix byte=" << i;
    }
    for (uint32_t i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], expected[i]) << "position=" << i;
    }
}
} // namespace

TEST_F(linear_index_v2_test, test_case_precision_valid_axis_index_near_fp32_boundary)
{
    // Every index is in [-size, size), and every integer is exactly representable in FP32.
    RunLinearIndexV2Precision<int32_t>({{16777214, -16777215, -16777214, -1, 0, 1, 16777213}}, {16777215}, {3});
    RunLinearIndexV2Precision<int64_t>({{16777214, -16777215, -16777214, -1, 0, 1, 16777213}}, {16777215}, {3});
}

TEST_F(linear_index_v2_test, test_case_precision_valid_axis_index_below_fp32_boundary)
{
    RunLinearIndexV2Precision<int32_t>({{13103684, -13103685, -1, 0, 1, 13103683}}, {13103685}, {1});
    RunLinearIndexV2Precision<int64_t>({{13103684, -13103685, -1, 0, 1, 13103683}}, {13103685}, {1});
}

TEST_F(linear_index_v2_test, test_case_precision_remainder_at_multiples)
{
    RunLinearIndexV2Precision<int32_t>({{-82, -42, -41, -40, -1, 0, 1, 40, 41, 42, 81, 82, 83}}, {41}, {3});
    RunLinearIndexV2Precision<int64_t>({{-82, -42, -41, -40, -1, 0, 1, 40, 41, 42, 81, 82, 83}}, {41}, {3});
}

TEST_F(linear_index_v2_test, test_case_precision_multiple_tensors_cores_and_tail)
{
    RunLinearIndexV2Precision<int32_t>({{16777214, -16777215, -1, 0}, {40, -41, -1, 0, 1}}, {16777215, 41}, {41, 1}, 2);
    RunLinearIndexV2Precision<int64_t>({{16777214, -16777215, -1, 0}, {40, -41, -1, 0, 1}}, {16777215, 41}, {41, 1}, 2);
}

TEST_F(linear_index_v2_test, test_case_precision_exact_divisors)
{
    RunLinearIndexV2Precision<int32_t>({{-1, 0}, {-256, -255, -1, 0, 1, 255}}, {1, 256}, {256, 1});
    RunLinearIndexV2Precision<int64_t>({{-1, 0}, {-256, -255, -1, 0, 1, 255}}, {1, 256}, {256, 1});
}

TEST_F(linear_index_v2_test, test_case_precision_zero_size_preserves_existing_behavior)
{
    RunLinearIndexV2Precision<int32_t>({{-41, -1, 0, 1, 41}}, {0}, {3});
    RunLinearIndexV2Precision<int64_t>({{-41, -1, 0, 1, 41}}, {0}, {3});
}

TEST_F(linear_index_v2_test, test_case_precision_signed_fp32_boundary)
{
    const std::vector<std::vector<int32_t>> patterns = {{-16777216, -16777215, 16777215, 16777216},
                                                        {-16777216, -16777215, 16777215, 16777216}};
    RunLinearIndexV2Precision<int32_t>(patterns, {3, 41}, {41, 1}, 2);
    RunLinearIndexV2Precision<int64_t>(patterns, {3, 41}, {41, 1}, 2);
}

TEST_F(linear_index_v2_test, test_case_precision_unaligned_core_boundaries)
{
    // The cores share a cache line at element 17; prefilled output must be overwritten before accumulation.
    const std::vector<std::vector<int32_t>> patterns = {
        {0, 16777214, -16777215, -1}, {-82, -41, 0, 40, 82}, {-41, 0, 41}};
    RunLinearIndexV2Precision<int32_t>(patterns, {16777215, 41, 0}, {3, 1, -3}, 2, 17);
    RunLinearIndexV2Precision<int64_t>(patterns, {16777215, 41, 0}, {3, 1, -3}, 2, 17);
}

TEST_F(linear_index_v2_test, test_case_precision_power_of_two_signed_boundaries)
{
    for (int32_t size : {1, 2, 256, 65536, 8388608, 16777216}) {
        SCOPED_TRACE(size);
        const std::vector<std::vector<int32_t>> patterns = {
            {-16777216, -16777215, -1, 0, 1, size - 1, size, 16777215, 16777216}};
        RunLinearIndexV2Precision<int32_t>(patterns, {size}, {3}, 2, 17);
        RunLinearIndexV2Precision<int64_t>(patterns, {size}, {3}, 2, 17);
    }
}

TEST_F(linear_index_v2_test, test_case_precision_ub_capacity_mixed_sizes)
{
    // The 910B UB limit gives blocks 10992+10991 and 10991+10991 on the two cores.
    // The power-of-two mask uses both uint16 halves of each int32 across vector repeats.
    const std::vector<std::vector<int32_t>> patterns = {
        {-16777216, -16777215, -41, -1, 0, 1, 16777214, 16777216},
        {-16777216, -131073, -131072, -1, 0, 65536, 131071, 131072, 16777215, 16777216},
        {-16777216, -41, -1, 0, 1, 41, 16777216}};
    RunLinearIndexV2Precision<int32_t>(patterns, {41, 131072, 0}, {131072, 3, -1}, 2, 21983, 43965, 10992);
    RunLinearIndexV2Precision<int64_t>(patterns, {41, 131072, 0}, {131072, 3, -1}, 2, 21983, 43965, 10992);
}
