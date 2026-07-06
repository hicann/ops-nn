/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_quant_max.cpp
 * \brief Unit test for GroupedQuantMax op_kernel
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

// Directly include the kernel source so the template entry
// `template <uint64_t RoundMode> grouped_quant_max(...)` is visible in this TU.
// The link error "undefined reference to grouped_quant_max" comes from declaring
// it as a non-template `extern "C"` symbol while the real kernel is a template.
#include "../../../op_kernel/grouped_quant_max.cpp"

// Tiling data structure (must match GroupedQuantMaxTilingData in
// op_kernel/arch35/grouped_quant_max_tiling_data.h: field order and types
// must be identical because the buffer is memcpy'd to the kernel).
struct TestGroupedQuantMaxTilingData {
    int64_t totalCoreNum;
    int64_t actualCoreNum;
    int64_t blockTailCoreNum;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t ubFactor;
    int64_t dim1;
    int64_t roundMode;
    int64_t numGroups;
};

class grouped_quant_max_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "grouped_quant_max_test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "grouped_quant_max_test TearDown" << std::endl; }
};

// Helper function to generate test data
template <typename T>
void GenerateGroupedInputData(void* buffer, size_t size, float minValue = -1.0f, float maxValue = 1.0f)
{
    T* data = static_cast<T*>(buffer);
    size_t count = size / sizeof(T);
    for (size_t i = 0; i < count; i++) {
        float val = minValue + (maxValue - minValue) * (static_cast<float>(i % 1000) / 1000.0f);
        if constexpr (std::is_same_v<T, float>) {
            data[i] = val;
        } else if constexpr (std::is_same_v<T, half>) {
            data[i] = static_cast<half>(val);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            data[i] = static_cast<bfloat16_t>(val);
        }
    }
}

// Compute per-group amax (golden reference)
template <typename T>
void ComputeGroupedAmax(const T* data, const int64_t* groupList, int64_t numGroups, int64_t blockSize, float* amaxOut)
{
    for (int64_t g = 0; g < numGroups; ++g) {
        int64_t groupStart = (g == 0) ? 0 : groupList[g - 1] * blockSize;
        int64_t groupEnd = groupList[g] * blockSize;
        float amax = 0.0f;
        for (int64_t i = groupStart; i < groupEnd; ++i) {
            float val;
            if constexpr (std::is_same_v<T, float>) {
                val = std::abs(data[i]);
            } else if constexpr (std::is_same_v<T, half>) {
                val = std::abs(static_cast<float>(data[i]));
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                val = std::abs(static_cast<float>(data[i]));
            }
            if (val > amax) {
                amax = val;
            }
        }
        amaxOut[g] = amax;
    }
}

// Test FP32 to FP8_E5M2 with 3 groups, single core
TEST_F(grouped_quant_max_test, test_fp32_to_fp8_e5m2_3groups)
{
    // x shape: [6, 128], blockSize = 128, groupList = [2, 4, 6]
    // Group 0: rows 0-1, Group 1: rows 2-3, Group 2: rows 4-5
    int64_t numGroups = 3;
    int64_t blockSize = 128;
    int64_t totalElements = 6 * 128;
    size_t xSize = totalElements * sizeof(float);
    size_t ySize = totalElements * sizeof(uint8_t); // FP8
    size_t scaleSize = numGroups * sizeof(float);
    size_t groupListSize = numGroups * sizeof(int64_t);
    size_t amaxSize = numGroups * sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    // Generate input data
    GenerateGroupedInputData<float>(x, xSize, -2.0f, 2.0f);

    // Set per-group scale values
    float scaleVals[3] = {1.0f, 0.5f, 2.0f};
    memcpy(scale, scaleVals, scaleSize);

    // Set groupList: cumulative row counts [2, 4, 6]
    int64_t groupListVals[3] = {2, 4, 6};
    memcpy(groupList, groupListVals, groupListSize);

    // Setup tiling data
    size_t tilingSize = sizeof(TestGroupedQuantMaxTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestGroupedQuantMaxTilingData tilingData;
    tilingData.totalCoreNum = blockDim;
    tilingData.actualCoreNum = 1;
    tilingData.blockTailCoreNum = 0;
    tilingData.blockFactor = totalElements;
    tilingData.blockTailFactor = totalElements;
    tilingData.ubFactor = 256;
    tilingData.dim1 = blockSize;
    tilingData.roundMode = 0; // Rint
    tilingData.numGroups = numGroups;
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR workspace,
                            GM_ADDR tiling) {
        ::grouped_quant_max<0>(x, scale, groupList, y, amax, workspace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, x, scale, groupList, y, amax, workspace, tiling);

    // Verify per-group amax
    float expectedAmax[3];
    ComputeGroupedAmax(reinterpret_cast<float*>(x), reinterpret_cast<int64_t*>(groupList), numGroups, blockSize,
                       expectedAmax);

    float* amaxOut = reinterpret_cast<float*>(amax);
    for (int64_t g = 0; g < numGroups; ++g) {
        EXPECT_NEAR(amaxOut[g], expectedAmax[g], 0.01f) << "Group " << g << " amax mismatch";
    }

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// Test FP32 to FP8_E5M2 with 4 groups, multi-core
TEST_F(grouped_quant_max_test, test_fp32_to_fp8_e5m2_4groups_multicore)
{
    // x shape: [8, 64], blockSize = 64, groupList = [2, 4, 6, 8]
    int64_t numGroups = 4;
    int64_t blockSize = 64;
    int64_t totalElements = 8 * 64;
    size_t xSize = totalElements * sizeof(float);
    size_t ySize = totalElements * sizeof(uint8_t);
    size_t scaleSize = numGroups * sizeof(float);
    size_t groupListSize = numGroups * sizeof(int64_t);
    size_t amaxSize = numGroups * sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    GenerateGroupedInputData<float>(x, xSize, -1.0f, 1.0f);

    float scaleVals[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    memcpy(scale, scaleVals, scaleSize);

    int64_t groupListVals[4] = {2, 4, 6, 8};
    memcpy(groupList, groupListVals, groupListSize);

    size_t tilingSize = sizeof(TestGroupedQuantMaxTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestGroupedQuantMaxTilingData tilingData;
    tilingData.totalCoreNum = blockDim;
    tilingData.actualCoreNum = 4;
    tilingData.blockTailCoreNum = 0;
    tilingData.blockFactor = totalElements / 4;
    tilingData.blockTailFactor = totalElements / 4;
    tilingData.ubFactor = 128;
    tilingData.dim1 = blockSize;
    tilingData.roundMode = 0;
    tilingData.numGroups = numGroups;
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR workspace,
                            GM_ADDR tiling) {
        ::grouped_quant_max<0>(x, scale, groupList, y, amax, workspace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, x, scale, groupList, y, amax, workspace, tiling);

    float expectedAmax[4];
    ComputeGroupedAmax(reinterpret_cast<float*>(x), reinterpret_cast<int64_t*>(groupList), numGroups, blockSize,
                       expectedAmax);

    float* amaxOut = reinterpret_cast<float*>(amax);
    for (int64_t g = 0; g < numGroups; ++g) {
        EXPECT_NEAR(amaxOut[g], expectedAmax[g], 0.01f) << "Group " << g << " amax mismatch";
    }

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// Test single group (degenerate case, equivalent to QuantMax)
TEST_F(grouped_quant_max_test, test_fp32_single_group)
{
    int64_t numGroups = 1;
    int64_t blockSize = 128;
    int64_t totalElements = 4 * 128;
    size_t xSize = totalElements * sizeof(float);
    size_t ySize = totalElements * sizeof(uint8_t);
    size_t scaleSize = numGroups * sizeof(float);
    size_t groupListSize = numGroups * sizeof(int64_t);
    size_t amaxSize = numGroups * sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    GenerateGroupedInputData<float>(x, xSize, -3.0f, 3.0f);

    float scaleVal = 0.5f;
    memcpy(scale, &scaleVal, scaleSize);

    int64_t groupListVal = 4; // All 4 rows in one group
    memcpy(groupList, &groupListVal, groupListSize);

    size_t tilingSize = sizeof(TestGroupedQuantMaxTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestGroupedQuantMaxTilingData tilingData;
    tilingData.totalCoreNum = blockDim;
    tilingData.actualCoreNum = 1;
    tilingData.blockTailCoreNum = 0;
    tilingData.blockFactor = totalElements;
    tilingData.blockTailFactor = totalElements;
    tilingData.ubFactor = 256;
    tilingData.dim1 = blockSize;
    tilingData.roundMode = 0;
    tilingData.numGroups = numGroups;
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR workspace,
                            GM_ADDR tiling) {
        ::grouped_quant_max<0>(x, scale, groupList, y, amax, workspace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, x, scale, groupList, y, amax, workspace, tiling);

    float expectedAmax[1];
    ComputeGroupedAmax(reinterpret_cast<float*>(x), reinterpret_cast<int64_t*>(groupList), numGroups, blockSize,
                       expectedAmax);

    float* amaxOut = reinterpret_cast<float*>(amax);
    EXPECT_NEAR(amaxOut[0], expectedAmax[0], 0.01f);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// Test uneven group sizes
TEST_F(grouped_quant_max_test, test_fp32_uneven_groups)
{
    // x shape: [10, 128], groupList = [3, 7, 10] (groups of 3, 4, 3 rows)
    int64_t numGroups = 3;
    int64_t blockSize = 128;
    int64_t totalElements = 10 * 128;
    size_t xSize = totalElements * sizeof(float);
    size_t ySize = totalElements * sizeof(uint8_t);
    size_t scaleSize = numGroups * sizeof(float);
    size_t groupListSize = numGroups * sizeof(int64_t);
    size_t amaxSize = numGroups * sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 2;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    GenerateGroupedInputData<float>(x, xSize, -5.0f, 5.0f);

    float scaleVals[3] = {0.1f, 0.2f, 0.3f};
    memcpy(scale, scaleVals, scaleSize);

    int64_t groupListVals[3] = {3, 7, 10};
    memcpy(groupList, groupListVals, groupListSize);

    size_t tilingSize = sizeof(TestGroupedQuantMaxTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestGroupedQuantMaxTilingData tilingData;
    tilingData.totalCoreNum = blockDim;
    tilingData.actualCoreNum = 2;
    tilingData.blockTailCoreNum = 0;
    tilingData.blockFactor = totalElements / 2;
    tilingData.blockTailFactor = totalElements / 2;
    tilingData.ubFactor = 256;
    tilingData.dim1 = blockSize;
    tilingData.roundMode = 0;
    tilingData.numGroups = numGroups;
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(0);
    auto kernel_lambda = [](GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR workspace,
                            GM_ADDR tiling) {
        ::grouped_quant_max<0>(x, scale, groupList, y, amax, workspace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, x, scale, groupList, y, amax, workspace, tiling);

    float expectedAmax[3];
    ComputeGroupedAmax(reinterpret_cast<float*>(x), reinterpret_cast<int64_t*>(groupList), numGroups, blockSize,
                       expectedAmax);

    float* amaxOut = reinterpret_cast<float*>(amax);
    for (int64_t g = 0; g < numGroups; ++g) {
        EXPECT_NEAR(amaxOut[g], expectedAmax[g], 0.01f) << "Group " << g << " amax mismatch";
    }

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
