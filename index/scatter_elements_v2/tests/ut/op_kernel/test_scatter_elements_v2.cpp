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
 * \file test_scatter_elements_v2.cpp
 * \brief
 */
#include <array>
#include <cmath>
#include <limits>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/arch22/scatter_elements_v2_tiling.h"
#include "../../../common/scatter_elements_v2_amax_policy.h"

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

using namespace std;

extern "C" __global__ __aicore__ void scatter_elements_v2(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR output,
                                                          GM_ADDR workspace, GM_ADDR tiling);
class scatter_elements_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "scatter_elements_v2_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "scatter_elements_v2_test TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./scatter_elements_v2_data");
    }
};

TEST_F(scatter_elements_v2_test, test_case_fp32)
{
    // inputs
    size_t data_size = 128 * 59;
    size_t var_size = data_size * sizeof(float);
    size_t indices_size = data_size * sizeof(long);
    size_t src_size = data_size * sizeof(float);
    size_t output_size = data_size * sizeof(float);
    size_t tiling_data_size = sizeof(ScatterElementsV2TilingData);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indices_size);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    kernel_ut::SetupTestEnvironment("index/scatter_elements_v2/tests/ut/op_kernel/scatter_elements_v2_data",
                                    "scatter_elements_v2_data");
    kernel_ut::RunGenData("./scatter_elements_v2_data", {"float32"});
    kernel_ut::RunGenTiling("./scatter_elements_v2_data", {});

    string path = kernel_ut::GetTestWorkDir();
    ReadFile(path + "/scatter_elements_v2_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/scatter_elements_v2_data/indices.bin", indices_size, indices, indices_size);
    ReadFile(path + "/scatter_elements_v2_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/scatter_elements_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ScatterElementsV2TilingData* tilingDatafromBin = reinterpret_cast<ScatterElementsV2TilingData*>(tiling);

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(indices);
    AscendC::GmFree(src);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

template <typename Index>
static void RunSumIncludeSelfCase(bool includeSelf, bool splitIndices, uint64_t mode = 2)
{
    constexpr uint32_t rows = 2;
    constexpr uint32_t columns = 16;
    constexpr uint32_t updatesPerRow = 24;
    vector<float> initial(rows * columns);
    vector<Index> indices(rows * updatesPerRow);
    vector<float> updates(rows * updatesPerRow);
    vector<float> expected(rows * columns);
    vector<bool> seen(rows * columns, includeSelf);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            initial[row * columns + column] = 100.0f + row * columns + column;
            expected[row * columns + column] = initial[row * columns + column];
        }
        for (uint32_t column : {0U, 9U, 15U}) {
            if (!includeSelf) {
                expected[row * columns + column] = 0.0f;
            }
        }
        for (uint32_t update = 0; update < updatesPerRow; ++update) {
            uint32_t column = update % 3 == 0 ? 0 : (update % 3 == 1 ? 9 : 15);
            indices[row * updatesPerRow + update] = static_cast<Index>(column);
            updates[row * updatesPerRow + update] = static_cast<float>(update + row + 1);
            const auto target = row * columns + column;
            if (mode == 4 || mode == 5) {
                auto& value = updates[row * updatesPerRow + update];
                if (update == 0 || update == 10) {
                    value = numeric_limits<float>::quiet_NaN();
                }
                if (!seen[target]) {
                    expected[target] = value;
                    seen[target] = true;
                } else if (std::isnan(expected[target]) || std::isnan(value)) {
                    expected[target] = numeric_limits<float>::quiet_NaN();
                } else {
                    expected[target] = mode == 4 ? std::min(expected[target], value) :
                                                   std::max(expected[target], value);
                }
            } else {
                expected[target] += updates[row * updatesPerRow + update];
            }
        }
    }

    auto* inputGm = static_cast<float*>(AscendC::GmAlloc(initial.size() * sizeof(float)));
    auto* indicesGm = static_cast<Index*>(AscendC::GmAlloc(indices.size() * sizeof(Index)));
    auto* updatesGm = static_cast<float*>(AscendC::GmAlloc(updates.size() * sizeof(float)));
    auto* tilingGm = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(ScatterElementsV2TilingData)));
    memcpy(inputGm, initial.data(), initial.size() * sizeof(float));
    memcpy(indicesGm, indices.data(), indices.size() * sizeof(Index));
    memcpy(updatesGm, updates.data(), updates.size() * sizeof(float));
    memset(tilingGm, 0, sizeof(ScatterElementsV2TilingData));
    auto* tiling = reinterpret_cast<ScatterElementsV2TilingData*>(tilingGm);
    tiling->usedCoreNum = splitIndices ? 4 : rows;
    tiling->eachNum = splitIndices ? 0 : 1;
    tiling->eachPiece = splitIndices ? 2 : 1;
    tiling->inputOnePiece = splitIndices ? columns / 2 : columns;
    tiling->inputCount = initial.size();
    tiling->indicesCount = indices.size();
    tiling->updatesCount = updates.size();
    tiling->inputOneTime = columns;
    tiling->indicesOneTime = updatesPerRow;
    tiling->updatesOneTime = updatesPerRow;
    tiling->inputEach = splitIndices ? columns / 2 : columns;
    tiling->inputLast = tiling->inputEach;
    tiling->inputLoop = 1;
    tiling->indicesEach = splitIndices ? 8 : 1;
    tiling->indicesLast = tiling->indicesEach;
    tiling->indicesLoop = splitIndices ? 3 : 1;
    tiling->inputAlign = splitIndices ? columns / 2 : columns;
    tiling->indicesAlign = splitIndices ? 8 : updatesPerRow;
    tiling->updatesAlign = tiling->indicesAlign;
    tiling->lastIndicesLoop = 1;
    tiling->lastIndicesEach = 1;
    tiling->lastIndicesLast = 1;
    tiling->oneTime = 1;
    tiling->lastOneTime = 1;
    tiling->modeFlag = splitIndices ? 0 : 1;
    tiling->includeSelf = includeSelf ? 1 : 0;
    tiling->mode = mode;

    ICPU_SET_TILING_KEY(sizeof(Index) == sizeof(int64_t) ? 120 : 110);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, static_cast<uint32_t>(tiling->usedCoreNum), reinterpret_cast<uint8_t*>(inputGm),
                reinterpret_cast<uint8_t*>(indicesGm), reinterpret_cast<uint8_t*>(updatesGm),
                reinterpret_cast<uint8_t*>(inputGm), nullptr, tilingGm);
    for (size_t index = 0; index < expected.size(); ++index) {
        if (std::isnan(expected[index])) {
            EXPECT_TRUE(std::isnan(inputGm[index])) << "output index=" << index;
        } else {
            EXPECT_FLOAT_EQ(inputGm[index], expected[index]) << "output index=" << index;
        }
    }
    AscendC::GmFree(inputGm);
    AscendC::GmFree(indicesGm);
    AscendC::GmFree(updatesGm);
    AscendC::GmFree(tilingGm);
}

TEST_F(scatter_elements_v2_test, sum_include_self_false_small)
{
    RunSumIncludeSelfCase<int32_t>(false, false);
    RunSumIncludeSelfCase<int64_t>(false, false);
}

TEST_F(scatter_elements_v2_test, sum_include_self_false_split_indices)
{
    RunSumIncludeSelfCase<int32_t>(false, true);
    RunSumIncludeSelfCase<int64_t>(false, true);
}

TEST_F(scatter_elements_v2_test, sum_include_self_true_regression)
{
    RunSumIncludeSelfCase<int32_t>(true, false);
    RunSumIncludeSelfCase<int64_t>(true, true);
}

TEST_F(scatter_elements_v2_test, max_layouts_repeatability)
{
    for (uint64_t key : {1100U, 2100U, 3100U}) {
        SCOPED_TRACE(key);
        const bool atomic = key == 2100;
        const uint32_t n = atomic ? 7 : 3;
        const uint32_t c = atomic ? 32 : 33;
        const uint32_t e = atomic ? 131 : 3;
        const auto plan = ScatterElementsV2Amax::Select(n, e, c, sizeof(float), key == 3100, 4, 196608);
        if (atomic) {
            ASSERT_TRUE(plan.valid);
            ASSERT_EQ(plan.algorithm, ScatterElementsV2Amax::Algorithm::InputPartition);
            ASSERT_GT(plan.cores, 1U);
        }
        for (bool includeSelf : {false, true}) {
            SCOPED_TRACE(includeSelf);
            vector<float> initial(n * c, 9.0f), source(e * c, 2.0f);
            vector<int64_t> index(key == 3100 ? e : e * c, 0);
            for (uint32_t row = 0; row < e; ++row) {
                for (uint32_t col = 0; col < c; ++col) {
                    source[row * c + col] = row + col;
                }
                source[row * c + 2] = -numeric_limits<float>::infinity();
                source[row * c + 3] = row == 0 ? -0.0f : 0.0f;
                source[row * c + 4] = row == 0 ? 0.0f : -0.0f;
            }
            source[0] = numeric_limits<float>::quiet_NaN();
            source[(e - 1) * c] = numeric_limits<float>::quiet_NaN();
            source[(e / 2) * c + 1] = numeric_limits<float>::infinity();
            initial[3] = 0.0f;
            initial[4] = -0.0f;
            initial[5] = numeric_limits<float>::quiet_NaN();
            initial[c] = -0.0f;
            initial[c + 1] = numeric_limits<float>::quiet_NaN();
            initial[c + 2] = numeric_limits<float>::infinity();
            initial[c + 3] = -numeric_limits<float>::infinity();
            auto expected = initial;
            expected[0] = numeric_limits<float>::quiet_NaN();
            expected[1] = numeric_limits<float>::infinity();
            expected[2] = includeSelf ? initial[2] : -numeric_limits<float>::infinity();
            expected[3] = includeSelf ? initial[3] : -0.0f;
            expected[4] = includeSelf ? initial[4] : 0.0f;
            expected[5] = includeSelf ? initial[5] : static_cast<float>(e - 1 + 5);
            for (uint32_t col = 6; col < c; ++col) {
                const float updated = static_cast<float>(e - 1 + col);
                expected[col] = includeSelf ? std::max(initial[col], updated) : updated;
            }
            auto* var = static_cast<float*>(AscendC::GmAlloc(initial.size() * sizeof(float)));
            auto* idx = static_cast<int64_t*>(AscendC::GmAlloc(index.size() * sizeof(int64_t)));
            auto* src = static_cast<float*>(AscendC::GmAlloc(source.size() * sizeof(float)));
            auto* td = static_cast<ScatterElementsV2TilingData*>(AscendC::GmAlloc(sizeof(ScatterElementsV2TilingData)));
            const size_t workspaceBytes = n * c * (sizeof(float) + 2 * sizeof(int32_t));
            auto* workspace = atomic ? static_cast<uint8_t*>(AscendC::GmAlloc(workspaceBytes)) : nullptr;
            ASSERT_NE(var, nullptr);
            ASSERT_NE(idx, nullptr);
            ASSERT_NE(src, nullptr);
            ASSERT_NE(td, nullptr);
            if (atomic) {
                ASSERT_NE(workspace, nullptr);
            }
            memcpy(idx, index.data(), index.size() * sizeof(int64_t));
            memcpy(src, source.data(), source.size() * sizeof(float));
            memset(td, 0, sizeof(*td));
            td->includeSelf = includeSelf;
            td->amaxInputRows = e;
            td->amaxOutputRows = n;
            td->amaxColumns = c;
            td->amaxChunkLength = atomic ? plan.chunk : (key == 3100 ? n : 104);
            td->amaxTileLength = atomic ? plan.tile : (key == 3100 ? 1024 : 2048);
            td->amaxCompactRows = key == 3100;
            vector<float> first(initial.size());
            for (int repeat = 0; repeat < 3; ++repeat) {
                SCOPED_TRACE(repeat);
                memcpy(var, initial.data(), initial.size() * sizeof(float));
                if (atomic) {
                    memset(workspace, 0, workspaceBytes);
                }
                ICPU_SET_TILING_KEY(key);
                AscendC::SetKernelMode(KernelMode::AIV_MODE);
                ICPU_RUN_KF(scatter_elements_v2, atomic ? plan.cores : 1, reinterpret_cast<uint8_t*>(var),
                            reinterpret_cast<uint8_t*>(idx), reinterpret_cast<uint8_t*>(src),
                            reinterpret_cast<uint8_t*>(var), workspace, reinterpret_cast<uint8_t*>(td));
                for (size_t element = 0; element < expected.size(); ++element) {
                    uint32_t expectedBits = 0;
                    uint32_t actualBits = 0;
                    memcpy(&expectedBits, &expected[element], sizeof(expectedBits));
                    memcpy(&actualBits, &var[element], sizeof(actualBits));
                    EXPECT_EQ(actualBits, expectedBits) << "output index=" << element;
                }
                if (repeat == 0) {
                    memcpy(first.data(), var, first.size() * sizeof(float));
                } else {
                    EXPECT_EQ(memcmp(first.data(), var, first.size() * sizeof(float)), 0);
                }
            }
            AscendC::GmFree(var);
            AscendC::GmFree(idx);
            AscendC::GmFree(src);
            AscendC::GmFree(td);
            if (atomic) {
                AscendC::GmFree(workspace);
            }
        }
    }
}

TEST_F(scatter_elements_v2_test, legacy_max_min_nan)
{
    for (uint64_t mode : {4U, 5U}) {
        for (bool includeSelf : {false, true}) {
            for (bool split : {false, true}) {
                RunSumIncludeSelfCase<int32_t>(includeSelf, split, mode);
                RunSumIncludeSelfCase<int64_t>(includeSelf, split, mode);
            }
        }
    }
}
