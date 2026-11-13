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
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/arch22/scatter_elements_v2_tiling.h"

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
static void RunSumIncludeSelfCase(bool includeSelf, bool splitIndices)
{
    constexpr uint32_t rows = 2;
    constexpr uint32_t columns = 16;
    constexpr uint32_t updatesPerRow = 24;
    vector<float> initial(rows * columns);
    vector<Index> indices(rows * updatesPerRow);
    vector<float> updates(rows * updatesPerRow);
    vector<float> expected(rows * columns);
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
            expected[row * columns + column] += updates[row * updatesPerRow + update];
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
    tiling->mode = 2; // ADD

    ICPU_SET_TILING_KEY(sizeof(Index) == sizeof(int64_t) ? 120 : 110);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, static_cast<uint32_t>(tiling->usedCoreNum), reinterpret_cast<uint8_t*>(inputGm),
                reinterpret_cast<uint8_t*>(indicesGm), reinterpret_cast<uint8_t*>(updatesGm),
                reinterpret_cast<uint8_t*>(inputGm), nullptr, tilingGm);
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_FLOAT_EQ(inputGm[index], expected[index]) << "output index=" << index;
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
