/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

TEST_F(scatter_elements_v2_test, test_case_bucket_bf16_int32)
{
    constexpr size_t rows = 2;
    constexpr size_t varN = 196608;
    constexpr size_t indicesN = 96;
    constexpr size_t tileLen = 65536;
    constexpr size_t numTiles = 3;
    constexpr size_t fifoDepth = 64;
    // 与 host 侧 BUCKET_PAD_PER_TILE_HOST 一致：kernel 每桶预留 ceil(cnt,16)*16 + 16 <= cnt + 31，
    // 故每核桶区容量须为 indicesN + 32 * numTiles；用旧式 16 * numTiles 会越界
    constexpr size_t bucketStride = indicesN + 32 * numTiles;
    constexpr size_t systemWorkspaceSize = 16 * 1024 * 1024;
    constexpr size_t bucketWorkspaceSize = rows * bucketStride * (sizeof(int32_t) + sizeof(uint16_t));
    constexpr size_t dataSize = rows * varN;
    constexpr size_t updatesSize = rows * indicesN;

    uint8_t* var = static_cast<uint8_t*>(AscendC::GmAlloc(dataSize * sizeof(uint16_t)));
    uint8_t* indices = static_cast<uint8_t*>(AscendC::GmAlloc(updatesSize * sizeof(int32_t)));
    uint8_t* updates = static_cast<uint8_t*>(AscendC::GmAlloc(updatesSize * sizeof(uint16_t)));
    uint8_t* output = static_cast<uint8_t*>(AscendC::GmAlloc(dataSize * sizeof(uint16_t)));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(systemWorkspaceSize + bucketWorkspaceSize));
    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(ScatterElementsV2TilingData)));

    auto* varData = reinterpret_cast<uint16_t*>(var);
    auto* indicesData = reinterpret_cast<int32_t*>(indices);
    auto* updatesData = reinterpret_cast<uint16_t*>(updates);
    vector<uint16_t> expected(dataSize);
    for (size_t i = 0; i < dataSize; ++i) {
        varData[i] = static_cast<uint16_t>(0x3F80U + i % 32);
        expected[i] = varData[i];
    }
    // 更新点故意分布在第 0 桶与第 2 桶，覆盖"跨桶 + 同桶内重复下标(末次写赢)"两种情形
    for (size_t r = 0; r < rows; ++r) {
        for (size_t i = 0; i < indicesN; ++i) {
            size_t offset = i < 70 ? i % 24 : 2 * tileLen + (i - 70) % 13;
            size_t updateOffset = r * indicesN + i;
            indicesData[updateOffset] = static_cast<int32_t>(offset);
            updatesData[updateOffset] = static_cast<uint16_t>(0x4000U + updateOffset);
            expected[r * varN + offset] = updatesData[updateOffset];
        }
    }

    memset(tiling, 0, sizeof(ScatterElementsV2TilingData));
    auto* tilingData = reinterpret_cast<ScatterElementsV2TilingData*>(tiling);
    tilingData->usedCoreNum = rows;
    tilingData->bktMode = 1;
    tilingData->bktRows = rows;
    tilingData->bktVarN = varN;
    tilingData->bktIndicesN = indicesN;
    tilingData->bktTileLen = tileLen;
    tilingData->bktNumTiles = numTiles;
    tilingData->bktShift = 16;
    tilingData->bktFifoDepth = fifoDepth;
    tilingData->bktStride = bucketStride;
    tilingData->bktRowsPerCore = 1;
    tilingData->bktFrontCore = 0;

    ICPU_SET_TILING_KEY(610);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, rows, var, indices, updates, output, workspace, tiling);

    size_t mismatch = dataSize;
    uint16_t actualValue = 0;
    uint16_t expectedValue = 0;
    for (size_t i = 0; i < dataSize; ++i) {
        if (varData[i] != expected[i]) {
            mismatch = i;
            actualValue = varData[i];
            expectedValue = expected[i];
            break;
        }
    }

    AscendC::GmFree(var);
    AscendC::GmFree(indices);
    AscendC::GmFree(updates);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    EXPECT_EQ(mismatch, dataSize) << "first mismatch at offset " << mismatch << ", actual raw bf16=" << actualValue
                                  << ", expected raw bf16=" << expectedValue;
}
