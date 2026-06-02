/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_kernel/data_format_dim_map_apt.cpp"

using namespace std;

class DataFormatDimMapKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "DataFormatDimMapKernelTest SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "DataFormatDimMapKernelTest TearDown\n" << endl; }
};

static constexpr int64_t kMinComparesElems = 256;  // Ascend950 Compares mask alignment

// Helper: fill TilingData for NHWC→NCHW (formatLen=4, table=[0,2,3,1,...])
static void FillTilingNh2Nc(int64_t totalNum, DataFormatDimMapTilingData* td,
                             int64_t blockFactor = 256, int64_t ubFactor = 256) {
    memset(td, 0, sizeof(DataFormatDimMapTilingData));
    td->totalNum = totalNum;
    td->blockFactor = blockFactor;
    td->ubFactor = ubFactor;
    td->formatLen = 4;
    td->expandedTable[0] = 0; td->expandedTable[1] = 2;
    td->expandedTable[2] = 3; td->expandedTable[3] = 1;
    td->expandedTable[4] = 0; td->expandedTable[5] = 2;
    td->expandedTable[6] = 3; td->expandedTable[7] = 1;
}

TEST_F(DataFormatDimMapKernelTest, nhwc_to_nchw_int32_256elem)
{
    int64_t totalNum = kMinComparesElems;
    size_t elemSize = sizeof(int32_t);
    size_t inputByteSize = totalNum * elemSize;
    size_t outputByteSize = totalNum * elemSize;
    size_t workspaceSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(DataFormatDimMapTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    // Fill repeating [0,1,2,3] pattern
    auto* xData = reinterpret_cast<int32_t*>(x);
    for (int64_t i = 0; i < totalNum; i++) {
        xData[i] = static_cast<int32_t>(i % 4);
    }

    auto* td = reinterpret_cast<DataFormatDimMapTilingData*>(tiling);
    FillTilingNh2Nc(totalNum, td);

    uint64_t tilingKey = 0;
    uint32_t blockDim = 1;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto KernelFn = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::data_format_dim_map<int32_t>(x, y, workspace, tiling);
    };
    ICPU_RUN_KF(KernelFn, blockDim, x, y, workspace, tiling);

    // NHWC→NCHW: [0,1,2,3] → [0,2,3,1]
    int32_t expected[4] = {0, 2, 3, 1};
    auto* output = reinterpret_cast<int32_t*>(y);
    for (int64_t i = 0; i < totalNum; i++) {
        ASSERT_EQ(output[i], expected[i % 4]) << "mismatch at index " << i;
    }

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(DataFormatDimMapKernelTest, nhwc_to_nchw_int64_256elem)
{
    int64_t totalNum = kMinComparesElems;
    size_t elemSize = sizeof(int64_t);
    size_t inputByteSize = totalNum * elemSize;
    size_t outputByteSize = totalNum * elemSize;
    size_t workspaceSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(DataFormatDimMapTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    auto* xData = reinterpret_cast<int64_t*>(x);
    for (int64_t i = 0; i < totalNum; i++) {
        xData[i] = static_cast<int64_t>(i % 4);
    }

    auto* td = reinterpret_cast<DataFormatDimMapTilingData*>(tiling);
    FillTilingNh2Nc(totalNum, td);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto KernelFn = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::data_format_dim_map<int64_t>(x, y, workspace, tiling);
    };
    ICPU_RUN_KF(KernelFn, blockDim, x, y, workspace, tiling);

    int64_t expected[4] = {0, 2, 3, 1};
    auto* output = reinterpret_cast<int64_t*>(y);
    for (int64_t i = 0; i < totalNum; i++) {
        ASSERT_EQ(output[i], expected[i % 4]) << "mismatch at index " << i;
    }

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(DataFormatDimMapKernelTest, nhwc_to_nchw_int32_negative_256elem)
{
    int64_t totalNum = kMinComparesElems;
    size_t inputByteSize = totalNum * sizeof(int32_t);
    size_t outputByteSize = totalNum * sizeof(int32_t);
    size_t workspaceSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(DataFormatDimMapTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    // Fill repeating [-1,-2,-3,-4], wrap to [3,2,1,0] → NHWC→NCHW map: [1,3,2,0]
    auto* xData = reinterpret_cast<int32_t*>(x);
    for (int64_t i = 0; i < totalNum; i++) {
        xData[i] = static_cast<int32_t>(-(i % 4) - 1);
    }

    auto* td = reinterpret_cast<DataFormatDimMapTilingData*>(tiling);
    FillTilingNh2Nc(totalNum, td);

    uint64_t tilingKey = 0;
    uint32_t blockDim = 1;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto KernelFn = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::data_format_dim_map<int32_t>(x, y, workspace, tiling);
    };
    ICPU_RUN_KF(KernelFn, blockDim, x, y, workspace, tiling);

    // [-1,-2,-3,-4] → wrap [3,2,1,0] → NHWC→NCHW map → [C→1, W→3, H→2, N→0] = [1,3,2,0]
    int32_t expected[4] = {1, 3, 2, 0};
    auto* output = reinterpret_cast<int32_t*>(y);
    for (int64_t i = 0; i < totalNum; i++) {
        ASSERT_EQ(output[i], expected[i % 4]) << "mismatch at index " << i;
    }

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(DataFormatDimMapKernelTest, nhwc_to_nchw_int32_multicore)
{
    int64_t totalNum = 1024;
    size_t inputByteSize = totalNum * sizeof(int32_t);
    size_t outputByteSize = totalNum * sizeof(int32_t);
    size_t workspaceSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(DataFormatDimMapTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    // All indices = 0 (N → 0), split across 2 "cores"
    memset(x, 0, inputByteSize);
    memset(y, 0xFF, outputByteSize);

    auto* td = reinterpret_cast<DataFormatDimMapTilingData*>(tiling);
    FillTilingNh2Nc(totalNum, td, /*blockFactor=*/512, /*ubFactor=*/256);

    uint64_t tilingKey = 0;
    uint32_t blockDim = 2;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto KernelFn = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::data_format_dim_map<int32_t>(x, y, workspace, tiling);
    };
    ICPU_RUN_KF(KernelFn, blockDim, x, y, workspace, tiling);

    auto* output = reinterpret_cast<int32_t*>(y);
    for (int64_t i = 0; i < totalNum; i++) {
        ASSERT_EQ(output[i], 0) << "mismatch at index " << i;
    }

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}
