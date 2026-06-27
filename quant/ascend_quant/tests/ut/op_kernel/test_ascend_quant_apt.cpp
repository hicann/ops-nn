/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_ascend_quant.cpp
 * \brief ascend_quant kernel unit test
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/ascend_quant_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void ascend_quant(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class ascend_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ascend_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ascend_quant_test TearDown\n" << endl;
    }
};

TEST_F(ascend_quant_test, test_case_fp16_to_int8)
{
    size_t inputXSize = 128 * 256 * sizeof(half);
    size_t outputYSize = 128 * 256 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(AscendQuantTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    AscendQuantTilingData* tilingData = reinterpret_cast<AscendQuantTilingData*>(tiling);

    tilingData->numCore = 2;
    tilingData->dim0 = 128 * 256;
    tilingData->blockFactor = 128 * 256 / 2;
    tilingData->blockTailFactor = 128 * 256 / 2;
    tilingData->baseLen = 256;
    tilingData->roundMode = 0;
    tilingData->scale = 1.0f;
    tilingData->offset = 0.0f;

    ICPU_SET_TILING_KEY(0);

    auto ascend_quant_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::ascend_quant<TPL_ROUND_MODE_ROUND, 0, TPL_OFFSET_ZERO>(x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(ascend_quant_kernel, blockDim, x, y, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingData);
    free(path_);
}

TEST_F(ascend_quant_test, test_case_fp32_to_int8)
{
    size_t inputXSize = 128 * 256 * sizeof(float);
    size_t outputYSize = 128 * 256 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(AscendQuantTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    AscendQuantTilingData* tilingData = reinterpret_cast<AscendQuantTilingData*>(tiling);

    tilingData->numCore = 2;
    tilingData->dim0 = 128 * 256;
    tilingData->blockFactor = 128 * 256 / 2;
    tilingData->blockTailFactor = 128 * 256 / 2;
    tilingData->baseLen = 256;
    tilingData->roundMode = 0;
    tilingData->scale = 0.5f;
    tilingData->offset = 128.0f;

    ICPU_SET_TILING_KEY(0);

    auto ascend_quant_kernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::ascend_quant<TPL_ROUND_MODE_ROUND, 0, TPL_OFFSET_NON_ZERO>(x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(ascend_quant_kernel, blockDim, x, y, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingData);
    free(path_);
}