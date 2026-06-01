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
 * \file test_gather_elements_v2.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <gtest/gtest.h>

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

extern "C" __global__ __aicore__ void gather_elements_v2(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling);

class gather_elements_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "gather_elements_v2_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "gather_elements_v2_test TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./data");
    }
};

TEST_F(gather_elements_v2_test, test_case_8_4096_1248_fp32)
{
    size_t xSize = 4096 * 1248 * sizeof(float);
    size_t indexSize = 4096 * 1248 * sizeof(int32_t);
    size_t ySize = 4096 * 1248 * sizeof(float);
    size_t tilingSize = sizeof(GatherElementsV2TilingData);
    size_t workspaceSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    kernel_ut::SetupTestEnvironment("index/gather_elements_v2/tests/ut/op_kernel/data", "data");
    ;
    kernel_ut::RunGenData("./data", {"1", "4096", "1248", "4096"});
    kernel_ut::RunGenTiling("./data", {"test_case_8_4096_1248_fp16"});

    string path = kernel_ut::GetTestWorkDir();
    ReadFile(path + "/data/x.bin", xSize, x, xSize);
    ReadFile(path + "/data/index.bin", indexSize, index, indexSize);
    ReadFile(path + "/data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(gather_elements_v2, 48, x, index, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(index);
    AscendC::GmFree(y);
    AscendC::GmFree(tiling);
}