/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_logit_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void logit(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
class logit_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "logit SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "logit TearDown\n" << endl;
    }
};

TEST_F(logit_test, test_logit_float_0)
{
    system(
        "cp -rf "
        "../../../../loss/logit/tests/ut/op_kernel/logit_data ./ ");
    system("chmod -R 755 ./logit_data/");
    system("cd ./logit_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'float32'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(float);

    size_t yFileSize = M * N * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 2;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(LogitTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName = "./logit_data/float32_input_logit.bin";

    ReadFile(fileName, xFileSize, x, xFileSize);

    LogitTilingDataTest* tilingDatafromBin = reinterpret_cast<LogitTilingDataTest*>(tiling);
    tilingDatafromBin->elementNum = 8;
    tilingDatafromBin->eps = -1.0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(logit, blockDim, x, y, workspace, (uint8_t*)tilingDatafromBin);
    fileName = "./logit_data/float32_output_logit.bin";
    WriteFile(fileName, y, yFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./logit_data/ && python3 compare_data.py 'float32'");
}

TEST_F(logit_test, test_logit_float16_1)
{
    system(
        "cp -rf "
        "../../../../loss/logit/tests/ut/op_kernel/logit_data ./");
    system("chmod -R 755 ./logit_data/");
    system("cd ./logit_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'float16'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(LogitTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName = "./logit_data/float16_input_logit.bin";

    ReadFile(fileName, xFileSize, x, xFileSize);

    LogitTilingDataTest* tilingDatafromBin = reinterpret_cast<LogitTilingDataTest*>(tiling);
    tilingDatafromBin->elementNum = M * N;
    tilingDatafromBin->eps = -1.0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(logit, blockDim, x, y, workspace, (uint8_t*)tilingDatafromBin);
    fileName = "./logit_data/float16_output_logit.bin";
    WriteFile(fileName, y, yFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./logit_data/ && python3 compare_data.py 'float16'");
}

TEST_F(logit_test, test_logit_bfloat16_2)
{
    system(
        "cp -rf "
        "../../../../loss/logit/tests/ut/op_kernel/logit_data ./");
    system("chmod -R 755 ./logit_data/");
    system("cd ./logit_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'bfloat16_t'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(bfloat16_t);

    size_t yFileSize = M * N * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 3;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(LogitTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName = "./logit_data/bfloat16_t_input_logit.bin";

    ReadFile(fileName, xFileSize, x, xFileSize);

    LogitTilingDataTest* tilingDatafromBin = reinterpret_cast<LogitTilingDataTest*>(tiling);
    tilingDatafromBin->elementNum = M * N;
    tilingDatafromBin->eps = -1.0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(logit, blockDim, x, y, workspace, (uint8_t*)tilingDatafromBin);
    fileName = "./logit_data/bfloat16_t_output_logit.bin";
    WriteFile(fileName, y, yFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./logit_data/ && python3 compare_data.py 'bfloat16_t'");
}