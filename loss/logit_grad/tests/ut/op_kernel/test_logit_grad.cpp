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
#include "test_logit_grad_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void logit_grad(GM_ADDR x, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling);
class logit_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "logit_grad SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "logit_grad TearDown\n" << endl;
    }
};

TEST_F(logit_grad_test, test_logit_grad_float_0)
{
    system(
        "cp -rf "
        "../../../../loss/logit_grad/tests/ut/op_kernel/logit_grad_data ./");
    system("chmod -R 755 ./logit_grad_data/");
    system("cd ./logit_grad_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'float32'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(float);

    size_t dyFileSize = M * N * sizeof(float);

    size_t dxFileSize = M * N * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyFileSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxFileSize);

    uint64_t tilingKey = 2;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(LogitGradTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string xFileName = "./logit_grad_data/float32_input_logit_grad_x.bin";
    std::string dyFileName = "./logit_grad_data/float32_input_logit_grad_dy.bin";

    ReadFile(xFileName, xFileSize, x, xFileSize);
    ReadFile(dyFileName, dyFileSize, dy, dyFileSize);

    LogitGradTilingDataTest* tilingDatafromBin = reinterpret_cast<LogitGradTilingDataTest*>(tiling);
    tilingDatafromBin->elementNum = 8;
    tilingDatafromBin->eps = -1.0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(logit_grad, blockDim, x, dy, dx, workspace, (uint8_t*)tilingDatafromBin);
    std::string dxFileName = "./logit_grad_data/float32_output_logit_grad.bin";
    WriteFile(dxFileName, dx, dxFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)dy);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./logit_grad_data/ && python3 compare_data.py 'float32'");
}

TEST_F(logit_grad_test, test_logit_grad_float16_1)
{
    system(
        "cp -rf "
        "../../../../loss/logit_grad/tests/ut/op_kernel/logit_grad_data ./");
    system("chmod -R 755 ./logit_grad_data/");
    system("cd ./logit_grad_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'float16'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(half);

    size_t dyFileSize = M * N * sizeof(half);

    size_t dxFileSize = M * N * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyFileSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxFileSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(LogitGradTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string xFileName = "./logit_grad_data/float16_input_logit_grad_x.bin";
    std::string dyFileName = "./logit_grad_data/float16_input_logit_grad_dy.bin";

    ReadFile(xFileName, xFileSize, x, xFileSize);
    ReadFile(dyFileName, dyFileSize, dy, dyFileSize);

    LogitGradTilingDataTest* tilingDatafromBin = reinterpret_cast<LogitGradTilingDataTest*>(tiling);
    tilingDatafromBin->elementNum = 8;
    tilingDatafromBin->eps = -1.0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(logit_grad, blockDim, x, dy, dx, workspace, (uint8_t*)tilingDatafromBin);
    std::string dxFileName = "./logit_grad_data/float16_output_logit_grad.bin";
    WriteFile(dxFileName, dx, dxFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)dy);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./logit_grad_data/ && python3 compare_data.py 'float16'");
}

TEST_F(logit_grad_test, test_logit_grad_bfloat16_2)
{
    system(
        "cp -rf "
        "../../../../loss/logit_grad/tests/ut/op_kernel/logit_grad_data ./");
    system("chmod -R 755 ./logit_grad_data/");
    system("cd ./logit_grad_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'bfloat16_t'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(bfloat16_t);

    size_t dyFileSize = M * N * sizeof(bfloat16_t);

    size_t dxFileSize = M * N * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyFileSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxFileSize);

    uint64_t tilingKey = 3;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(LogitGradTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string xFileName = "./logit_grad_data/bfloat16_t_input_logit_grad_x.bin";
    std::string dyFileName = "./logit_grad_data/bfloat16_t_input_logit_grad_dy.bin";

    ReadFile(xFileName, xFileSize, x, xFileSize);
    ReadFile(dyFileName, dyFileSize, dy, dyFileSize);

    LogitGradTilingDataTest* tilingDatafromBin = reinterpret_cast<LogitGradTilingDataTest*>(tiling);
    tilingDatafromBin->elementNum = 8;
    tilingDatafromBin->eps = -1.0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(logit_grad, blockDim, x, dy, dx, workspace, (uint8_t*)tilingDatafromBin);
    std::string dxFileName = "./logit_grad_data/bfloat16_t_output_logit_grad.bin";
    WriteFile(dxFileName, dx, dxFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)dy);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./logit_grad_data/ && python3 compare_data.py 'bfloat16_t'");
}