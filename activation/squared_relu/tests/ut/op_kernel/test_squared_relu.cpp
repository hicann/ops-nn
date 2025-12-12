/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "squared_relu_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void squared_relu(GM_ADDR input,
                                            GM_ADDR output,
                                            GM_ADDR workspace,
                                            GM_ADDR tiling);

class squared_relu_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "squared_relu SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "squared_relu TearDown\n" << endl;
  }
};

TEST_F(squared_relu_test, test_squared_relu_float_0)
{
    system(
        "cp -rf "
        "../../../../activation/squared_relu/tests/ut/op_kernel/squared_relu_data ./");
    system("chmod -R 755 ./squared_relu_data/");
    system("cd ./squared_relu_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'float32'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(float);

    size_t yFileSize = M * N * sizeof(float);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 2;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(SquaredReluTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    std::string fileName = "./squared_relu_data/float32_input_squared_relu.bin";

    ReadFile(fileName, xFileSize, x, xFileSize);

    SquaredReluTilingData* tilingDatafromBin = reinterpret_cast<SquaredReluTilingData*>(tiling);
    tilingDatafromBin->elementNum = 8;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(squared_relu, blockDim, x, y, workspace, (uint8_t*)tilingDatafromBin);
    fileName = "./squared_relu_data/float32_output_squared_relu.bin";
    WriteFile(fileName, y, yFileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./squared_relu_data/ && python3 compare_data.py 'float32'");
}

TEST_F(squared_relu_test, test_squared_relu_float16_1)
{
    system(
        "cp -rf "
        "../../../../activation/squared_relu/tests/ut/op_kernel/squared_relu_data ./");
    system("chmod -R 755 ./squared_relu_data/");
    system("cd ./squared_relu_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'float16'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(SquaredReluTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    std::string fileName = "./squared_relu_data/float16_input_squared_relu.bin";

    ReadFile(fileName, xFileSize, x, xFileSize);

    SquaredReluTilingData* tilingDatafromBin = reinterpret_cast<SquaredReluTilingData*>(tiling);
    tilingDatafromBin->elementNum = M * N;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(squared_relu, blockDim, x, y, workspace, (uint8_t*)tilingDatafromBin);
    fileName = "./squared_relu_data/float16_output_squared_relu.bin";
    WriteFile(fileName, y, yFileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./squared_relu_data/ && python3 compare_data.py 'float16'");
}

TEST_F(squared_relu_test, test_squared_relu_bfloat16_2)
{
    system(
        "cp -rf "
        "../../../../activation/squared_relu/tests/ut/op_kernel/squared_relu_data ./");
    system("chmod -R 755 ./squared_relu_data/");
    system("cd ./squared_relu_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'bfloat16_t'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(bfloat16_t);

    size_t yFileSize = M * N * sizeof(bfloat16_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 3;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(SquaredReluTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    std::string fileName = "./squared_relu_data/bfloat16_t_input_squared_relu.bin";

    ReadFile(fileName, xFileSize, x, xFileSize);

    SquaredReluTilingData* tilingDatafromBin = reinterpret_cast<SquaredReluTilingData*>(tiling);
    tilingDatafromBin->elementNum = M * N;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(squared_relu, blockDim, x, y, workspace, (uint8_t*)tilingDatafromBin);
    fileName = "./squared_relu_data/bfloat16_t_output_squared_relu.bin";
    WriteFile(fileName, y, yFileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./squared_relu_data/ && python3 compare_data.py 'bfloat16_t'");
}