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
#include "heaviside_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void heaviside(
    GM_ADDR input, GM_ADDR values, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
class heaviside_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "heaviside SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "heaviside TearDown\n" << endl;
    }
};

TEST_F(heaviside_test, test_heaviside_float_0)
{
#undef DTYPE_INPUT
#define DTYPE_INPUT DT_FP32

    system(
        "cp -rf "
        "../../../../activation/heaviside/tests/ut/op_kernel/heaviside_data ./ ");
    system("chmod -R 755 ./heaviside_data/");
    system("cd ./heaviside_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'float32'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(float);

    size_t yFileSize = M * N * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* values = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(HeavisideTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName = "./heaviside_data/float32_input_heaviside.bin";
    ReadFile(fileName, xFileSize, x, xFileSize);
    fileName = "./heaviside_data/float32_values_heaviside.bin";
    ReadFile(fileName, xFileSize, values, xFileSize);

    HeavisideTilingData* tilingDatafromBin = reinterpret_cast<HeavisideTilingData*>(tiling);
    tilingDatafromBin->elementNum = 8;
    tilingDatafromBin->valuesType = false;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(heaviside, blockDim, x, values, y, workspace, (uint8_t*)tilingDatafromBin);
    fileName = "./heaviside_data/float32_output_heaviside.bin";
    WriteFile(fileName, y, yFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)values);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./heaviside_data/ && python3 compare_data.py 'float32'");
}

TEST_F(heaviside_test, test_heaviside_float16_1)
{
#undef DTYPE_INPUT
#define DTYPE_INPUT DT_FP16

    system(
        "cp -rf "
        "../../../../activation/heaviside/tests/ut/op_kernel/heaviside_data ./");
    system("chmod -R 755 ./heaviside_data/");
    system("cd ./heaviside_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'float16'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(half);

    size_t yFileSize = M * N * sizeof(half);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* values = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(HeavisideTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName = "./heaviside_data/float16_input_heaviside.bin";
    ReadFile(fileName, xFileSize, x, xFileSize);
    fileName = "./heaviside_data/float16_values_heaviside.bin";
    ReadFile(fileName, xFileSize, values, xFileSize);

    HeavisideTilingData* tilingDatafromBin = reinterpret_cast<HeavisideTilingData*>(tiling);
    tilingDatafromBin->elementNum = M * N;
    tilingDatafromBin->valuesType = false;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(heaviside, blockDim, x, values, y, workspace, (uint8_t*)tilingDatafromBin);
    fileName = "./heaviside_data/float16_output_heaviside.bin";
    WriteFile(fileName, y, yFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)values);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./heaviside_data/ && python3 compare_data.py 'float16'");
}

TEST_F(heaviside_test, test_heaviside_bfloat16_2)
{
#undef DTYPE_INPUT
#define DTYPE_INPUT DT_BF16

    system(
        "cp -rf "
        "../../../../activation/heaviside/tests/ut/op_kernel/heaviside_data ./");
    system("chmod -R 755 ./heaviside_data/");
    system("cd ./heaviside_data/ && python3 gen_data.py '(2, 4)' '(2, 4)' 'bfloat16_t'");
    size_t M = 2;
    size_t N = 4;

    size_t xFileSize = M * N * sizeof(bfloat16_t);

    size_t yFileSize = M * N * sizeof(bfloat16_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* values = (uint8_t*)AscendC::GmAlloc(xFileSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(HeavisideTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName = "./heaviside_data/bfloat16_t_input_heaviside.bin";
    ReadFile(fileName, xFileSize, x, xFileSize);
    fileName = "./heaviside_data/bfloat16_t_values_heaviside.bin";
    ReadFile(fileName, xFileSize, values, xFileSize);

    HeavisideTilingData* tilingDatafromBin = reinterpret_cast<HeavisideTilingData*>(tiling);
    tilingDatafromBin->elementNum = M * N;
    tilingDatafromBin->valuesType = false;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(heaviside, blockDim, x, values, y, workspace, (uint8_t*)tilingDatafromBin);
    fileName = "./heaviside_data/bfloat16_t_output_heaviside.bin";
    WriteFile(fileName, y, yFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)values);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    system("cd ./heaviside_data/ && python3 compare_data.py 'bfloat16_t'");
}