/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cross_v2.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "test_cross_v2_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

using namespace std;
class cross_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "cross_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "cross_v2_test TearDown\n" << endl;
    }
};

extern "C" __global__ __aicore__ void cross_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

static bool CompareData()
{
    std::string cmd = "cd ./cross_v2_data/ && python3 verify.py y.bin golden.bin ";
    return system(cmd.c_str()) == 0;
}

static void InitEnv(std::string caseName, std::string dtype)
{
    std::string gen = "cd ./cross_v2_data/ && python3 gen_data.py " + caseName + " " + dtype;
    system("cp -r ../../../../loss/cross_v2/tests/ut/op_kernel/cross_v2_data ./");
    system("chmod -R 755 ./cross_v2_data/");
    system("cd ./cross_v2_data/ && rm -rf ./*bin");
    system(gen.c_str());
    std::string genTi = "cd ./cross_v2_data/ && python3 gen_tiling.py " + caseName;
    system(genTi.c_str());
}

TEST_F(cross_v2_test, test_case_float_0)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 40;
    string caseName = "test_case_float_0";
    string dtype = "float";
    std::vector<int64_t> input_x1{320, 3};
    std::vector<int64_t> intput_x2{320, 3};
    std::vector<int64_t> output_y{320, 3};

    size_t x1Size = ShapeSize(input_x1) * sizeof(float);
    size_t x2Size = ShapeSize(intput_x2) * sizeof(float);
    size_t ySize = ShapeSize(output_y) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingFileSize = 5 * sizeof(int64_t);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1Size);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2Size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingFileSize);
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    ReadFile(path + "/cross_v2_data/x1.bin", x1Size, x1, x1Size);
    ReadFile(path + "/cross_v2_data/x2.bin", x2Size, x2, x2Size);
    ReadFile(path + "/cross_v2_data/tiling.bin", tilingFileSize, tiling, tilingFileSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(cross_v2, blockDim, x1, x2, y, workspace, tiling);
    WriteFile(path + "/cross_v2_data/y.bin", y, ySize);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    EXPECT_EQ(CompareData(), true);
    free(cpath);
}

TEST_F(cross_v2_test, test_case_float_1)
{
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    string caseName = "test_case_float_1";
    string dtype = "float";
    std::vector<int64_t> input_x1{3, 320};
    std::vector<int64_t> intput_x2{3, 320};
    std::vector<int64_t> output_y{3, 320};

    size_t x1Size = ShapeSize(input_x1) * sizeof(float);
    size_t x2Size = ShapeSize(intput_x2) * sizeof(float);
    size_t ySize = ShapeSize(output_y) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingFileSize = 5 * sizeof(int64_t);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1Size);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2Size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingFileSize);
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    ReadFile(path + "/cross_v2_data/x1.bin", x1Size, x1, x1Size);
    ReadFile(path + "/cross_v2_data/x2.bin", x2Size, x2, x2Size);
    ReadFile(path + "/cross_v2_data/tiling.bin", tilingFileSize, tiling, tilingFileSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(cross_v2, blockDim, x1, x2, y, workspace, tiling);
    WriteFile(path + "/cross_v2_data/y.bin", y, ySize);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    EXPECT_EQ(CompareData(), true);
    free(cpath);
}