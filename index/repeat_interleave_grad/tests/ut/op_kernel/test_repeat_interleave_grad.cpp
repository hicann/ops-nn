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
 * \file repeat_interleave_grad_proto.h
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_repeat_interleave_grad.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void repeat_interleave_grad(
    GM_ADDR input_grad, GM_ADDR repeats, GM_ADDR output_grad, GM_ADDR workspace, GM_ADDR tiling);

class repeat_interleave_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "repeat_interleave_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "repeat_interleave_grad_test TearDown\n" << endl;
    }
};

TEST_F(repeat_interleave_grad_test, test_case_0)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/repeat_interleave_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // batch, repeat_dim, data_num, axis, data_type
    system("cd ./gen_data/ && python3 gen_data.py 2 16 16 1 float16, int32");

    size_t inputByteSize = 2 * 16 * 2 * 16 * sizeof(half);
    size_t repeatsByteSize = 16 * sizeof(int32_t);
    size_t outputByteSize = 2 * 16 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(RepeatInterleaveGradTilingDataDef);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize + 32);
    uint8_t* repeats = (uint8_t*)AscendC::GmAlloc(repeatsByteSize + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);
    auto read_file_0 = ReadFile(path + "/gen_data/y_grad.bin", inputByteSize, x, inputByteSize);
    auto read_file_1 = ReadFile(path + "/gen_data/repeats.bin", repeatsByteSize, repeats, repeatsByteSize);
    cout << read_file_0 << " read_file_0 SetUp\n" << endl;
    cout << read_file_1 << " read_file_1 SetUp\n" << endl;
    RepeatInterleaveGradTilingDataDef* tilingDatafromBin = reinterpret_cast<RepeatInterleaveGradTilingDataDef*>(tiling);

    tilingDatafromBin->batch_dim_num = 2;
    tilingDatafromBin->repeats_i_grad_dim_num = 32;
    tilingDatafromBin->repeats_o_grad_dim_num = 16;
    tilingDatafromBin->data_dim_num = 16;

    tilingDatafromBin->core_num = 1;
    tilingDatafromBin->batch_dim_num_each_core = 1;
    tilingDatafromBin->batch_dim_num_last_core = 1;
    tilingDatafromBin->core_num_each_batch = 1;
    tilingDatafromBin->element_num_each_core = 64;
    tilingDatafromBin->element_num_last_core = 16;

    tilingDatafromBin->element_num_each_loop = 128;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(repeat_interleave_grad, blockDim, x, repeats, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(repeats);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(repeat_interleave_grad_test, test_case_1)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/repeat_interleave_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // batch, repeat_dim, data_num, axis, data_type
    system("cd ./gen_data/ && python3 gen_data.py 2 16 16 1 float16, int32");

    size_t inputByteSize = 2 * 16 * 2 * 16 * sizeof(half);
    size_t repeatsByteSize = 16 * sizeof(int32_t);
    size_t outputByteSize = 2 * 16 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(RepeatInterleaveGradTilingDataDef);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize + 32);
    uint8_t* repeats = (uint8_t*)AscendC::GmAlloc(repeatsByteSize + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);
    auto read_file_0 = ReadFile(path + "/gen_data/y_grad.bin", inputByteSize, x, inputByteSize);
    auto read_file_1 = ReadFile(path + "/gen_data/repeats.bin", repeatsByteSize, repeats, repeatsByteSize);
    cout << read_file_0 << " read_file_0 SetUp\n" << endl;
    cout << read_file_1 << " read_file_1 SetUp\n" << endl;
    RepeatInterleaveGradTilingDataDef* tilingDatafromBin = reinterpret_cast<RepeatInterleaveGradTilingDataDef*>(tiling);

    tilingDatafromBin->batch_dim_num = 2;
    tilingDatafromBin->repeats_i_grad_dim_num = 32;
    tilingDatafromBin->repeats_o_grad_dim_num = 16;
    tilingDatafromBin->data_dim_num = 16;

    tilingDatafromBin->core_num = 1;
    tilingDatafromBin->batch_dim_num_each_core = 1;
    tilingDatafromBin->batch_dim_num_last_core = 1;
    tilingDatafromBin->core_num_each_batch = 1;
    tilingDatafromBin->element_num_each_core = 64;
    tilingDatafromBin->element_num_last_core = 16;

    tilingDatafromBin->element_num_each_loop = 128;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(repeat_interleave_grad, blockDim, x, repeats, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(repeats);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(repeat_interleave_grad_test, test_case_2)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/repeat_interleave_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // batch, repeat_dim, data_num, axis, data_type
    system("cd ./gen_data/ && python3 gen_data.py 2 16 16 1 float32, int32");

    size_t inputByteSize = 2 * 16 * 2 * 16 * sizeof(float);
    size_t repeatsByteSize = 16 * sizeof(int32_t);
    size_t outputByteSize = 2 * 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(RepeatInterleaveGradTilingDataDef);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize + 32);
    uint8_t* repeats = (uint8_t*)AscendC::GmAlloc(repeatsByteSize + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);
    auto read_file_0 = ReadFile(path + "/gen_data/y_grad.bin", inputByteSize, x, inputByteSize);
    auto read_file_1 = ReadFile(path + "/gen_data/repeats.bin", repeatsByteSize, repeats, repeatsByteSize);
    cout << read_file_0 << " read_file_0 SetUp\n" << endl;
    cout << read_file_1 << " read_file_1 SetUp\n" << endl;
    RepeatInterleaveGradTilingDataDef* tilingDatafromBin = reinterpret_cast<RepeatInterleaveGradTilingDataDef*>(tiling);

    tilingDatafromBin->batch_dim_num = 2;
    tilingDatafromBin->repeats_i_grad_dim_num = 32;
    tilingDatafromBin->repeats_o_grad_dim_num = 16;
    tilingDatafromBin->data_dim_num = 16;

    tilingDatafromBin->core_num = 1;
    tilingDatafromBin->batch_dim_num_each_core = 1;
    tilingDatafromBin->batch_dim_num_last_core = 1;
    tilingDatafromBin->core_num_each_batch = 1;
    tilingDatafromBin->element_num_each_core = 64;
    tilingDatafromBin->element_num_last_core = 16;

    tilingDatafromBin->element_num_each_loop = 128;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(repeat_interleave_grad, blockDim, x, repeats, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(repeats);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(repeat_interleave_grad_test, test_case_10)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/repeat_interleave_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // batch, repeat_dim, data_num, axis, data_type
    system("cd ./gen_data/ && python3 gen_data.py 2 16 16 1 float16, int64");

    size_t inputByteSize = 2 * 16 * 2 * 16 * sizeof(half);
    size_t repeatsByteSize = 16 * sizeof(int64_t);
    size_t outputByteSize = 2 * 16 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(RepeatInterleaveGradTilingDataDef);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize + 32);
    uint8_t* repeats = (uint8_t*)AscendC::GmAlloc(repeatsByteSize + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);
    auto read_file_0 = ReadFile(path + "/gen_data/y_grad.bin", inputByteSize, x, inputByteSize);
    auto read_file_1 = ReadFile(path + "/gen_data/repeats.bin", repeatsByteSize, repeats, repeatsByteSize);
    cout << read_file_0 << " read_file_0 SetUp\n" << endl;
    cout << read_file_1 << " read_file_1 SetUp\n" << endl;
    RepeatInterleaveGradTilingDataDef* tilingDatafromBin = reinterpret_cast<RepeatInterleaveGradTilingDataDef*>(tiling);

    tilingDatafromBin->batch_dim_num = 2;
    tilingDatafromBin->repeats_i_grad_dim_num = 32;
    tilingDatafromBin->repeats_o_grad_dim_num = 16;
    tilingDatafromBin->data_dim_num = 16;

    tilingDatafromBin->core_num = 1;
    tilingDatafromBin->batch_dim_num_each_core = 1;
    tilingDatafromBin->batch_dim_num_last_core = 1;
    tilingDatafromBin->core_num_each_batch = 1;
    tilingDatafromBin->element_num_each_core = 64;
    tilingDatafromBin->element_num_last_core = 16;

    tilingDatafromBin->element_num_each_loop = 128;

    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(repeat_interleave_grad, blockDim, x, repeats, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(repeats);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(repeat_interleave_grad_test, test_case_11)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/repeat_interleave_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // batch, repeat_dim, data_num, axis, data_type
    system("cd ./gen_data/ && python3 gen_data.py 2 16 16 1 float16, int64");

    size_t inputByteSize = 2 * 16 * 2 * 16 * sizeof(half);
    size_t repeatsByteSize = 16 * sizeof(int64_t);
    size_t outputByteSize = 2 * 16 * 16 * sizeof(half);
    size_t tiling_data_size = sizeof(RepeatInterleaveGradTilingDataDef);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize + 32);
    uint8_t* repeats = (uint8_t*)AscendC::GmAlloc(repeatsByteSize + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);
    auto read_file_0 = ReadFile(path + "/gen_data/y_grad.bin", inputByteSize, x, inputByteSize);
    auto read_file_1 = ReadFile(path + "/gen_data/repeats.bin", repeatsByteSize, repeats, repeatsByteSize);
    cout << read_file_0 << " read_file_0 SetUp\n" << endl;
    cout << read_file_1 << " read_file_1 SetUp\n" << endl;
    RepeatInterleaveGradTilingDataDef* tilingDatafromBin = reinterpret_cast<RepeatInterleaveGradTilingDataDef*>(tiling);

    tilingDatafromBin->batch_dim_num = 2;
    tilingDatafromBin->repeats_i_grad_dim_num = 32;
    tilingDatafromBin->repeats_o_grad_dim_num = 16;
    tilingDatafromBin->data_dim_num = 16;

    tilingDatafromBin->core_num = 1;
    tilingDatafromBin->batch_dim_num_each_core = 1;
    tilingDatafromBin->batch_dim_num_last_core = 1;
    tilingDatafromBin->core_num_each_batch = 1;
    tilingDatafromBin->element_num_each_core = 64;
    tilingDatafromBin->element_num_last_core = 16;

    tilingDatafromBin->element_num_each_loop = 128;

    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(repeat_interleave_grad, blockDim, x, repeats, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(repeats);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(repeat_interleave_grad_test, test_case_12)
{
    system(
        "cp -rf "
        "../../../../../../../ops/built-in/tests/ut/fast_op_test/repeat_interleave_grad/gen_data "
        "./");
    system("chmod -R 755 ./gen_data/");
    // batch, repeat_dim, data_num, axis, data_type
    system("cd ./gen_data/ && python3 gen_data.py 2 16 16 1 float, int64");

    size_t inputByteSize = 2 * 16 * 2 * 16 * sizeof(float);
    size_t repeatsByteSize = 16 * sizeof(int64_t);
    size_t outputByteSize = 2 * 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(RepeatInterleaveGradTilingDataDef);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize + 32);
    uint8_t* repeats = (uint8_t*)AscendC::GmAlloc(repeatsByteSize + 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize + 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);
    auto read_file_0 = ReadFile(path + "/gen_data/y_grad.bin", inputByteSize, x, inputByteSize);
    auto read_file_1 = ReadFile(path + "/gen_data/repeats.bin", repeatsByteSize, repeats, repeatsByteSize);
    cout << read_file_0 << " read_file_0 SetUp\n" << endl;
    cout << read_file_1 << " read_file_1 SetUp\n" << endl;
    RepeatInterleaveGradTilingDataDef* tilingDatafromBin = reinterpret_cast<RepeatInterleaveGradTilingDataDef*>(tiling);

    tilingDatafromBin->batch_dim_num = 2;
    tilingDatafromBin->repeats_i_grad_dim_num = 32;
    tilingDatafromBin->repeats_o_grad_dim_num = 16;
    tilingDatafromBin->data_dim_num = 16;

    tilingDatafromBin->core_num = 1;
    tilingDatafromBin->batch_dim_num_each_core = 1;
    tilingDatafromBin->batch_dim_num_last_core = 1;
    tilingDatafromBin->core_num_each_batch = 1;
    tilingDatafromBin->element_num_each_core = 64;
    tilingDatafromBin->element_num_last_core = 16;

    tilingDatafromBin->element_num_each_loop = 128;

    ICPU_SET_TILING_KEY(12);
    ICPU_RUN_KF(repeat_interleave_grad, blockDim, x, repeats, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(repeats);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
