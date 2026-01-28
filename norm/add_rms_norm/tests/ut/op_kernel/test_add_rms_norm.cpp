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
#include "add_rms_norm_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void add_rms_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling);

class add_rms_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "add_rms_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_rms_norm_test TearDown\n" << endl;
    }
};

TEST_F(add_rms_norm_test, test_case_10)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, test_case_20)
{
    size_t inputByteSize = 2 * 256 * sizeof(float);
    size_t gammaByteSize = 256 * sizeof(float);
    size_t outputByteSize = 2 * 256 * sizeof(float);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 10240;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(20);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, test_case_30)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, test_case_11)
{
    size_t inputByteSize = 2 * 25600 * sizeof(int16_t);
    size_t gammaByteSize = 25600 * sizeof(int16_t);
    size_t outputByteSize = 2 * 25600 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 25600;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 8544;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(11);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, test_case_21)
{
    size_t inputByteSize = 2 * 25600 * sizeof(float);
    size_t gammaByteSize = 25600 * sizeof(float);
    size_t outputByteSize = 2 * 25600 * sizeof(float);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 25600;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 8544;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(21);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// TEST_F(add_rms_norm_test, test_case_31)
// {
//     size_t inputByteSize = 2 * 25600 * sizeof(int16_t);
//     size_t gammaByteSize = 25600 * sizeof(int16_t);
//     size_t outputByteSize = 2 * 25600 * sizeof(int16_t);
//     size_t rstdByteSize = 2 * sizeof(float);
//     size_t tiling_data_size = sizeof(AddRMSNormTilingData);

//     uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 2;
//     // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
//     // system("chmod -R 755 ./rms_norm_data/");
//     // system("cd ./rms_norm_data/ && rm -rf ./*bin");
//     // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
//     // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

//     tilingDatafromBin->num_row = 2;
//     tilingDatafromBin->num_col = 25600;
//     tilingDatafromBin->block_factor = 1;
//     tilingDatafromBin->row_factor = 64;
//     tilingDatafromBin->ub_factor = 8544;
//     tilingDatafromBin->epsilon = 0.01;
//     tilingDatafromBin->avg_factor = 0.01;

//     // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(31);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x1);
//     AscendC::GmFree(x2);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(y);
//     AscendC::GmFree(rstd);
//     AscendC::GmFree(x);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

TEST_F(add_rms_norm_test, test_case_14)
{
    size_t inputByteSize = 48 * 2016 * sizeof(int16_t);
    size_t gammaByteSize = 2016 * sizeof(int16_t);
    size_t outputByteSize = 48 * 2016 * sizeof(int16_t);
    size_t rstdByteSize = 48 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 48;
    tilingDatafromBin->num_col = 2016;
    tilingDatafromBin->block_factor = 2;
    tilingDatafromBin->row_factor = 5;
    tilingDatafromBin->ub_factor = 10080;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avg_factor = 0.000496;
    tilingDatafromBin->num_col_align = 2016;
    tilingDatafromBin->last_block_factor = 2;
    tilingDatafromBin->row_loop = 1;
    tilingDatafromBin->last_block_row_loop = 1;
    tilingDatafromBin->row_tail = 2;
    tilingDatafromBin->last_block_row_tail = 2;

    ICPU_SET_TILING_KEY(14);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// TEST_F(add_rms_norm_test, test_case_34)
// {
//     size_t inputByteSize = 48 * 2016 * sizeof(int16_t);
//     size_t gammaByteSize = 48 * sizeof(int16_t);
//     size_t outputByteSize = 48 * 2016 * sizeof(int16_t);
//     size_t rstdByteSize = 48 * sizeof(float);
//     size_t tiling_data_size = sizeof(AddRMSNormTilingData);

//     uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 1;

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

//     tilingDatafromBin->num_row = 48;
//     tilingDatafromBin->num_col = 2016;
//     tilingDatafromBin->block_factor = 2;
//     tilingDatafromBin->row_factor = 5;
//     tilingDatafromBin->ub_factor = 10080;
//     tilingDatafromBin->epsilon = 1e-5;
//     tilingDatafromBin->avg_factor = 0.000496;
//     tilingDatafromBin->num_col_align = 2016;
//     tilingDatafromBin->last_block_factor = 2;
//     tilingDatafromBin->row_loop = 1;
//     tilingDatafromBin->last_block_row_loop = 1;
//     tilingDatafromBin->row_tail = 2;
//     tilingDatafromBin->last_block_row_tail = 2;


//     ICPU_SET_TILING_KEY(34);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x1);
//     AscendC::GmFree(x2);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(y);
//     AscendC::GmFree(rstd);
//     AscendC::GmFree(x);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(add_rms_norm_test, test_case_22)
// {
//     size_t inputByteSize = 50 * 2 * sizeof(float);
//     size_t gammaByteSize = 2 * sizeof(float);
//     size_t outputByteSize = 50 * 2 * sizeof(float);
//     size_t rstdByteSize = 50 * sizeof(float);
//     size_t tiling_data_size = sizeof(AddRMSNormTilingData);

//     uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 25;
//     // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
//     // system("chmod -R 755 ./rms_norm_data/");
//     // system("cd ./rms_norm_data/ && rm -rf ./*bin");
//     // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
//     // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

//     tilingDatafromBin->num_row = 50;
//     tilingDatafromBin->num_col = 2;
//     tilingDatafromBin->block_factor = 2;
//     tilingDatafromBin->row_factor = 432;
//     tilingDatafromBin->ub_factor = 3456;
//     tilingDatafromBin->epsilon = 1e-5;
//     tilingDatafromBin->avg_factor = 0.5;

//     // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(22);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x1);
//     AscendC::GmFree(x2);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(y);
//     AscendC::GmFree(rstd);
//     AscendC::GmFree(x);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

// TEST_F(add_rms_norm_test, test_case_32)
// {
//     size_t inputByteSize = 50 * 2 * sizeof(int16_t);
//     size_t gammaByteSize = 2 * sizeof(int16_t);
//     size_t outputByteSize = 50 * 2 * sizeof(int16_t);
//     size_t rstdByteSize = 50 * sizeof(float);
//     size_t tiling_data_size = sizeof(AddRMSNormTilingData);

//     uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 25;
//     // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
//     // system("chmod -R 755 ./rms_norm_data/");
//     // system("cd ./rms_norm_data/ && rm -rf ./*bin");
//     // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
//     // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

//     tilingDatafromBin->num_row = 50;
//     tilingDatafromBin->num_col = 2;
//     tilingDatafromBin->block_factor = 2;
//     tilingDatafromBin->row_factor = 356;
//     tilingDatafromBin->ub_factor = 5696;
//     tilingDatafromBin->epsilon = 1e-5;
//     tilingDatafromBin->avg_factor = 0.5;

//     // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(32);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x1);
//     AscendC::GmFree(x2);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(y);
//     AscendC::GmFree(rstd);
//     AscendC::GmFree(x);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

TEST_F(add_rms_norm_test, test_case_13)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avg_factor = 0.004;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(13);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// TEST_F(add_rms_norm_test, test_case_23)
// {
//     size_t inputByteSize = 2 * 256 * sizeof(float);
//     size_t gammaByteSize = 256 * sizeof(float);
//     size_t outputByteSize = 2 * 256 * sizeof(float);
//     size_t rstdByteSize = 2 * sizeof(float);
//     size_t tiling_data_size = sizeof(AddRMSNormTilingData);

//     uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 2;
//     // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
//     // system("chmod -R 755 ./rms_norm_data/");
//     // system("cd ./rms_norm_data/ && rm -rf ./*bin");
//     // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
//     // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

//     tilingDatafromBin->num_row = 2;
//     tilingDatafromBin->num_col = 256;
//     tilingDatafromBin->block_factor = 1;
//     tilingDatafromBin->row_factor = 64;
//     tilingDatafromBin->ub_factor = 10240;
//     tilingDatafromBin->epsilon = 1e-5;
//     tilingDatafromBin->avg_factor = 0.004;

//     // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(23);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x1);
//     AscendC::GmFree(x2);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(y);
//     AscendC::GmFree(rstd);
//     AscendC::GmFree(x);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

TEST_F(add_rms_norm_test, test_case_33)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avg_factor = 0.004;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(33);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, test_case_12)
{
    size_t inputByteSize = 42 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 42 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 42 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 42;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 2;
    tilingDatafromBin->row_factor = 40;
    tilingDatafromBin->ub_factor = 10240;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.003906;
    tilingDatafromBin->num_col_align = 256;
    tilingDatafromBin->last_block_factor = 2;
    tilingDatafromBin->row_loop = 1;
    tilingDatafromBin->last_block_row_loop = 1;
    tilingDatafromBin->row_tail = 2;
    tilingDatafromBin->last_block_row_tail = 2;
    tilingDatafromBin->mul_loop_fp32 = 4;
    tilingDatafromBin->mul_tail_fp32 = 0;
    tilingDatafromBin->dst_rep_stride_fp32 = 32;
    tilingDatafromBin->mul_loop_fp16 = 2;
    tilingDatafromBin->mul_tail_fp16 = 0;
    tilingDatafromBin->dst_rep_stride_fp16 = 16;
    tilingDatafromBin->is_performance = 1;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(12);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// TEST_F(add_rms_norm_test, test_case_12_2)
// {
//     size_t inputByteSize = 1280 * 256 * sizeof(int16_t);
//     size_t gammaByteSize = 256 * sizeof(int16_t);
//     size_t outputByteSize = 1280 * 256 * sizeof(int16_t);
//     size_t rstdByteSize = 1280 * sizeof(float);
//     size_t tiling_data_size = sizeof(AddRMSNormTilingData);

//     uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
//     uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
//     uint32_t blockDim = 1;

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

//     tilingDatafromBin->num_row = 1280;
//     tilingDatafromBin->num_col = 256;
//     tilingDatafromBin->block_factor = 32;
//     tilingDatafromBin->row_factor = 40;
//     tilingDatafromBin->ub_factor = 10240;
//     tilingDatafromBin->epsilon = 0.01;
//     tilingDatafromBin->avg_factor = 0.003906;
//     tilingDatafromBin->num_col_align = 256;
//     tilingDatafromBin->last_block_factor = 32;
//     tilingDatafromBin->row_loop = 1;
//     tilingDatafromBin->last_block_row_loop = 1;
//     tilingDatafromBin->row_tail = 32;
//     tilingDatafromBin->last_block_row_tail = 32;
//     tilingDatafromBin->mul_loop_fp32 = 4;
//     tilingDatafromBin->mul_tail_fp32 = 0;
//     tilingDatafromBin->dst_rep_stride_fp32 = 32;
//     tilingDatafromBin->mul_loop_fp16 = 2;
//     tilingDatafromBin->mul_tail_fp16 = 0;
//     tilingDatafromBin->dst_rep_stride_fp16 = 16;
//     tilingDatafromBin->is_performance = 0;

//     // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
//     ICPU_SET_TILING_KEY(12);
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x1);
//     AscendC::GmFree(x2);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(y);
//     AscendC::GmFree(rstd);
//     AscendC::GmFree(x);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }


TEST_F(add_rms_norm_test, test_case_32)
{
    size_t inputByteSize = 42 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 42 * 256 * sizeof(int16_t);
    size_t rstdByteSize = 42 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 42;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 2;
    tilingDatafromBin->row_factor = 40;
    tilingDatafromBin->ub_factor = 10240;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.003906;
    tilingDatafromBin->num_col_align = 256;
    tilingDatafromBin->last_block_factor = 2;
    tilingDatafromBin->row_loop = 1;
    tilingDatafromBin->last_block_row_loop = 1;
    tilingDatafromBin->row_tail = 2;
    tilingDatafromBin->last_block_row_tail = 2;
    tilingDatafromBin->mul_loop_fp32 = 4;
    tilingDatafromBin->mul_tail_fp32 = 0;
    tilingDatafromBin->dst_rep_stride_fp32 = 32;
    tilingDatafromBin->mul_loop_fp16 = 2;
    tilingDatafromBin->mul_tail_fp16 = 0;
    tilingDatafromBin->dst_rep_stride_fp16 = 16;
    tilingDatafromBin->is_performance = 1;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(32);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, pre_test_case_0)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(110);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, nullptr, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, pre_test_case_1)
{
    size_t inputByteSize = 42 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 42 * 256 * sizeof(int16_t);
    // size_t rstdByteSize = 42 * sizeof(float);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    // uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 42;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 2;
    tilingDatafromBin->row_factor = 40;
    tilingDatafromBin->ub_factor = 10240;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.003906;
    tilingDatafromBin->num_col_align = 256;
    tilingDatafromBin->last_block_factor = 2;
    tilingDatafromBin->row_loop = 1;
    tilingDatafromBin->last_block_row_loop = 1;
    tilingDatafromBin->row_tail = 2;
    tilingDatafromBin->last_block_row_tail = 2;
    tilingDatafromBin->mul_loop_fp32 = 4;
    tilingDatafromBin->mul_tail_fp32 = 0;
    tilingDatafromBin->dst_rep_stride_fp32 = 32;
    tilingDatafromBin->mul_loop_fp16 = 2;
    tilingDatafromBin->mul_tail_fp16 = 0;
    tilingDatafromBin->dst_rep_stride_fp16 = 16;
    tilingDatafromBin->is_performance = 1;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(132);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, nullptr, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    // AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, pre_test_case_2)
{
    size_t inputByteSize = 2 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 2 * 256 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 12288;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avg_factor = 0.004;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(133);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, nullptr, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, pre_test_case_3)
{
    size_t inputByteSize = 42 * 256 * sizeof(int16_t);
    size_t gammaByteSize = 256 * sizeof(int16_t);
    size_t outputByteSize = 42 * 256 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 42;
    tilingDatafromBin->num_col = 256;
    tilingDatafromBin->block_factor = 2;
    tilingDatafromBin->row_factor = 40;
    tilingDatafromBin->ub_factor = 10240;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.003906;
    tilingDatafromBin->num_col_align = 256;
    tilingDatafromBin->last_block_factor = 2;
    tilingDatafromBin->row_loop = 1;
    tilingDatafromBin->last_block_row_loop = 1;
    tilingDatafromBin->row_tail = 2;
    tilingDatafromBin->last_block_row_tail = 2;
    tilingDatafromBin->mul_loop_fp32 = 4;
    tilingDatafromBin->mul_tail_fp32 = 0;
    tilingDatafromBin->dst_rep_stride_fp32 = 32;
    tilingDatafromBin->mul_loop_fp16 = 2;
    tilingDatafromBin->mul_tail_fp16 = 0;
    tilingDatafromBin->dst_rep_stride_fp16 = 16;
    tilingDatafromBin->is_performance = 1;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(112);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, nullptr, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, pre_test_case_4)
{
    size_t inputByteSize = 2 * 25600 * sizeof(int16_t);
    size_t gammaByteSize = 25600 * sizeof(int16_t);
    size_t outputByteSize = 2 * 25600 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 25600;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 8544;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(111);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, nullptr, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, pre_test_case_5)
{
    size_t inputByteSize = 48 * 2016 * sizeof(int16_t);
    size_t gammaByteSize = 2016 * sizeof(int16_t);
    size_t outputByteSize = 48 * 2016 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 48;
    tilingDatafromBin->num_col = 2016;
    tilingDatafromBin->block_factor = 2;
    tilingDatafromBin->row_factor = 5;
    tilingDatafromBin->ub_factor = 10080;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avg_factor = 0.000496;
    tilingDatafromBin->num_col_align = 2016;
    tilingDatafromBin->last_block_factor = 2;
    tilingDatafromBin->row_loop = 1;
    tilingDatafromBin->last_block_row_loop = 1;
    tilingDatafromBin->row_tail = 2;
    tilingDatafromBin->last_block_row_tail = 2;

    ICPU_SET_TILING_KEY(114);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, nullptr, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_test, post_test_case_1)
{
    size_t inputByteSize = 2 * 25600 * sizeof(int16_t);
    size_t gammaByteSize = 25600 * sizeof(int16_t);
    size_t outputByteSize = 2 * 25600 * sizeof(int16_t);
    size_t tiling_data_size = sizeof(AddRMSNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2 + 2 * 25600 * sizeof(int16_t));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRMSNormTilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormTilingData*>(tiling);

    tilingDatafromBin->num_row = 2;
    tilingDatafromBin->num_col = 25600;
    tilingDatafromBin->block_factor = 1;
    tilingDatafromBin->row_factor = 64;
    tilingDatafromBin->ub_factor = 8544;
    tilingDatafromBin->epsilon = 0.01;
    tilingDatafromBin->avg_factor = 0.01;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(1011);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm, blockDim, x1, x2, gamma, y, nullptr, nullptr, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
