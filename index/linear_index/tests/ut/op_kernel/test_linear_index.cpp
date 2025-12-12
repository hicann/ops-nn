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
 * \file test_linear_index.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "test_linear_index_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void linear_index(
    GM_ADDR indices, GM_ADDR var, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
class linear_index_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "linear_index_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "linear_index_test TearDown\n" << endl;
    }
};

TEST_F(linear_index_test, test_case_int32)
{
    // inputs
    size_t ind_size = 63806 * sizeof(int);
    size_t var_size = 65535 * 4096 * sizeof(float);
    size_t output_size = 63806 * sizeof(int);
    size_t tiling_data_size = sizeof(LinearIndexTilingDataDef);

    uint8_t* ind = (uint8_t*)AscendC::GmAlloc(ind_size);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    system("cp -r ../../../../index/linear_index/tests/ut/op_kernel/linear_index_data ./");
    system("chmod -R 755 ./linear_index_data/");
    system("cd ./linear_index_data/ && rm -rf ./*bin");
    system("cd ./linear_index_data/ && python3 gen_data.py int32");
    system("cd ./linear_index_data/ && python3 gen_tiling.py");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/linear_index_data/indices.bin", ind_size, ind, ind_size);
    ReadFile(path + "/linear_index_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/linear_index_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    LinearIndexTilingDataDef* tilingDatafromBin = reinterpret_cast<LinearIndexTilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(11);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(21);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(ind);
    AscendC::GmFree(var);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(linear_index_test, test_case_int64)
{
    // inputs
    size_t ind_size = 63806 * sizeof(int64_t);
    size_t var_size = 65535 * 4096 * sizeof(float);
    size_t output_size = 63806 * sizeof(int);
    size_t tiling_data_size = sizeof(LinearIndexTilingDataDef);

    uint8_t* ind = (uint8_t*)AscendC::GmAlloc(ind_size);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    system("cp -r ../../../../index/linear_index/tests/ut/op_kernel/linear_index_data ./");
    system("chmod -R 755 ./linear_index_data/");
    system("cd ./linear_index_data/ && rm -rf ./*bin");
    system("cd ./linear_index_data/ && python3 gen_data.py int64");
    system("cd ./linear_index_data/ && python3 gen_tiling.py");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/linear_index_data/indices.bin", ind_size, ind, ind_size);
    ReadFile(path + "/linear_index_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/linear_index_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    LinearIndexTilingDataDef* tilingDatafromBin = reinterpret_cast<LinearIndexTilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(21);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(22);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index, blockDim, ind, var, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(ind);
    AscendC::GmFree(var);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
