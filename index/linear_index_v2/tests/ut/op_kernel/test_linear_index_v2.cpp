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
 * \file test_linear_index_v2.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "linear_index_v2_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "tensor_list_operate.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void linear_index_v2(
    GM_ADDR indexList, GM_ADDR stride, GM_ADDR valueSize, GM_ADDR output, GM_ADDR workSpace, GM_ADDR tiling);
class linear_index_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "linear_index_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "linear_index_v2_test TearDown\n" << endl;
    }
};

TEST_F(linear_index_v2_test, test_case_0)
{
    std::vector<std::vector<uint64_t>> idx_shape{{3}, {3}};
    size_t stride_size = 2 * sizeof(int);
    size_t value_size = 2 * sizeof(int);
    size_t output_size = 3 * sizeof(int);
    size_t tiling_size = sizeof(LinearIndexV2TilingData);
    
    system("cp -r ../../../../index/linear_index_v2/tests/ut/op_kernel/linear_index_v2_data ./");
    system("chmod -R 755 ./linear_index_v2_data/");
    system("cd ./linear_index_v2_data/ && rm -rf ./*bin");
    system("cd ./linear_index_v2_data/ && python3 gen_data.py");
    system("cd ./linear_index_v2_data/ && python3 gen_tiling.py test_case_continuous");


    uint8_t* idx_list = CreateTensorList<int32_t>(idx_shape);
    uint8_t* stride = (uint8_t*)AscendC::GmAlloc(stride_size);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);
    uint32_t blockDim = 3;

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/linear_index_v2_data/stride.bin", stride_size, stride, stride_size);
    ReadFile(path + "/linear_index_v2_data/value.bin", value_size, value, value_size);
    ReadFile(path + "/linear_index_v2_data/tiling.bin", tiling_size, tiling, tiling_size);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(linear_index_v2, blockDim, idx_list, stride, value, output, workspace, tiling);

    FreeTensorList<int32_t>(idx_list, idx_shape);
    AscendC::GmFree(stride);
    AscendC::GmFree(value);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
