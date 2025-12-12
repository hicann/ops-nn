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
 * \file test_scatter_add_with_sorted.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "test_scatter_add_with_sorted_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void scatter_add_with_sorted(
    GM_ADDR var, GM_ADDR value, GM_ADDR sorted_index, GM_ADDR pos, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
class scatter_add_with_sorted_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "scatter_add_with_sorted_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "scatter_add_with_sorted_test TearDown\n" << endl;
    }
};

TEST_F(scatter_add_with_sorted_test, test_case_fp32)
{
    // inputs
    size_t var_size = 65 * 4096 * sizeof(float);
    size_t src_size = 63 * 4096 * sizeof(float);
    size_t ind_size = 63 * sizeof(int);
    size_t pos_size = 63 * sizeof(int);
    size_t output_size = 65 * 4096 * sizeof(float);
    size_t tiling_data_size = sizeof(ScatterAddWithSortedTilingDataDef);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t* ind = (uint8_t*)AscendC::GmAlloc(ind_size);
    uint8_t* pos = (uint8_t*)AscendC::GmAlloc(pos_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    system(
        "cp -r "
        "../../../../index/scatter_add_with_sorted/tests/ut/op_kernel/scatter_add_with_sorted_data "
        "./");
    system("chmod -R 755 ./scatter_add_with_sorted_data/");
    system("cd ./scatter_add_with_sorted_data/ && rm -rf ./*bin");
    system("cd ./scatter_add_with_sorted_data/ && python3 gen_data.py float32");
    system("cd ./scatter_add_with_sorted_data/ && python3 gen_tiling.py");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_add_with_sorted_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/scatter_add_with_sorted_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/scatter_add_with_sorted_data/ind.bin", ind_size, ind, ind_size);
    ReadFile(path + "/scatter_add_with_sorted_data/pos.bin", pos_size, pos, pos_size);
    ReadFile(path + "/scatter_add_with_sorted_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ScatterAddWithSortedTilingDataDef* tilingDatafromBin = reinterpret_cast<ScatterAddWithSortedTilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(11);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        scatter_add_with_sorted, blockDim, var, src, ind, pos, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(src);
    AscendC::GmFree(ind);
    AscendC::GmFree(pos);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

