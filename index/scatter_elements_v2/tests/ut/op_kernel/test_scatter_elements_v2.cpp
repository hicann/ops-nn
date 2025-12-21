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
 * \file test_scatter_elements_v2.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/scatter_elements_v2_tiling.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void scatter_elements_v2(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
class scatter_elements_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "scatter_elements_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "scatter_elements_v2_test TearDown\n" << endl;
    }
};

TEST_F(scatter_elements_v2_test, test_case_fp32)
{
    // inputs
    size_t data_size = 128 * 59;
    size_t var_size = data_size * sizeof(float);
    size_t indices_size = data_size * sizeof(long);
    size_t src_size = data_size * sizeof(float);
    size_t output_size = data_size * sizeof(float);
    size_t tiling_data_size = sizeof(ScatterElementsV2TilingData);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indices_size);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    system(
        "cp -r ../../../../index/scatter_elements_v2/tests/ut/op_kernel/scatter_elements_v2_data "
        "./");
    system("chmod -R 755 ./scatter_elements_v2_data/");
    system("cd ./scatter_elements_v2_data/ && rm -rf ./*bin");
    system("cd ./scatter_elements_v2_data/ && python3 gen_data.py float32");
    system("cd ./scatter_elements_v2_data/ && python3 gen_tiling.py");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_elements_v2_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/scatter_elements_v2_data/indices.bin", indices_size, indices, indices_size);
    ReadFile(path + "/scatter_elements_v2_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/scatter_elements_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ScatterElementsV2TilingData* tilingDatafromBin = reinterpret_cast<ScatterElementsV2TilingData*>(tiling);

    ICPU_SET_TILING_KEY(111);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(112);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(121);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(122);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(indices);
    AscendC::GmFree(src);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}