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
 * \file test_scatter_elements_v2.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "test_scatter_elements_v2_tiling_def.h"

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
    size_t tiling_data_size = sizeof(ScatterElementsV2TilingDataDef);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indices_size);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    system(
        "cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/scatter_elements_v2/scatter_elements_v2_data "
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

    ScatterElementsV2TilingDataDef* tilingDatafromBin = reinterpret_cast<ScatterElementsV2TilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(111);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(112);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(121);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(122);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(indices);
    AscendC::GmFree(src);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scatter_elements_v2_test, test_case_fp16)
{
    // inputs
    size_t data_size = 128 * 59;
    size_t var_size = data_size * sizeof(half);
    size_t indices_size = data_size * sizeof(long);
    size_t src_size = data_size * sizeof(half);
    size_t output_size = data_size * sizeof(half);
    size_t tiling_data_size = sizeof(ScatterElementsV2TilingDataDef);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indices_size);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    system(
        "cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/scatter_elements_v2/scatter_elements_v2_data "
        "./");
    system("chmod -R 755 ./scatter_elements_v2_data/");
    system("cd ./scatter_elements_v2_data/ && rm -rf ./*bin");
    system("cd ./scatter_elements_v2_data/ && python3 gen_data.py float16");
    system("cd ./scatter_elements_v2_data/ && python3 gen_tiling.py");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_elements_v2_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/scatter_elements_v2_data/indices.bin", indices_size, indices, indices_size);
    ReadFile(path + "/scatter_elements_v2_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/scatter_elements_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ScatterElementsV2TilingDataDef* tilingDatafromBin = reinterpret_cast<ScatterElementsV2TilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(211);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(212);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(221);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(222);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(indices);
    AscendC::GmFree(src);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scatter_elements_v2_test, test_case_int32)
{
    // inputs
    size_t data_size = 128 * 59;
    size_t var_size = data_size * sizeof(int);
    size_t indices_size = data_size * sizeof(long);
    size_t src_size = data_size * sizeof(int);
    size_t output_size = data_size * sizeof(int);
    size_t tiling_data_size = sizeof(ScatterElementsV2TilingDataDef);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indices_size);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    system(
        "cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/scatter_elements_v2/scatter_elements_v2_data "
        "./");
    system("chmod -R 755 ./scatter_elements_v2_data/");
    system("cd ./scatter_elements_v2_data/ && rm -rf ./*bin");
    system("cd ./scatter_elements_v2_data/ && python3 gen_data.py int32");
    system("cd ./scatter_elements_v2_data/ && python3 gen_tiling.py");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_elements_v2_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/scatter_elements_v2_data/indices.bin", indices_size, indices, indices_size);
    ReadFile(path + "/scatter_elements_v2_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/scatter_elements_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ScatterElementsV2TilingDataDef* tilingDatafromBin = reinterpret_cast<ScatterElementsV2TilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(311);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(312);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(321);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(322);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(indices);
    AscendC::GmFree(src);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scatter_elements_v2_test, test_case_uint8)
{
    // inputs
    size_t data_size = 128 * 59;
    size_t var_size = data_size * sizeof(uint8_t);
    size_t indices_size = data_size * sizeof(long);
    size_t src_size = data_size * sizeof(uint8_t);
    size_t output_size = data_size * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(ScatterElementsV2TilingDataDef);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indices_size);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    system(
        "cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/scatter_elements_v2/scatter_elements_v2_data "
        "./");
    system("chmod -R 755 ./scatter_elements_v2_data/");
    system("cd ./scatter_elements_v2_data/ && rm -rf ./*bin");
    system("cd ./scatter_elements_v2_data/ && python3 gen_data.py uint8");
    system("cd ./scatter_elements_v2_data/ && python3 gen_tiling.py");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_elements_v2_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/scatter_elements_v2_data/indices.bin", indices_size, indices, indices_size);
    ReadFile(path + "/scatter_elements_v2_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/scatter_elements_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ScatterElementsV2TilingDataDef* tilingDatafromBin = reinterpret_cast<ScatterElementsV2TilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(411);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(412);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(421);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(422);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(indices);
    AscendC::GmFree(src);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scatter_elements_v2_test, test_case_int8)
{
    // inputs
    size_t data_size = 128 * 59;
    size_t var_size = data_size * sizeof(int8_t);
    size_t indices_size = data_size * sizeof(long);
    size_t src_size = data_size * sizeof(int8_t);
    size_t output_size = data_size * sizeof(int8_t);
    size_t tiling_data_size = sizeof(ScatterElementsV2TilingDataDef);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indices_size);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    system(
        "cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/scatter_elements_v2/scatter_elements_v2_data "
        "./");
    system("chmod -R 755 ./scatter_elements_v2_data/");
    system("cd ./scatter_elements_v2_data/ && rm -rf ./*bin");
    system("cd ./scatter_elements_v2_data/ && python3 gen_data.py int8");
    system("cd ./scatter_elements_v2_data/ && python3 gen_tiling.py");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_elements_v2_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/scatter_elements_v2_data/indices.bin", indices_size, indices, indices_size);
    ReadFile(path + "/scatter_elements_v2_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/scatter_elements_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ScatterElementsV2TilingDataDef* tilingDatafromBin = reinterpret_cast<ScatterElementsV2TilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(511);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(512);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(521);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(522);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(indices);
    AscendC::GmFree(src);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scatter_elements_v2_test, test_case_bf16)
{
    // inputs
    size_t data_size = 128 * 59;
    size_t var_size = data_size * sizeof(half);
    size_t indices_size = data_size * sizeof(long);
    size_t src_size = data_size * sizeof(half);
    size_t output_size = data_size * sizeof(half);
    size_t tiling_data_size = sizeof(ScatterElementsV2TilingDataDef);

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(var_size);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indices_size);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(src_size);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 32;

    system(
        "cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/scatter_elements_v2/scatter_elements_v2_data "
        "./");
    system("chmod -R 755 ./scatter_elements_v2_data/");
    system("cd ./scatter_elements_v2_data/ && rm -rf ./*bin");
    system("cd ./scatter_elements_v2_data/ && python3 gen_data.py bfloat16");
    system("cd ./scatter_elements_v2_data/ && python3 gen_tiling.py");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scatter_elements_v2_data/var.bin", var_size, var, var_size);
    ReadFile(path + "/scatter_elements_v2_data/indices.bin", indices_size, indices, indices_size);
    ReadFile(path + "/scatter_elements_v2_data/src.bin", src_size, src, src_size);
    ReadFile(path + "/scatter_elements_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ScatterElementsV2TilingDataDef* tilingDatafromBin = reinterpret_cast<ScatterElementsV2TilingDataDef*>(tiling);

    ICPU_SET_TILING_KEY(611);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(612);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(621);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(622);
    ICPU_RUN_KF(scatter_elements_v2, blockDim, var, indices, src, output, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(indices);
    AscendC::GmFree(src);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
