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
 * \file test_foreach_minimum_scalar_list.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "foreach_minimum_scalar_list_tiling_function.h"
#include "tensor_list_operate.h"

extern "C" __global__ __aicore__ void foreach_minimum_scalar_list(
    GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling);

class foreach_minimum_scalar_list_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "foreach_minimum_scalar_list_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "foreach_minimum_scalar_list_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_minimum_scalar_list_test, test_case_float_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_minimum_scalar_list/tests/ut/op_kernel/minimum_scalar_list_data ./");
    system("chmod -R 755 ./minimum_scalar_list_data/");
    system("cd ./minimum_scalar_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, optiling::FOREACH_BINARY_SCALAR_OP_CODE);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachMinimumScalarList<float>(shapeInfos, "float32"); // input tensor
    uint8_t* x2 = CreateTensorListForeachMinimumScalarList<float>(shapeInfos, "float32"); // output tensor
    float* scalar = (float*)AscendC::GmAlloc(sizeof(float) * 3);

    for (int i = 0; i < 3; i++) {
        scalar[i] = i;
    }

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_minimum_scalar_list, blockDim, x1, reinterpret_cast<uint8_t*>(scalar), x2, workspace, tiling);

    FreeTensorListForeachMinimumScalarList<float>(x2, shapeInfos, "float32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./minimum_scalar_list_data/ && python3 compare_data.py 'float32'");
}

TEST_F(foreach_minimum_scalar_list_test, test_case_float16_2)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_minimum_scalar_list/tests/ut/op_kernel/minimum_scalar_list_data ./");
    system("chmod -R 755 ./minimum_scalar_list_data/");
    system("cd ./minimum_scalar_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 2, optiling::FOREACH_BINARY_SCALAR_OP_CODE);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachMinimumScalarList<half>(shapeInfos, "float16"); // input tensor
    uint8_t* x2 = CreateTensorListForeachMinimumScalarList<half>(shapeInfos, "float16"); // output tensor
    float* scalar = (float*)AscendC::GmAlloc(sizeof(float) * 3);

    for (int i = 0; i < 3; i++) {
        scalar[i] = i;
    }

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(foreach_minimum_scalar_list, blockDim, x1, reinterpret_cast<uint8_t*>(scalar), x2, workspace, tiling);

    FreeTensorListForeachMinimumScalarList<half>(x2, shapeInfos, "float16");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./minimum_scalar_list_data/ && python3 compare_data.py 'float16'");
}

TEST_F(foreach_minimum_scalar_list_test, test_case_int32_3)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_minimum_scalar_list/tests/ut/op_kernel/minimum_scalar_list_data ./");
    system("chmod -R 755 ./minimum_scalar_list_data/");
    system("cd ./minimum_scalar_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'int32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 3, optiling::FOREACH_BINARY_SCALAR_OP_CODE);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachMinimumScalarList<int32_t>(shapeInfos, "int32"); // input tensor
    uint8_t* x2 = CreateTensorListForeachMinimumScalarList<int32_t>(shapeInfos, "int32"); // output tensor
    float* scalar = (float*)AscendC::GmAlloc(sizeof(float) * 3);

    for (int i = 0; i < 3; i++) {
        scalar[i] = i;
    }

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(foreach_minimum_scalar_list, blockDim, x1, reinterpret_cast<uint8_t*>(scalar), x2, workspace, tiling);

    FreeTensorListForeachMinimumScalarList<int32_t>(x2, shapeInfos, "int32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./minimum_scalar_list_data/ && python3 compare_data.py 'int32'");
}

TEST_F(foreach_minimum_scalar_list_test, test_case_bfloat16_4)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_minimum_scalar_list/tests/ut/op_kernel/minimum_scalar_list_data ./");
    system("chmod -R 755 ./minimum_scalar_list_data/");
    system("cd ./minimum_scalar_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'bfloat16_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, optiling::FOREACH_BINARY_SCALAR_OP_CODE);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachMinimumScalarList<bfloat16_t>(shapeInfos, "bfloat16_t"); // input tensor
    uint8_t* x2 = CreateTensorListForeachMinimumScalarList<bfloat16_t>(shapeInfos, "bfloat16_t"); // output tensor
    float* scalar = (float*)AscendC::GmAlloc(sizeof(float) * 3);

    for (int i = 0; i < 3; i++) {
        scalar[i] = i;
    }

    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(foreach_minimum_scalar_list, blockDim, x1, reinterpret_cast<uint8_t*>(scalar), x2, workspace, tiling);

    FreeTensorListForeachMinimumScalarList<bfloat16_t>(x2, shapeInfos, "bfloat16_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./minimum_scalar_list_data/ && python3 compare_data.py 'bfloat16_t'");
}