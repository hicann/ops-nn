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
 * \file test_foreach_minimum_scalar.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "foreach_minimum_scalar_tiling_function.h"
#include "foreach_minimum_scalar_tensorlist.h"

extern "C" __global__ __aicore__ void foreach_minimum_scalar(
    GM_ADDR inputs_1, GM_ADDR inputs_2, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling);

class foreach_minimum_scalar_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "foreach_minimum_scalar_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "foreach_minimum_scalar_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_minimum_scalar_test, test_case_float_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_minimum_scalar/tests/ut/op_kernel/minimum_scalar_data ./");
    system("chmod -R 755 ./minimum_scalar_data/");
    system("cd ./minimum_scalar_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'float32'");
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

    uint8_t* x = CreateTensorListForeachMinimumScalar<float>(shapeInfos, "float32", 1);
    uint8_t* xs = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)xs) = 64;
    uint8_t* y = CreateTensorListForeachMinimumScalar<float>(shapeInfos, "float32", 0);
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_minimum_scalar, blockDim, x, xs, y, workspace, tiling);

    FreeTensorListForeachMinimumScalar<float>(y, shapeInfos, "float32", 0);
    FreeTensorListForeachMinimumScalar<float>(x, shapeInfos, "float32", 1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xs);
    system("cd ./minimum_scalar_data/ && python3 compare_data.py 'float32'");
}

TEST_F(foreach_minimum_scalar_test, test_case_float16_2)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_minimum_scalar/tests/ut/op_kernel/minimum_scalar_data ./");
    system("chmod -R 755 ./minimum_scalar_data/");
    system("cd ./minimum_scalar_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'float16'");
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

    uint8_t* x = CreateTensorListForeachMinimumScalar<half>(shapeInfos, "float16", 1);
    uint8_t* xs = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)xs) = 64;
    uint8_t* y = CreateTensorListForeachMinimumScalar<half>(shapeInfos, "float16", 0);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(foreach_minimum_scalar, blockDim, x, xs, y, workspace, tiling);

    FreeTensorListForeachMinimumScalar<half>(y, shapeInfos, "float16", 0);
    FreeTensorListForeachMinimumScalar<half>(x, shapeInfos, "float16", 1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xs);

    system("cd ./minimum_scalar_data/ && python3 compare_data.py 'float16'");
}

TEST_F(foreach_minimum_scalar_test, test_case_int32_3)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_minimum_scalar/tests/ut/op_kernel/minimum_scalar_data ./");
    system("chmod -R 755 ./minimum_scalar_data/");
    system("cd ./minimum_scalar_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'int32'");
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

    uint8_t* x = CreateTensorListForeachMinimumScalar<int32_t>(shapeInfos, "int32", 1);
    uint8_t* xs = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)xs) = 64;
    uint8_t* y = CreateTensorListForeachMinimumScalar<int32_t>(shapeInfos, "int32", 0);
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(foreach_minimum_scalar, blockDim, x, xs, y, workspace, tiling);

    FreeTensorListForeachMinimumScalar<int32_t>(y, shapeInfos, "int32", 0);
    FreeTensorListForeachMinimumScalar<int32_t>(x, shapeInfos, "int32", 1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xs);
    system("cd ./minimum_scalar_data/ && python3 compare_data.py 'int32'");
}

TEST_F(foreach_minimum_scalar_test, test_case_bfloat16_4)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_minimum_scalar/tests/ut/op_kernel/minimum_scalar_data ./");
    system("chmod -R 755 ./minimum_scalar_data/");
    system("cd ./minimum_scalar_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'bfloat16_t'");
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

    uint8_t* x = CreateTensorListForeachMinimumScalar<bfloat16_t>(shapeInfos, "bfloat16_t", 1);
    uint8_t* xs = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)xs) = 64;
    uint8_t* y = CreateTensorListForeachMinimumScalar<bfloat16_t>(shapeInfos, "bfloat16_t", 0);
    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(foreach_minimum_scalar, blockDim, x, xs, y, workspace, tiling);

    FreeTensorListForeachMinimumScalar<bfloat16_t>(y, shapeInfos, "bfloat16_t", 0);
    FreeTensorListForeachMinimumScalar<bfloat16_t>(x, shapeInfos, "bfloat16_t", 1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xs);
    system("cd ./minimum_scalar_data/ && python3 compare_data.py 'bfloat16_t'");
}
