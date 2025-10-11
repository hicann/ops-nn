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
 * \file test_foreach_mul_list.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "foreach_mul_list_tiling_function.h"
#include "foreach_mul_list_tensorlist.h"

extern "C" __global__ __aicore__ void foreach_mul_list(
    GM_ADDR inputs_1, GM_ADDR inputs_2, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling);

class foreach_mul_list_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "foreach_mul_list_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "foreach_mul_list_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_mul_list_test, test_case_float_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_mul_list/tests/ut/op_kernel/mul_list_data ./");
    system("chmod -R 755 ./mul_list_data/");
    system("cd ./mul_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, 3);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = ForeachMulList::CreateTensorList<float>(shapeInfos, "float32", 1);
    uint8_t* x2 = ForeachMulList::CreateTensorList<float>(shapeInfos, "float32", 2);

    uint8_t* y = ForeachMulList::CreateTensorList<float>(shapeInfos, "float32", 0);
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_mul_list, blockDim, x1, x2, y, workspace, tiling);

    ForeachMulList::FreeTensorList<float>(y, shapeInfos, "float32", 0);
    ForeachMulList::FreeTensorList<float>(x1, shapeInfos, "float32", 1);
    ForeachMulList::FreeTensorList<float>(x2, shapeInfos, "float32", 2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./mul_list_data/ && python3 compare_data.py 'float32'");
}

TEST_F(foreach_mul_list_test, test_case_float16_2)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_mul_list/tests/ut/op_kernel/mul_list_data ./");
    system("chmod -R 755 ./mul_list_data/");
    system("cd ./mul_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 2, 3);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = ForeachMulList::CreateTensorList<half>(shapeInfos, "float16", 1);
    uint8_t* x2 = ForeachMulList::CreateTensorList<half>(shapeInfos, "float16", 2);

    uint8_t* y = ForeachMulList::CreateTensorList<half>(shapeInfos, "float16", 0);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(foreach_mul_list, blockDim, x1, x2, y, workspace, tiling);

    ForeachMulList::FreeTensorList<half>(y, shapeInfos, "float16", 0);
    ForeachMulList::FreeTensorList<half>(x1, shapeInfos, "float16", 1);
    ForeachMulList::FreeTensorList<half>(x2, shapeInfos, "float16", 2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./mul_list_data/ && python3 compare_data.py 'float16'");
}

TEST_F(foreach_mul_list_test, test_case_int32_3)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_mul_list/tests/ut/op_kernel/mul_list_data ./");
    system("chmod -R 755 ./mul_list_data/");
    system("cd ./mul_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'int32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 3, 3);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = ForeachMulList::CreateTensorList<int32_t>(shapeInfos, "int32", 1);
    uint8_t* x2 = ForeachMulList::CreateTensorList<int32_t>(shapeInfos, "int32", 2);

    uint8_t* y = ForeachMulList::CreateTensorList<int32_t>(shapeInfos, "int32", 0);
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(foreach_mul_list, blockDim, x1, x2, y, workspace, tiling);

    ForeachMulList::FreeTensorList<int32_t>(y, shapeInfos, "int32", 0);
    ForeachMulList::FreeTensorList<int32_t>(x1, shapeInfos, "int32", 1);
    ForeachMulList::FreeTensorList<int32_t>(x2, shapeInfos, "int32", 2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./mul_list_data/ && python3 compare_data.py 'int32'");
}

TEST_F(foreach_mul_list_test, test_case_bfloat_4)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_mul_list/tests/ut/op_kernel/mul_list_data ./");
    system("chmod -R 755 ./mul_list_data/");
    system("cd ./mul_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'bfloat16_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 3);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = ForeachMulList::CreateTensorList<bfloat16_t>(shapeInfos, "bfloat16_t", 1);
    uint8_t* x2 = ForeachMulList::CreateTensorList<bfloat16_t>(shapeInfos, "bfloat16_t", 2);

    uint8_t* y = ForeachMulList::CreateTensorList<bfloat16_t>(shapeInfos, "bfloat16_t", 0);
    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(foreach_mul_list, blockDim, x1, x2, y, workspace, tiling);

    ForeachMulList::FreeTensorList<bfloat16_t>(y, shapeInfos, "bfloat16_t", 0);
    ForeachMulList::FreeTensorList<bfloat16_t>(x1, shapeInfos, "bfloat16_t", 1);
    ForeachMulList::FreeTensorList<bfloat16_t>(x2, shapeInfos, "bfloat16_t", 2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./mul_list_data/ && python3 compare_data.py 'bfloat16_t'");
}
