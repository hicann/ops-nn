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
 * \file test_foreach_reciprocal.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../foreach_abs/tests/ut/op_kernel/foreach_abs_tiling_function.h"
#include "tensor_list_operate.h"

extern "C" __global__ __aicore__ void foreach_reciprocal(
    GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling);

class foreach_reciprocal_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "foreach_reciprocal_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "foreach_reciprocal_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_reciprocal_test, test_case_float_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_reciprocal/tests/ut/op_kernel/reciprocal_data ./");
    system("chmod -R 755 ./reciprocal_data/");
    system("cd ./reciprocal_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, 7); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachReciprocal<float>(shapeInfos, "float32"); // input tensor
    uint8_t* x2 = CreateTensorListForeachReciprocal<float>(shapeInfos, "float32"); // output tensor

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_reciprocal, blockDim, x1, x2, workspace, tiling);

    FreeTensorListForeachReciprocal<float>(x2, shapeInfos, "float32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./reciprocal_data/ && python3 compare_data.py 'float32'");
}

TEST_F(foreach_reciprocal_test, test_case_float16_2)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_reciprocal/tests/ut/op_kernel/reciprocal_data ./");
    system("chmod -R 755 ./reciprocal_data/");
    system("cd ./reciprocal_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 2, 7); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachReciprocal<half>(shapeInfos, "float16"); // input tensor
    uint8_t* x2 = CreateTensorListForeachReciprocal<half>(shapeInfos, "float16"); // output tensor

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(foreach_reciprocal, blockDim, x1, x2, workspace, tiling);

    FreeTensorListForeachReciprocal<half>(x2, shapeInfos, "float16");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./reciprocal_data/ && python3 compare_data.py 'float16'");
}

TEST_F(foreach_reciprocal_test, test_case_bfloat16_3)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_reciprocal/tests/ut/op_kernel/reciprocal_data ./");
    system("chmod -R 755 ./reciprocal_data/");
    system("cd ./reciprocal_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'bfloat16_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 7);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachReciprocal<bfloat16_t>(shapeInfos, "bfloat16_t");
    uint8_t* x2 = CreateTensorListForeachReciprocal<bfloat16_t>(shapeInfos, "bfloat16_t");

    ICPU_SET_TILING_KEY(4); // bfloat16_t
    ICPU_RUN_KF(foreach_reciprocal, blockDim, x1, x2, workspace, tiling);

    FreeTensorListForeachReciprocal<bfloat16_t>(x2, shapeInfos, "bfloat16_t");
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./reciprocal_data/ && python3 compare_data.py 'bfloat16_t'");
}
