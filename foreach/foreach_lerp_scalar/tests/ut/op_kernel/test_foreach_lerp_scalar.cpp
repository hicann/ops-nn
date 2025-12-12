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
 * \file test_foreach_lerp_scalar.cpp
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
#include "foreach_lerp_scalar_tensorlist.h"

extern "C" __global__ __aicore__ void foreach_lerp_scalar(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class foreach_lerp_scalar_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "foreach_lerp_scalar_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "foreach_lerp_scalar_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_lerp_scalar_test, test_case_float_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_lerp_scalar/tests/ut/op_kernel/lerp_scalar_data ./");
    system("chmod -R 755 ./lerp_scalar_data/");
    system("cd ./lerp_scalar_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, 19);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* tensor1 = ForeachLerpScalar::CreateTensorList<float>(shapeInfos, "float32", 1);
    uint8_t* tensor2 = ForeachLerpScalar::CreateTensorList<float>(shapeInfos, "float32", 2);
    uint8_t* out = ForeachLerpScalar::CreateTensorList<float>(shapeInfos, "float32", 0);
    uint8_t* scalar = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)scalar) = 3;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_lerp_scalar, blockDim, tensor1, tensor2, scalar, out, workspace, tiling);

    ForeachLerpScalar::FreeTensorList<float>(out, shapeInfos, "float32", 0);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)tensor1);
    AscendC::GmFree((void*)tensor2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./lerp_scalar_data/ && python3 compare_data.py 'float32'");
}

TEST_F(foreach_lerp_scalar_test, test_case_float16_2)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_lerp_scalar/tests/ut/op_kernel/lerp_scalar_data ./");
    system("chmod -R 755 ./lerp_scalar_data/");
    system("cd ./lerp_scalar_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 2, 19);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* tensor1 = ForeachLerpScalar::CreateTensorList<half>(shapeInfos, "float16", 1);
    uint8_t* tensor2 = ForeachLerpScalar::CreateTensorList<half>(shapeInfos, "float16", 2);
    uint8_t* out = ForeachLerpScalar::CreateTensorList<half>(shapeInfos, "float16", 0);
    uint8_t* scalar = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)scalar) = 3;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(foreach_lerp_scalar, blockDim, tensor1, tensor2, scalar, out, workspace, tiling);

    ForeachLerpScalar::FreeTensorList<half>(out, shapeInfos, "float16", 0);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)tensor1);
    AscendC::GmFree((void*)tensor2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./lerp_scalar_data/ && python3 compare_data.py 'float16'");
}

TEST_F(foreach_lerp_scalar_test, test_case_bfloat16_3)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_lerp_scalar/tests/ut/op_kernel/lerp_scalar_data ./");
    system("chmod -R 755 ./lerp_scalar_data/");
    system("cd ./lerp_scalar_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'bfloat16_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 19);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* tensor1 = ForeachLerpScalar::CreateTensorList<bfloat16_t>(shapeInfos, "bfloat16_t", 1);
    uint8_t* tensor2 = ForeachLerpScalar::CreateTensorList<bfloat16_t>(shapeInfos, "bfloat16_t", 2);
    uint8_t* out = ForeachLerpScalar::CreateTensorList<bfloat16_t>(shapeInfos, "bfloat16_t", 0);
    uint8_t* scalar = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)scalar) = 3;

    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(foreach_lerp_scalar, blockDim, tensor1, tensor2, scalar, out, workspace, tiling);

    ForeachLerpScalar::FreeTensorList<bfloat16_t>(out, shapeInfos, "bfloat16_t", 0);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)tensor1);
    AscendC::GmFree((void*)tensor2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./lerp_scalar_data/ && python3 compare_data.py 'bfloat16_t'");
}
