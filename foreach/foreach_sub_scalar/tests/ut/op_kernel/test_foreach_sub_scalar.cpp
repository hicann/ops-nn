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
 * \file test_foreach_sub_scalar.cpp
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

extern "C" __global__ __aicore__ void foreach_sub_scalar(
    GM_ADDR x, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class foreach_sub_scalar_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "foreach_sub_scalar_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "foreach_sub_scalar_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_sub_scalar_test, test_case_float_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_sub_scalar/tests/ut/op_kernel/sub_scalar_data ./");
    system("chmod -R 755 ./sub_scalar_data/");
    system("cd ./sub_scalar_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;

    tilingFuncObj.Init(shapeInfos, 1, optiling::FOREACH_SUB_SCALAR_OP_CODE);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachSubScalar<float>(shapeInfos, "float32");
    uint8_t* x2 = CreateTensorListForeachSubScalar<float>(shapeInfos, "float32");
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(sizeof(float));

    *((float*)x3) = 3;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_sub_scalar, blockDim, x1, x3, x2, workspace, tiling);

    FreeTensorListForeachSubScalar<float>(x2, shapeInfos, "float32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    int ret = system("cd ./sub_scalar_data && python compare_data.py 'float32'");
    std::cout << "exit = " << ret << std::endl;
}

TEST_F(foreach_sub_scalar_test, test_case_float16_2)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_sub_scalar/tests/ut/op_kernel/sub_scalar_data ./");
    system("chmod -R 755 ./sub_scalar_data/");
    system("cd ./sub_scalar_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 15 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;

    tilingFuncObj.Init(shapeInfos, 2, optiling::FOREACH_SUB_SCALAR_OP_CODE);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachSubScalar<half>(shapeInfos, "float16");
    uint8_t* x2 = CreateTensorListForeachSubScalar<half>(shapeInfos, "float16");
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(sizeof(float));

    *((float*)x3) = 3;

    ICPU_SET_TILING_KEY(1); // half
    ICPU_RUN_KF(foreach_sub_scalar, blockDim, x1, x3, x2, workspace, tiling);

    FreeTensorListForeachSubScalar<half>(x2, shapeInfos, "float16");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    int ret = system("cd ./sub_scalar_data && python compare_data.py 'float16'");
    std::cout << "exit = " << ret << std::endl;
}
