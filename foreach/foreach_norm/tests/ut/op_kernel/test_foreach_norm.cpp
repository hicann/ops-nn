/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_foreach_norm.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "foreach_norm_tiling_function.h"
#include "tensor_list_operate.h"

extern "C" __global__ __aicore__ void foreach_norm(
    GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling);

class foreach_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "foreach_norm_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "foreach_norm_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_norm_test, test_case_float_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{1, 4}, {1, 4}, {1, 4}};
    std::vector<std::vector<uint64_t>> shapeInfos_out = {{1}, {1}, {1}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_norm/tests/ut/op_kernel/norm_data ./");
    system("chmod -R 755 ./norm_data/");
    system("cd ./norm_data/ && python3 gen_data.py '{{1, 4}, {1, 4}, {1, 4}}' 2 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachNormTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, 0);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachReduceTilingData*>(tiling));

    uint8_t* x1 = CreateNormTensorList<float>(shapeInfos, "float32");
    uint8_t* x2 = CreateNormTensorList<float>(shapeInfos, "float32");
    uint8_t* scalar = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)scalar) = 2.0;

    ICPU_SET_TILING_KEY(2); // float
    ICPU_RUN_KF(foreach_norm, blockDim, x1, (uint8_t*)scalar, x2, workspace, tiling);

    FreeNormTensorList<float>(x2, shapeInfos_out, "float32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./norm_data/ && python3 compare_data.py 'float32'");
}

TEST_F(foreach_norm_test, test_case_float16_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{1, 4}, {1, 4}, {1, 4}};
    std::vector<std::vector<uint64_t>> shapeInfos_out = {{1}, {1}, {1}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_norm/tests/ut/op_kernel/norm_data ./");
    system("chmod -R 755 ./norm_data/");
    system("cd ./norm_data/ && python3 gen_data.py '{{1, 4}, {1, 4}, {1, 4}}' 2 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachNormTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 2, 0);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachReduceTilingData*>(tiling));

    uint8_t* x1 = CreateNormTensorList<half>(shapeInfos, "float16");
    uint8_t* x2 = CreateNormTensorList<half>(shapeInfos, "float16");
    uint8_t* scalar = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)scalar) = 2.0;

    ICPU_SET_TILING_KEY(1); // half
    ICPU_RUN_KF(foreach_norm, blockDim, x1, (uint8_t*)scalar, x2, workspace, tiling);

    FreeNormTensorList<half>(x2, shapeInfos_out, "float16");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./norm_data/ && python3 compare_data.py 'float16'");
}

TEST_F(foreach_norm_test, test_case_float16_2)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{1, 4}, {1, 4}, {1, 4}};
    std::vector<std::vector<uint64_t>> shapeInfos_out = {{1}, {1}, {1}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_norm/tests/ut/op_kernel/norm_data ./");
    system("chmod -R 755 ./norm_data/");
    system("cd ./norm_data/ && python3 gen_data.py '{{1, 4}, {1, 4}, {1, 4}}' 1 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachNormTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 2, 0);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachReduceTilingData*>(tiling));

    uint8_t* x1 = CreateNormTensorList<half>(shapeInfos, "float16");
    uint8_t* x2 = CreateNormTensorList<half>(shapeInfos, "float16");
    uint8_t* scalar = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)scalar) = 1.0;

    ICPU_SET_TILING_KEY(1); // half
    ICPU_RUN_KF(foreach_norm, blockDim, x1, (uint8_t*)scalar, x2, workspace, tiling);

    FreeNormTensorList<half>(x2, shapeInfos_out, "float16");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./norm_data/ && python3 compare_data.py 'float16'");
}

TEST_F(foreach_norm_test, test_case_bfloat16_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{1, 4}, {1, 4}, {1, 4}};
    std::vector<std::vector<uint64_t>> shapeInfos_out = {{1}, {1}, {1}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_norm/tests/ut/op_kernel/norm_data ./");
    system("chmod -R 755 ./norm_data/");
    system("cd ./norm_data/ && python3 gen_data.py '{{1, 4}, {1, 4}, {1, 4}}' 2 'bfloat16_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachNormTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 0);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachReduceTilingData*>(tiling));

    uint8_t* x1 = CreateNormTensorList<bfloat16_t>(shapeInfos, "bfloat16_t");
    uint8_t* x2 = CreateNormTensorList<bfloat16_t>(shapeInfos, "bfloat16_t");
    uint8_t* scalar = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)scalar) = 2.0;

    ICPU_SET_TILING_KEY(4); // bfloat16
    ICPU_RUN_KF(foreach_norm, blockDim, x1, (uint8_t*)scalar, x2, workspace, tiling);

    FreeNormTensorList<bfloat16_t>(x2, shapeInfos_out, "bfloat16_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./norm_data/ && python3 compare_data.py 'bfloat16_t'");
}

TEST_F(foreach_norm_test, test_case_bfloat16_2)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{1, 4}, {1, 4}, {1, 4}};
    std::vector<std::vector<uint64_t>> shapeInfos_out = {{1}, {1}, {1}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_norm/tests/ut/op_kernel/norm_data ./");
    system("chmod -R 755 ./norm_data/");
    system("cd ./norm_data/ && python3 gen_data.py '{{1, 4}, {1, 4}, {1, 4}}' 1 'bfloat16_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachReduceTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachNormTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 0);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachReduceTilingData*>(tiling));

    uint8_t* x1 = CreateNormTensorList<bfloat16_t>(shapeInfos, "bfloat16_t");
    uint8_t* x2 = CreateNormTensorList<bfloat16_t>(shapeInfos, "bfloat16_t");
    uint8_t* scalar = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)scalar) = 1.0;

    ICPU_SET_TILING_KEY(4); // bfloat16
    ICPU_RUN_KF(foreach_norm, blockDim, x1, (uint8_t*)scalar, x2, workspace, tiling);

    FreeNormTensorList<bfloat16_t>(x2, shapeInfos_out, "bfloat16_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./norm_data/ && python3 compare_data.py 'bfloat16_t'");
}