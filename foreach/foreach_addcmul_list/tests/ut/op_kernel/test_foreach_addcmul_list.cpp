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
 * \file test_foreach_addcmul_list.cpp
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

extern "C" __global__ __aicore__ void foreach_addcmul_list(GM_ADDR inputs, GM_ADDR tensor1,
                                                            GM_ADDR tensor2, GM_ADDR scalar,
                                                            GM_ADDR outputs,
                                                            GM_ADDR workspace, GM_ADDR tiling);

class foreach_addcmul_list_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "foreach_addcmul_list_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "foreach_addcmul_list_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_addcmul_list_test, test_case_float_1) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_addcmul_list/tests/ut/op_kernel/addcmul_list_data ./");
    system("chmod -R 755 ./addcmul_list_data/");
    system("cd ./addcmul_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, 10);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachAddcmulList<float>(shapeInfos, "float32", "input");
    uint8_t* tensor1 = CreateTensorListForeachAddcmulList<float>(shapeInfos, "float32", "tensor1");
    uint8_t* tensor2 = CreateTensorListForeachAddcmulList<float>(shapeInfos, "float32", "tensor2");
    uint8_t* out = CreateTensorListForeachAddcmulList<float>(shapeInfos, "float32", "input");
    float * scalar = (float *)AscendC::GmAlloc(sizeof(float) * 3);
    for (int i = 0; i < 3; i ++ ) {
        scalar[i] = (float)3;
    }

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_addcmul_list, blockDim, x1, tensor1, tensor2, reinterpret_cast<uint8_t*>(scalar), out, workspace, tiling);

    FreeTensorListForeachAddcmulList<float>(out, shapeInfos, "float32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)tensor1);
    AscendC::GmFree((void*)tensor2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./addcmul_list_data/ && python3 compare_data.py 'float32'");
}

TEST_F(foreach_addcmul_list_test, test_case_float16_2) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_addcmul_list/tests/ut/op_kernel/addcmul_list_data ./");
    system("chmod -R 755 ./addcmul_list_data/");
    system("cd ./addcmul_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 2, 10);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachAddcmulList<half>(shapeInfos, "float16", "input");
    uint8_t* tensor1 = CreateTensorListForeachAddcmulList<half>(shapeInfos, "float16", "tensor1");
    uint8_t* tensor2 = CreateTensorListForeachAddcmulList<half>(shapeInfos, "float16", "tensor2");
    uint8_t* out = CreateTensorListForeachAddcmulList<half>(shapeInfos, "float16", "input");
    half * scalar = (half *)AscendC::GmAlloc(sizeof(half) * 3);
    for (int i = 0; i < 3; i ++ ) {
        scalar[i] = (half)3;
    }

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(foreach_addcmul_list, blockDim, x1, tensor1, tensor2, reinterpret_cast<uint8_t*>(scalar), out, workspace, tiling);

    FreeTensorListForeachAddcmulList<half>(out, shapeInfos, "float16");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)tensor1);
    AscendC::GmFree((void*)tensor2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./addcmul_list_data/ && python3 compare_data.py 'float16'");
}

TEST_F(foreach_addcmul_list_test, test_case_int32_3) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_addcmul_list/tests/ut/op_kernel/addcmul_list_data ./");
    system("chmod -R 755 ./addcmul_list_data/");
    system("cd ./addcmul_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'int32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 3, 10);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachAddcmulList<int32_t>(shapeInfos, "int32", "input");
    uint8_t* tensor1 = CreateTensorListForeachAddcmulList<int32_t>(shapeInfos, "int32", "tensor1");
    uint8_t* tensor2 = CreateTensorListForeachAddcmulList<int32_t>(shapeInfos, "int32", "tensor2");
    uint8_t* out = CreateTensorListForeachAddcmulList<int32_t>(shapeInfos, "int32", "input");
    int32_t * scalar = (int32_t *)AscendC::GmAlloc(sizeof(int32_t) * 3);
    for (int i = 0; i < 3; i ++ ) {
        scalar[i] = (int32_t)3;
    }

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(foreach_addcmul_list, blockDim, x1, tensor1, tensor2, reinterpret_cast<uint8_t*>(scalar), out, workspace, tiling);

    FreeTensorListForeachAddcmulList<int32_t>(out, shapeInfos, "int32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)tensor1);
    AscendC::GmFree((void*)tensor2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./addcmul_list_data/ && python3 compare_data.py 'int32'");
}

TEST_F(foreach_addcmul_list_test, test_case_bloat16_4) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_addcmul_list/tests/ut/op_kernel/addcmul_list_data ./");
    system("chmod -R 755 ./addcmul_list_data/");
    system("cd ./addcmul_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'bfloat16_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 10);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachAddcmulList<bfloat16_t>(shapeInfos, "bfloat16_t", "input");
    uint8_t* tensor1 = CreateTensorListForeachAddcmulList<bfloat16_t>(shapeInfos, "bfloat16_t", "tensor1");
    uint8_t* tensor2 = CreateTensorListForeachAddcmulList<bfloat16_t>(shapeInfos, "bfloat16_t", "tensor2");
    uint8_t* out = CreateTensorListForeachAddcmulList<bfloat16_t>(shapeInfos, "bfloat16_t", "input");
    bfloat16_t * scalar = (bfloat16_t *)AscendC::GmAlloc(sizeof(bfloat16_t) * 3);
    for (int i = 0; i < 3; i ++ ) {
        scalar[i] = (bfloat16_t)3;
    }

    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(foreach_addcmul_list, blockDim, x1, tensor1, tensor2, reinterpret_cast<uint8_t*>(scalar), out, workspace, tiling);

    FreeTensorListForeachAddcmulList<bfloat16_t>(out, shapeInfos, "bfloat16_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)tensor1);
    AscendC::GmFree((void*)tensor2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./addcmul_list_data/ && python3 compare_data.py 'bfloat16_t'");
}

TEST_F(foreach_addcmul_list_test, test_case_float_5_for_not_aligned) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{7}, {9}, {17}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_addcmul_list/tests/ut/op_kernel/addcmul_list_data ./");
    system("chmod -R 755 ./addcmul_list_data/");
    system("cd ./addcmul_list_data/ && python3 gen_data.py '{{7}, {9}, {17}}' 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, 10);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachAddcmulList<float>(shapeInfos, "float32", "input");
    uint8_t* tensor1 = CreateTensorListForeachAddcmulList<float>(shapeInfos, "float32", "tensor1");
    uint8_t* tensor2 = CreateTensorListForeachAddcmulList<float>(shapeInfos, "float32", "tensor2");
    uint8_t* out = CreateTensorListForeachAddcmulList<float>(shapeInfos, "float32", "input");
    float * scalar = (float *)AscendC::GmAlloc(sizeof(float) * 3);
    for (int i = 0; i < 3; i ++ ) {
        scalar[i] = (float)3;
    }

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_addcmul_list, blockDim, x1, tensor1, tensor2, reinterpret_cast<uint8_t*>(scalar), out, workspace, tiling);

    FreeTensorListForeachAddcmulList<float>(out, shapeInfos, "float32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)tensor1);
    AscendC::GmFree((void*)tensor2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./addcmul_list_data/ && python3 compare_data.py 'float32'");
}