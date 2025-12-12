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
 * \file test_foreach_copy.cpp
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

extern "C" __global__ __aicore__ void foreach_copy(GM_ADDR inputs, GM_ADDR outputs,
                                                                GM_ADDR workspace,
                                                                GM_ADDR tiling);

class foreach_copy_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "foreach_copy_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "foreach_copy_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_copy_test, test_case_float_1) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<float>(shapeInfos, "float32"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<float>(shapeInfos, "float32"); // output tensor

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<float>(x2, shapeInfos, "float32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'float32'");
}

TEST_F(foreach_copy_test, test_case_float16_2) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 2, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<half>(shapeInfos, "float16"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<half>(shapeInfos, "float16"); // output tensor

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<half>(x2, shapeInfos, "float16");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'float16'");
}

TEST_F(foreach_copy_test, test_case_bfloat16_3) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'bfloat16_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<bfloat16_t>(shapeInfos, "bfloat16_t"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<bfloat16_t>(shapeInfos, "bfloat16_t"); // output tensor

    ICPU_SET_TILING_KEY(4);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<bfloat16_t>(x2, shapeInfos, "bfloat16_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'bfloat16_t'");
}

TEST_F(foreach_copy_test, test_case_int8_4) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'int8_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<int8_t>(shapeInfos, "int8_t"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<int8_t>(shapeInfos, "int8_t"); // output tensor

    ICPU_SET_TILING_KEY(7);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<int8_t>(x2, shapeInfos, "int8_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'int8_t'");
}

TEST_F(foreach_copy_test, test_case_uint8_5) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{64, 32}, {8, 64}, {16, 64}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{64, 32}, {8, 64}, {16, 64}}' 'uint8_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<uint8_t>(shapeInfos, "uint8_t"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<uint8_t>(shapeInfos, "uint8_t"); // output tensor

    ICPU_SET_TILING_KEY(8);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<uint8_t>(x2, shapeInfos, "uint8_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'uint8_t'");
}

TEST_F(foreach_copy_test, test_case_int16_6) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'int16_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<int16_t>(shapeInfos, "int16_t"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<int16_t>(shapeInfos, "int16_t"); // output tensor

    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<int16_t>(x2, shapeInfos, "int16_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'int16_t'");
}

TEST_F(foreach_copy_test, test_case_uint16_7) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'uint16_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<uint16_t>(shapeInfos, "uint16_t"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<uint16_t>(shapeInfos, "uint16_t"); // output tensor

    ICPU_SET_TILING_KEY(6);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<uint16_t>(x2, shapeInfos, "uint16_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'uint16_t'");
}

TEST_F(foreach_copy_test, test_case_int32_8) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'int32_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<int32_t>(shapeInfos, "int32_t"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<int32_t>(shapeInfos, "int32_t"); // output tensor

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<int32_t>(x2, shapeInfos, "int32_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'int32_t'");
}

TEST_F(foreach_copy_test, test_case_uint32_9) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'uint32_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<uint32_t>(shapeInfos, "uint32_t"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<uint32_t>(shapeInfos, "uint32_t"); // output tensor

    ICPU_SET_TILING_KEY(9);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<uint32_t>(x2, shapeInfos, "uint32_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'uint32_t'");
}

TEST_F(foreach_copy_test, test_case_int64_10) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'int64_t'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 10, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<int64_t>(shapeInfos, "int64_t"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<int64_t>(shapeInfos, "int64_t"); // output tensor

    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<int64_t>(x2, shapeInfos, "int64_t");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'int64_t'");
}

TEST_F(foreach_copy_test, test_case_float64_11) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float64'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 11, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<double>(shapeInfos, "float64"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<double>(shapeInfos, "float64"); // output tensor

    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<double>(x2, shapeInfos, "float64");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'float64'");
}

TEST_F(foreach_copy_test, test_case_bool_12) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_copy/tests/ut/op_kernel/copy_data ./");
    system("chmod -R 755 ./copy_data/");
    system("cd ./copy_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'bool'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 12, 32); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateCopyTensorList<bool>(shapeInfos, "bool"); // input tensor
    uint8_t* x2 = CreateCopyTensorList<bool>(shapeInfos, "bool"); // output tensor

    ICPU_SET_TILING_KEY(12);
    ICPU_RUN_KF(foreach_copy, blockDim, x1, x2, workspace, tiling);

    FreeCopyTensorList<bool>(x2, shapeInfos, "bool");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./copy_data/ && python3 compare_data.py 'bool'");
}
