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
 * \file test_foreach_neg.cpp
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

extern "C" __global__ __aicore__ void foreach_neg(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling);

class foreach_neg_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "foreach_neg_test SetUp\n" << std::endl; }
    static void TearDownTestCase() { std::cout << "foreach_neg_test TearDown\n" << std::endl; }
};

static void RunCastTiling(optiling::ForeachCommonTiling& tilingFuncObj, uint32_t blockDim)
{
    constexpr uint64_t ubSize = 196608;
    tilingFuncObj.RunBigKernelTiling(blockDim, ubSize);
    uint32_t totalSize = static_cast<uint32_t>(ubSize - sizeof(ForeachCommonTilingData));
    totalSize /= optiling::UB_DIVIDER_FOR_TEMP_CASTING;
    uint32_t canUseUbSize = totalSize / 2U - optiling::BYTE_PER_BLOCK;
    tilingFuncObj.inputsTensorUbSize = canUseUbSize / optiling::BYTE_BLOCK_FOR_BF16 * optiling::BYTE_BLOCK_FOR_BF16;
}

TEST_F(foreach_neg_test, test_case_float_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system("cp -rf "
           "../../../../foreach/foreach_neg/tests/ut/op_kernel/neg_data ./");
    system("chmod -R 755 ./neg_data/");
    system("cd ./neg_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float32'");
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

    uint8_t* x1 = CreateTensorListForeachNeg<float>(shapeInfos, "float32", true);  // input tensor
    uint8_t* x2 = CreateTensorListForeachNeg<float>(shapeInfos, "float32", false); // output tensor

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_neg, blockDim, x1, x2, workspace, tiling);

    FreeTensorListForeachNeg<float>(x2, shapeInfos, "float32", true);
    FreeTensorListForeachNeg<float>(x1, shapeInfos, "float32", false);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    EXPECT_EQ(system("cd ./neg_data/ && python3 compare_data.py 'float32'"), 0);
}

TEST_F(foreach_neg_test, test_case_float16_2)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system("cp -rf "
           "../../../../foreach/foreach_neg/tests/ut/op_kernel/neg_data ./");
    system("chmod -R 755 ./neg_data/");
    system("cd ./neg_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float16'");
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

    uint8_t* x1 = CreateTensorListForeachNeg<half>(shapeInfos, "float16", true);  // input tensor
    uint8_t* x2 = CreateTensorListForeachNeg<half>(shapeInfos, "float16", false); // output tensor

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(foreach_neg, blockDim, x1, x2, workspace, tiling);

    FreeTensorListForeachNeg<half>(x2, shapeInfos, "float16", true);
    FreeTensorListForeachNeg<half>(x1, shapeInfos, "float16", false);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    EXPECT_EQ(system("cd ./neg_data/ && python3 compare_data.py 'float16'"), 0);
}

TEST_F(foreach_neg_test, test_case_int32_3)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system("cp -rf "
           "../../../../foreach/foreach_neg/tests/ut/op_kernel/neg_data ./");
    system("chmod -R 755 ./neg_data/");
    system("cd ./neg_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'int32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 3, 7);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachNeg<int32_t>(shapeInfos, "int32", true);
    uint8_t* x2 = CreateTensorListForeachNeg<int32_t>(shapeInfos, "int32", false);

    ICPU_SET_TILING_KEY(3); // int32
    ICPU_RUN_KF(foreach_neg, blockDim, x1, x2, workspace, tiling);

    FreeTensorListForeachNeg<int32_t>(x2, shapeInfos, "int32", true);
    FreeTensorListForeachNeg<int32_t>(x1, shapeInfos, "int32", false);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    EXPECT_EQ(system("cd ./neg_data/ && python3 compare_data.py 'int32'"), 0);
}

TEST_F(foreach_neg_test, test_case_bfloat16_4)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system("cp -rf "
           "../../../../foreach/foreach_neg/tests/ut/op_kernel/neg_data ./");
    system("chmod -R 755 ./neg_data/");
    system("cd ./neg_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'bfloat16_t'");
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

    uint8_t* x1 = CreateTensorListForeachNeg<bfloat16_t>(shapeInfos, "bfloat16_t", true);
    uint8_t* x2 = CreateTensorListForeachNeg<bfloat16_t>(shapeInfos, "bfloat16_t", false);

    ICPU_SET_TILING_KEY(4); // bfloat16_t
    ICPU_RUN_KF(foreach_neg, blockDim, x1, x2, workspace, tiling);

    FreeTensorListForeachNeg<bfloat16_t>(x2, shapeInfos, "bfloat16_t", true);
    FreeTensorListForeachNeg<bfloat16_t>(x1, shapeInfos, "bfloat16_t", false);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    EXPECT_EQ(system("cd ./neg_data/ && python3 compare_data.py 'bfloat16_t'"), 0);
}

TEST_F(foreach_neg_test, test_case_int16_5)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{125, 58}, {12}, {3, 2, 2, 2}};
    system("cp -rf "
           "../../../../foreach/foreach_neg/tests/ut/op_kernel/neg_data ./");
    system("chmod -R 755 ./neg_data/");
    system("cd ./neg_data/ && python3 gen_data.py '{{125, 58}, {12}, {3, 2, 2, 2}}' 'int16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 5, 7);
    RunCastTiling(tilingFuncObj, blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachNeg<int16_t>(shapeInfos, "int16", true);
    uint8_t* x2 = CreateTensorListForeachNeg<int16_t>(shapeInfos, "int16", false);

    ICPU_SET_TILING_KEY(5);
    ICPU_RUN_KF(foreach_neg, blockDim, x1, x2, workspace, tiling);

    FreeTensorListForeachNeg<int16_t>(x2, shapeInfos, "int16", true);
    FreeTensorListForeachNeg<int16_t>(x1, shapeInfos, "int16", false);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    EXPECT_EQ(system("cd ./neg_data/ && python3 compare_data.py 'int16'"), 0);
}

TEST_F(foreach_neg_test, test_case_int8_6)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{125, 58}, {12}, {3, 2, 2, 2}};
    system("cp -rf "
           "../../../../foreach/foreach_neg/tests/ut/op_kernel/neg_data ./");
    system("chmod -R 755 ./neg_data/");
    system("cd ./neg_data/ && python3 gen_data.py '{{125, 58}, {12}, {3, 2, 2, 2}}' 'int8'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 7, 7);
    RunCastTiling(tilingFuncObj, blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachNeg<int8_t>(shapeInfos, "int8", true);
    uint8_t* x2 = CreateTensorListForeachNeg<int8_t>(shapeInfos, "int8", false);

    ICPU_SET_TILING_KEY(7);
    ICPU_RUN_KF(foreach_neg, blockDim, x1, x2, workspace, tiling);

    FreeTensorListForeachNeg<int8_t>(x2, shapeInfos, "int8", true);
    FreeTensorListForeachNeg<int8_t>(x1, shapeInfos, "int8", false);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    EXPECT_EQ(system("cd ./neg_data/ && python3 compare_data.py 'int8'"), 0);
}

TEST_F(foreach_neg_test, test_case_uint8_7)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{125, 58}, {12}, {3, 2, 2, 2}};
    system("cp -rf "
           "../../../../foreach/foreach_neg/tests/ut/op_kernel/neg_data ./");
    system("chmod -R 755 ./neg_data/");
    system("cd ./neg_data/ && python3 gen_data.py '{{125, 58}, {12}, {3, 2, 2, 2}}' 'uint8'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 8, 7);
    RunCastTiling(tilingFuncObj, blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachNeg<uint8_t>(shapeInfos, "uint8", true);
    uint8_t* x2 = CreateTensorListForeachNeg<uint8_t>(shapeInfos, "uint8", false);

    ICPU_SET_TILING_KEY(8);
    ICPU_RUN_KF(foreach_neg, blockDim, x1, x2, workspace, tiling);

    FreeTensorListForeachNeg<uint8_t>(x2, shapeInfos, "uint8", true);
    FreeTensorListForeachNeg<uint8_t>(x1, shapeInfos, "uint8", false);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    EXPECT_EQ(system("cd ./neg_data/ && python3 compare_data.py 'uint8'"), 0);
}
