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
 * \file test_foreach_zero_inplace.cpp
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

extern "C" __global__ __aicore__ void foreach_zero_inplace(GM_ADDR x, GM_ADDR xRef, GM_ADDR workspace, GM_ADDR tiling);

class foreach_zero_inplace_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "foreach_zero_inplace_test SetUp\n" << std::endl; }
    static void TearDownTestCase() { std::cout << "foreach_zero_inplace_test TearDown\n" << std::endl; }
};

TEST_F(foreach_zero_inplace_test, test_case_float_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    ASSERT_EQ(system("cp -rf "
                     "../../../../foreach/foreach_zero_inplace/tests/ut/op_kernel/zero_inplace_data ./"),
              0);
    ASSERT_EQ(system("chmod -R 755 ./zero_inplace_data/"), 0);
    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float32'"),
              0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, 1); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachZeroInplace<float>(shapeInfos, "float32"); // input tensor
    ASSERT_NE(x1, nullptr);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_zero_inplace, blockDim, x1, x1, workspace, tiling);

    FreeTensorListForeachZeroInplace<float>(x1, shapeInfos, "float32");
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 compare_data.py 'float32'"), 0);
}

TEST_F(foreach_zero_inplace_test, test_case_float_ref_output)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    ASSERT_EQ(system("cp -rf "
                     "../../../../foreach/foreach_zero_inplace/tests/ut/op_kernel/zero_inplace_data ./"),
              0);
    ASSERT_EQ(system("chmod -R 755 ./zero_inplace_data/"), 0);
    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float32'"),
              0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    constexpr uint32_t blockDim = 4;
    constexpr size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(sysWorkspaceSize));
    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(ForeachCommonTilingData)));

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, 1);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x = CreateTensorListForeachZeroInplace<float>(shapeInfos, "float32");
    uint8_t* xRef = CreateTensorListForeachZeroInplace<float>(shapeInfos, "float32");
    ASSERT_NE(x, nullptr);
    ASSERT_NE(xRef, nullptr);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_zero_inplace, blockDim, x, xRef, workspace, tiling);

    FreeTensorListForeachZeroInplace<float>(x, shapeInfos, "float32", false);
    FreeTensorListForeachZeroInplace<float>(xRef, shapeInfos, "float32");
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 compare_data.py 'float32'"), 0);
}

TEST_F(foreach_zero_inplace_test, test_case_float16_2)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    ASSERT_EQ(system("cp -rf "
                     "../../../../foreach/foreach_zero_inplace/tests/ut/op_kernel/zero_inplace_data ./"),
              0);
    ASSERT_EQ(system("chmod -R 755 ./zero_inplace_data/"), 0);
    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float16'"),
              0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 2, 1); // shape info, dataType, the tiling code
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachZeroInplace<half>(shapeInfos, "float16"); // input tensor
    ASSERT_NE(x1, nullptr);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(foreach_zero_inplace, blockDim, x1, x1, workspace, tiling);

    FreeTensorListForeachZeroInplace<half>(x1, shapeInfos, "float16");
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 compare_data.py 'float16'"), 0);
}

TEST_F(foreach_zero_inplace_test, test_case_int32_3)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    ASSERT_EQ(system("cp -rf "
                     "../../../../foreach/foreach_zero_inplace/tests/ut/op_kernel/zero_inplace_data ./"),
              0);
    ASSERT_EQ(system("chmod -R 755 ./zero_inplace_data/"), 0);
    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'int32'"), 0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 3, 1);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachZeroInplace<int32_t>(shapeInfos, "int32");
    ASSERT_NE(x1, nullptr);

    ICPU_SET_TILING_KEY(3); // int32
    ICPU_RUN_KF(foreach_zero_inplace, blockDim, x1, x1, workspace, tiling);

    FreeTensorListForeachZeroInplace<int32_t>(x1, shapeInfos, "int32");
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 compare_data.py 'int32'"), 0);
}

TEST_F(foreach_zero_inplace_test, test_case_int16_5)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    ASSERT_EQ(system("cp -rf "
                     "../../../../foreach/foreach_zero_inplace/tests/ut/op_kernel/zero_inplace_data ./"),
              0);
    ASSERT_EQ(system("chmod -R 755 ./zero_inplace_data/"), 0);
    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'int16'"), 0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 5, 1);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachZeroInplace<int16_t>(shapeInfos, "int16");
    ASSERT_NE(x1, nullptr);

    ICPU_SET_TILING_KEY(5); // int16
    ICPU_RUN_KF(foreach_zero_inplace, blockDim, x1, x1, workspace, tiling);

    FreeTensorListForeachZeroInplace<int16_t>(x1, shapeInfos, "int16");
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 compare_data.py 'int16'"), 0);
}

TEST_F(foreach_zero_inplace_test, test_case_bfloat16_4)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    ASSERT_EQ(system("cp -rf "
                     "../../../../foreach/foreach_zero_inplace/tests/ut/op_kernel/zero_inplace_data ./"),
              0);
    ASSERT_EQ(system("chmod -R 755 ./zero_inplace_data/"), 0);
    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'bfloat16_t'"),
              0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 4, 1);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachZeroInplace<bfloat16_t>(shapeInfos, "bfloat16_t");
    ASSERT_NE(x1, nullptr);

    ICPU_SET_TILING_KEY(4); // bfloat16_t
    ICPU_RUN_KF(foreach_zero_inplace, blockDim, x1, x1, workspace, tiling);

    FreeTensorListForeachZeroInplace<bfloat16_t>(x1, shapeInfos, "bfloat16_t");
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    ASSERT_EQ(system("cd ./zero_inplace_data/ && python3 compare_data.py 'bfloat16_t'"), 0);
}
