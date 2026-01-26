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
 * \file test_foreach_non_finite_check_and_unscale.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "foreach_non_finite_check_and_unscale_tiling_funtion.h"
#include "tensor_list_operate.h"

using namespace ForeachNonFiniteCheckAndUnscaleTest;

extern "C" __global__ __aicore__ void foreach_non_finite_check_and_unscale(
    GM_ADDR scaled_grads, GM_ADDR found_inf, GM_ADDR inv_scale, GM_ADDR workspace, GM_ADDR tiling);

class foreach_non_finite_check_and_unscale_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "foreach_non_finite_check_and_unscale_test SetUp\n" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "foreach_non_finite_check_and_unscale_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_non_finite_check_and_unscale_test, test_case_half_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_non_finite_check_and_unscale/tests/ut/op_kernel/f_n_f_c_a_u_data ./");
    system("chmod -R 755 ./f_n_f_c_a_u_data/");
    system("cd ./f_n_f_c_a_u_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 2");

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachNonFiniteCheckAndUnscaleTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    ForeachNonFiniteCheckAndUnscaleTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, sizeof(half));
    uint32_t needBlockNum = tilingFuncObj.RunBigKernelTiling();
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachNonFiniteCheckAndUnscaleTilingData*>(tiling));

    uint8_t* x1 = CreateTensorList<half>(shapeInfos);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)x2) = 0;
    *((float*)x3) = 3;

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // ICPU_RUN_KF(foreach_non_finite_check_and_unscale, needBlockNum, x1, x2, x3, workspace, tiling);

    FreeTensorList<half>(x1, shapeInfos);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    // EXPECT_EQ(system("cd ./f_n_f_c_a_u_data/ && python3 compare_data.py 3 2"), 0);
}

TEST_F(foreach_non_finite_check_and_unscale_test, test_case_bfloat16_1)
{
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_non_finite_check_and_unscale/tests/ut/op_kernel/f_n_f_c_a_u_data ./");
    system("chmod -R 755 ./f_n_f_c_a_u_data/");
    system("cd ./f_n_f_c_a_u_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 3 3");

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachNonFiniteCheckAndUnscaleTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    ForeachNonFiniteCheckAndUnscaleTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, sizeof(bfloat16_t));
    uint32_t needBlockNum = tilingFuncObj.RunBigKernelTiling();
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachNonFiniteCheckAndUnscaleTilingData*>(tiling));

    uint8_t* x1 = CreateTensorList<bfloat16_t>(shapeInfos);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    *((float*)x2) = 0;
    *((float*)x3) = 3;

    ICPU_SET_TILING_KEY(3);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // ICPU_RUN_KF(foreach_non_finite_check_and_unscale, needBlockNum, x1, x2, x3, workspace, tiling);

    FreeTensorList<bfloat16_t>(x1, shapeInfos);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    // EXPECT_EQ(system("cd ./f_n_f_c_a_u_data/ && python3 compare_data.py 3 3"), 0);
}