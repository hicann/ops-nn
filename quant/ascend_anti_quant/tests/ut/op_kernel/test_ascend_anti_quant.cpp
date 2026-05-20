/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_ascend_anti_quant.cpp
 * \brief CPU-side mock invocation of the AscendAntiQuant regbase kernel.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/arch35/ascend_anti_quant.cpp"

using namespace std;

extern "C" __global__ __aicore__ void ascend_anti_quant(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class ascend_anti_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ascend_anti_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ascend_anti_quant_test TearDown\n" << endl;
    }
};

static void FillBaseTiling(AscendAntiQuantTilingData* tilingDatafromBin, int64_t elemNum, int64_t blockNum)
{
    auto& bt = tilingDatafromBin->baseTiling;
    bt.dim0 = elemNum;
    bt.coreNum = static_cast<int32_t>(blockNum);
    bt.ubFormer = 1024;
    bt.blockFormer = elemNum / blockNum;
    bt.blockNum = blockNum;
    bt.ubLoopOfFormerBlock = 1;
    bt.ubLoopOfTailBlock = 1;
    bt.ubTailOfFormerBlock = 0;
    bt.ubTailOfTailBlock = 0;
    bt.elemNum = elemNum;
    bt.scheMode = 0;
}

TEST_F(ascend_anti_quant_test, test_case_no_sqrt_mode)
{
    int64_t elemNum = 128 * 256;
    uint32_t blockDim = 2;

    size_t inputXSize = elemNum * sizeof(int8_t);
    size_t outputYSize = elemNum * sizeof(half);
    size_t tiling_data_size = sizeof(AscendAntiQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    AscendAntiQuantTilingData* tilingDatafromBin = reinterpret_cast<AscendAntiQuantTilingData*>(tiling);
    FillBaseTiling(tilingDatafromBin, elemNum, blockDim);
    tilingDatafromBin->scale = 0.5f;
    tilingDatafromBin->offset = 1.0f;
    tilingDatafromBin->sqrtMode = 0;
    tilingDatafromBin->reserved = 0;

    ICPU_SET_TILING_KEY(0);

    auto kernel_lambda = [](GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::ascend_anti_quant<0>(x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, x, y, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}

TEST_F(ascend_anti_quant_test, test_case_sqrt_mode)
{
    int64_t elemNum = 128 * 256;
    uint32_t blockDim = 2;

    size_t inputXSize = elemNum * sizeof(int8_t);
    size_t outputYSize = elemNum * sizeof(half);
    size_t tiling_data_size = sizeof(AscendAntiQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    AscendAntiQuantTilingData* tilingDatafromBin = reinterpret_cast<AscendAntiQuantTilingData*>(tiling);
    FillBaseTiling(tilingDatafromBin, elemNum, blockDim);
    tilingDatafromBin->scale = 2.0f;
    tilingDatafromBin->offset = 0.0f;
    tilingDatafromBin->sqrtMode = 1;
    tilingDatafromBin->reserved = 0;

    ICPU_SET_TILING_KEY(1);

    auto kernel_lambda = [](GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::ascend_anti_quant<1>(x, y, workSpace, tiling);
    };
    ICPU_RUN_KF(kernel_lambda, blockDim, x, y, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
}
