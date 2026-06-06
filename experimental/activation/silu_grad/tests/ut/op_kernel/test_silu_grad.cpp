/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif
#include "../../../op_kernel/silu_grad.h"
#include <cstdint>

extern "C" __global__ __aicore__ void silu_grad(
    GM_ADDR dy, GM_ADDR x, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling);

using namespace std;

class silu_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "silu_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "silu_grad_test TearDown\n" << endl;
    }
};

TEST_F(silu_grad_test, test_case_0)
{
    size_t elementNum = 32 * 4 * 4 * 4;
    size_t dyByteSize = elementNum * sizeof(float);
    size_t dxByteSize = elementNum * sizeof(float);
    size_t tilingDataSize = sizeof(SiluGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    auto* tilingDataFromBin = reinterpret_cast<SiluGradTilingData*>(tiling);
    tilingDataFromBin->smallCoreDataNum = elementNum;
    tilingDataFromBin->bigCoreDataNum = elementNum;
    tilingDataFromBin->tileDataNum = elementNum;
    tilingDataFromBin->tailBlockNum = 0;

    auto siluGradKernel = silu_grad;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(siluGradKernel, blockDim, dy, x, dx, workspace, (uint8_t*)(tilingDataFromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(silu_grad_test, test_case_1_double_buffer)
{
    size_t elementNum = 32 * 4 * 4 * 4;
    size_t dyByteSize = elementNum * sizeof(float);
    size_t dxByteSize = elementNum * sizeof(float);
    size_t tilingDataSize = sizeof(SiluGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    auto* tilingDataFromBin = reinterpret_cast<SiluGradTilingData*>(tiling);
    tilingDataFromBin->smallCoreDataNum = elementNum;
    tilingDataFromBin->bigCoreDataNum = elementNum;
    tilingDataFromBin->tileDataNum = 2048;
    tilingDataFromBin->tailBlockNum = 0;

    auto siluGradKernel = silu_grad;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(siluGradKernel, blockDim, dy, x, dx, workspace, (uint8_t*)(tilingDataFromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(dx);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
