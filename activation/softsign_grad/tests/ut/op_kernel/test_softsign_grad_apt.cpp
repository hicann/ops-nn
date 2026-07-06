/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/arch35/softsign_grad_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void softsign_grad(GM_ADDR gradients, GM_ADDR features, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling);

class SoftsignGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SoftsignGradKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "SoftsignGradKernelTest TearDown" << endl; }
};

TEST_F(SoftsignGradKernelTest, test_fp32_basic)
{
    constexpr size_t numElements = 256;
    size_t dataSize = numElements * sizeof(float);
    size_t tilingSize = sizeof(SoftsignGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    SoftsignGradTilingData* tilingData = reinterpret_cast<SoftsignGradTilingData*>(tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = numElements;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(softsign_grad, blockDim, gradients, features, output, workspace, tiling);

    AscendC::GmFree(gradients);
    AscendC::GmFree(features);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SoftsignGradKernelTest, test_fp32_large)
{
    constexpr size_t numElements = 2048;
    size_t dataSize = numElements * sizeof(float);
    size_t tilingSize = sizeof(SoftsignGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    SoftsignGradTilingData* tilingData = reinterpret_cast<SoftsignGradTilingData*>(tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = 1024;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(softsign_grad, blockDim, gradients, features, output, workspace, tiling);

    AscendC::GmFree(gradients);
    AscendC::GmFree(features);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
