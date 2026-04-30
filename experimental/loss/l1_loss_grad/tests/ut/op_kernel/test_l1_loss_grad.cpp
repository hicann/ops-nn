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
#include "../../../op_kernel/l1_loss_grad.h"
#include <cstdint>

extern "C" __global__ __aicore__ void l1_loss_grad(
    GM_ADDR grads, GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

using namespace std;

class l1_loss_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l1_loss_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l1_loss_grad_test TearDown\n" << endl;
    }
};

TEST_F(l1_loss_grad_test, test_case_0)
{
    size_t gradsByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t predictByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t labelByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tilingDataSize = sizeof(L1LossGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* grads = (uint8_t*)AscendC::GmAlloc(gradsByteSize);
    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictByteSize);
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(labelByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    auto* tilingDataFromBin = reinterpret_cast<L1LossGradTilingData*>(tiling);
    tilingDataFromBin->smallCoreDataNum = 2048;
    tilingDataFromBin->bigCoreDataNum = 2112;
    tilingDataFromBin->tileDataNum = 4032;
    tilingDataFromBin->smallTailDataNum = 2048;
    tilingDataFromBin->bigTailDataNum = 2112;
    tilingDataFromBin->finalSmallTileNum = 1;
    tilingDataFromBin->finalBigTileNum = 1;
    tilingDataFromBin->tailBlockNum = 0;
    tilingDataFromBin->scaleValue = 1.0f;

    auto l1LossGradKernel = l1_loss_grad;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(l1LossGradKernel, blockDim, grads, predict, label, y, workspace, (uint8_t*)(tilingDataFromBin));

    AscendC::GmFree(grads);
    AscendC::GmFree(predict);
    AscendC::GmFree(label);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}