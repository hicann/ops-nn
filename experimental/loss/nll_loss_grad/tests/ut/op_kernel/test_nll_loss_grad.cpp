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
#include <cstdint>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <string>
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"
#endif
#include "../../../op_kernel/nll_loss_grad.cpp"
#include "../../../op_kernel/nll_loss_grad_tiling_data.h"

using namespace std;

class NllLossGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment("experimental/loss/nll_loss_grad/tests/ut/op_kernel/nll_loss_grad_data",
                                        "nll_loss_grad_data");
    }
    static void TearDownTestCase() {}
};

TEST_F(NllLossGradTest, test_case_float32_mean)
{
    kernel_ut::RunGenData("./nll_loss_grad_data", {"'(4, 7)'", "float32", "mean", "-100"});

    uint32_t N = 4;
    uint32_t C = 7;
    uint32_t totalNum = N * C;

    size_t xByteSize = totalNum * sizeof(float);
    size_t yGradByteSize = 1 * sizeof(float);
    size_t targetByteSize = N * sizeof(int32_t);
    size_t weightByteSize = C * sizeof(float);
    size_t totalWeightByteSize = 1 * sizeof(float);
    size_t xGradByteSize = totalNum * sizeof(float);

    std::string xFile = "./nll_loss_grad_data/float32_x_t_nll_loss_grad.bin";
    uint8_t* xBuf = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(xFile, xByteSize, xBuf, xByteSize);

    std::string yGradFile = "./nll_loss_grad_data/float32_y_grad_t_nll_loss_grad.bin";
    uint8_t* yGradBuf = (uint8_t*)AscendC::GmAlloc(yGradByteSize);
    ReadFile(yGradFile, yGradByteSize, yGradBuf, yGradByteSize);

    std::string targetFile = "./nll_loss_grad_data/float32_target_t_nll_loss_grad.bin";
    uint8_t* targetBuf = (uint8_t*)AscendC::GmAlloc(targetByteSize);
    ReadFile(targetFile, targetByteSize, targetBuf, targetByteSize);

    std::string weightFile = "./nll_loss_grad_data/float32_weight_t_nll_loss_grad.bin";
    uint8_t* weightBuf = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    ReadFile(weightFile, weightByteSize, weightBuf, weightByteSize);

    std::string twFile = "./nll_loss_grad_data/float32_total_weight_t_nll_loss_grad.bin";
    uint8_t* twBuf = (uint8_t*)AscendC::GmAlloc(totalWeightByteSize);
    ReadFile(twFile, totalWeightByteSize, twBuf, totalWeightByteSize);

    uint8_t* xGradBuf = (uint8_t*)AscendC::GmAlloc(xGradByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(NllLossGradTilingData));

    NllLossGradTilingData* tilingData = reinterpret_cast<NllLossGradTilingData*>(tl);
    tilingData->nDim = 4;
    tilingData->cDim = 7;
    tilingData->coreNum = 1;
    tilingData->reduction = 2;
    tilingData->ignoreIndex = -100;
    tilingData->bigWeight = 0;
    tilingData->maxLine = 4;
    tilingData->lowerLine = 4;
    tilingData->redundantLine = 0;
    tilingData->lineTile = 4;
    tilingData->cAlign = 8;
    tilingData->outUbSize = 32;
    tilingData->colTile = 0;
    tilingData->moveOutTime = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(nll_loss_grad<0>, 1, xBuf, yGradBuf, targetBuf, weightBuf, twBuf, xGradBuf, ws, tl);

    std::string outFile = "./nll_loss_grad_data/float32_output_x_grad_t_nll_loss_grad.bin";
    WriteFile(outFile, xGradBuf, xGradByteSize);

    AscendC::GmFree((void*)xBuf);
    AscendC::GmFree((void*)yGradBuf);
    AscendC::GmFree((void*)targetBuf);
    AscendC::GmFree((void*)weightBuf);
    AscendC::GmFree((void*)twBuf);
    AscendC::GmFree((void*)xGradBuf);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./nll_loss_grad_data", {"float32"});
}
