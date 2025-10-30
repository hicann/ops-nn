/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_ctc_loss_v3_grad_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" void ctc_loss_v3_grad(
    GM_ADDR grad_out, GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths,
    GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, GM_ADDR grad, GM_ADDR workspace, GM_ADDR tiling);

class CTCLossV3GradKernel : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "CTCLossV3Grad Kernel SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "CTCLossV3Grad Kernel TearDown\n" << endl;
    }
};

TEST_F(CTCLossV3GradKernel, ctc_loss_v3_grad_case_float_0)
{
    size_t gradoutputSize = 4 * sizeof(float);
    size_t logProbsSize = 12 * 4 * 5 * sizeof(float);
    size_t targetsSize = 4 * 7 * sizeof(int64_t);
    size_t inputLengthsSize = 10 * 10 * 10 * 10 * sizeof(float);
    size_t targetLengthSize = 2 * 3 * 1 * 5 * sizeof(float);
    size_t lossSize = 4 * sizeof(float);
    size_t logAlphaSize = 4 * 12 * 16 * sizeof(float);
    size_t gradSize = 12 * 4 * 5 * sizeof(float);
    size_t tilingDataSize = sizeof(CTCLossV3GradTilingDataTest) + 1024;

    uint8_t* gradoutput = (uint8_t*)AscendC::GmAlloc(gradoutputSize);
    uint8_t* logProbs = (uint8_t*)AscendC::GmAlloc(logProbsSize);
    uint8_t* targets = (uint8_t*)AscendC::GmAlloc(targetsSize);
    uint8_t* inputLengths = (uint8_t*)AscendC::GmAlloc(inputLengthsSize);
    uint8_t* targetLength = (uint8_t*)AscendC::GmAlloc(targetLengthSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossSize);
    uint8_t* logAlpha = (uint8_t*)AscendC::GmAlloc(logAlphaSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    CTCLossV3GradTilingDataTest* tilingDatafromBin = reinterpret_cast<CTCLossV3GradTilingDataTest*>(tiling);

    tilingDatafromBin->sliceLength = 64;
    tilingDatafromBin->sliceLengthTail = 5;
    tilingDatafromBin->probSliceNum = 1;
    tilingDatafromBin->alphaLength = 16;
    tilingDatafromBin->maxInputLength = 12;
    tilingDatafromBin->symbolSet = 5;
    tilingDatafromBin->batchSize = 4;
    tilingDatafromBin->targetsDimNum = 2;
    tilingDatafromBin->sDimRange = 7;
    tilingDatafromBin->targetsNum = 28;
    tilingDatafromBin->taskPerCore = 0;
    tilingDatafromBin->taskTailCore = 4;
    tilingDatafromBin->BLANK = 0;
    tilingDatafromBin->zeroInfinity = 0;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(
        ctc_loss_v3_grad, blockDim, gradoutput, logProbs, targets, inputLengths, targetLength, loss, logAlpha, grad,
        workspace, (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(gradoutput);
    AscendC::GmFree(logProbs);
    AscendC::GmFree(targets);
    AscendC::GmFree(inputLengths);
    AscendC::GmFree(targetLength);
    AscendC::GmFree(loss);
    AscendC::GmFree(logAlpha);
    AscendC::GmFree(grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}