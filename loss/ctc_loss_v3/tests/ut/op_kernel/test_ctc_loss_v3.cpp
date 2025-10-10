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
#include "test_ctc_loss_v3_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void ctc_loss_v3(
    GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths, GM_ADDR neg_log_likelihood,
    GM_ADDR log_alpha, GM_ADDR workspace, GM_ADDR tiling);

class CTCLossV3Kernel : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "CTCLossV3 Kernel SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "CTCLossV3 Kernel TearDown\n" << endl;
    }
};

template <typename T>
void TestCTCLossV3Kernel(
    vector<uint64_t>& logProbsShape, vector<uint64_t>& targetsShape, vector<uint64_t>& inputLengthsShape,
    vector<uint64_t>& targetLengthsShape, vector<uint64_t>& lossShape, vector<uint64_t>& logAlphaShape,
    uint64_t workspaceSize, uint64_t blockDim, uint64_t tilingKey)
{
    uint64_t N = logProbsShape[1];
    uint64_t Time = logProbsShape[0];
    uint64_t C = logProbsShape[2];
    uint64_t alpha = logAlphaShape[2];
    uint64_t S = (logAlphaShape[2] - 1) / 2;

    size_t logProbsByteSize = N * Time * C * sizeof(T);
    size_t targetsByteSize = N * S * sizeof(int64_t);
    size_t inputLengthsByteSize = N * sizeof(int64_t);
    size_t targetLengthsByteSize = N * sizeof(int64_t);
    size_t lossByteSize = N * sizeof(T);
    size_t logAlphaByteSize = N * S * sizeof(T);
    size_t tilingDataSize = sizeof(CTCLossV3TilingDataTest);

    uint8_t* log_probs = (uint8_t*)AscendC::GmAlloc(logProbsByteSize);
    uint8_t* targets = (uint8_t*)AscendC::GmAlloc(targetsByteSize);
    uint8_t* input_lengths = (uint8_t*)AscendC::GmAlloc(inputLengthsByteSize);
    uint8_t* target_lengths = (uint8_t*)AscendC::GmAlloc(targetLengthsByteSize);
    int64_t two = 2;
    for (size_t i = 0; i < targetLengthsByteSize; i += sizeof(int64_t)) {
        std::memcpy(target_lengths + i, &two, sizeof(int64_t));
    }
    for (size_t i = 0; i < inputLengthsByteSize; i += sizeof(int64_t)) {
        std::memcpy(input_lengths + i, &two, sizeof(int64_t));
    }
    for (size_t i = 0; i < targetsByteSize; i += sizeof(int64_t)) {
        std::memcpy(targets + i, &two, sizeof(int64_t));
    }
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossByteSize);
    uint8_t* log_alpha = (uint8_t*)AscendC::GmAlloc(logAlphaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);
    CTCLossV3TilingDataTest* tilingDatafromBin = reinterpret_cast<CTCLossV3TilingDataTest*>(tiling);

    tilingDatafromBin->sliceLength = 1024;
    tilingDatafromBin->sliceLengthTail = 1024;
    tilingDatafromBin->probSliceNum = 1;
    // tilingDatafromBin->alphaLength = alpha;
    tilingDatafromBin->maxTargetLength = S;
    tilingDatafromBin->timeStep = Time;
    tilingDatafromBin->symbleSet = C;
    tilingDatafromBin->batchSize = N;
    tilingDatafromBin->targetsDimNum = 2;
    tilingDatafromBin->targetsDimLength = S;
    tilingDatafromBin->targetsNum = N * S;
    tilingDatafromBin->taskPerCore = 1;
    tilingDatafromBin->taskTailCore = 0;
    tilingDatafromBin->blank = 0;
    tilingDatafromBin->zeroInfinity = 0;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        ctc_loss_v3, blockDim, log_probs, targets, input_lengths, target_lengths, loss, log_alpha, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(log_probs);
    AscendC::GmFree(targets);
    AscendC::GmFree(input_lengths);
    AscendC::GmFree(target_lengths);
    AscendC::GmFree(loss);
    AscendC::GmFree(log_alpha);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(CTCLossV3Kernel, ctc_loss_v3_case_float_0)
{
    vector<uint64_t> logProbsShape = {32, 1, 1024};
    vector<uint64_t> targetsShape = {1, 32};
    vector<uint64_t> inputLengthsShape = {1};
    vector<uint64_t> targetLengthsShape = {1};
    vector<uint64_t> lossShape = {1};
    vector<uint64_t> logAlphaShape = {1, 32, 65};
    uint64_t workspaceSize = (65 + 32) * 4 * 4 * 1;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 0;
    TestCTCLossV3Kernel<float>(
        logProbsShape, targetsShape, inputLengthsShape, targetLengthsShape, lossShape, logAlphaShape, workspaceSize,
        blockDim, tilingKey);
}

TEST_F(CTCLossV3Kernel, ctc_loss_v3_case_float_large_symbleSet_0)
{
    vector<uint64_t> logProbsShape = {32, 1, 25240};
    vector<uint64_t> targetsShape = {1, 32};
    vector<uint64_t> inputLengthsShape = {1};
    vector<uint64_t> targetLengthsShape = {1};
    vector<uint64_t> lossShape = {1};
    vector<uint64_t> logAlphaShape = {1, 32, 65};
    uint64_t workspaceSize = (65 + 32) * 4 * 4 * 1;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 0;
    TestCTCLossV3Kernel<float>(
        logProbsShape, targetsShape, inputLengthsShape, targetLengthsShape, lossShape, logAlphaShape, workspaceSize,
        blockDim, tilingKey);
}