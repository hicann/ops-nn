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
#include "sigmoid_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

#define SIGMOID_F16_TILING_KEY 100000000000100
#define SIGMOID_BF16_TILING_KEY 200000000000100
#define SIGMOID_F32_TILING_KEY 300000000000100

class sigmoid_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "sigmoid_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "sigmoid_test TearDown\n" << endl; }
};

TEST_F(sigmoid_test, test_case_fp32)
{
    size_t xByteSize = 256 * sizeof(float);
    size_t yByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(SigmoidTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    SigmoidTilingData* tilingDatafromBin = reinterpret_cast<SigmoidTilingData*>(tiling);

    tilingDatafromBin->dim0 = 256;
    tilingDatafromBin->blockFormer = 256;
    tilingDatafromBin->ubFormer = 256;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 256;
    tilingDatafromBin->ubTailOfTailBlock = 256;
    tilingDatafromBin->elemNum = 256;

    auto KernelSigmoid = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid(x, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(SIGMOID_F32_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoid, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sigmoid_test, test_case_fp16)
{
    size_t xByteSize = 256 * sizeof(half);
    size_t yByteSize = 256 * sizeof(half);
    size_t tiling_data_size = sizeof(SigmoidTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    SigmoidTilingData* tilingDatafromBin = reinterpret_cast<SigmoidTilingData*>(tiling);

    tilingDatafromBin->dim0 = 256;
    tilingDatafromBin->blockFormer = 256;
    tilingDatafromBin->ubFormer = 256;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 256;
    tilingDatafromBin->ubTailOfTailBlock = 256;
    tilingDatafromBin->elemNum = 256;

    auto KernelSigmoid = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid(x, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(SIGMOID_F16_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoid, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sigmoid_test, test_case_bf16)
{
    size_t xByteSize = 256 * sizeof(bfloat16_t);
    size_t yByteSize = 256 * sizeof(bfloat16_t);
    size_t tiling_data_size = sizeof(SigmoidTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    SigmoidTilingData* tilingDatafromBin = reinterpret_cast<SigmoidTilingData*>(tiling);

    tilingDatafromBin->dim0 = 256;
    tilingDatafromBin->blockFormer = 256;
    tilingDatafromBin->ubFormer = 256;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 256;
    tilingDatafromBin->ubTailOfTailBlock = 256;
    tilingDatafromBin->elemNum = 256;

    auto KernelSigmoid = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid(x, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(SIGMOID_BF16_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoid, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sigmoid_test, test_case_fp32_large)
{
    size_t xByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(SigmoidTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    SigmoidTilingData* tilingDatafromBin = reinterpret_cast<SigmoidTilingData*>(tiling);

    tilingDatafromBin->dim0 = 2048;
    tilingDatafromBin->blockFormer = 2048;
    tilingDatafromBin->ubFormer = 2048;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 2048;
    tilingDatafromBin->ubTailOfTailBlock = 2048;
    tilingDatafromBin->elemNum = 2048;

    auto KernelSigmoid = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid(x, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(SIGMOID_F32_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoid, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sigmoid_test, test_case_fp16_large)
{
    size_t xByteSize = 32 * 4 * 4 * 4 * sizeof(half);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(half);
    size_t tiling_data_size = sizeof(SigmoidTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    SigmoidTilingData* tilingDatafromBin = reinterpret_cast<SigmoidTilingData*>(tiling);

    tilingDatafromBin->dim0 = 2048;
    tilingDatafromBin->blockFormer = 2048;
    tilingDatafromBin->ubFormer = 2048;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 2048;
    tilingDatafromBin->ubTailOfTailBlock = 2048;
    tilingDatafromBin->elemNum = 2048;

    auto KernelSigmoid = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid(x, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(SIGMOID_F16_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoid, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
