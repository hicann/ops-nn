/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void sigmoid_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

#define SIGMOID_GRAD_F16_TILING_KEY 100000000000100
#define SIGMOID_GRAD_BF16_TILING_KEY 200000000000100
#define SIGMOID_GRAD_F32_TILING_KEY 300000000000100

class sigmoid_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "sigmoid_grad_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "sigmoid_grad_test TearDown\n" << endl; }
};

TEST_F(sigmoid_grad_test, test_case_fp32)
{
    size_t yByteSize = 256 * sizeof(float);
    size_t dyByteSize = 256 * sizeof(float);
    size_t zByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(SigmoidGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    SigmoidGradTilingData* tilingDatafromBin = reinterpret_cast<SigmoidGradTilingData*>(tiling);

    tilingDatafromBin->dim0 = 256;
    tilingDatafromBin->blockFormer = 256;
    tilingDatafromBin->ubFormer = 256;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 256;
    tilingDatafromBin->ubTailOfTailBlock = 256;
    tilingDatafromBin->elemNum = 256;

    auto KernelSigmoidGrad = [](GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid_grad(y, dy, z, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(SIGMOID_GRAD_F32_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoidGrad, blockDim, y, dy, z, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sigmoid_grad_test, test_case_fp16)
{
    size_t yByteSize = 256 * sizeof(half);
    size_t dyByteSize = 256 * sizeof(half);
    size_t zByteSize = 256 * sizeof(half);
    size_t tiling_data_size = sizeof(SigmoidGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    SigmoidGradTilingData* tilingDatafromBin = reinterpret_cast<SigmoidGradTilingData*>(tiling);

    tilingDatafromBin->dim0 = 256;
    tilingDatafromBin->blockFormer = 256;
    tilingDatafromBin->ubFormer = 256;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 256;
    tilingDatafromBin->ubTailOfTailBlock = 256;
    tilingDatafromBin->elemNum = 256;

    auto KernelSigmoidGrad = [](GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid_grad(y, dy, z, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(SIGMOID_GRAD_F16_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoidGrad, blockDim, y, dy, z, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sigmoid_grad_test, test_case_bf16)
{
    size_t yByteSize = 256 * sizeof(bfloat16_t);
    size_t dyByteSize = 256 * sizeof(bfloat16_t);
    size_t zByteSize = 256 * sizeof(bfloat16_t);
    size_t tiling_data_size = sizeof(SigmoidGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    SigmoidGradTilingData* tilingDatafromBin = reinterpret_cast<SigmoidGradTilingData*>(tiling);

    tilingDatafromBin->dim0 = 256;
    tilingDatafromBin->blockFormer = 256;
    tilingDatafromBin->ubFormer = 256;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 256;
    tilingDatafromBin->ubTailOfTailBlock = 256;
    tilingDatafromBin->elemNum = 256;

    auto KernelSigmoidGrad = [](GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid_grad(y, dy, z, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(SIGMOID_GRAD_BF16_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoidGrad, blockDim, y, dy, z, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sigmoid_grad_test, test_case_fp32_large)
{
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t dyByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t zByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(SigmoidGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    SigmoidGradTilingData* tilingDatafromBin = reinterpret_cast<SigmoidGradTilingData*>(tiling);

    tilingDatafromBin->dim0 = 2048;
    tilingDatafromBin->blockFormer = 2048;
    tilingDatafromBin->ubFormer = 2048;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 2048;
    tilingDatafromBin->ubTailOfTailBlock = 2048;
    tilingDatafromBin->elemNum = 2048;

    auto KernelSigmoidGrad = [](GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid_grad(y, dy, z, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(SIGMOID_GRAD_F32_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoidGrad, blockDim, y, dy, z, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sigmoid_grad_test, test_case_fp16_large)
{
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(half);
    size_t dyByteSize = 32 * 4 * 4 * 4 * sizeof(half);
    size_t zByteSize = 32 * 4 * 4 * 4 * sizeof(half);
    size_t tiling_data_size = sizeof(SigmoidGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    SigmoidGradTilingData* tilingDatafromBin = reinterpret_cast<SigmoidGradTilingData*>(tiling);

    tilingDatafromBin->dim0 = 2048;
    tilingDatafromBin->blockFormer = 2048;
    tilingDatafromBin->ubFormer = 2048;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 2048;
    tilingDatafromBin->ubTailOfTailBlock = 2048;
    tilingDatafromBin->elemNum = 2048;

    auto KernelSigmoidGrad = [](GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid_grad(y, dy, z, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(SIGMOID_GRAD_F16_TILING_KEY);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoidGrad, blockDim, y, dy, z, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}