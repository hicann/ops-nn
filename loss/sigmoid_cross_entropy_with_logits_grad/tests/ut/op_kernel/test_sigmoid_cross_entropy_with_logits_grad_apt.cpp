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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"

#include "kernel_operator.h"
#include "../../../op_kernel/arch35/sigmoid_cross_entropy_with_logits_grad_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void sigmoid_cross_entropy_with_logits_grad(GM_ADDR predict, GM_ADDR target,
                                                                             GM_ADDR dout, GM_ADDR gradient,
                                                                             GM_ADDR workspace, GM_ADDR tiling);

class sigmoid_cross_entropy_with_logits_grad_kernel_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "sigmoid_cross_entropy_with_logits_grad_kernel_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "sigmoid_cross_entropy_with_logits_grad_kernel_test TearDown" << endl; }
};

TEST_F(sigmoid_cross_entropy_with_logits_grad_kernel_test, fp32_basic)
{
    size_t numElements = 256;
    size_t elemSize = sizeof(float);
    size_t bufferSize = numElements * elemSize;
    size_t tilingSize = sizeof(SigmoidCrossEntropyWithLogitsGradTilingData);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(bufferSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(bufferSize);
    uint8_t* dout = (uint8_t*)AscendC::GmAlloc(bufferSize);
    uint8_t* gradient = (uint8_t*)AscendC::GmAlloc(bufferSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    SigmoidCrossEntropyWithLogitsGradTilingData*
        tilingData = reinterpret_cast<SigmoidCrossEntropyWithLogitsGradTilingData*>(tiling);
    tilingData->totalLength = numElements;
    tilingData->tileNum = 256;

    auto kernelFn = [](GM_ADDR predict, GM_ADDR target, GM_ADDR dout, GM_ADDR gradient, GM_ADDR workspace,
                       GM_ADDR tiling) {
        ::sigmoid_cross_entropy_with_logits_grad(predict, target, dout, gradient, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernelFn, 1, predict, target, dout, gradient, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(predict);
    AscendC::GmFree(target);
    AscendC::GmFree(dout);
    AscendC::GmFree(gradient);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sigmoid_cross_entropy_with_logits_grad_kernel_test, fp32_basic_another)
{
    size_t numElements = 256;
    size_t elemSize = sizeof(float);
    size_t bufferSize = numElements * elemSize;
    size_t tilingSize = sizeof(SigmoidCrossEntropyWithLogitsGradTilingData);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(bufferSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(bufferSize);
    uint8_t* dout = (uint8_t*)AscendC::GmAlloc(bufferSize);
    uint8_t* gradient = (uint8_t*)AscendC::GmAlloc(bufferSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    SigmoidCrossEntropyWithLogitsGradTilingData*
        tilingData = reinterpret_cast<SigmoidCrossEntropyWithLogitsGradTilingData*>(tiling);
    tilingData->totalLength = numElements;
    tilingData->tileNum = 256;

    auto kernelFn = [](GM_ADDR predict, GM_ADDR target, GM_ADDR dout, GM_ADDR gradient, GM_ADDR workspace,
                       GM_ADDR tiling) {
        ::sigmoid_cross_entropy_with_logits_grad(predict, target, dout, gradient, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernelFn, 1, predict, target, dout, gradient, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(predict);
    AscendC::GmFree(target);
    AscendC::GmFree(dout);
    AscendC::GmFree(gradient);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
