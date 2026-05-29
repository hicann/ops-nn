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

#include "../../../op_kernel/sigmoid_cross_entropy_with_logits_grad_v2_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void sigmoid_cross_entropy_with_logits_grad_v2(GM_ADDR predict, GM_ADDR target, GM_ADDR dout, GM_ADDR weight,
                    GM_ADDR posWeight, GM_ADDR gradient, GM_ADDR workspace, GM_ADDR tiling);

class sigmoid_cross_entropy_with_logits_grad_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "sigmoid_cross_entropy_with_logits_grad_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "sigmoid_cross_entropy_with_logits_grad_v2_test TearDown\n" << endl;
    }
};


TEST_F(sigmoid_cross_entropy_with_logits_grad_v2_test, test_case_fp32_1)
{
    size_t predictByteSize = 256 * sizeof(float);
    size_t targetByteSize = 256 * sizeof(float);
    size_t doutByteSize = 256 * sizeof(float);
    size_t weightByteSize = 256 * sizeof(float);
    size_t posWeightByteSize = 256 * sizeof(float);
    size_t gradientByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(BroadcastOneDimTilingData);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictByteSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetByteSize);
    uint8_t* dout = (uint8_t*)AscendC::GmAlloc(doutByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* posWeight = (uint8_t*)AscendC::GmAlloc(posWeightByteSize);
    uint8_t* gradient = (uint8_t*)AscendC::GmAlloc(gradientByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path = kernel_ut::GetTestWorkDir();

    BroadcastOneDimTilingData* tilingDatafromBin = reinterpret_cast<BroadcastOneDimTilingData*>(tiling);

    tilingDatafromBin->scalarFlag = 0;
    tilingDatafromBin->ubSplitAxis = 0;
    tilingDatafromBin->ubFormer = 160;
    tilingDatafromBin->ubTail = 96;
    tilingDatafromBin->blockNum = 2;
    tilingDatafromBin->blockFormer = 1;
    tilingDatafromBin->blockTail = 1;
    *reinterpret_cast<float*>(tilingDatafromBin->scalarData) = 0.01f;
    auto KernelSigmoidCrossEntropyWithLogitsGradV2 = [](GM_ADDR predict, GM_ADDR target, GM_ADDR dout, GM_ADDR weight, GM_ADDR posWeight, GM_ADDR gradient, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid_cross_entropy_with_logits_grad_v2<201, 1, 1, 0>(predict, target, dout, weight, posWeight, gradient, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(196615);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSigmoidCrossEntropyWithLogitsGradV2, blockDim, predict, target, dout, weight, posWeight, gradient, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(predict);
    AscendC::GmFree(target);
    AscendC::GmFree(dout);
    AscendC::GmFree(weight);
    AscendC::GmFree(posWeight);
    AscendC::GmFree(gradient);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
