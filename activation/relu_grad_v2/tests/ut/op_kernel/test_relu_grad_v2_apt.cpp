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

#include "../../../op_kernel/relu_grad_v2_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void relu_grad_v2(GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops,
                                                   GM_ADDR workspace, GM_ADDR tiling);

class relu_grad_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "relu_grad_v2_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "relu_grad_v2 TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./relu_grad_v2_data");
    }
};

TEST_F(relu_grad_v2_test, test_case_fp32_1)
{
    size_t gradientsByteSize = 256 * sizeof(float);
    size_t maskByteSize = 256 * sizeof(uint8_t);
    size_t backpropsByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(EleBaseTilingDataV2);

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradientsByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(backpropsByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/relu_grad_v2/tests/ut/op_kernel/relu_grad_v2_data",
                                    "relu_grad_v2_data");
    kernel_ut::RunGenData("./relu_grad_v2_data", {"'(256)'", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    EleBaseTilingDataV2* tilingDatafromBin = reinterpret_cast<EleBaseTilingDataV2*>(tiling);

    tilingDatafromBin->dim0 = 256;
    tilingDatafromBin->coreNum = 1;
    tilingDatafromBin->ubFormer = 1024;
    tilingDatafromBin->blockFormer = 256;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 256;
    tilingDatafromBin->ubTailOfTailBlock = 256;
    tilingDatafromBin->elemNum = 256;
    tilingDatafromBin->scheMode = 0;
    ReadFile(path + "/relu_grad_v2_data/input_gradients.bin", gradientsByteSize, gradients, gradientsByteSize);
    ReadFile(path + "/relu_grad_v2_data/input_mask.bin", maskByteSize, mask, maskByteSize);
    auto KernelReluGradV2 = [](GM_ADDR gradients, GM_ADDR mask, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::relu_grad_v2<0, TPL_FP32>(gradients, mask, backprops, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelReluGradV2, blockDim, gradients, mask, backprops, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/relu_grad_v2_data/output.bin", backprops, backpropsByteSize);

    AscendC::GmFree(gradients);
    AscendC::GmFree(mask);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}