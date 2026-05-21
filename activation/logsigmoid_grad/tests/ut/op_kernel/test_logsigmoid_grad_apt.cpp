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

#include "../../../op_kernel/logsigmoid_grad.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void log_sigmoid_grad(GM_ADDR grads, GM_ADDR features, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling);

class logsigmoid_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "logsigmoid_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "logsigmoid_grad TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./logsigmoid_grad_data");
    }
};


TEST_F(logsigmoid_grad_test, test_case_fp32_1)
{
    size_t gradsByteSize = 256 * sizeof(float);
    size_t featuresByteSize = 256 * sizeof(float);
    size_t backpropsByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(BroadcastOneDimTilingDataAdvance);

    uint8_t* grads = (uint8_t*)AscendC::GmAlloc(gradsByteSize);
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(featuresByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(backpropsByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/logsigmoid_grad/tests/ut/op_kernel/logsigmoid_grad_data", "logsigmoid_grad_data");
    kernel_ut::RunGenData("./logsigmoid_grad_data", {"'(256)'", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    BroadcastOneDimTilingDataAdvance* tilingDatafromBin = reinterpret_cast<BroadcastOneDimTilingDataAdvance*>(tiling);

    tilingDatafromBin->dimLen = 256;
    tilingDatafromBin->tileNum = 1024;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->scalarFlag = 0;
    ReadFile(path + "/logsigmoid_grad_data/input_grads.bin", gradsByteSize, grads, gradsByteSize);
    ReadFile(path + "/logsigmoid_grad_data/input_features.bin", featuresByteSize, features, featuresByteSize);
    auto KernelLogSigmoidGrad = [](GM_ADDR grads, GM_ADDR features, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::log_sigmoid_grad<202>(grads, features, backprops, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelLogSigmoidGrad, blockDim, grads, features, backprops, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/logsigmoid_grad_data/output.bin", backprops, backpropsByteSize);

    AscendC::GmFree(grads);
    AscendC::GmFree(features);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}