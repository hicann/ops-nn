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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/leaky_relu_grad_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void leaky_relu_grad(GM_ADDR gradients, GM_ADDR features, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling);

class leaky_relu_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "leaky_relu_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "leaky_relu_grad_test TearDown\n" << endl;
    }
};

TEST_F(leaky_relu_grad_test, test_case_fp32_1)
{
    size_t gradientsByteSize = 256 * sizeof(float);
    size_t featuresByteSize = 256 * sizeof(float);
    size_t backpropsByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(BroadcastOneDimTilingData);

    uint8_t* gradients = (uint8_t*)AscendC::GmAlloc(gradientsByteSize);
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(featuresByteSize);
    uint8_t* backprops = (uint8_t*)AscendC::GmAlloc(backpropsByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../activation/leaky_relu_grad/tests/ut/op_kernel/leaky_relu_grad_data ./");
    system("chmod -R 755 ./leaky_relu_grad_data/");
    system("cd ./leaky_relu_grad_data/ && rm -rf ./*bin");
    system("cd ./leaky_relu_grad_data/ && python3 gen_data.py '(256)' float32 0.01");

    char* path_ = get_current_dir_name();
    string path(path_);

    BroadcastOneDimTilingData* tilingDatafromBin = reinterpret_cast<BroadcastOneDimTilingData*>(tiling);

    tilingDatafromBin->scalarFlag = 0;
    tilingDatafromBin->ubSplitAxis = 0;
    tilingDatafromBin->ubFormer = 256;
    tilingDatafromBin->ubTail = 256;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->blockFormer = 1;
    tilingDatafromBin->blockTail = 1;
    *reinterpret_cast<float*>(tilingDatafromBin->scalarData) = 0.01f;
    
    ReadFile(path + "/leaky_relu_grad_data/input_gradients.bin", gradientsByteSize, gradients, gradientsByteSize);
    ReadFile(path + "/leaky_relu_grad_data/input_features.bin", featuresByteSize, features, featuresByteSize);
    auto KernelLeakyReluGrad = [](GM_ADDR gradients, GM_ADDR features, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling) {
        ::leaky_relu_grad<201, LEAKY_RELU_GRAD_TPL_FP32>(gradients, features, backprops, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelLeakyReluGrad, blockDim, gradients, features, backprops, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/leaky_relu_grad_data/output.bin", backprops, backpropsByteSize);

    AscendC::GmFree(gradients);
    AscendC::GmFree(features);
    AscendC::GmFree(backprops);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}