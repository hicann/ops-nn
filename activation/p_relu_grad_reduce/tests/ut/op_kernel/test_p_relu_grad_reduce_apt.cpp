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

#include "../../../op_kernel/p_relu_grad_reduce_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void prelu_grad_reduce(GM_ADDR grads, GM_ADDR features, GM_ADDR weights,
                                                        GM_ADDR updates, GM_ADDR da, GM_ADDR workspace, GM_ADDR tiling);

class p_relu_grad_reduce_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "p_relu_grad_reduce_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "p_relu_grad_reduce_test TearDown\n" << endl;
    }
};

TEST_F(p_relu_grad_reduce_test, test_case_fp32_1)
{
    size_t gradsByteSize = 256 * sizeof(float);
    size_t featuresByteSize = 256 * sizeof(float);
    size_t weightsByteSize = 256 * sizeof(float);
    size_t updatesByteSize = 256 * sizeof(float);
    size_t daByteSize = 1 * sizeof(float);
    size_t tiling_data_size = sizeof(Ops::Base::ReduceOpTilingData);

    uint8_t* grads = (uint8_t*)AscendC::GmAlloc(gradsByteSize);
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(featuresByteSize);
    uint8_t* weights = (uint8_t*)AscendC::GmAlloc(weightsByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* da = (uint8_t*)AscendC::GmAlloc(daByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    
    system("cp -r ../../../../activation/p_relu_grad_reduce/tests/ut/op_kernel/p_relu_grad_reduce_data ./");
    system("chmod -R 755 ./p_relu_grad_reduce_data/");
    system("cd ./p_relu_grad_reduce_data/ && rm -rf ./*bin");
    system("cd ./p_relu_grad_reduce_data/ && python3 gen_data.py '(256)' float32");

    char* path_ = get_current_dir_name();
    string path(path_);

    Ops::Base::ReduceOpTilingData* tilingDatafromBin = reinterpret_cast<Ops::Base::ReduceOpTilingData*>(tiling);

    tilingDatafromBin->factorACntPerCore = 256;
    tilingDatafromBin->factorATotalCnt = 256;
    tilingDatafromBin->ubFactorA = 256;
    tilingDatafromBin->factorRCntPerCore = 256;
    tilingDatafromBin->factorRTotalCnt = 256;
    tilingDatafromBin->ubFactorR = 256;
    tilingDatafromBin->groupR = 1;
    tilingDatafromBin->outSize = 1;
    tilingDatafromBin->basicBlock = 256;
    tilingDatafromBin->resultBlock = 1;
    tilingDatafromBin->coreNum = 1;
    tilingDatafromBin->useNddma = 0;
    tilingDatafromBin->meanVar = 0.0f;
    tilingDatafromBin->shape[0] = 256;
    tilingDatafromBin->stride[0] = 1;
    tilingDatafromBin->dstStride[0] = 1;
    tilingDatafromBin->sliceNum[0] = 1;
    tilingDatafromBin->sliceShape[0] = 256;
    tilingDatafromBin->sliceStride[0] = 1;
    
    ReadFile(path + "/p_relu_grad_reduce_data/input_grads.bin", gradsByteSize, grads, gradsByteSize);
    ReadFile(path + "/p_relu_grad_reduce_data/input_features.bin", featuresByteSize, features, featuresByteSize);
    ReadFile(path + "/p_relu_grad_reduce_data/input_weights.bin", weightsByteSize, weights, weightsByteSize);
    ReadFile(path + "/p_relu_grad_reduce_data/input_updates.bin", updatesByteSize, updates, updatesByteSize);
    
    auto KernelPReluGradReduce = [](GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR updates, GM_ADDR da, GM_ADDR workspace, GM_ADDR tiling) {
        ::prelu_grad_reduce<true, 10, 10, 0>(grads, features, weights, updates, da, workspace, tiling);
    };
    
    ICPU_SET_TILING_KEY(10);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelPReluGradReduce, blockDim, grads, features, weights, updates, da, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/p_relu_grad_reduce_data/output_da.bin", da, daByteSize);

    AscendC::GmFree(grads);
    AscendC::GmFree(features);
    AscendC::GmFree(weights);
    AscendC::GmFree(updates);
    AscendC::GmFree(da);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
