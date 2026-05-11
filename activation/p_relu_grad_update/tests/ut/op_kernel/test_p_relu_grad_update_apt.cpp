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

#include "../../../op_kernel/p_relu_grad_update_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void prelu_grad_update(GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx, GM_ADDR update, GM_ADDR workspace, GM_ADDR tiling);

class p_relu_grad_update_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "p_relu_grad_update_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "p_relu_grad_update_test TearDown\n" << endl;
    }
};

TEST_F(p_relu_grad_update_test, test_case_fp32_1)
{
    size_t gradsByteSize = 256 * sizeof(float);
    size_t featuresByteSize = 256 * sizeof(float);
    size_t weightsByteSize = 256 * sizeof(float);
    size_t dxByteSize = 256 * sizeof(float);
    size_t updateByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(BroadcastOneDimTilingDataAdvance);

    uint8_t* grads = (uint8_t*)AscendC::GmAlloc(gradsByteSize);
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(featuresByteSize);
    uint8_t* weights = (uint8_t*)AscendC::GmAlloc(weightsByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* update = (uint8_t*)AscendC::GmAlloc(updateByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../activation/p_relu_grad_update/tests/ut/op_kernel/p_relu_grad_update_data ./");
    system("chmod -R 755 ./p_relu_grad_update_data/");
    system("cd ./p_relu_grad_update_data/ && rm -rf ./*bin");
    system("cd ./p_relu_grad_update_data/ && python3 gen_data.py '(256)' float32");

    char* path_ = get_current_dir_name();
    string path(path_);

    BroadcastOneDimTilingDataAdvance* tilingDatafromBin = reinterpret_cast<BroadcastOneDimTilingDataAdvance*>(tiling);

    tilingDatafromBin->dimLen = 256;
    tilingDatafromBin->tileNum = 1024;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->scalarFlag = 0;
    ReadFile(path + "/p_relu_grad_update_data/input_grads.bin", gradsByteSize, grads, gradsByteSize);
    ReadFile(path + "/p_relu_grad_update_data/input_features.bin", featuresByteSize, features, featuresByteSize);
    ReadFile(path + "/p_relu_grad_update_data/input_weights.bin", weightsByteSize, weights, weightsByteSize);
    auto KernelPReluGradUpdate = [](GM_ADDR grads, GM_ADDR features, GM_ADDR weights, GM_ADDR dx, GM_ADDR update, GM_ADDR workspace, GM_ADDR tiling) {
        ::prelu_grad_update<202>(grads, features, weights, dx, update, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelPReluGradUpdate, blockDim, grads, features, weights, dx, update, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/p_relu_grad_update_data/output_dx.bin", dx, dxByteSize);
    WriteFile(path + "/p_relu_grad_update_data/output_update.bin", update, updateByteSize);

    AscendC::GmFree(grads);
    AscendC::GmFree(features);
    AscendC::GmFree(weights);
    AscendC::GmFree(dx);
    AscendC::GmFree(update);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}