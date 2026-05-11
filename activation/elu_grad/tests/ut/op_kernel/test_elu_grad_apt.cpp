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

#include "../../../op_kernel/elu_grad_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void elu_grad(GM_ADDR grads, GM_ADDR activations, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class elu_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "elu_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "elu_grad_test TearDown\n" << endl;
    }
};

TEST_F(elu_grad_test, test_case_fp32_1)
{
    size_t gradsByteSize = 256 * sizeof(float);
    size_t activationsByteSize = 256 * sizeof(float);
    size_t yByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(EleBaseTilingDataV2);

    uint8_t* grads = (uint8_t*)AscendC::GmAlloc(gradsByteSize);
    uint8_t* activations = (uint8_t*)AscendC::GmAlloc(activationsByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../activation/elu_grad/tests/ut/op_kernel/elu_grad_data ./");
    system("chmod -R 755 ./elu_grad_data/");
    system("cd ./elu_grad_data/ && rm -rf ./*bin");
    system("cd ./elu_grad_data/ && python3 gen_data.py '(256)' float32");

    char* path_ = get_current_dir_name();
    string path(path_);

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
    ReadFile(path + "/elu_grad_data/input_grads.bin", gradsByteSize, grads, gradsByteSize);
    ReadFile(path + "/elu_grad_data/input_activations.bin", activationsByteSize, activations, activationsByteSize);
    auto KernelEluGrad = [](GM_ADDR grads, GM_ADDR activations, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::elu_grad<0, TPL_FP32>(grads, activations, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelEluGrad, blockDim, grads, activations, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/elu_grad_data/output.bin", y, yByteSize);

    AscendC::GmFree(grads);
    AscendC::GmFree(activations);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
