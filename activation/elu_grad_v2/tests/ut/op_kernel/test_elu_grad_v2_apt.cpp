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

#include "../../../op_kernel/elu_grad_v2_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void elu_grad_v2(GM_ADDR grads, GM_ADDR activations, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class elu_grad_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "elu_grad_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "elu_grad_v2_test TearDown\n" << endl;
    }
};

TEST_F(elu_grad_v2_test, test_case_fp32_is_result_true)
{
    size_t gradsByteSize = 256 * sizeof(float);
    size_t activationsByteSize = 256 * sizeof(float);
    size_t yByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(EluGradV2Ns::EluGradV2TilingData);

    uint8_t* grads = (uint8_t*)AscendC::GmAlloc(gradsByteSize);
    uint8_t* activations = (uint8_t*)AscendC::GmAlloc(activationsByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../activation/elu_grad_v2/tests/ut/op_kernel/elu_grad_v2_data ./");
    system("chmod -R 755 ./elu_grad_v2_data/");
    system("cd ./elu_grad_v2_data/ && rm -rf ./*bin");
    system("cd ./elu_grad_v2_data/ && python3 gen_data.py '(256)' float32 1.0 1.0 1.0 True");

    char* path_ = get_current_dir_name();
    string path(path_);

    EluGradV2Ns::EluGradV2TilingData* tilingDatafromBin = reinterpret_cast<EluGradV2Ns::EluGradV2TilingData*>(tiling);

    tilingDatafromBin->baseTiling.dim0 = 256;
    tilingDatafromBin->baseTiling.coreNum = 1;
    tilingDatafromBin->baseTiling.ubFormer = 1024;
    tilingDatafromBin->baseTiling.blockFormer = 256;
    tilingDatafromBin->baseTiling.blockNum = 1;
    tilingDatafromBin->baseTiling.ubLoopOfFormerBlock = 1;
    tilingDatafromBin->baseTiling.ubLoopOfTailBlock = 1;
    tilingDatafromBin->baseTiling.ubTailOfFormerBlock = 256;
    tilingDatafromBin->baseTiling.ubTailOfTailBlock = 256;
    tilingDatafromBin->baseTiling.elemNum = 256;
    tilingDatafromBin->baseTiling.scheMode = 0;
    tilingDatafromBin->negcoef = 1.0;
    tilingDatafromBin->scale = 1.0;
    tilingDatafromBin->inputScale = 1.0;
    tilingDatafromBin->alignHolder = 0;

    ReadFile(path + "/elu_grad_v2_data/input_grads.bin", gradsByteSize, grads, gradsByteSize);
    ReadFile(path + "/elu_grad_v2_data/input_activations.bin", activationsByteSize, activations, activationsByteSize);
    auto KernelEluGradV2 = [](GM_ADDR grads, GM_ADDR activations, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::elu_grad_v2<0, EluGradV2_TPL_FP32>(grads, activations, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelEluGradV2, blockDim, grads, activations, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/elu_grad_v2_data/output.bin", y, yByteSize);

    AscendC::GmFree(grads);
    AscendC::GmFree(activations);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
