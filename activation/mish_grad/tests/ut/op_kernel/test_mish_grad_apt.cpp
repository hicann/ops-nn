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

#include "../../../op_kernel/mish_grad_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void mish_grad(
    GM_ADDR grad, GM_ADDR x, GM_ADDR tanhx, GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling);

class mish_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "mish_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "mish_grad_test TearDown\n" << endl;
    }
};

TEST_F(mish_grad_test, test_case_fp32_1)
{
    size_t gradByteSize = 256 * sizeof(float);
    size_t xByteSize = 256 * sizeof(float);
    size_t x_gradByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(EleBaseTilingData16B);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* tanhx = nullptr;
    uint8_t* x_grad = (uint8_t*)AscendC::GmAlloc(x_gradByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../activation/mish_grad/tests/ut/op_kernel/mish_grad_data ./");
    system("chmod -R 755 ./mish_grad_data/");
    system("cd ./mish_grad_data/ && rm -rf ./*bin");
    system("cd ./mish_grad_data/ && python3 gen_data.py '(256)' float32");

    char* path_ = get_current_dir_name();
    string path(path_);

    EleBaseTilingData16B* tilingDatafromBin = reinterpret_cast<EleBaseTilingData16B*>(tiling);

    tilingDatafromBin->dim0 = 256;
    tilingDatafromBin->coreNum = 1;
    tilingDatafromBin->ubFormer = 1024;
    
    ReadFile(path + "/mish_grad_data/input_grad.bin", gradByteSize, grad, gradByteSize);
    ReadFile(path + "/mish_grad_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelMishGrad = [](GM_ADDR grad, GM_ADDR x, GM_ADDR tanhx, GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling) {
        ::mish_grad<0, TPL_FP32>(grad, x, tanhx, x_grad, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelMishGrad, blockDim, grad, x, tanhx, x_grad, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/mish_grad_data/output.bin", x_grad, x_gradByteSize);

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(x_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
