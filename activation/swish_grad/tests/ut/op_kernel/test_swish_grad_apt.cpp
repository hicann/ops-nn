/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in the License.
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

#include "../../../op_kernel/swish_grad_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void swish_grad(GM_ADDR grad, GM_ADDR x, GM_ADDR y, GM_ADDR grad_x, GM_ADDR workspace,
                                                 GM_ADDR tiling);

class swish_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "swish_grad_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "swish_grad TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./swish_grad_data");
    }
};

TEST_F(swish_grad_test, test_case_fp32_1)
{
    size_t gradByteSize = 256 * sizeof(float);
    size_t xByteSize = 256 * sizeof(float);
    size_t yByteSize = 256 * sizeof(float);
    size_t grad_xByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(SwishGradTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = nullptr;
    uint8_t* grad_x = (uint8_t*)AscendC::GmAlloc(grad_xByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/swish_grad/tests/ut/op_kernel/swish_grad_data", "swish_grad_data");
    kernel_ut::RunGenData("./swish_grad_data", {"'(256)'", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    SwishGradTilingData* tilingDatafromBin = reinterpret_cast<SwishGradTilingData*>(tiling);

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
    tilingDatafromBin->scale = 1.0;

    ReadFile(path + "/swish_grad_data/input_grad.bin", gradByteSize, grad, gradByteSize);
    ReadFile(path + "/swish_grad_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelSwishGrad = [](GM_ADDR grad, GM_ADDR x, GM_ADDR y, GM_ADDR grad_x, GM_ADDR workspace, GM_ADDR tiling) {
        ::swish_grad<0, TPL_FP32>(grad, x, y, grad_x, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSwishGrad, blockDim, grad, x, y, grad_x, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/swish_grad_data/output.bin", grad_x, grad_xByteSize);

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(grad_x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
