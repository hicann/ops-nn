/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except the License.
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

#include "../../../op_kernel/hardtanh_grad_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void hardtanh_grad(
    GM_ADDR result, GM_ADDR grad, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class hardtanh_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "hardtanh_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "hardtanh_grad TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./hardtanh_grad_data");
    }
};


TEST_F(hardtanh_grad_test, test_case_fp32_1)
{
    size_t resultByteSize = 256 * sizeof(float);
    size_t gradByteSize = 256 * sizeof(float);
    size_t yByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(HardtanhGradTilingData);

    uint8_t* result = (uint8_t*)AscendC::GmAlloc(resultByteSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/hardtanh_grad/tests/ut/op_kernel/hardtanh_grad_data", "hardtanh_grad_data");
    kernel_ut::RunGenData("./hardtanh_grad_data", {"'(256)'", "float32", "-1.0", "1.0"});

    std::string path = kernel_ut::GetTestWorkDir();

    HardtanhGradTilingData* tilingDatafromBin = reinterpret_cast<HardtanhGradTilingData*>(tiling);

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
    tilingDatafromBin->minVal = -1.0f;
    tilingDatafromBin->maxVal = 1.0f;
    
    ReadFile(path + "/hardtanh_grad_data/input_result.bin", resultByteSize, result, resultByteSize);
    ReadFile(path + "/hardtanh_grad_data/input_grad.bin", gradByteSize, grad, gradByteSize);
    auto KernelHardtanhGrad = [](GM_ADDR result, GM_ADDR grad, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::hardtanh_grad<0, HardtanhGrad_TPL>(result, grad, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelHardtanhGrad, blockDim, result, grad, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/hardtanh_grad_data/output.bin", y, yByteSize);

    AscendC::GmFree(result);
    AscendC::GmFree(grad);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}