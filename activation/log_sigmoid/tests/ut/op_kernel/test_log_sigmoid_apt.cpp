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
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"

#include "../../../op_kernel/log_sigmoid_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void log_sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class log_sigmoid_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "log_sigmoid_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "log_sigmoid TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./log_sigmoid_data");
    }
};


TEST_F(log_sigmoid_test, test_case_fp32_1)
{
    size_t xByteSize = 256 * sizeof(float);
    size_t yByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(EleBaseTilingDataV2);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/log_sigmoid/tests/ut/op_kernel/log_sigmoid_data", "log_sigmoid_data");
    kernel_ut::RunGenData("./log_sigmoid_data", {"'(256)'", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    EleBaseTilingDataV2* tilingDatafromBin = reinterpret_cast<EleBaseTilingDataV2*>(tiling);

    tilingDatafromBin->dim0 = 256;
    tilingDatafromBin->coreNum = 1;
    tilingDatafromBin->ubFormer = 256;
    tilingDatafromBin->blockFormer = 256;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 256;
    tilingDatafromBin->ubTailOfTailBlock = 256;
    tilingDatafromBin->elemNum = 256;
    tilingDatafromBin->scheMode = 0;
    ReadFile(path + "/log_sigmoid_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelLogSigmoid = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::log_sigmoid<0, TPL_FP32>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelLogSigmoid, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/log_sigmoid_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}