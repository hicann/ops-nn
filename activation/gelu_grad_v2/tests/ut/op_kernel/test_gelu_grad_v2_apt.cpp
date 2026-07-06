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

#include "../../../op_kernel/gelu_grad_v2_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void gelu_grad_v2(GM_ADDR dy, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

class gelu_grad_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "gelu_grad_v2_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "gelu_grad_v2 TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./gelu_grad_v2_data");
    }
};

TEST_F(gelu_grad_v2_test, test_case_fp32_tanh)
{
    size_t dyByteSize = 256 * sizeof(float);
    size_t xByteSize = 256 * sizeof(float);
    size_t zByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(BroadcastOneDimTilingDataAdvance);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/gelu_grad_v2/tests/ut/op_kernel/gelu_grad_v2_data",
                                    "gelu_grad_v2_data");
    kernel_ut::RunGenData("./gelu_grad_v2_data", {"'(256)'", "float32", "tanh"});

    std::string path = kernel_ut::GetTestWorkDir();

    BroadcastOneDimTilingDataAdvance* tilingDatafromBin = reinterpret_cast<BroadcastOneDimTilingDataAdvance*>(tiling);

    tilingDatafromBin->dimLen = 256;
    tilingDatafromBin->tileNum = 1024;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->scalarFlag = 0;
    ReadFile(path + "/gelu_grad_v2_data/input_dy.bin", dyByteSize, dy, dyByteSize);
    ReadFile(path + "/gelu_grad_v2_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelGeluGradV2 = [](GM_ADDR dy, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        ::gelu_grad_v2<202, TPL_TANH>(dy, x, z, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelGeluGradV2, blockDim, dy, x, z, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/gelu_grad_v2_data/output.bin", z, zByteSize);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
