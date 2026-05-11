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

#include "../../../op_kernel/leaky_relu_apt.cpp"
#include "../../../op_kernel/arch35/leaky_relu_tilingdata.h"
#include <cstdint>

using namespace std;
using namespace AscendC;
using namespace LeakyReluOp;

extern "C" __global__ __aicore__ void leaky_relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class leaky_relu_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "leaky_relu_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "leaky_relu_test TearDown\n" << endl;
    }
};

TEST_F(leaky_relu_test, test_case_fp32_1)
{
    size_t inputNum = 256;
    size_t xByteSize = inputNum * sizeof(float);
    size_t yByteSize = inputNum * sizeof(float);
    size_t tiling_data_size = sizeof(LeakyReluTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    system("cp -r ../../../../activation/leaky_relu/tests/ut/op_kernel/leaky_relu_data ./");
    system("chmod -R 755 ./leaky_relu_data/");
    system("cd ./leaky_relu_data/ && rm -rf ./*bin");
    system("cd ./leaky_relu_data/ && python3 gen_data.py '(256)' float32 0.01");

    char* path_ = get_current_dir_name();
    string path(path_);

    LeakyReluTilingData* tilingDatafromBin = reinterpret_cast<LeakyReluTilingData*>(tiling);
    
    tilingDatafromBin->baseTiling.dim0 = inputNum;
    tilingDatafromBin->baseTiling.coreNum = 1;
    tilingDatafromBin->baseTiling.ubFormer = 1024;
    tilingDatafromBin->baseTiling.blockFormer = 256;
    tilingDatafromBin->baseTiling.blockNum = 1;
    tilingDatafromBin->baseTiling.ubLoopOfFormerBlock = 1;
    tilingDatafromBin->baseTiling.ubLoopOfTailBlock = 1;
    tilingDatafromBin->baseTiling.ubTailOfFormerBlock = 256;
    tilingDatafromBin->baseTiling.ubTailOfTailBlock = 256;
    tilingDatafromBin->baseTiling.elemNum = inputNum;
    tilingDatafromBin->baseTiling.scheMode = 0;
    tilingDatafromBin->negativeSlope = 0.01f;
    
    ReadFile(path + "/leaky_relu_data/input_x.bin", xByteSize, x, xByteSize);
    
    auto KernelLeakyRelu = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::leaky_relu<201, TPL_FP32>(x, y, workspace, tiling);
    };
    
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelLeakyRelu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/leaky_relu_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
