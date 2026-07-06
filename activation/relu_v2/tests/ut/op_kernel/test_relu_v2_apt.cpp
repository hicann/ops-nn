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

#include "atvoss/elewise/elewise_sch_16b.h"
#include <cstdint>

using namespace std;
using namespace Ops::Base;

extern "C" __global__ __aicore__ void relu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

struct ReluV2TilingData {
    int64_t dim0;
    int32_t coreNum;
    int32_t ubFormer;
    int64_t blockFormer;
    int64_t blockNum;
    int64_t ubLoopOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubTailOfTailBlock;
    int64_t elemNum;
    uint64_t scheMode;
};

class relu_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "relu_v2_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "relu_v2 TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./relu_v2_data");
    }
};

TEST_F(relu_v2_test, test_case_fp32_1)
{
    size_t inputByteSize = 256 * sizeof(float);
    size_t outputByteSize = 256 * sizeof(float);
    size_t maskByteSize = 256 * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(ReluV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/relu_v2/tests/ut/op_kernel/relu_v2_data", "relu_v2_data");
    kernel_ut::RunGenData("./relu_v2_data", {"'(256)'", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    ReluV2TilingData* tilingDatafromBin = reinterpret_cast<ReluV2TilingData*>(tiling);

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

    ReadFile(path + "/relu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(103);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(relu_v2, blockDim, x, y, z, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/relu_v2_data/output.bin", y, outputByteSize);
    WriteFile(path + "/relu_v2_data/mask.bin", z, maskByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(relu_v2_test, test_case_fp16_1)
{
    size_t inputByteSize = 256 * sizeof(uint16_t);
    size_t outputByteSize = 256 * sizeof(uint16_t);
    size_t maskByteSize = 256 * sizeof(uint8_t);
    size_t tiling_data_size = sizeof(ReluV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/relu_v2/tests/ut/op_kernel/relu_v2_data", "relu_v2_data");
    kernel_ut::RunGenData("./relu_v2_data", {"'(256)'", "float16"});

    std::string path = kernel_ut::GetTestWorkDir();

    ReluV2TilingData* tilingDatafromBin = reinterpret_cast<ReluV2TilingData*>(tiling);

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

    ReadFile(path + "/relu_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(101);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(relu_v2, blockDim, x, y, z, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/relu_v2_data/output.bin", y, outputByteSize);
    WriteFile(path + "/relu_v2_data/mask.bin", z, maskByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
