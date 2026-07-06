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

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void elu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

struct EluTilingData {
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
    float alpha;
    float scale;
    float inputScale;
};

class elu_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "elu_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "elu TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./elu_data");
    }
};

TEST_F(elu_test, test_case_fp32_1)
{
    size_t inputByteSize = 10 * sizeof(float);
    size_t outputByteSize = 10 * sizeof(float);
    size_t tiling_data_size = sizeof(EluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/elu/tests/ut/op_kernel/elu_data", "elu_data");
    kernel_ut::RunGenData("./elu_data", {"'(10)'", "1.0", "1.0", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    EluTilingData* tilingDatafromBin = reinterpret_cast<EluTilingData*>(tiling);

    tilingDatafromBin->dim0 = 10;
    tilingDatafromBin->coreNum = 1;
    tilingDatafromBin->ubFormer = 256;
    tilingDatafromBin->blockFormer = 10;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 10;
    tilingDatafromBin->ubTailOfTailBlock = 10;
    tilingDatafromBin->elemNum = 10;
    tilingDatafromBin->scheMode = 0;
    tilingDatafromBin->alpha = 1.0;
    tilingDatafromBin->scale = 1.0;
    tilingDatafromBin->inputScale = 1.0;

    ReadFile(path + "/elu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(101);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(elu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/elu_data/output.bin", y, outputByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    // system("cd ./elu_data/ && python3 compare_data.py 'float32'");
}

TEST_F(elu_test, test_case_fp32_2)
{
    size_t inputByteSize = 12 * 4 * sizeof(float);
    size_t outputByteSize = 12 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(EluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/elu/tests/ut/op_kernel/elu_data", "elu_data");
    kernel_ut::RunGenData("./elu_data", {"'(12, 4)'", "1.0", "1.0", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    EluTilingData* tilingDatafromBin = reinterpret_cast<EluTilingData*>(tiling);

    tilingDatafromBin->dim0 = 12 * 4;
    tilingDatafromBin->coreNum = 1;
    tilingDatafromBin->ubFormer = 2048;
    tilingDatafromBin->blockFormer = (12 * 4) / 1;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 12 * 4;
    tilingDatafromBin->ubTailOfTailBlock = 12 * 4;
    tilingDatafromBin->elemNum = 12 * 4;
    tilingDatafromBin->scheMode = 0;
    tilingDatafromBin->alpha = 1.0;
    tilingDatafromBin->scale = 1.0;
    tilingDatafromBin->inputScale = 1.0;

    ReadFile(path + "/elu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(101);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(elu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/elu_data/output.bin", y, outputByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    // system("cd ./elu_data/ && python3 compare_data.py 'float32'");
}

TEST_F(elu_test, test_case_fp16_1)
{
    size_t inputByteSize = 2 * 32 * sizeof(uint16_t);
    size_t outputByteSize = 2 * 32 * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(EluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/elu/tests/ut/op_kernel/elu_data", "elu_data");
    kernel_ut::RunGenData("./elu_data", {"'(2, 32)'", "1.0", "1.0", "float16"});

    std::string path = kernel_ut::GetTestWorkDir();

    EluTilingData* tilingDatafromBin = reinterpret_cast<EluTilingData*>(tiling);

    tilingDatafromBin->dim0 = 2 * 32;
    tilingDatafromBin->coreNum = 1;
    tilingDatafromBin->ubFormer = 2048;
    tilingDatafromBin->blockFormer = (2 * 32) / 1;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 2 * 32;
    tilingDatafromBin->ubTailOfTailBlock = 2 * 32;
    tilingDatafromBin->elemNum = 2 * 32;
    tilingDatafromBin->scheMode = 0;
    tilingDatafromBin->alpha = 1.0;
    tilingDatafromBin->scale = 1.0;
    tilingDatafromBin->inputScale = 1.0;

    ReadFile(path + "/elu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(101);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(elu, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/elu_data/output.bin", y, outputByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    // system("cd ./elu_data/ && python3 compare_data.py 'float16'");
}
