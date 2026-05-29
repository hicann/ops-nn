/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
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

#define __RELU6_TILING_KEY_H__
#include "../../../../op_kernel/arch35/relu6_tiling_data.h"
#include "../../../../op_kernel/arch35/relu6.h"

using namespace std;

template <typename T>
static void relu6_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    const Relu6TilingData* tilingData = reinterpret_cast<const Relu6TilingData*>(tiling);
    NsRelu6::Relu6<T> op;
    op.Init(x, y, tilingData);
    op.Process();
}

class relu6_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "relu6_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "relu6 TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./relu6_data");
    }
};

TEST_F(relu6_test, test_case_fp32_small)
{
    size_t dataNum = 64;
    size_t xByteSize = dataNum * sizeof(float);
    size_t yByteSize = dataNum * sizeof(float);
    size_t tiling_data_size = sizeof(Relu6TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/relu6/tests/ut/op_kernel/relu6_data", "relu6_data");
    kernel_ut::RunGenData("./relu6_data", {"'(64)'", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    Relu6TilingData* tilingData = reinterpret_cast<Relu6TilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(dataNum);
    tilingData->dataType = 1;

    ReadFile(path + "/relu6_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelRelu6 = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        relu6_kernel<float>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelRelu6, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/relu6_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(relu6_test, test_case_fp32_multi_loop)
{
    size_t dataNum = 256;
    size_t xByteSize = dataNum * sizeof(float);
    size_t yByteSize = dataNum * sizeof(float);
    size_t tiling_data_size = sizeof(Relu6TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/relu6/tests/ut/op_kernel/relu6_data", "relu6_data");
    kernel_ut::RunGenData("./relu6_data", {"'(256)'", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    Relu6TilingData* tilingData = reinterpret_cast<Relu6TilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(128);
    tilingData->dataType = 1;

    ReadFile(path + "/relu6_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelRelu6 = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        relu6_kernel<float>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelRelu6, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/relu6_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(relu6_test, test_case_fp16_small)
{
    size_t dataNum = 128;
    size_t xByteSize = dataNum * sizeof(uint16_t);
    size_t yByteSize = dataNum * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(Relu6TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/relu6/tests/ut/op_kernel/relu6_data", "relu6_data");
    kernel_ut::RunGenData("./relu6_data", {"'(128)'", "float16"});

    std::string path = kernel_ut::GetTestWorkDir();

    Relu6TilingData* tilingData = reinterpret_cast<Relu6TilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(dataNum);
    tilingData->dataType = 0;

    ReadFile(path + "/relu6_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelRelu6 = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        relu6_kernel<half>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelRelu6, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/relu6_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(relu6_test, test_case_int32_small)
{
    size_t dataNum = 64;
    size_t xByteSize = dataNum * sizeof(int32_t);
    size_t yByteSize = dataNum * sizeof(int32_t);
    size_t tiling_data_size = sizeof(Relu6TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/relu6/tests/ut/op_kernel/relu6_data", "relu6_data");
    kernel_ut::RunGenData("./relu6_data", {"'(64)'", "int32"});

    std::string path = kernel_ut::GetTestWorkDir();

    Relu6TilingData* tilingData = reinterpret_cast<Relu6TilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(dataNum);
    tilingData->dataType = 2;

    ReadFile(path + "/relu6_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelRelu6 = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        relu6_kernel<int32_t>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelRelu6, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/relu6_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(relu6_test, test_case_bf16_small)
{
    size_t dataNum = 128;
    size_t xByteSize = dataNum * sizeof(uint16_t);
    size_t yByteSize = dataNum * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(Relu6TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/relu6/tests/ut/op_kernel/relu6_data", "relu6_data");
    kernel_ut::RunGenData("./relu6_data", {"'(128)'", "bfloat16"});

    std::string path = kernel_ut::GetTestWorkDir();

    Relu6TilingData* tilingData = reinterpret_cast<Relu6TilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(dataNum);
    tilingData->dataType = 3;

    ReadFile(path + "/relu6_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelRelu6 = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        relu6_kernel<bfloat16_t>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelRelu6, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/relu6_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(relu6_test, test_case_fp16_multi_loop)
{
    size_t dataNum = 512;
    size_t xByteSize = dataNum * sizeof(uint16_t);
    size_t yByteSize = dataNum * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(Relu6TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/relu6/tests/ut/op_kernel/relu6_data", "relu6_data");
    kernel_ut::RunGenData("./relu6_data", {"'(512)'", "float16"});

    std::string path = kernel_ut::GetTestWorkDir();

    Relu6TilingData* tilingData = reinterpret_cast<Relu6TilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(128);
    tilingData->dataType = 0;

    ReadFile(path + "/relu6_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelRelu6 = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        relu6_kernel<half>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelRelu6, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/relu6_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}