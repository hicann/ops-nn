/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
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

#define __CELU_TILING_KEY_H__
#include "../../../../op_kernel/arch35/celu_tiling_data.h"
#include "../../../../op_kernel/arch35/celu.h"

using namespace std;

template <typename T>
static void celu_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    const CeluTilingData* tilingData = reinterpret_cast<const CeluTilingData*>(tiling);
    NsCelu::Celu<T> op;
    op.Init(x, y, tilingData);
    op.Process();
}

class celu_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "celu_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "celu TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./celu_data");
    }
};

TEST_F(celu_test, test_case_fp32_1)
{
    size_t dataNum = 10;
    size_t xByteSize = dataNum * sizeof(float);
    size_t yByteSize = dataNum * sizeof(float);
    size_t tiling_data_size = sizeof(CeluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/celu/tests/ut/op_kernel/celu_data", "celu_data");
    kernel_ut::RunGenData("./celu_data", {"'(10)'", "1.0", "1.0", "1.0", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    CeluTilingData* tilingData = reinterpret_cast<CeluTilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(256);
    tilingData->alpha1 = 1.0f;
    tilingData->alpha2 = 1.0f;
    tilingData->alpha3 = 1.0f;

    ReadFile(path + "/celu_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelCelu = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        celu_kernel<float>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelCelu, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/celu_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(celu_test, test_case_fp32_2)
{
    size_t dataNum = 12 * 4;
    size_t xByteSize = dataNum * sizeof(float);
    size_t yByteSize = dataNum * sizeof(float);
    size_t tiling_data_size = sizeof(CeluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/celu/tests/ut/op_kernel/celu_data", "celu_data");
    kernel_ut::RunGenData("./celu_data", {"'(12, 4)'", "1.0", "1.0", "1.0", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    CeluTilingData* tilingData = reinterpret_cast<CeluTilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(2048);
    tilingData->alpha1 = 1.0f;
    tilingData->alpha2 = 1.0f;
    tilingData->alpha3 = 1.0f;

    ReadFile(path + "/celu_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelCelu = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        celu_kernel<float>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelCelu, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/celu_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(celu_test, test_case_fp16_1)
{
    size_t dataNum = 2 * 32;
    size_t xByteSize = dataNum * sizeof(uint16_t);
    size_t yByteSize = dataNum * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CeluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/celu/tests/ut/op_kernel/celu_data", "celu_data");
    kernel_ut::RunGenData("./celu_data", {"'(2, 32)'", "1.0", "1.0", "1.0", "float16"});

    std::string path = kernel_ut::GetTestWorkDir();

    CeluTilingData* tilingData = reinterpret_cast<CeluTilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(2048);
    tilingData->alpha1 = 1.0f;
    tilingData->alpha2 = 1.0f;
    tilingData->alpha3 = 1.0f;

    ReadFile(path + "/celu_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelCelu = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        celu_kernel<half>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelCelu, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/celu_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(celu_test, test_case_fp32_alpha2)
{
    size_t dataNum = 10;
    size_t xByteSize = dataNum * sizeof(float);
    size_t yByteSize = dataNum * sizeof(float);
    size_t tiling_data_size = sizeof(CeluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/celu/tests/ut/op_kernel/celu_data", "celu_data");
    kernel_ut::RunGenData("./celu_data", {"'(10)'", "1.0", "2.0", "1.0", "float32"});

    std::string path = kernel_ut::GetTestWorkDir();

    CeluTilingData* tilingData = reinterpret_cast<CeluTilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(256);
    tilingData->alpha1 = 1.0f;
    tilingData->alpha2 = 2.0f;
    tilingData->alpha3 = 1.0f;

    ReadFile(path + "/celu_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelCelu = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        celu_kernel<float>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelCelu, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/celu_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(celu_test, test_case_fp16_alpha1_half)
{
    size_t dataNum = 2 * 32;
    size_t xByteSize = dataNum * sizeof(uint16_t);
    size_t yByteSize = dataNum * sizeof(uint16_t);
    size_t tiling_data_size = sizeof(CeluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    kernel_ut::SetupTestEnvironment("activation/celu/tests/ut/op_kernel/celu_data", "celu_data");
    kernel_ut::RunGenData("./celu_data", {"'(2, 32)'", "0.5", "1.0", "1.0", "float16"});

    std::string path = kernel_ut::GetTestWorkDir();

    CeluTilingData* tilingData = reinterpret_cast<CeluTilingData*>(tiling);
    tilingData->totalNum = static_cast<int64_t>(dataNum);
    tilingData->blockFactor = static_cast<int64_t>(dataNum);
    tilingData->ubFactor = static_cast<int64_t>(2048);
    tilingData->alpha1 = 0.5f;
    tilingData->alpha2 = 1.0f;
    tilingData->alpha3 = 1.0f;

    ReadFile(path + "/celu_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelCelu = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        celu_kernel<half>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelCelu, blockDim, x, y, workspace, (uint8_t*)(tilingData));
    WriteFile(path + "/celu_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}