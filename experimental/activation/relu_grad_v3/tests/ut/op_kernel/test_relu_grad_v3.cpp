/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <cstdint>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <string>
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"
#endif
#include "../../../op_kernel/relu_grad_v3.cpp"
#include "../../../op_kernel/relu_grad_v3_tiling_data.h"

using namespace std;

class ReluGradV3Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment("experimental/activation/relu_grad_v3/tests/ut/op_kernel/relu_grad_v3_data",
                                        "relu_grad_v3_data");
    }
    static void TearDownTestCase() {}
};

static void FillSameShapeTiling(ReluGradV3TilingData* tilingData, uint64_t totalNum)
{
    tilingData->totalLength = totalNum;
    tilingData->smallCoreDataNum = totalNum;
    tilingData->bigCoreDataNum = 0;
    tilingData->ubPartDataNum = 2048;
    tilingData->smallCoreTailDataNum = 2048;
    tilingData->bigCoreTailDataNum = 0;
    tilingData->smallCoreLoopNum = totalNum / 2048;
    tilingData->bigCoreLoopNum = 0;
    tilingData->tailBlockNum = 0;
    tilingData->broadcastMode = 0;
    tilingData->dimNum = 2;
    tilingData->xElementNum = totalNum;
    tilingData->yElementNum = totalNum;
    tilingData->outShape[0] = 256;
    tilingData->outShape[1] = 32;
    tilingData->xStrides[0] = 32;
    tilingData->xStrides[1] = 1;
    tilingData->yStrides[0] = 32;
    tilingData->yStrides[1] = 1;
}

static void FillScalarBroadcastTiling(ReluGradV3TilingData* tilingData)
{
    FillSameShapeTiling(tilingData, 8192);
    tilingData->broadcastMode = 1;
    tilingData->yElementNum = 1;
    tilingData->yStrides[0] = 0;
    tilingData->yStrides[1] = 0;
}

static void Fill4DBroadcastTiling(ReluGradV3TilingData* tilingData)
{
    tilingData->totalLength = 120;
    tilingData->smallCoreDataNum = 120;
    tilingData->bigCoreDataNum = 0;
    tilingData->ubPartDataNum = 128;
    tilingData->smallCoreTailDataNum = 120;
    tilingData->bigCoreTailDataNum = 0;
    tilingData->smallCoreLoopNum = 1;
    tilingData->bigCoreLoopNum = 0;
    tilingData->tailBlockNum = 0;
    tilingData->broadcastMode = 1;
    tilingData->dimNum = 4;
    tilingData->xElementNum = 24;
    tilingData->yElementNum = 60;
    tilingData->outShape[0] = 2;
    tilingData->outShape[1] = 3;
    tilingData->outShape[2] = 5;
    tilingData->outShape[3] = 4;
    tilingData->xStrides[0] = 12;
    tilingData->xStrides[1] = 4;
    tilingData->xStrides[2] = 0;
    tilingData->xStrides[3] = 1;
    tilingData->yStrides[0] = 0;
    tilingData->yStrides[1] = 20;
    tilingData->yStrides[2] = 4;
    tilingData->yStrides[3] = 1;
}

// ============================================================================
// float32 测试用例
// ============================================================================

TEST_F(ReluGradV3Test, test_case_float32_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./relu_grad_v3_data", {"'(256, 32)'", "float32"});

    // 2. 申请内存并加载输入
    uint32_t totalNum = 256 * 32; // 8192
    size_t inputByteSize = totalNum * sizeof(float);

    std::string xFile = "./relu_grad_v3_data/float32_x_t_relu_grad_v3.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(xFile, inputByteSize, x, inputByteSize);

    std::string yFile = "./relu_grad_v3_data/float32_y_t_relu_grad_v3.bin";
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(yFile, inputByteSize, y, inputByteSize);

    size_t outputByteSize = totalNum * sizeof(float);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(ReluGradV3TilingData));

    // 3. 手动构造 TilingData（blockDim=1，单核处理全部数据）
    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tl);
    tilingData->totalLength = 8192;
    tilingData->smallCoreDataNum = 8192;
    tilingData->bigCoreDataNum = 0;
    tilingData->ubPartDataNum = 2048;
    tilingData->smallCoreTailDataNum = 2048;
    tilingData->bigCoreTailDataNum = 0;
    tilingData->smallCoreLoopNum = 4;
    tilingData->bigCoreLoopNum = 0;
    tilingData->tailBlockNum = 0;

    // 4. 执行 kernel
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR w, GM_ADDR tl) { ::relu_grad_v3<0>(x, y, z, w, tl); };
    ICPU_RUN_KF(kf, 1, x, y, z, ws, tl);

    // 5. 写出结果并比对
    std::string outFile = "./relu_grad_v3_data/float32_output_z_t_relu_grad_v3.bin";
    WriteFile(outFile, z, outputByteSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)z);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./relu_grad_v3_data", {"float32"});
}

TEST_F(ReluGradV3Test, test_case_float32_y_scalar_broadcast)
{
    kernel_ut::RunGenData("./relu_grad_v3_data", {"'(256, 32)'", "float32", "'(1)'"});

    uint32_t totalNum = 256 * 32;
    size_t xByteSize = totalNum * sizeof(float);
    size_t yByteSize = sizeof(float);

    std::string xFile = "./relu_grad_v3_data/float32_x_t_relu_grad_v3.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(xFile, xByteSize, x, xByteSize);

    std::string yFile = "./relu_grad_v3_data/float32_y_t_relu_grad_v3.bin";
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    ReadFile(yFile, yByteSize, y, yByteSize);

    size_t outputByteSize = totalNum * sizeof(float);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(ReluGradV3TilingData));

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tl);
    FillScalarBroadcastTiling(tilingData);

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR w, GM_ADDR tl) { ::relu_grad_v3<1>(x, y, z, w, tl); };
    ICPU_RUN_KF(kf, 1, x, y, z, ws, tl);

    std::string outFile = "./relu_grad_v3_data/float32_output_z_t_relu_grad_v3.bin";
    WriteFile(outFile, z, outputByteSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)z);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./relu_grad_v3_data", {"float32"});
}

TEST_F(ReluGradV3Test, test_case_float32_4d_broadcast)
{
    kernel_ut::RunGenData("./relu_grad_v3_data", {"'(2, 3, 1, 4)'", "float32", "'(1, 3, 5, 4)'"});

    size_t xByteSize = 24 * sizeof(float);
    size_t yByteSize = 60 * sizeof(float);
    size_t outputByteSize = 120 * sizeof(float);

    std::string xFile = "./relu_grad_v3_data/float32_x_t_relu_grad_v3.bin";
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    ReadFile(xFile, xByteSize, x, xByteSize);

    std::string yFile = "./relu_grad_v3_data/float32_y_t_relu_grad_v3.bin";
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    ReadFile(yFile, yByteSize, y, yByteSize);

    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(ReluGradV3TilingData));

    ReluGradV3TilingData* tilingData = reinterpret_cast<ReluGradV3TilingData*>(tl);
    Fill4DBroadcastTiling(tilingData);

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR w, GM_ADDR tl) { ::relu_grad_v3<1>(x, y, z, w, tl); };
    ICPU_RUN_KF(kf, 1, x, y, z, ws, tl);

    std::string outFile = "./relu_grad_v3_data/float32_output_z_t_relu_grad_v3.bin";
    WriteFile(outFile, z, outputByteSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)z);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./relu_grad_v3_data", {"float32"});
}
