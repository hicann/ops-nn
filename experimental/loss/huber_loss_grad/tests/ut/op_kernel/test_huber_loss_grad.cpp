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
#include "../../../op_kernel/huber_loss_grad.cpp"
#include "../../../op_kernel/huber_loss_grad_tiling_data.h"

using namespace std;

class HuberLossGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment("experimental/loss/huber_loss_grad/tests/ut/op_kernel/huber_loss_grad_data",
                                        "huber_loss_grad_data");
    }
    static void TearDownTestCase() {}
};

// ============================================================================
// float32 测试用例
// ============================================================================

TEST_F(HuberLossGradTest, test_case_float32_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./huber_loss_grad_data", {"'(256, 32)'", "float32"});

    // 2. 申请内存并加载输入
    uint32_t totalNum = 256 * 32; // 8192
    size_t inputByteSize = totalNum * sizeof(float);

    std::string predFile = "./huber_loss_grad_data/float32_predictions_t_huber_loss_grad.bin";
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(predFile, inputByteSize, x1, inputByteSize);

    std::string targetFile = "./huber_loss_grad_data/float32_targets_t_huber_loss_grad.bin";
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(targetFile, inputByteSize, x2, inputByteSize);

    size_t outputByteSize = totalNum * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(HuberLossGradTilingData));

    // 3. 手动构造 TilingData
    HuberLossGradTilingData* tilingData = reinterpret_cast<HuberLossGradTilingData*>(tl);
    tilingData->smallCoreDataNum = 8192;
    tilingData->bigCoreDataNum = 0;
    tilingData->finalBigTileNum = 0;
    tilingData->finalSmallTileNum = 4;
    tilingData->tileDataNum = 2048;
    tilingData->smallTailDataNum = 2048;
    tilingData->bigTailDataNum = 0;
    tilingData->tailBlockNum = 0;
    tilingData->dataTypeId = 0;
    tilingData->inputNum = 8192;
    tilingData->delta = 1.0f;
    // Sign高阶接口临时空间：对齐32B，不小于GetSignMaxMinTmpSize的minValue以保证kernel正确
    tilingData->signTmpSize = ((tilingData->tileDataNum * sizeof(float) + 31U) / 32U) * 32U;

    // 4. 执行 kernel
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR p, GM_ADDR t, GM_ADDR g, GM_ADDR w, GM_ADDR tl) { ::huber_loss_grad<0>(p, t, g, w, tl); };
    ICPU_RUN_KF(kf, 1, x1, x2, y, ws, tl);

    // 5. 写出结果并比对
    std::string outFile = "./huber_loss_grad_data/float32_output_t_huber_loss_grad.bin";
    WriteFile(outFile, y, outputByteSize);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./huber_loss_grad_data", {"float32"});
}

// ============================================================================
// float16 测试用例
// ============================================================================

TEST_F(HuberLossGradTest, test_case_float16_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./huber_loss_grad_data", {"'(256, 32)'", "float16"});

    // 2. 申请内存并加载输入
    uint32_t totalNum = 256 * 32; // 8192
    size_t inputByteSize = totalNum * sizeof(half);

    std::string predFile = "./huber_loss_grad_data/float16_predictions_t_huber_loss_grad.bin";
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(predFile, inputByteSize, x1, inputByteSize);

    std::string targetFile = "./huber_loss_grad_data/float16_targets_t_huber_loss_grad.bin";
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(targetFile, inputByteSize, x2, inputByteSize);

    size_t outputByteSize = totalNum * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(HuberLossGradTilingData));

    // 3. 手动构造 TilingData
    // float16: 2B/elem, UB=196608, tileDataNum=4096, blockDim=1
    HuberLossGradTilingData* tilingData = reinterpret_cast<HuberLossGradTilingData*>(tl);
    tilingData->smallCoreDataNum = 8192;
    tilingData->bigCoreDataNum = 0;
    tilingData->finalBigTileNum = 0;
    tilingData->finalSmallTileNum = 2;
    tilingData->tileDataNum = 4096;
    tilingData->smallTailDataNum = 4096;
    tilingData->bigTailDataNum = 0;
    tilingData->tailBlockNum = 0;
    tilingData->dataTypeId = 1;
    tilingData->inputNum = 8192;
    tilingData->delta = 1.0f;
    // Sign高阶接口临时空间（float16按2B对齐32B）
    tilingData->signTmpSize = ((tilingData->tileDataNum * sizeof(half) + 31U) / 32U) * 32U;

    // 4. 执行 kernel
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // float16: schMode=1 → if constexpr 分支 → KernelHuberLossGrad<half>
    auto kf = [](GM_ADDR p, GM_ADDR t, GM_ADDR g, GM_ADDR w, GM_ADDR tl) { ::huber_loss_grad<1>(p, t, g, w, tl); };
    ICPU_RUN_KF(kf, 1, x1, x2, y, ws, tl);

    // 5. 写出结果并比对
    std::string outFile = "./huber_loss_grad_data/float16_output_t_huber_loss_grad.bin";
    WriteFile(outFile, y, outputByteSize);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./huber_loss_grad_data", {"float16"});
}

// ============================================================================
// bfloat16 测试用例
// ============================================================================

TEST_F(HuberLossGradTest, test_case_bfloat16_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./huber_loss_grad_data", {"'(256, 32)'", "bfloat16"});

    // 2. 申请内存并加载输入
    uint32_t totalNum = 256 * 32; // 8192
    size_t inputByteSize = totalNum * sizeof(bfloat16_t);

    std::string predFile = "./huber_loss_grad_data/bfloat16_predictions_t_huber_loss_grad.bin";
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(predFile, inputByteSize, x1, inputByteSize);

    std::string targetFile = "./huber_loss_grad_data/bfloat16_targets_t_huber_loss_grad.bin";
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(targetFile, inputByteSize, x2, inputByteSize);

    size_t outputByteSize = totalNum * sizeof(bfloat16_t);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(HuberLossGradTilingData));

    // 3. 手动构造 TilingData
    // bfloat16: 2B/elem (IO), 内部 float32 计算, UB=196608, tileDataNum=4096
    HuberLossGradTilingData* tilingData = reinterpret_cast<HuberLossGradTilingData*>(tl);
    tilingData->smallCoreDataNum = 8192;
    tilingData->bigCoreDataNum = 0;
    tilingData->finalBigTileNum = 0;
    tilingData->finalSmallTileNum = 2;
    tilingData->tileDataNum = 4096;
    tilingData->smallTailDataNum = 4096;
    tilingData->bigTailDataNum = 0;
    tilingData->tailBlockNum = 0;
    tilingData->dataTypeId = 2;
    tilingData->inputNum = 8192;
    tilingData->delta = 1.0f;
    // Sign高阶接口临时空间（bf16特化下Sign在float上计算，按4B对齐32B）
    tilingData->signTmpSize = ((tilingData->tileDataNum * sizeof(float) + 31U) / 32U) * 32U;

    // 4. 执行 kernel
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // bfloat16: schMode=2 → if constexpr 分支 → KernelHuberLossGrad<bfloat16_t>
    auto kf = [](GM_ADDR p, GM_ADDR t, GM_ADDR g, GM_ADDR w, GM_ADDR tl) { ::huber_loss_grad<2>(p, t, g, w, tl); };
    ICPU_RUN_KF(kf, 1, x1, x2, y, ws, tl);

    // 5. 写出结果并比对
    std::string outFile = "./huber_loss_grad_data/bfloat16_output_t_huber_loss_grad.bin";
    WriteFile(outFile, y, outputByteSize);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./huber_loss_grad_data", {"bfloat16"});
}
