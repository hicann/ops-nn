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
#include "../../../op_kernel/cross_entropy_grad.cpp"
#include "../../../op_kernel/cross_entropy_grad_tiling_data.h"

using namespace std;

class CrossEntropyGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment(
            "experimental/loss/cross_entropy_grad/tests/ut/op_kernel/cross_entropy_grad_data",
            "cross_entropy_grad_data");
    }
    static void TearDownTestCase() {}
};

// ============================================================================
// float32 测试用例
// ============================================================================

TEST_F(CrossEntropyGradTest, test_case_float32_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./cross_entropy_grad_data", {"'(256, 32)'", "float32"});

    // 2. 申请内存并加载输入
    uint32_t totalNum = 256 * 32; // 8192
    size_t logitsByteSize = totalNum * sizeof(float);

    std::string logitsFile = "./cross_entropy_grad_data/float32_logits_t_cross_entropy_grad.bin";
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(logitsByteSize);
    ReadFile(logitsFile, logitsByteSize, x1, logitsByteSize);

    uint32_t targetNum = 256;
    size_t targetByteSize = targetNum * sizeof(int32_t);

    std::string targetFile = "./cross_entropy_grad_data/float32_target_t_cross_entropy_grad.bin";
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(targetByteSize);
    ReadFile(targetFile, targetByteSize, x2, targetByteSize);

    size_t outputByteSize = totalNum * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(CrossEntropyGradTilingData));

    // 3. 手动构造 TilingData (blockDim=1, 所有 256 行在一个 core 上)
    //    tileRowNum=16: 每 tile 处理 16 行, 共 256/16=16 个 tile
    CrossEntropyGradTilingData* tilingData = reinterpret_cast<CrossEntropyGradTilingData*>(tl);
    tilingData->rowLen = 32;
    tilingData->totalRows = 256;
    tilingData->smallCoreRowNum = 256;
    tilingData->bigCoreRowNum = 0;
    tilingData->bigCoreNum = 0;
    tilingData->tileRowNum = 16;
    tilingData->smallCoreTileNum = 16;
    tilingData->bigCoreTileNum = 0;
    tilingData->smallTailRowNum = 0;
    tilingData->bigTailRowNum = 0;
    tilingData->colSplitMode = 0;
    tilingData->tileCols = 32;
    tilingData->numColPasses = 1;
    tilingData->dataTypeId = 0;
    tilingData->typeBytes = 4;

    // 4. 执行 kernel
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR p, GM_ADDR t, GM_ADDR g, GM_ADDR w, GM_ADDR tl) {
        ::cross_entropy_grad<0>(p, t, g, w, tl);
    };
    ICPU_RUN_KF(kf, 1, x1, x2, y, ws, tl);

    // 5. 写出结果并比对
    std::string outFile = "./cross_entropy_grad_data/float32_output_t_cross_entropy_grad.bin";
    WriteFile(outFile, y, outputByteSize);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./cross_entropy_grad_data", {"float32"});
}

// ============================================================================
// float16 测试用例
// ============================================================================

TEST_F(CrossEntropyGradTest, test_case_float16_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./cross_entropy_grad_data", {"'(256, 32)'", "float16"});

    // 2. 申请内存并加载输入
    uint32_t totalNum = 256 * 32; // 8192
    size_t logitsByteSize = totalNum * sizeof(half);

    std::string logitsFile = "./cross_entropy_grad_data/float16_logits_t_cross_entropy_grad.bin";
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(logitsByteSize);
    ReadFile(logitsFile, logitsByteSize, x1, logitsByteSize);

    uint32_t targetNum = 256;
    size_t targetByteSize = targetNum * sizeof(int32_t);

    std::string targetFile = "./cross_entropy_grad_data/float16_target_t_cross_entropy_grad.bin";
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(targetByteSize);
    ReadFile(targetFile, targetByteSize, x2, targetByteSize);

    size_t outputByteSize = totalNum * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(CrossEntropyGradTilingData));

    // 3. 手动构造 TilingData
    //    float16: 2B/elem, UB ~196608, tileRowNum 可更大
    CrossEntropyGradTilingData* tilingData = reinterpret_cast<CrossEntropyGradTilingData*>(tl);
    tilingData->rowLen = 32;
    tilingData->totalRows = 256;
    tilingData->smallCoreRowNum = 256;
    tilingData->bigCoreRowNum = 0;
    tilingData->bigCoreNum = 0;
    tilingData->tileRowNum = 32;
    tilingData->smallCoreTileNum = 8;
    tilingData->bigCoreTileNum = 0;
    tilingData->smallTailRowNum = 0;
    tilingData->bigTailRowNum = 0;
    tilingData->colSplitMode = 0;
    tilingData->tileCols = 32;
    tilingData->numColPasses = 1;
    tilingData->dataTypeId = 1;
    tilingData->typeBytes = 2;

    // 4. 执行 kernel
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR p, GM_ADDR t, GM_ADDR g, GM_ADDR w, GM_ADDR tl) {
        ::cross_entropy_grad<1>(p, t, g, w, tl);
    };
    ICPU_RUN_KF(kf, 1, x1, x2, y, ws, tl);

    // 5. 写出结果并比对
    std::string outFile = "./cross_entropy_grad_data/float16_output_t_cross_entropy_grad.bin";
    WriteFile(outFile, y, outputByteSize);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./cross_entropy_grad_data", {"float16"});
}

// ============================================================================
// bfloat16 测试用例
// ============================================================================

TEST_F(CrossEntropyGradTest, test_case_bfloat16_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./cross_entropy_grad_data", {"'(256, 32)'", "bfloat16"});

    // 2. 申请内存并加载输入
    uint32_t totalNum = 256 * 32; // 8192
    size_t logitsByteSize = totalNum * sizeof(bfloat16_t);

    std::string logitsFile = "./cross_entropy_grad_data/bfloat16_logits_t_cross_entropy_grad.bin";
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(logitsByteSize);
    ReadFile(logitsFile, logitsByteSize, x1, logitsByteSize);

    uint32_t targetNum = 256;
    size_t targetByteSize = targetNum * sizeof(int32_t);

    std::string targetFile = "./cross_entropy_grad_data/bfloat16_target_t_cross_entropy_grad.bin";
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(targetByteSize);
    ReadFile(targetFile, targetByteSize, x2, targetByteSize);

    size_t outputByteSize = totalNum * sizeof(bfloat16_t);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(CrossEntropyGradTilingData));

    // 3. 手动构造 TilingData
    //    bfloat16: 2B/elem IO, 内部 float32 计算
    //    bf16 的 UB 计算模型不同（CalcUbUsageBf16），tileRowNum 需要更保守
    CrossEntropyGradTilingData* tilingData = reinterpret_cast<CrossEntropyGradTilingData*>(tl);
    tilingData->rowLen = 32;
    tilingData->totalRows = 256;
    tilingData->smallCoreRowNum = 256;
    tilingData->bigCoreRowNum = 0;
    tilingData->bigCoreNum = 0;
    tilingData->tileRowNum = 16;
    tilingData->smallCoreTileNum = 16;
    tilingData->bigCoreTileNum = 0;
    tilingData->smallTailRowNum = 0;
    tilingData->bigTailRowNum = 0;
    tilingData->colSplitMode = 0;
    tilingData->tileCols = 32;
    tilingData->numColPasses = 1;
    tilingData->dataTypeId = 2;
    tilingData->typeBytes = 2;

    // 4. 执行 kernel
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR p, GM_ADDR t, GM_ADDR g, GM_ADDR w, GM_ADDR tl) {
        ::cross_entropy_grad<2>(p, t, g, w, tl);
    };
    ICPU_RUN_KF(kf, 1, x1, x2, y, ws, tl);

    // 5. 写出结果并比对
    std::string outFile = "./cross_entropy_grad_data/bfloat16_output_t_cross_entropy_grad.bin";
    WriteFile(outFile, y, outputByteSize);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./cross_entropy_grad_data", {"bfloat16"});
}
