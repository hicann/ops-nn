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
#include "../../../op_kernel/batch_normalization_grad.cpp"
#include "../../../op_kernel/batch_normalization_grad_tiling_data.h"

using namespace std;

class BatchNormalizationGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment(
            "experimental/norm/batch_normalization_grad/tests/ut/op_kernel/batch_normalization_grad_data",
            "batch_normalization_grad_data");
    }
    static void TearDownTestCase() {}
};

// ============================================================================
// float32 测试用例
// ============================================================================

TEST_F(BatchNormalizationGradTest, test_case_float32_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./batch_normalization_grad_data", {"'(16, 16, 64, 2)'", "float32"});

    // 2. 申请内存并加载 6 个输入
    uint32_t spatialElems = 16 * 16 * 64 * 2; // 32768
    uint32_t channelElems = 16;
    size_t spatialByteSize = spatialElems * sizeof(float);
    size_t channelByteSize = channelElems * sizeof(float);

    uint8_t* grad_out = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./batch_normalization_grad_data/float32_grad_output_t_batch_normalization_grad.bin",
             spatialByteSize, grad_out, spatialByteSize);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./batch_normalization_grad_data/float32_input_t_batch_normalization_grad.bin",
             spatialByteSize, input, spatialByteSize);

    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/float32_weight_t_batch_normalization_grad.bin",
             channelByteSize, weight, channelByteSize);

    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/float32_bias_t_batch_normalization_grad.bin",
             channelByteSize, bias, channelByteSize);

    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/float32_save_mean_t_batch_normalization_grad.bin",
             channelByteSize, save_mean, channelByteSize);

    uint8_t* save_invstd = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/float32_save_invstd_t_batch_normalization_grad.bin",
             channelByteSize, save_invstd, channelByteSize);

    // 3. 分配 3 个输出 buffer
    uint8_t* grad_input = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    uint8_t* grad_weight = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    uint8_t* grad_bias = (uint8_t*)AscendC::GmAlloc(channelByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(BatchNormalizationGradTilingData));

    // 4. 手动构造 TilingData（[16,16,64,2] 整空间单核模式）
    BatchNormalizationGradTilingData* tilingData =
        reinterpret_cast<BatchNormalizationGradTilingData*>(tl);
    tilingData->numChannels = 16;
    tilingData->numBatches = 16;
    tilingData->numSpatial = 128;
    tilingData->m = 2048;
    tilingData->mFloat = 2048.0f;
    tilingData->coreNum = 1;
    tilingData->channelsPerCore = 16;
    tilingData->tailChannels = 0;
    tilingData->spatialSplitMode = 0;
    tilingData->spatialTileSize = 128;
    tilingData->spatialTileNum = 1;
    tilingData->spatialTailSize = 128;
    tilingData->batchStrideAligned = 1;
    tilingData->batchesPerTile = 16;
    tilingData->batchGroups = 1;
    tilingData->tailBatches = 16;
    tilingData->paddedSpatial = 128;
    tilingData->epsilon = 1e-5f;
    tilingData->blockSize = 32;
    tilingData->typeLength = 4;
    tilingData->elementsPerBlock = 8;
    tilingData->needPerBatchLoad = 0;

    // 5. lambda 包装模板实例化，blockDim=1
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR go, GM_ADDR in, GM_ADDR w, GM_ADDR b,
                  GM_ADDR sm, GM_ADDR si, GM_ADDR gi, GM_ADDR gw,
                  GM_ADDR gb, GM_ADDR ws, GM_ADDR tl) {
        ::batch_normalization_grad<0>(go, in, w, b, sm, si, gi, gw, gb, ws, tl);
    };
    ICPU_RUN_KF(kf, 1, grad_out, input, weight, bias, save_mean,
                save_invstd, grad_input, grad_weight, grad_bias, ws, tl);

    // 6. 写出 3 个输出 bin
    WriteFile("./batch_normalization_grad_data/float32_output_grad_input_t_batch_normalization_grad.bin",
              grad_input, spatialByteSize);
    WriteFile("./batch_normalization_grad_data/float32_output_grad_weight_t_batch_normalization_grad.bin",
              grad_weight, channelByteSize);
    WriteFile("./batch_normalization_grad_data/float32_output_grad_bias_t_batch_normalization_grad.bin",
              grad_bias, channelByteSize);

    // 7. 释放所有 GM 内存
    AscendC::GmFree((void*)grad_out);
    AscendC::GmFree((void*)input);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)save_mean);
    AscendC::GmFree((void*)save_invstd);
    AscendC::GmFree((void*)grad_input);
    AscendC::GmFree((void*)grad_weight);
    AscendC::GmFree((void*)grad_bias);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    // 8. 比对结果
    kernel_ut::RunCompareData("./batch_normalization_grad_data", {"float32"});
}

// ============================================================================
// float16 测试用例
// ============================================================================

TEST_F(BatchNormalizationGradTest, test_case_float16_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./batch_normalization_grad_data", {"'(16, 16, 64, 2)'", "float16"});

    // 2. 申请内存并加载 6 个输入
    uint32_t spatialElems = 16 * 16 * 64 * 2; // 32768
    uint32_t channelElems = 16;
    size_t spatialByteSize = spatialElems * sizeof(half);
    size_t channelByteSize = channelElems * sizeof(half);

    uint8_t* grad_out = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./batch_normalization_grad_data/float16_grad_output_t_batch_normalization_grad.bin",
             spatialByteSize, grad_out, spatialByteSize);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./batch_normalization_grad_data/float16_input_t_batch_normalization_grad.bin",
             spatialByteSize, input, spatialByteSize);

    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/float16_weight_t_batch_normalization_grad.bin",
             channelByteSize, weight, channelByteSize);

    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/float16_bias_t_batch_normalization_grad.bin",
             channelByteSize, bias, channelByteSize);

    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/float16_save_mean_t_batch_normalization_grad.bin",
             channelByteSize, save_mean, channelByteSize);

    uint8_t* save_invstd = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/float16_save_invstd_t_batch_normalization_grad.bin",
             channelByteSize, save_invstd, channelByteSize);

    // 3. 分配 3 个输出 buffer
    uint8_t* grad_input = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    uint8_t* grad_weight = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    uint8_t* grad_bias = (uint8_t*)AscendC::GmAlloc(channelByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(BatchNormalizationGradTilingData));

    // 4. 手动构造 TilingData（[16,16,64,2] 整空间单核模式）
    //    float16: 2B/elem, elementsPerBlock=32/2=16
    BatchNormalizationGradTilingData* tilingData =
        reinterpret_cast<BatchNormalizationGradTilingData*>(tl);
    tilingData->numChannels = 16;
    tilingData->numBatches = 16;
    tilingData->numSpatial = 128;
    tilingData->m = 2048;
    tilingData->mFloat = 2048.0f;
    tilingData->coreNum = 1;
    tilingData->channelsPerCore = 16;
    tilingData->tailChannels = 0;
    tilingData->spatialSplitMode = 0;
    tilingData->spatialTileSize = 128;
    tilingData->spatialTileNum = 1;
    tilingData->spatialTailSize = 128;
    tilingData->batchStrideAligned = 1;
    tilingData->batchesPerTile = 16;
    tilingData->batchGroups = 1;
    tilingData->tailBatches = 16;
    tilingData->paddedSpatial = 128;
    tilingData->epsilon = 1e-5f;
    tilingData->blockSize = 32;
    tilingData->typeLength = 2;
    tilingData->elementsPerBlock = 16;
    tilingData->needPerBatchLoad = 0;

    // 5. lambda 包装模板实例化，blockDim=1
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR go, GM_ADDR in, GM_ADDR w, GM_ADDR b,
                  GM_ADDR sm, GM_ADDR si, GM_ADDR gi, GM_ADDR gw,
                  GM_ADDR gb, GM_ADDR ws, GM_ADDR tl) {
        ::batch_normalization_grad<1>(go, in, w, b, sm, si, gi, gw, gb, ws, tl);
    };
    ICPU_RUN_KF(kf, 1, grad_out, input, weight, bias, save_mean,
                save_invstd, grad_input, grad_weight, grad_bias, ws, tl);

    // 6. 写出 3 个输出 bin
    WriteFile("./batch_normalization_grad_data/float16_output_grad_input_t_batch_normalization_grad.bin",
              grad_input, spatialByteSize);
    WriteFile("./batch_normalization_grad_data/float16_output_grad_weight_t_batch_normalization_grad.bin",
              grad_weight, channelByteSize);
    WriteFile("./batch_normalization_grad_data/float16_output_grad_bias_t_batch_normalization_grad.bin",
              grad_bias, channelByteSize);

    // 7. 释放所有 GM 内存
    AscendC::GmFree((void*)grad_out);
    AscendC::GmFree((void*)input);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)save_mean);
    AscendC::GmFree((void*)save_invstd);
    AscendC::GmFree((void*)grad_input);
    AscendC::GmFree((void*)grad_weight);
    AscendC::GmFree((void*)grad_bias);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    // 8. 比对结果
    kernel_ut::RunCompareData("./batch_normalization_grad_data", {"float16"});
}

// ============================================================================
// bfloat16 测试用例
// ============================================================================

TEST_F(BatchNormalizationGradTest, test_case_bfloat16_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./batch_normalization_grad_data", {"'(16, 16, 64, 2)'", "bfloat16"});

    // 2. 申请内存并加载 6 个输入
    uint32_t spatialElems = 16 * 16 * 64 * 2; // 32768
    uint32_t channelElems = 16;
    size_t spatialByteSize = spatialElems * sizeof(bfloat16_t);
    size_t channelByteSize = channelElems * sizeof(bfloat16_t);

    uint8_t* grad_out = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./batch_normalization_grad_data/bfloat16_grad_output_t_batch_normalization_grad.bin",
             spatialByteSize, grad_out, spatialByteSize);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./batch_normalization_grad_data/bfloat16_input_t_batch_normalization_grad.bin",
             spatialByteSize, input, spatialByteSize);

    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/bfloat16_weight_t_batch_normalization_grad.bin",
             channelByteSize, weight, channelByteSize);

    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/bfloat16_bias_t_batch_normalization_grad.bin",
             channelByteSize, bias, channelByteSize);

    uint8_t* save_mean = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/bfloat16_save_mean_t_batch_normalization_grad.bin",
             channelByteSize, save_mean, channelByteSize);

    uint8_t* save_invstd = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    ReadFile("./batch_normalization_grad_data/bfloat16_save_invstd_t_batch_normalization_grad.bin",
             channelByteSize, save_invstd, channelByteSize);

    // 3. 分配 3 个输出 buffer
    uint8_t* grad_input = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    uint8_t* grad_weight = (uint8_t*)AscendC::GmAlloc(channelByteSize);
    uint8_t* grad_bias = (uint8_t*)AscendC::GmAlloc(channelByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(BatchNormalizationGradTilingData));

    // 4. 手动构造 TilingData（[16,16,64,2] 整空间单核模式）
    //    bfloat16: 2B/elem, elementsPerBlock=32/2=16
    BatchNormalizationGradTilingData* tilingData =
        reinterpret_cast<BatchNormalizationGradTilingData*>(tl);
    tilingData->numChannels = 16;
    tilingData->numBatches = 16;
    tilingData->numSpatial = 128;
    tilingData->m = 2048;
    tilingData->mFloat = 2048.0f;
    tilingData->coreNum = 1;
    tilingData->channelsPerCore = 16;
    tilingData->tailChannels = 0;
    tilingData->spatialSplitMode = 0;
    tilingData->spatialTileSize = 128;
    tilingData->spatialTileNum = 1;
    tilingData->spatialTailSize = 128;
    tilingData->batchStrideAligned = 1;
    tilingData->batchesPerTile = 16;
    tilingData->batchGroups = 1;
    tilingData->tailBatches = 16;
    tilingData->paddedSpatial = 128;
    tilingData->epsilon = 1e-5f;
    tilingData->blockSize = 32;
    tilingData->typeLength = 2;
    tilingData->elementsPerBlock = 16;
    tilingData->needPerBatchLoad = 0;

    // 5. lambda 包装模板实例化，blockDim=1
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR go, GM_ADDR in, GM_ADDR w, GM_ADDR b,
                  GM_ADDR sm, GM_ADDR si, GM_ADDR gi, GM_ADDR gw,
                  GM_ADDR gb, GM_ADDR ws, GM_ADDR tl) {
        ::batch_normalization_grad<2>(go, in, w, b, sm, si, gi, gw, gb, ws, tl);
    };
    ICPU_RUN_KF(kf, 1, grad_out, input, weight, bias, save_mean,
                save_invstd, grad_input, grad_weight, grad_bias, ws, tl);

    // 6. 写出 3 个输出 bin
    WriteFile("./batch_normalization_grad_data/bfloat16_output_grad_input_t_batch_normalization_grad.bin",
              grad_input, spatialByteSize);
    WriteFile("./batch_normalization_grad_data/bfloat16_output_grad_weight_t_batch_normalization_grad.bin",
              grad_weight, channelByteSize);
    WriteFile("./batch_normalization_grad_data/bfloat16_output_grad_bias_t_batch_normalization_grad.bin",
              grad_bias, channelByteSize);

    // 7. 释放所有 GM 内存
    AscendC::GmFree((void*)grad_out);
    AscendC::GmFree((void*)input);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)bias);
    AscendC::GmFree((void*)save_mean);
    AscendC::GmFree((void*)save_invstd);
    AscendC::GmFree((void*)grad_input);
    AscendC::GmFree((void*)grad_weight);
    AscendC::GmFree((void*)grad_bias);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    // 8. 比对结果
    kernel_ut::RunCompareData("./batch_normalization_grad_data", {"bfloat16"});
}
