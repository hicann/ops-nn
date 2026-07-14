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
#include "../../../op_kernel/layer_normalization_grad.cpp"
#include "../../../op_kernel/layer_normalization_grad_tiling_data.h"

using namespace std;

class LayerNormalizationGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment(
            "experimental/norm/layer_normalization_grad/tests/ut/op_kernel/layer_normalization_grad_data",
            "layer_normalization_grad_data");
    }
    static void TearDownTestCase() {}
};

// ============================================================================
// float32 测试用例
// ============================================================================

TEST_F(LayerNormalizationGradTest, test_case_float32_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./layer_normalization_grad_data", {"'(256, 128)'", "float32"});

    // 2. 申请内存并加载 5 个输入
    //    [256,128] → N=256 rows, D=128 features
    uint32_t N = 256;            // 256
    uint32_t D = 128;            // 128
    uint32_t totalElems = N * D; // 32768
    size_t spatialByteSize = totalElems * sizeof(float);
    size_t rowByteSize = N * sizeof(float);
    size_t featByteSize = D * sizeof(float);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./layer_normalization_grad_data/float32_dy_t_layer_normalization_grad.bin", spatialByteSize, dy,
             spatialByteSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./layer_normalization_grad_data/float32_x_t_layer_normalization_grad.bin", spatialByteSize, x,
             spatialByteSize);

    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(featByteSize);
    ReadFile("./layer_normalization_grad_data/float32_gamma_t_layer_normalization_grad.bin", featByteSize, gamma,
             featByteSize);

    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rowByteSize);
    ReadFile("./layer_normalization_grad_data/float32_mean_t_layer_normalization_grad.bin", rowByteSize, mean,
             rowByteSize);

    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rowByteSize);
    ReadFile("./layer_normalization_grad_data/float32_rstd_t_layer_normalization_grad.bin", rowByteSize, rstd,
             rowByteSize);

    // 3. 分配 3 个输出 buffer
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(featByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(featByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(LayerNormalizationGradTilingData));

    // 4. 手动构造 TilingData（[256,128] float32 单核整列模式）
    LayerNormalizationGradTilingData* tilingData = reinterpret_cast<LayerNormalizationGradTilingData*>(tl);
    tilingData->N = 256;
    tilingData->D = 128;
    tilingData->DPadded = 128;
    tilingData->rowsPerCore = 256;
    tilingData->tailCoreRows = 0;
    tilingData->usedCoreNum = 1;
    tilingData->bufferNum = 1;
    tilingData->needFloatConvert = 0;
    tilingData->colSplitMode = 0;
    tilingData->tileCols = 128;
    tilingData->tileColsAligned = 128;
    tilingData->numColTiles = 1;
    tilingData->workspaceTileStride = 0;
    tilingData->maxCoreRows = 256;

    // 5. lambda 包装模板实例化，blockDim=1
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR dy, GM_ADDR x, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd, GM_ADDR dx, GM_ADDR dgamma,
                 GM_ADDR dbeta, GM_ADDR ws,
                 GM_ADDR tl) { ::layer_normalization_grad<0>(dy, x, gamma, mean, rstd, dx, dgamma, dbeta, ws, tl); };
    ICPU_RUN_KF(kf, 1, dy, x, gamma, mean, rstd, dx, dgamma, dbeta, ws, tl);

    // 6. 写出 3 个输出 bin
    WriteFile("./layer_normalization_grad_data/float32_output_dx_t_layer_normalization_grad.bin", dx, spatialByteSize);
    WriteFile("./layer_normalization_grad_data/float32_output_dgamma_t_layer_normalization_grad.bin", dgamma,
              featByteSize);
    WriteFile("./layer_normalization_grad_data/float32_output_dbeta_t_layer_normalization_grad.bin", dbeta,
              featByteSize);

    // 7. 释放所有 GM 内存
    AscendC::GmFree((void*)dy);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)gamma);
    AscendC::GmFree((void*)mean);
    AscendC::GmFree((void*)rstd);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)dgamma);
    AscendC::GmFree((void*)dbeta);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    // 8. 比对结果
    kernel_ut::RunCompareData("./layer_normalization_grad_data", {"float32"});
}

// ============================================================================
// float16 测试用例
// ============================================================================

TEST_F(LayerNormalizationGradTest, test_case_float16_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./layer_normalization_grad_data", {"'(256, 128)'", "float16"});

    // 2. 申请内存并加载 5 个输入
    uint32_t N = 256;
    uint32_t D = 128;
    uint32_t totalElems = N * D;
    size_t spatialByteSize = totalElems * sizeof(half);
    size_t rowByteSize = N * sizeof(half);
    size_t featByteSize = D * sizeof(half);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./layer_normalization_grad_data/float16_dy_t_layer_normalization_grad.bin", spatialByteSize, dy,
             spatialByteSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./layer_normalization_grad_data/float16_x_t_layer_normalization_grad.bin", spatialByteSize, x,
             spatialByteSize);

    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(featByteSize);
    ReadFile("./layer_normalization_grad_data/float16_gamma_t_layer_normalization_grad.bin", featByteSize, gamma,
             featByteSize);

    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rowByteSize);
    ReadFile("./layer_normalization_grad_data/float16_mean_t_layer_normalization_grad.bin", rowByteSize, mean,
             rowByteSize);

    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rowByteSize);
    ReadFile("./layer_normalization_grad_data/float16_rstd_t_layer_normalization_grad.bin", rowByteSize, rstd,
             rowByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(featByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(featByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(LayerNormalizationGradTilingData));

    // float16: needFloatConvert=1
    LayerNormalizationGradTilingData* tilingData = reinterpret_cast<LayerNormalizationGradTilingData*>(tl);
    tilingData->N = 256;
    tilingData->D = 128;
    tilingData->DPadded = 128;
    tilingData->rowsPerCore = 256;
    tilingData->tailCoreRows = 0;
    tilingData->usedCoreNum = 1;
    tilingData->bufferNum = 1;
    tilingData->needFloatConvert = 1;
    tilingData->colSplitMode = 0;
    tilingData->tileCols = 128;
    tilingData->tileColsAligned = 128;
    tilingData->numColTiles = 1;
    tilingData->workspaceTileStride = 0;
    tilingData->maxCoreRows = 256;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR dy, GM_ADDR x, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd, GM_ADDR dx, GM_ADDR dgamma,
                 GM_ADDR dbeta, GM_ADDR ws,
                 GM_ADDR tl) { ::layer_normalization_grad<1>(dy, x, gamma, mean, rstd, dx, dgamma, dbeta, ws, tl); };
    ICPU_RUN_KF(kf, 1, dy, x, gamma, mean, rstd, dx, dgamma, dbeta, ws, tl);

    WriteFile("./layer_normalization_grad_data/float16_output_dx_t_layer_normalization_grad.bin", dx, spatialByteSize);
    WriteFile("./layer_normalization_grad_data/float16_output_dgamma_t_layer_normalization_grad.bin", dgamma,
              featByteSize);
    WriteFile("./layer_normalization_grad_data/float16_output_dbeta_t_layer_normalization_grad.bin", dbeta,
              featByteSize);

    AscendC::GmFree((void*)dy);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)gamma);
    AscendC::GmFree((void*)mean);
    AscendC::GmFree((void*)rstd);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)dgamma);
    AscendC::GmFree((void*)dbeta);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./layer_normalization_grad_data", {"float16"});
}

// ============================================================================
// bfloat16 测试用例
// ============================================================================

TEST_F(LayerNormalizationGradTest, test_case_bfloat16_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./layer_normalization_grad_data", {"'(256, 128)'", "bfloat16"});

    // 2. 申请内存并加载 5 个输入
    uint32_t N = 256;
    uint32_t D = 128;
    uint32_t totalElems = N * D;
    size_t spatialByteSize = totalElems * sizeof(bfloat16_t);
    size_t rowByteSize = N * sizeof(bfloat16_t);
    size_t featByteSize = D * sizeof(bfloat16_t);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./layer_normalization_grad_data/bfloat16_dy_t_layer_normalization_grad.bin", spatialByteSize, dy,
             spatialByteSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    ReadFile("./layer_normalization_grad_data/bfloat16_x_t_layer_normalization_grad.bin", spatialByteSize, x,
             spatialByteSize);

    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(featByteSize);
    ReadFile("./layer_normalization_grad_data/bfloat16_gamma_t_layer_normalization_grad.bin", featByteSize, gamma,
             featByteSize);

    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(rowByteSize);
    ReadFile("./layer_normalization_grad_data/bfloat16_mean_t_layer_normalization_grad.bin", rowByteSize, mean,
             rowByteSize);

    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rowByteSize);
    ReadFile("./layer_normalization_grad_data/bfloat16_rstd_t_layer_normalization_grad.bin", rowByteSize, rstd,
             rowByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(spatialByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(featByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(featByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(LayerNormalizationGradTilingData));

    // bfloat16: needFloatConvert=1（同 float16，通过 Cast 模式区分）
    LayerNormalizationGradTilingData* tilingData = reinterpret_cast<LayerNormalizationGradTilingData*>(tl);
    tilingData->N = 256;
    tilingData->D = 128;
    tilingData->DPadded = 128;
    tilingData->rowsPerCore = 256;
    tilingData->tailCoreRows = 0;
    tilingData->usedCoreNum = 1;
    tilingData->bufferNum = 1;
    tilingData->needFloatConvert = 1;
    tilingData->colSplitMode = 0;
    tilingData->tileCols = 128;
    tilingData->tileColsAligned = 128;
    tilingData->numColTiles = 1;
    tilingData->workspaceTileStride = 0;
    tilingData->maxCoreRows = 256;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR dy, GM_ADDR x, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd, GM_ADDR dx, GM_ADDR dgamma,
                 GM_ADDR dbeta, GM_ADDR ws,
                 GM_ADDR tl) { ::layer_normalization_grad<2>(dy, x, gamma, mean, rstd, dx, dgamma, dbeta, ws, tl); };
    ICPU_RUN_KF(kf, 1, dy, x, gamma, mean, rstd, dx, dgamma, dbeta, ws, tl);

    WriteFile("./layer_normalization_grad_data/bfloat16_output_dx_t_layer_normalization_grad.bin", dx, spatialByteSize);
    WriteFile("./layer_normalization_grad_data/bfloat16_output_dgamma_t_layer_normalization_grad.bin", dgamma,
              featByteSize);
    WriteFile("./layer_normalization_grad_data/bfloat16_output_dbeta_t_layer_normalization_grad.bin", dbeta,
              featByteSize);

    AscendC::GmFree((void*)dy);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)gamma);
    AscendC::GmFree((void*)mean);
    AscendC::GmFree((void*)rstd);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)dgamma);
    AscendC::GmFree((void*)dbeta);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./layer_normalization_grad_data", {"bfloat16"});
}
