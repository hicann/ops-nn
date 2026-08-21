/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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
#include "situ_mx_quant_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

#ifdef __CCE_KT_TEST__
#include "../../../op_kernel/situ_mx_quant_apt.cpp"
#endif

using namespace std;
using namespace SituMxQuantOp;

class SituMxQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SituMxQuantKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "SituMxQuantKernelTest TearDown" << endl; }
};

static void InitTilingData(SituMxQuantTilingData* tilingData, size_t batchSize, size_t seqLen, size_t hiddenDim,
                           float beta, float linearBeta, int64_t hasLinearBeta)
{
    int64_t M = batchSize * seqLen;
    int64_t N = hiddenDim;

    tilingData->usedCoreNum = 24;
    tilingData->inputDim0 = 1;
    tilingData->inputDim1 = M;
    tilingData->inputDim2 = N;

    int64_t dimNBlockNum = (N + 255) / 256;
    tilingData->dimNBlockNum = dimNBlockNum;
    tilingData->maxBasicNumUbDim2 = 1;
    tilingData->maxBasicNumUbDim1 = 1;
    tilingData->nCoreNum = 1;
    tilingData->mCorePerB = 24;
    tilingData->frontCoreNum = 0;
    tilingData->tailCoreBasicNumDim1 = 0;

    tilingData->activateLeft = 0;
    tilingData->beta = beta;
    tilingData->linearBeta = linearBeta;
    tilingData->hasLinearBeta = hasLinearBeta;
}

// Test BF16 -> FP8_E4M3FN without linear_beta
TEST_F(SituMxQuantKernelTest, test_situ_mx_quant_bf16_to_fp8_e4m3)
{
    size_t batchSize = 8;
    size_t seqLen = 128;
    size_t hiddenDim = 4096;

    size_t inputSize = batchSize * seqLen * hiddenDim * 2 * sizeof(uint16_t);
    size_t outputSize = batchSize * seqLen * hiddenDim * sizeof(uint8_t);
    size_t scaleSize = batchSize * seqLen * ((hiddenDim + 63) / 64) * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(SituMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* y_scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    for (size_t i = 0; i < inputSize; i++) {
        x[i] = static_cast<uint8_t>(rand() % 256);
    }

    SituMxQuantTilingData* tilingData = reinterpret_cast<SituMxQuantTilingData*>(tiling);
    InitTilingData(tilingData, batchSize, seqLen, hiddenDim, 1.0f, 0.0f, 0);

#undef DTYPE_X
#undef DTYPE_Y
#define DTYPE_X bfloat16_t
#define DTYPE_Y fp8_e4m3fn_t

    auto KernelFunc = [](GM_ADDR x, GM_ADDR y, GM_ADDR y_scale, GM_ADDR workspace, GM_ADDR tiling) {
        ::situ_mx_quant<TPL_NO_LINEAR_BETA, TPL_RINT>(x, y, y_scale, workspace, tiling);
    };

    uint32_t blockDim = 24;
    uint64_t tilingKey = GET_TPL_TILING_KEY(TPL_NO_LINEAR_BETA, TPL_RINT);
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, y, y_scale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(y_scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test BF16 -> FP8_E5M2 without linear_beta
TEST_F(SituMxQuantKernelTest, test_situ_mx_quant_bf16_to_fp8_e5m2)
{
    size_t batchSize = 8;
    size_t seqLen = 128;
    size_t hiddenDim = 4096;

    size_t inputSize = batchSize * seqLen * hiddenDim * 2 * sizeof(uint16_t);
    size_t outputSize = batchSize * seqLen * hiddenDim * sizeof(uint8_t);
    size_t scaleSize = batchSize * seqLen * ((hiddenDim + 63) / 64) * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(SituMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* y_scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    for (size_t i = 0; i < inputSize; i++) {
        x[i] = static_cast<uint8_t>(rand() % 256);
    }

    SituMxQuantTilingData* tilingData = reinterpret_cast<SituMxQuantTilingData*>(tiling);
    InitTilingData(tilingData, batchSize, seqLen, hiddenDim, 1.0f, 0.0f, 0);

#undef DTYPE_X
#undef DTYPE_Y
#define DTYPE_X bfloat16_t
#define DTYPE_Y fp8_e5m2_t

    auto KernelFunc = [](GM_ADDR x, GM_ADDR y, GM_ADDR y_scale, GM_ADDR workspace, GM_ADDR tiling) {
        ::situ_mx_quant<TPL_NO_LINEAR_BETA, TPL_RINT>(x, y, y_scale, workspace, tiling);
    };

    uint32_t blockDim = 24;
    uint64_t tilingKey = GET_TPL_TILING_KEY(TPL_NO_LINEAR_BETA, TPL_RINT);
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, y, y_scale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(y_scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test BF16 -> FP8_E4M3FN with linear_beta
TEST_F(SituMxQuantKernelTest, test_situ_mx_quant_bf16_to_fp8_e4m3_with_linear_beta)
{
    size_t batchSize = 8;
    size_t seqLen = 128;
    size_t hiddenDim = 4096;

    size_t inputSize = batchSize * seqLen * hiddenDim * 2 * sizeof(uint16_t);
    size_t outputSize = batchSize * seqLen * hiddenDim * sizeof(uint8_t);
    size_t scaleSize = batchSize * seqLen * ((hiddenDim + 63) / 64) * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(SituMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* y_scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    for (size_t i = 0; i < inputSize; i++) {
        x[i] = static_cast<uint8_t>(rand() % 256);
    }

    SituMxQuantTilingData* tilingData = reinterpret_cast<SituMxQuantTilingData*>(tiling);
    InitTilingData(tilingData, batchSize, seqLen, hiddenDim, 1.0f, 2.0f, 1);

#undef DTYPE_X
#undef DTYPE_Y
#define DTYPE_X bfloat16_t
#define DTYPE_Y fp8_e4m3fn_t

    auto KernelFunc = [](GM_ADDR x, GM_ADDR y, GM_ADDR y_scale, GM_ADDR workspace, GM_ADDR tiling) {
        ::situ_mx_quant<TPL_HAS_LINEAR_BETA, TPL_RINT>(x, y, y_scale, workspace, tiling);
    };

    uint32_t blockDim = 24;
    uint64_t tilingKey = GET_TPL_TILING_KEY(TPL_HAS_LINEAR_BETA, TPL_RINT);
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, y, y_scale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(y_scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
