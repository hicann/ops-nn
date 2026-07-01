/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "swiglu_mx_quant_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

#ifdef __CCE_KT_TEST__
#include "../../../op_kernel/swiglu_mx_quant_apt.cpp"
#endif

using namespace std;
using namespace SwigluMxQuantOp;

class SwigluMxQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        cout << "SwigluMxQuantKernelTest SetUp" << endl;
    }
    static void TearDownTestCase() {
        cout << "SwigluMxQuantKernelTest TearDown" << endl;
    }
};

static void InitTilingData(SwigluMxQuantTilingData* tilingData, size_t batchSize, size_t seqLen,
                            size_t hiddenDim, int64_t scaleAlg)
{
    int64_t M = batchSize * seqLen;
    int64_t N = hiddenDim;

    tilingData->usedCoreNum = 24;
    tilingData->inputDim0 = 1;
    tilingData->inputDim1 = M;
    tilingData->inputDim2 = N;

    int64_t dimNBlockNum = (N + 255) / 256;
    int64_t dimMBlockNum = (M + 63) / 64;
    tilingData->dimNBlockNum = dimNBlockNum;
    tilingData->dimNTail = (N % 256 == 0) ? 256 : N % 256;
    tilingData->dimMBlockNum = dimMBlockNum;
    tilingData->dimMTail = (M % 64 == 0) ? 64 : M % 64;
    tilingData->blockCountPerBatch = 0;

    tilingData->nCoreNum = 1;
    tilingData->bCoreNum = 1;
    tilingData->mCorePerB = 24;
    tilingData->maxBasicNumUbDim2 = 1;
    tilingData->maxBasicNumUbDim1 = 1;
    tilingData->frontCoreNum = 0;
    tilingData->tailCoreBasicNumDim1 = 0;
    tilingData->groupIndexNum = 0;

    tilingData->activateLeft = 0;
    tilingData->swigluMode = 0;
    tilingData->scaleAlg = scaleAlg;
    tilingData->groupMode = 0;
    tilingData->clampLimit = 7.0f;
    tilingData->gluAlpha = 1.702f;
    tilingData->gluBias = 1.0f;
}

// tilingKey=12: bitpacked by GET_TPL_TILING_KEY: gi_idx=0, ax_idx=1, act_idx=1, rm_idx=0
TEST_F(SwigluMxQuantKernelTest, test_swiglu_mx_quant_fp16_to_fp8_e4m3)
{
    size_t batchSize = 8;
    size_t seqLen = 128;
    size_t hiddenDim = 4096;
    int64_t scaleAlg = 0;
    uint64_t tilingKey = 12;

    size_t inputSize = batchSize * seqLen * hiddenDim * 2 * sizeof(uint16_t);
    size_t outputSize = batchSize * seqLen * hiddenDim * sizeof(uint8_t);
    size_t scaleSize = batchSize * seqLen * ((hiddenDim + 63) / 64) * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(SwigluMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* group_index = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    SwigluMxQuantTilingData* tilingData = reinterpret_cast<SwigluMxQuantTilingData*>(tiling);
    InitTilingData(tilingData, batchSize, seqLen, hiddenDim, scaleAlg);

    auto KernelFunc = [](GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale,
                         GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_mx_quant<TPL_NO_GROUP_INDEX, TPL_AXIS_LAST, TPL_ACTIVATE_LAST, TPL_RINT>(
            x, group_index, y, mxscale, workspace, tiling);
    };
    uint32_t blockDim = 24;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, group_index, y, mxscale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluMxQuantKernelTest, test_swiglu_mx_quant_bf16_to_fp8_e5m2)
{
    size_t batchSize = 4;
    size_t seqLen = 64;
    size_t hiddenDim = 1024;
    int64_t scaleAlg = 0;
    uint64_t tilingKey = 12;

    size_t inputSize = batchSize * seqLen * hiddenDim * 2 * sizeof(uint16_t);
    size_t outputSize = batchSize * seqLen * hiddenDim * sizeof(uint8_t);
    size_t scaleSize = batchSize * seqLen * ((hiddenDim + 63) / 64) * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(SwigluMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* group_index = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    SwigluMxQuantTilingData* tilingData = reinterpret_cast<SwigluMxQuantTilingData*>(tiling);
    InitTilingData(tilingData, batchSize, seqLen, hiddenDim, scaleAlg);

    auto KernelFunc = [](GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale,
                         GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_mx_quant<TPL_NO_GROUP_INDEX, TPL_AXIS_LAST, TPL_ACTIVATE_LAST, TPL_RINT>(
            x, group_index, y, mxscale, workspace, tiling);
    };
    uint32_t blockDim = 24;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, group_index, y, mxscale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluMxQuantKernelTest, test_swiglu_mx_quant_fp16_to_fp4_e2m1)
{
    size_t batchSize = 4;
    size_t seqLen = 256;
    size_t hiddenDim = 2048;
    int64_t scaleAlg = 0;
    uint64_t tilingKey = 44;  // FLOOR

    size_t inputSize = batchSize * seqLen * hiddenDim * 2 * sizeof(uint16_t);
    size_t outputSize = batchSize * seqLen * hiddenDim * sizeof(uint8_t);
    size_t scaleSize = batchSize * seqLen * ((hiddenDim + 63) / 64) * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(SwigluMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* group_index = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    SwigluMxQuantTilingData* tilingData = reinterpret_cast<SwigluMxQuantTilingData*>(tiling);
    InitTilingData(tilingData, batchSize, seqLen, hiddenDim, scaleAlg);

    auto KernelFunc = [](GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale,
                         GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_mx_quant<TPL_NO_GROUP_INDEX, TPL_AXIS_LAST, TPL_ACTIVATE_LAST, TPL_FLOOR>(
            x, group_index, y, mxscale, workspace, tiling);
    };
    uint32_t blockDim = 24;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, group_index, y, mxscale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluMxQuantKernelTest, test_swiglu_mx_quant_scale_alg_1)
{
    size_t batchSize = 4;
    size_t seqLen = 128;
    size_t hiddenDim = 2048;
    int64_t scaleAlg = 1;
    uint64_t tilingKey = 12;

    size_t inputSize = batchSize * seqLen * hiddenDim * 2 * sizeof(uint16_t);
    size_t outputSize = batchSize * seqLen * hiddenDim * sizeof(uint8_t);
    size_t scaleSize = batchSize * seqLen * ((hiddenDim + 63) / 64) * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(SwigluMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* group_index = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    SwigluMxQuantTilingData* tilingData = reinterpret_cast<SwigluMxQuantTilingData*>(tiling);
    InitTilingData(tilingData, batchSize, seqLen, hiddenDim, scaleAlg);

    auto KernelFunc = [](GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale,
                         GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_mx_quant<TPL_NO_GROUP_INDEX, TPL_AXIS_LAST, TPL_ACTIVATE_LAST, TPL_RINT>(
            x, group_index, y, mxscale, workspace, tiling);
    };
    uint32_t blockDim = 24;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, group_index, y, mxscale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluMxQuantKernelTest, test_swiglu_mx_quant_small_shape)
{
    size_t batchSize = 1;
    size_t seqLen = 1;
    size_t hiddenDim = 64;
    int64_t scaleAlg = 0;
    uint64_t tilingKey = 12;

    size_t inputSize = batchSize * seqLen * hiddenDim * 2 * sizeof(uint16_t);
    size_t outputSize = batchSize * seqLen * hiddenDim * sizeof(uint8_t);
    size_t scaleSize = batchSize * seqLen * ((hiddenDim + 63) / 64) * 2 * sizeof(uint8_t);
    size_t tilingDataSize = sizeof(SwigluMxQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* group_index = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* mxscale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    SwigluMxQuantTilingData* tilingData = reinterpret_cast<SwigluMxQuantTilingData*>(tiling);
    InitTilingData(tilingData, batchSize, seqLen, hiddenDim, scaleAlg);

    auto KernelFunc = [](GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale,
                         GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_mx_quant<TPL_NO_GROUP_INDEX, TPL_AXIS_LAST, TPL_ACTIVATE_LAST, TPL_RINT>(
            x, group_index, y, mxscale, workspace, tiling);
    };
    uint32_t blockDim = 24;
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, group_index, y, mxscale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(mxscale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
