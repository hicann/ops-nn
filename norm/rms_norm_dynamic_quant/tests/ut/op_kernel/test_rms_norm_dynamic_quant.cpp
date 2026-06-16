/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "rms_norm_dynamic_quant_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void rms_norm_dynamic_quant(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR smooth, GM_ADDR beta, GM_ADDR y, GM_ADDR outScale,
    GM_ADDR workspace, GM_ADDR tiling);

class RmsNormDynamicQuantKernelTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "RmsNormDynamicQuantKernelTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "RmsNormDynamicQuantKernelTest TearDown\n" << endl;
    }
};

TEST_F(RmsNormDynamicQuantKernelTest, test_case_basic)
{
    int N = 3;
    int D = 256;
    size_t rowsByteSize = N * D * sizeof(int16_t);
    size_t weightBetaByteSize = D * sizeof(int16_t);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t reducedByteSize = N * sizeof(float);
    size_t tilingDataSize = sizeof(RmsNormDynamicQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* outScale = (uint8_t*)AscendC::GmAlloc(reducedByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 4096 + 1);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    char* path_ = get_current_dir_name();
    string path(path_);

    RmsNormDynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<RmsNormDynamicQuantTilingData*>(tiling);

    tilingDatafromBin->useCore = blockDim;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numLastDimAligned = (D + 32 - 1) / 32 * 32;
    tilingDatafromBin->firstDimPerCore = 1;
    tilingDatafromBin->firstDimPerCoreTail = 1;
    tilingDatafromBin->firstDimPerLoop = 1;
    tilingDatafromBin->lastDimLoopNum = 1;
    tilingDatafromBin->lastDimSliceLen = 4096;
    tilingDatafromBin->lastDimSliceLenTail = D;
    tilingDatafromBin->smoothNum1 = 1;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->outQuant1Flag = -1;
    tilingDatafromBin->avgFactor = (1.0 / D);
    tilingDatafromBin->betaFlag = 1;

    // normal tiling
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        rms_norm_dynamic_quant, blockDim, x, gamma, smooth, beta, y, outScale,
        workspace, (uint8_t*)(tilingDatafromBin));
    // single row tiling
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        rms_norm_dynamic_quant, blockDim, x, gamma, smooth, beta, y, outScale,
        workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(smooth);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(outScale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
