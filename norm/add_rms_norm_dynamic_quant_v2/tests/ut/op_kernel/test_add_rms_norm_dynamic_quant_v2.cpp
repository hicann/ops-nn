/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "add_rms_norm_dynamic_quant_v2_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void add_rms_norm_dynamic_quant_v2(
    uint8_t* x1, uint8_t* x2, uint8_t* gamma, uint8_t* scales1, uint8_t* scales2, uint8_t* y1, uint8_t* y2, uint8_t* y3,
    uint8_t* y4, uint8_t* x, uint8_t* outScale1, uint8_t* outScale2, uint8_t* workspace, uint8_t* tiling);

class add_rms_norm_dynamic_quant_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "add_rms_norm_dynamic_quant_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_rms_norm_dynamic_quant_v2_test TearDown\n" << endl;
    }
};

TEST_F(add_rms_norm_dynamic_quant_v2_test, test_case_dynamic_dual_smooth)
{
    int N = 3;
    int D = 256;
    size_t rowsByteSize = N * D * sizeof(int16_t);
    size_t weightBetaByteSize = D * sizeof(int16_t);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t reducedByteSize = N * D * sizeof(float);
    size_t outY3ByteSize = N * D * sizeof(float);
    size_t tilingDataSize = sizeof(AddRmsNormDynamicQuantV2TilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth1 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth2 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(outY3ByteSize);
    uint8_t* y4 = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 1);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRmsNormDynamicQuantV2TilingData* tilingDatafromBin =
        reinterpret_cast<AddRmsNormDynamicQuantV2TilingData*>(tiling);

    tilingDatafromBin->useCore = blockDim;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numLastDimAligned = (D + 32 - 1) / 32 * 32;
    tilingDatafromBin->firstDimPerCore = 1;
    tilingDatafromBin->firstDimPerCoreTail = 1;
    tilingDatafromBin->firstDimPerLoop = 1;
    tilingDatafromBin->lastDimLoopNum = 1;
    tilingDatafromBin->lastDimSliceLen = 8864;
    tilingDatafromBin->lastDimSliceLenTail = D;
    tilingDatafromBin->smoothNum = 2;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = (1.0 / D);

    // dual normal bf16/fp16
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        add_rms_norm_dynamic_quant_v2, blockDim, x1, x2, gamma, smooth1, smooth2, y1, y2, y3, y4, x, outScale1,
        outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        add_rms_norm_dynamic_quant_v2, blockDim, x1, x2, gamma, smooth1, smooth2, y1, y2, y3, y4, x, outScale1,
        outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(smooth1);
    AscendC::GmFree(smooth2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(y3);
    AscendC::GmFree(y4);
    AscendC::GmFree(x);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_rms_norm_dynamic_quant_v2_test, test_case_dynamic_case_3)
{
    int N = 2;
    int D = 25600;
    size_t rowsByteSize = N * D * sizeof(int16_t);
    size_t weightBetaByteSize = D * sizeof(int16_t);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t reducedByteSize = N * D * sizeof(float);
    size_t outY3ByteSize = N * D * sizeof(float);
    size_t tilingDataSize = sizeof(AddRmsNormDynamicQuantV2TilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth1 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth2 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(outY3ByteSize);
    uint8_t* y4 = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 1);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 2;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRmsNormDynamicQuantV2TilingData* tilingDatafromBin =
        reinterpret_cast<AddRmsNormDynamicQuantV2TilingData*>(tiling);

    tilingDatafromBin->useCore = blockDim;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numLastDimAligned = D;
    tilingDatafromBin->firstDimPerCore = 1;
    tilingDatafromBin->firstDimPerCoreTail = 1;
    tilingDatafromBin->firstDimPerLoop = 1;
    tilingDatafromBin->lastDimLoopNum = 2;
    tilingDatafromBin->lastDimSliceLen = 8864;
    tilingDatafromBin->lastDimSliceLenTail = 7872;
    tilingDatafromBin->smoothNum = 2;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = (1.0 / D);

    // dual normal bf16/fp16
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(
        add_rms_norm_dynamic_quant_v2, blockDim, x1, x2, gamma, smooth1, smooth2, y1, y2, y3, y4, x, outScale1,
        outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(smooth1);
    AscendC::GmFree(smooth2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(y3);
    AscendC::GmFree(y4);
    AscendC::GmFree(x);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}