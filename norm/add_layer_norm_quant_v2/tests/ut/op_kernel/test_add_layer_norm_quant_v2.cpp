/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "add_layer_norm_quant_v2_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void add_layer_norm_quant_v2(
    uint8_t* x1, uint8_t* x2, uint8_t* gamma, uint8_t* beta, uint8_t* bias, uint8_t* scales1, uint8_t* scale2,
    uint8_t* zeroOffset1, uint8_t* zeroOffset2, uint8_t* y1, uint8_t* y2, uint8_t* x, uint8_t* layernormRes,
    uint8_t* outScale1, uint8_t* outScale2, uint8_t* workspace, uint8_t* tiling);

class add_layer_norm_quant_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_layer_norm_quant_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_layer_norm_quant_v2_test TearDown\n" << endl;
    }
};

TEST_F(add_layer_norm_quant_v2_test, test_case_static_normal_fp32)
{
    int N = 3;
    int D = 5120;
    size_t inByteSize = N * D * sizeof(float);
    size_t gammaBetaByteSize = D * sizeof(float);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t outDataByteSize = N * D * sizeof(float);
    size_t tilingDataSize = sizeof(AddLayerNormQuantV2TilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroOffset1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroOffset2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outDataByteSize);
    uint8_t* layernormRes = (uint8_t*)AscendC::GmAlloc(outDataByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(32);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormQuantV2TilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormQuantV2TilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 1;
    tilingDatafromBin->firstDimPerCoreTail = 1;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->lastDimPerTime = D;
    tilingDatafromBin->eps = 0.01;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->colMoveCnt = 1;
    tilingDatafromBin->colTail = D;
    tilingDatafromBin->isXOut = 1;
    tilingDatafromBin->isPerTensor = 0;

    tilingDatafromBin->numLastDimAlign = D;
    tilingDatafromBin->numLastDimAlign32 = D;
    tilingDatafromBin->mulLoopFp32 = 80;
    tilingDatafromBin->mulTailFp32 = 0;
    tilingDatafromBin->dstRepStrideFp32 = 128;
    tilingDatafromBin->firstDimPerTimeTail = 1;
    tilingDatafromBin->rowTailPerBlock = 1;
    tilingDatafromBin->rowTailLastBlock = 1;
    tilingDatafromBin->gmOffset = D;

    tilingDatafromBin->scaleOffsetMode = 200;
    ICPU_SET_TILING_KEY(2002);
    ICPU_RUN_KF(
        add_layer_norm_quant_v2, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, zeroOffset1, zeroOffset2, y1, y2, x,
        layernormRes, outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    tilingDatafromBin->scaleOffsetMode = 100;
    ICPU_SET_TILING_KEY(2002);
    ICPU_RUN_KF(
        add_layer_norm_quant_v2, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, zeroOffset1, zeroOffset2, y1, y2, x,
        layernormRes, outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zeroOffset1);
    AscendC::GmFree(zeroOffset2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(layernormRes);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_quant_v2_test, test_case_static_normal_fp16)
{
    int N = 3;
    int D = 5120;
    size_t inByteSize = N * D * sizeof(int16_t);
    size_t gammaBetaByteSize = D * sizeof(int16_t);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t outDataByteSize = N * D * sizeof(int16_t);
    size_t tilingDataSize = sizeof(AddLayerNormQuantV2TilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroOffset1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroOffset2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outDataByteSize);
    uint8_t* layernormRes = (uint8_t*)AscendC::GmAlloc(outDataByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(32);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormQuantV2TilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormQuantV2TilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 1;
    tilingDatafromBin->firstDimPerCoreTail = 1;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->lastDimPerTime = D;
    tilingDatafromBin->eps = 0.01;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->colMoveCnt = 1;
    tilingDatafromBin->colTail = D;
    tilingDatafromBin->isXOut = 1;
    tilingDatafromBin->isPerTensor = 0;

    tilingDatafromBin->numLastDimAlign = D;
    tilingDatafromBin->numLastDimAlign32 = D;
    tilingDatafromBin->mulLoopFp32 = 80;
    tilingDatafromBin->mulTailFp32 = 0;
    tilingDatafromBin->dstRepStrideFp32 = 128;
    tilingDatafromBin->firstDimPerTimeTail = 1;
    tilingDatafromBin->rowTailPerBlock = 1;
    tilingDatafromBin->rowTailLastBlock = 1;
    tilingDatafromBin->gmOffset = D;

    tilingDatafromBin->scaleOffsetMode = 200;
    ICPU_SET_TILING_KEY(1002);
    ICPU_RUN_KF(
        add_layer_norm_quant_v2, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, zeroOffset1, zeroOffset2, y1, y2, x,
        layernormRes, outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    tilingDatafromBin->scaleOffsetMode = 100;
    ICPU_SET_TILING_KEY(1002);
    ICPU_RUN_KF(
        add_layer_norm_quant_v2, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, zeroOffset1, zeroOffset2, y1, y2, x,
        layernormRes, outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zeroOffset1);
    AscendC::GmFree(zeroOffset2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(layernormRes);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
