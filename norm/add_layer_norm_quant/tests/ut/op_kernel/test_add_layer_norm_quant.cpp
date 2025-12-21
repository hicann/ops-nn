/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "add_layer_norm_quant_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void add_layer_norm_quant(
    uint8_t* x1, uint8_t* x2, uint8_t* gamma, uint8_t* beta, uint8_t* bias, uint8_t* scales1, uint8_t* scales2,
    uint8_t* zeroOffset1, uint8_t* zeroOffset2, uint8_t* y1, uint8_t* y2, uint8_t* x, uint8_t* outScale1,
    uint8_t* outScale2, uint8_t* workspace, uint8_t* tiling);

class add_layer_norm_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_layer_norm_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_layer_norm_quant_test TearDown\n" << endl;
    }
};

TEST_F(add_layer_norm_quant_test, test_case_dynamic_normal)
{
    int N = 3;
    int D = 5120;
    size_t inByteSize = N * D * sizeof(int16_t);
    size_t gammaBetaByteSize = D * sizeof(int16_t);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t outDataByteSize = N * D * sizeof(int16_t);
    size_t reducedByteSize = N * sizeof(float);
    size_t tilingDataSize = sizeof(AddLayerNormQuantTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* smooth1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* smooth2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outDataByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormQuantTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormQuantTilingData*>(tiling);

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
    tilingDatafromBin->scaleOffsetMode = 200;
    tilingDatafromBin->isPerTensor = 0;
    // dual normal bf16/fp16
    // ICPU_SET_TILING_KEY(1100);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(1101);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(1102);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(3100);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(3101);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(3102);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // Sole normal bf16/fp16
    tilingDatafromBin->scaleOffsetMode = 100;
    ICPU_SET_TILING_KEY(1102);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, nullptr, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // None normal bf16/fp16
    tilingDatafromBin->scaleOffsetMode = 0;
    ICPU_SET_TILING_KEY(1102);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, nullptr, nullptr, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(bias1);
    AscendC::GmFree(smooth1);
    AscendC::GmFree(smooth2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_quant_test, test_case_dynamic_single_row)
{
    int N = 3;
    int D = 11264;
    size_t inByteSize = N * D * sizeof(int16_t);
    size_t gammaBetaByteSize = D * sizeof(int16_t);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t outDataByteSize = N * D * sizeof(int16_t);
    size_t reducedByteSize = N * sizeof(float);
    size_t tilingDataSize = sizeof(AddLayerNormQuantTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* smooth1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* smooth2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outDataByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormQuantTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormQuantTilingData*>(tiling);

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
    tilingDatafromBin->scaleOffsetMode = 200;
    tilingDatafromBin->isPerTensor = 0;
    // dual normal bf16/fp16
    // ICPU_SET_TILING_KEY(1120);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(1121);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(1122);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(3120);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(3121);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(3122);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // Sole normal bf16/fp16
    tilingDatafromBin->scaleOffsetMode = 100;
    ICPU_SET_TILING_KEY(1122);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, nullptr, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // None normal bf16/fp16
    tilingDatafromBin->scaleOffsetMode = 0;
    ICPU_SET_TILING_KEY(1122);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, nullptr, nullptr, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(bias1);
    AscendC::GmFree(smooth1);
    AscendC::GmFree(smooth2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_quant_test, test_case_static_normal)
{
    int N = 3;
    int D = 5120;
    size_t inByteSize = N * D * sizeof(int16_t);
    size_t gammaBetaByteSize = D * sizeof(int16_t);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t outDataByteSize = N * D * sizeof(int16_t);
    size_t reducedByteSize = N * sizeof(float);
    size_t tilingDataSize = sizeof(AddLayerNormQuantTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* offsets1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* offsets2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outDataByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(1);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(1);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormQuantTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormQuantTilingData*>(tiling);

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
    tilingDatafromBin->scaleOffsetMode = 211;
    tilingDatafromBin->isPerTensor = 0;
    // dual normal bf16/fp16
    // ICPU_SET_TILING_KEY(1000);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(1001);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, scales1, scales2, offsets1, offsets2, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(1002);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(3000);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(3001);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, scales1, scales2, offsets1, offsets2, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // ICPU_SET_TILING_KEY(3002);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    // Dual and No offset
    tilingDatafromBin->scaleOffsetMode = 200;
    ICPU_SET_TILING_KEY(1002);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // Sole and has offset
    tilingDatafromBin->scaleOffsetMode = 110;
    // ICPU_SET_TILING_KEY(1002);
    // ICPU_RUN_KF(
    //     add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, nullptr, offsets1, nullptr, y1, y2, x,
    //     outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    // Sole and no offset
    tilingDatafromBin->scaleOffsetMode = 100;
    ICPU_SET_TILING_KEY(1002);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, nullptr, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(bias1);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(offsets1);
    AscendC::GmFree(offsets2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// TEST_F(add_layer_norm_quant_test, test_case_static_single_row)
// {
//     int N = 3;
//     int D = 11264;
//     size_t inByteSize = N * D * sizeof(int16_t);
//     size_t gammaBetaByteSize = D * sizeof(int16_t);
//     size_t outQuantByteSize = N * D * sizeof(int8_t);
//     size_t outDataByteSize = N * D * sizeof(int16_t);
//     size_t reducedByteSize = N * sizeof(float);
//     size_t tilingDataSize = sizeof(AddLayerNormQuantTilingData);

//     uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
//     uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* bias1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
//     uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* offsets1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* offsets2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);

//     uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
//     uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(outDataByteSize);
//     uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(1);
//     uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(1);

//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
//     uint32_t blockDim = 3;
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     AddLayerNormQuantTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormQuantTilingData*>(tiling);

//     tilingDatafromBin->numCore = blockDim;
//     tilingDatafromBin->numLastDim = D;
//     tilingDatafromBin->numFirstDim = N;
//     tilingDatafromBin->firstDimPerCore = 1;
//     tilingDatafromBin->firstDimPerCoreTail = 1;
//     tilingDatafromBin->firstDimPerTime = 1;
//     tilingDatafromBin->lastDimPerTime = D;
//     tilingDatafromBin->eps = 0.01;
//     tilingDatafromBin->aveFactor = 1.0 / D;
//     tilingDatafromBin->colMoveCnt = 1;
//     tilingDatafromBin->colTail = D;
//     tilingDatafromBin->isXOut = 1;
//     tilingDatafromBin->scaleOffsetMode = 211;
//     tilingDatafromBin->isPerTensor = 0;
//     // dual normal bf16/fp16
//     ICPU_SET_TILING_KEY(1020);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(1021);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(1022);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(3020);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(3021);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(3022);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

//     tilingDatafromBin->scaleOffsetMode = 110;
//     // dual normal bf16/fp16
//     ICPU_SET_TILING_KEY(1020);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(1021);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(1022);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(3020);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(3021);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(3022);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

//     // Dual and No offset
//     tilingDatafromBin->scaleOffsetMode = 200;
//     ICPU_SET_TILING_KEY(1022);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, nullptr, nullptr, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     // Sole and no offset
//     tilingDatafromBin->scaleOffsetMode = 100;
//     ICPU_SET_TILING_KEY(1022);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, nullptr, nullptr, nullptr, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x1);
//     AscendC::GmFree(x2);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(beta);
//     AscendC::GmFree(bias);
//     AscendC::GmFree(bias1);
//     AscendC::GmFree(scales1);
//     AscendC::GmFree(scales2);
//     AscendC::GmFree(offsets1);
//     AscendC::GmFree(offsets2);
//     AscendC::GmFree(y1);
//     AscendC::GmFree(y2);
//     AscendC::GmFree(x);
//     AscendC::GmFree(outScale1);
//     AscendC::GmFree(outScale2);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

TEST_F(add_layer_norm_quant_test, test_case_dynamic_fp32)
{
    int N = 3;
    int D = 5120;
    size_t inByteSize = N * D * sizeof(float);
    size_t gammaBetaByteSize = D * sizeof(float);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t outDataByteSize = N * D * sizeof(float);
    size_t reducedByteSize = N * sizeof(float);
    size_t tilingDataSize = sizeof(AddLayerNormQuantTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
    uint8_t* smooth1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* smooth2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outDataByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormQuantTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormQuantTilingData*>(tiling);

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
    tilingDatafromBin->scaleOffsetMode = 200;
    tilingDatafromBin->isPerTensor = 0;
    // dual normal fp32
    ICPU_SET_TILING_KEY(2100);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(2101);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(2102);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(2120);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(2121);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(2122);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, smooth1, smooth2, nullptr, nullptr, y1, y2, x,
        outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(bias1);
    AscendC::GmFree(smooth1);
    AscendC::GmFree(smooth2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// TEST_F(add_layer_norm_quant_test, test_case_static_fp32)
// {
//     int N = 3;
//     int D = 9264;
//     size_t inByteSize = N * D * sizeof(float);
//     size_t gammaBetaByteSize = D * sizeof(float);
//     size_t outQuantByteSize = N * D * sizeof(int8_t);
//     size_t outDataByteSize = N * D * sizeof(float);
//     size_t reducedByteSize = N * sizeof(float);
//     size_t tilingDataSize = sizeof(AddLayerNormQuantTilingData);

//     uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
//     uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inByteSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* bias1 = (uint8_t*)AscendC::GmAlloc(inByteSize);
//     uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* offsets1 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
//     uint8_t* offsets2 = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);

//     uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
//     uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(outDataByteSize);
//     uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(1);
//     uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(1);

//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
//     uint32_t blockDim = 3;
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     AddLayerNormQuantTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormQuantTilingData*>(tiling);

//     tilingDatafromBin->numCore = blockDim;
//     tilingDatafromBin->numLastDim = D;
//     tilingDatafromBin->numFirstDim = N;
//     tilingDatafromBin->firstDimPerCore = 1;
//     tilingDatafromBin->firstDimPerCoreTail = 1;
//     tilingDatafromBin->firstDimPerTime = 1;
//     tilingDatafromBin->lastDimPerTime = D;
//     tilingDatafromBin->eps = 0.01;
//     tilingDatafromBin->aveFactor = 1.0 / D;
//     tilingDatafromBin->colMoveCnt = 1;
//     tilingDatafromBin->colTail = D;
//     tilingDatafromBin->isXOut = 1;
//     tilingDatafromBin->scaleOffsetMode = 211;
//     tilingDatafromBin->isPerTensor = 0;
//     // dual normal bf16/fp16
//     ICPU_SET_TILING_KEY(2020);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(2021);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(2022);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

//     tilingDatafromBin->scaleOffsetMode = 110;
//     // dual normal bf16/fp16
//     ICPU_SET_TILING_KEY(2020);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(2021);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias1, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));
//     ICPU_SET_TILING_KEY(2022);
//     ICPU_RUN_KF(
//         add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, offsets1, offsets2, y1, y2, x,
//         outScale1, outScale2, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x1);
//     AscendC::GmFree(x2);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(beta);
//     AscendC::GmFree(bias);
//     AscendC::GmFree(bias1);
//     AscendC::GmFree(scales1);
//     AscendC::GmFree(scales2);
//     AscendC::GmFree(offsets1);
//     AscendC::GmFree(offsets2);
//     AscendC::GmFree(y1);
//     AscendC::GmFree(y2);
//     AscendC::GmFree(x);
//     AscendC::GmFree(outScale1);
//     AscendC::GmFree(outScale2);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }
