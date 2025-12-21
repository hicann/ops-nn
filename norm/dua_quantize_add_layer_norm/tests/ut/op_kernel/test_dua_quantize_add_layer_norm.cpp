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
#include "dua_quantize_add_layer_norm_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void dua_quantize_add_layer_norm(
    uint8_t* x1, uint8_t* x2, uint8_t* gamma, uint8_t* beta, uint8_t* bias, uint8_t* scales1, uint8_t* scales2,
    uint8_t* zeroPoints1, uint8_t* zeroPoints2, uint8_t* y1, uint8_t* y2, uint8_t* x, uint8_t* workspace,
    uint8_t* tiling);

class dua_quantize_add_layer_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "dua_quantize_add_layer_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "dua_quantize_add_layer_norm_test TearDown\n" << endl;
    }
};

TEST_F(dua_quantize_add_layer_norm_test, test_case_bf16)
{
    int N = 8;
    int D = 1024;
    size_t inputByteSize = N * D * sizeof(int16_t);
    size_t gammaByteSize = D * sizeof(int16_t);
    size_t betaByteSize = D * sizeof(int16_t);
    size_t outputByteSize = N * D * sizeof(int16_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t tiling_data_size = sizeof(DuaQuantizeAddLayerNormTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* zeroPoints1 = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* zeroPoints2 = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 2;

    char* path_ = get_current_dir_name();
    string path(path_);

    DuaQuantizeAddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<DuaQuantizeAddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numRow = N;
    tilingDatafromBin->nlFirstDimPerCore = 5;
    tilingDatafromBin->lFirstDimPerCore = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->lastDimPerTime = 1;
    tilingDatafromBin->aveNum = 1.0 / D;
    tilingDatafromBin->eps = 0.00001;
    tilingDatafromBin->colMoveCnt = 1;
    tilingDatafromBin->colTail = 1;
    tilingDatafromBin->isZeroPoint1Exist = 1;
    tilingDatafromBin->isZeroPoint2Exist = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    // normal fp16
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(
        dua_quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, zeroPoints1, zeroPoints2,
        y1, y2, x, workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1001);
    ICPU_RUN_KF(
        dua_quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales1, scales2, zeroPoints1, zeroPoints2,
        y1, y2, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales1);
    AscendC::GmFree(scales2);
    AscendC::GmFree(zeroPoints1);
    AscendC::GmFree(zeroPoints2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
