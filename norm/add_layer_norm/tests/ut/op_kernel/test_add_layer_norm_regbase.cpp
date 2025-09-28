/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_add_layer_norm.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "add_layer_norm_regbase_tiling.h"

using namespace std;

extern "C" void add_layer_norm(
    uint8_t* x1, uint8_t* x2, uint8_t* gamma, uint8_t* beta, uint8_t* bias, uint8_t* y, uint8_t* mean, uint8_t* rstd,
    uint8_t* x, uint8_t* workspace, uint8_t* tiling);

class add_layer_norm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_layer_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_layer_norm_test TearDown\n" << endl;
    }
};

TEST_F(add_layer_norm_test, test_case_fp32_full_load)
{
    int N = 37;
    int D = 13;
    size_t inputByteSize = N * D * sizeof(int32_t);
    size_t gammaByteSize = D * sizeof(int32_t);
    size_t betaByteSize = D * sizeof(int32_t);
    size_t outputByteSize = N * D * sizeof(int32_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormRegbaseTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormRegbaseTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormRegbaseTilingData*>(tiling);

    tilingDatafromBin->blockSize = 32;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->vlFp32 = 64;
    tilingDatafromBin->tailCoreStartIndex = 1;
    tilingDatafromBin->rowsPerCore = 37;
    tilingDatafromBin->rowsPerTailCore = 37;
    tilingDatafromBin->rowsPerLoop = 37;
    tilingDatafromBin->cols = 13;
    tilingDatafromBin->colsPerLoop = 13;
    tilingDatafromBin->colsLoopCount = 1;
    tilingDatafromBin->colsTail = 13;
    tilingDatafromBin->binaryAddNum = 64;
    tilingDatafromBin->binaryAddK = 0;
    tilingDatafromBin->binaryAddLastNum = 1;
    tilingDatafromBin->eps = 1e-5;
    tilingDatafromBin->outputX = 1;

    // normal fp32
    ICPU_SET_TILING_KEY(8000);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8001);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8002);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_test, test_case_fp32_welford)
{
    int N = 37;
    int D = 13000;
    size_t inputByteSize = N * D * sizeof(int32_t);
    size_t gammaByteSize = D * sizeof(int32_t);
    size_t betaByteSize = D * sizeof(int32_t);
    size_t outputByteSize = N * D * sizeof(int32_t);
    size_t meanByteSize = N * sizeof(float);
    size_t rstdByteSize = N * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormRegbaseTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormRegbaseTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormRegbaseTilingData*>(tiling);

    tilingDatafromBin->blockSize = 32;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->vlFp32 = 64;
    tilingDatafromBin->tailCoreStartIndex = 1;
    tilingDatafromBin->rowsPerCore = 37;
    tilingDatafromBin->rowsPerTailCore = 37;
    tilingDatafromBin->rowsPerLoop = 37;
    tilingDatafromBin->cols = 13;
    tilingDatafromBin->colsPerLoop = 13;
    tilingDatafromBin->colsLoopCount = 1;
    tilingDatafromBin->colsTail = 13;
    tilingDatafromBin->binaryAddNum = 64;
    tilingDatafromBin->binaryAddK = 0;
    tilingDatafromBin->binaryAddLastNum = 1;
    tilingDatafromBin->eps = 1e-5;
    tilingDatafromBin->outputX = 1;

    // normal fp32
    ICPU_SET_TILING_KEY(8100);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8101);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8102);
    ICPU_RUN_KF(
        add_layer_norm, blockDim, x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}