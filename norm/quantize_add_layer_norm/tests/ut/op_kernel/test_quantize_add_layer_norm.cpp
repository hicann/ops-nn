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
#include "quantize_add_layer_norm_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void quantize_add_layer_norm(
    uint8_t* x1, uint8_t* x2, uint8_t* gamma, uint8_t* beta, uint8_t* bias, uint8_t* scales, uint8_t* zeroPoints,
    uint8_t* y, uint8_t* x, uint8_t* workspace, uint8_t* tiling);

class quantize_add_layer_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "quantize_add_layer_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "quantize_add_layer_norm_test TearDown\n" << endl;
    }
};

TEST_F(quantize_add_layer_norm_test, test_case_bf16_per_channel)
{
    int N = 8;
    int D = 1024;
    size_t inputByteSize = N * D * sizeof(short);
    size_t gammaBetaByteSize = D * sizeof(short);
    size_t xOutByteSize = N * D * sizeof(short);
    size_t yOutByteSize = N * D * sizeof(char);
    size_t tiling_data_size = sizeof(QuantizeAddLayerNormTilingData);

    uint32_t blockDim = 2;
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroPoints = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yOutByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(D * blockDim * sizeof(float) * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizeAddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<QuantizeAddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 5;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->eps = 0.00001;

    // single row bf16 per channel
    ICPU_SET_TILING_KEY(3100);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(3000);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales);
    AscendC::GmFree(zeroPoints);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantize_add_layer_norm_test, test_case_fp32_per_channel)
{
    int N = 8;
    int D = 1024;
    size_t inputByteSize = N * D * sizeof(float);
    size_t gammaBetaByteSize = D * sizeof(float);
    size_t xOutByteSize = N * D * sizeof(float);
    size_t yOutByteSize = N * D * sizeof(char);
    size_t tiling_data_size = sizeof(QuantizeAddLayerNormTilingData);

    uint32_t blockDim = 2;
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroPoints = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yOutByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(D * blockDim * sizeof(float) * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizeAddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<QuantizeAddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 5;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->eps = 0.00001;

    // single row bf16 per channel
    ICPU_SET_TILING_KEY(2100);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales);
    AscendC::GmFree(zeroPoints);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantize_add_layer_norm_test, test_case_fp16_per_channel)
{
    int N = 8;
    int D = 1024;
    size_t inputByteSize = N * D * sizeof(short);
    size_t gammaBetaByteSize = D * sizeof(short);
    size_t xOutByteSize = N * D * sizeof(short);
    size_t yOutByteSize = N * D * sizeof(char);
    size_t tiling_data_size = sizeof(QuantizeAddLayerNormTilingData);

    uint32_t blockDim = 2;
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroPoints = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yOutByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(D * blockDim * sizeof(float) * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizeAddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<QuantizeAddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 5;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->eps = 0.00001;

    // single row bf16 per channel
    ICPU_SET_TILING_KEY(1100);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales);
    AscendC::GmFree(zeroPoints);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantize_add_layer_norm_test, test_case_fp16_per_tensor)
{
    int N = 8;
    int D = 1024;
    size_t inputByteSize = N * D * sizeof(short);
    size_t gammaBetaByteSize = D * sizeof(short);
    size_t xOutByteSize = N * D * sizeof(short);
    size_t yOutByteSize = N * D * sizeof(char);
    size_t tiling_data_size = sizeof(QuantizeAddLayerNormTilingData);

    uint32_t blockDim = 2;
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroPoints = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yOutByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(D * blockDim * sizeof(float) * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizeAddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<QuantizeAddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 5;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->eps = 0.00001;

    // single row bf16 per channel
    ICPU_SET_TILING_KEY(1101);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales);
    AscendC::GmFree(zeroPoints);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantize_add_layer_norm_test, test_case_bf16_per_tensor)
{
    int N = 8;
    int D = 1024;
    size_t inputByteSize = N * D * sizeof(short);
    size_t gammaBetaByteSize = D * sizeof(short);
    size_t xOutByteSize = N * D * sizeof(short);
    size_t yOutByteSize = N * D * sizeof(char);
    size_t tiling_data_size = sizeof(QuantizeAddLayerNormTilingData);

    uint32_t blockDim = 2;
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroPoints = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yOutByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(D * blockDim * sizeof(float) * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizeAddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<QuantizeAddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 5;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->eps = 0.00001;

    // single row bf16 per channel
    ICPU_SET_TILING_KEY(3101);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales);
    AscendC::GmFree(zeroPoints);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantize_add_layer_norm_test, test_case_fp32_per_tensor)
{
    int N = 8;
    int D = 1024;
    size_t inputByteSize = N * D * sizeof(float);
    size_t gammaBetaByteSize = D * sizeof(float);
    size_t xOutByteSize = N * D * sizeof(float);
    size_t yOutByteSize = N * D * sizeof(char);
    size_t tiling_data_size = sizeof(QuantizeAddLayerNormTilingData);

    uint32_t blockDim = 2;
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroPoints = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yOutByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(D * blockDim * sizeof(float) * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizeAddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<QuantizeAddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 5;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->eps = 0.00001;

    // single row bf16 per channel
    ICPU_SET_TILING_KEY(2101);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales);
    AscendC::GmFree(zeroPoints);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantize_add_layer_norm_test, test_case_bf16_fp16_per_channel_v2)
{
    int N = 8;
    int D = 1024;
    size_t inputByteSize = N * D * sizeof(short);
    size_t gammaBetaByteSize = D * sizeof(short);
    size_t xOutByteSize = N * D * sizeof(short);
    size_t yOutByteSize = N * D * sizeof(char);
    size_t tiling_data_size = sizeof(QuantizeAddLayerNormTilingData);

    uint32_t blockDim = 2;
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroPoints = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yOutByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(D * blockDim * sizeof(float) * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizeAddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<QuantizeAddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 5;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->eps = 0.00001;

    // single row bf16 per channel
    ICPU_SET_TILING_KEY(3002);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(3102);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(1002);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(1102);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales);
    AscendC::GmFree(zeroPoints);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(quantize_add_layer_norm_test, test_case_fp32_per_channel_v2)
{
    int N = 8;
    int D = 1024;
    size_t inputByteSize = N * D * sizeof(short);
    size_t gammaBetaByteSize = D * sizeof(short);
    size_t xOutByteSize = N * D * sizeof(short);
    size_t yOutByteSize = N * D * sizeof(char);
    size_t tiling_data_size = sizeof(QuantizeAddLayerNormTilingData);

    uint32_t blockDim = 2;
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* scales = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* zeroPoints = (uint8_t*)AscendC::GmAlloc(gammaBetaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yOutByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(D * blockDim * sizeof(float) * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    QuantizeAddLayerNormTilingData* tilingDatafromBin = reinterpret_cast<QuantizeAddLayerNormTilingData*>(tiling);

    tilingDatafromBin->numCore = blockDim;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->firstDimPerCore = 5;
    tilingDatafromBin->firstDimPerCoreTail = 3;
    tilingDatafromBin->firstDimPerTime = 1;
    tilingDatafromBin->aveFactor = 1.0 / D;
    tilingDatafromBin->eps = 0.00001;

    // single row bf16 per channel
    ICPU_SET_TILING_KEY(2002);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(2102);
    ICPU_RUN_KF(
        quantize_add_layer_norm, blockDim, x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(scales);
    AscendC::GmFree(zeroPoints);
    AscendC::GmFree(y);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
