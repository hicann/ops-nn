/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_add_layer_norm_quant.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_add_layer_norm_quant_regbase.h"

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

TEST_F(add_layer_norm_quant_test, test_case_fp32_full_load)
{
    int N = 3;
    int D = 128;
    size_t inputByteSize = N * D * sizeof(int32_t);
    size_t weightByteSize = D * sizeof(int32_t);
    size_t outputByteSize = N * D * sizeof(int32_t);
    size_t quantByteSize = N * D * sizeof(int8_t);

    size_t tiling_data_size = sizeof(AddLayerNormQuantRegbaseTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* s1 = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* s2 = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* o1 = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* o2 = (uint8_t*)AscendC::GmAlloc(weightByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(quantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(quantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* out_scale1 = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* out_scale2 = (uint8_t*)AscendC::GmAlloc(32);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormQuantRegbaseTilingData* tilingDatafromBin =
        reinterpret_cast<AddLayerNormQuantRegbaseTilingData*>(tiling);

    tilingDatafromBin->blockSize = 32;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->vlFp32 = 64;
    tilingDatafromBin->rowsPerCore = 1;
    tilingDatafromBin->rowsPerTailCore = 1;
    tilingDatafromBin->rowsPerLoop = 1;
    tilingDatafromBin->cols = 128;
    tilingDatafromBin->colsPerLoop = 128;
    tilingDatafromBin->colsLoopCount = 1;
    tilingDatafromBin->colsTail = 128;
    tilingDatafromBin->binaryAddNum = 64;
    tilingDatafromBin->binaryAddK = 0;
    tilingDatafromBin->binaryAddLastNum = 1;
    tilingDatafromBin->eps = 1e-5;
    tilingDatafromBin->outputX = 1;

    // normal fp32
    ICPU_SET_TILING_KEY(8000);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8001);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8002);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8010);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8011);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8012);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8022);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, nullptr, nullptr, y1, y2, x, out_scale1,
        out_scale2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(s1);
    AscendC::GmFree(s2);
    AscendC::GmFree(o1);
    AscendC::GmFree(o2);

    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(out_scale1);
    AscendC::GmFree(out_scale2);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_quant_test, test_case_fp32_welford)
{
    int N = 1;
    int D = 8000;
    size_t inputByteSize = N * D * sizeof(int32_t);
    size_t weightByteSize = D * sizeof(int32_t);
    size_t outputByteSize = N * D * sizeof(int32_t);
    size_t quantByteSize = N * D * sizeof(int8_t);

    size_t tiling_data_size = sizeof(AddLayerNormQuantRegbaseTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* bias = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* s1 = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* s2 = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* o1 = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* o2 = (uint8_t*)AscendC::GmAlloc(weightByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(quantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(quantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* out_scale1 = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* out_scale2 = (uint8_t*)AscendC::GmAlloc(32);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormQuantRegbaseTilingData* tilingDatafromBin =
        reinterpret_cast<AddLayerNormQuantRegbaseTilingData*>(tiling);

    tilingDatafromBin->blockSize = 32;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->vlFp32 = 64;
    tilingDatafromBin->rowsPerCore = 1;
    tilingDatafromBin->rowsPerTailCore = 1;
    tilingDatafromBin->rowsPerLoop = 1;
    tilingDatafromBin->cols = 128;
    tilingDatafromBin->colsPerLoop = 128;
    tilingDatafromBin->colsLoopCount = 1;
    tilingDatafromBin->colsTail = 128;
    tilingDatafromBin->binaryAddNum = 64;
    tilingDatafromBin->binaryAddK = 0;
    tilingDatafromBin->binaryAddLastNum = 1;
    tilingDatafromBin->eps = 1e-5;
    tilingDatafromBin->outputX = 1;

    // normal fp32
    ICPU_SET_TILING_KEY(8100);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8101);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8102);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8110);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8111);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8112);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, o1, o2, y1, y2, x, out_scale1, out_scale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(8122);
    ICPU_RUN_KF(
        add_layer_norm_quant, blockDim, x1, x2, gamma, beta, bias, s1, s2, nullptr, nullptr, y1, y2, x, out_scale1,
        out_scale2, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(bias);
    AscendC::GmFree(s1);
    AscendC::GmFree(s2);
    AscendC::GmFree(o1);
    AscendC::GmFree(o2);

    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(out_scale1);
    AscendC::GmFree(out_scale2);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}