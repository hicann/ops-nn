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
 * \file test_layer_norm_v4.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "layer_norm_v4_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void layer_norm_v4(
    GM_ADDR x, GM_ADDR normalized_shape, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
    GM_ADDR workspace, GM_ADDR tiling);

class layer_norm_v4_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "layer_norm_v4_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "layer_norm_v4_test TearDown\n" << endl;
    }
};

TEST_F(layer_norm_v4_test, test_case_0001_100)
{
    size_t xByteSize = 1 * 960 * 8192 * sizeof(float);
    size_t gammaByteSize = 8192 * sizeof(float);
    size_t betaByteSize = 8192 * sizeof(float);
    size_t yByteSize = 1 * 960 * 8192 * sizeof(float);
    size_t meanByteSize = 1 * 960 * sizeof(float);
    size_t rstdByteSize = 1 * 960 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4TilingDataSingleRead);
    uint32_t blockDim = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4TilingDataSingleRead* tilingDatafromBin = reinterpret_cast<LayerNormV4TilingDataSingleRead*>(tiling);

    tilingDatafromBin->blockDim = 48;
    tilingDatafromBin->colSize = 960;
    tilingDatafromBin->rowSize = 8192;
    tilingDatafromBin->eps = 1e-05;
    tilingDatafromBin->coefficient = 0.0009765625;
    tilingDatafromBin->rowAlign = 8192;
    tilingDatafromBin->nRow = 2;
    tilingDatafromBin->tailNRow = 0;
    tilingDatafromBin->loopCount = 10;
    tilingDatafromBin->tailLoop = 0;
    tilingDatafromBin->tileLength = 16384;
    tilingDatafromBin->blockLength = 16384;
    tilingDatafromBin->nullptrGamma = 0;
    tilingDatafromBin->nullptrBeta = 0;

    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(
        layer_norm_v4, blockDim, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_v4_test, test_case_0002)
{
    size_t xByteSize = 48 * 2 * sizeof(float);
    size_t gammaByteSize = 2 * sizeof(float);
    size_t betaByteSize = 2 * sizeof(float);
    size_t yByteSize = 48 * 2 * sizeof(float);
    size_t meanByteSize = 48 * sizeof(float);
    size_t rstdByteSize = 48 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4TilingDataTranspose);
    uint32_t blockDim = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormV4TilingDataTranspose* tilingDatafromBin = reinterpret_cast<LayerNormV4TilingDataTranspose*>(tiling);

    tilingDatafromBin->col = 48;                // 输入tensor的行
    tilingDatafromBin->row = 2;                 // 输入tensor的列，即reduce的轴
    tilingDatafromBin->blockDim = 48;           // 实际使用的core数量
    tilingDatafromBin->blockFormer = 1;         // 整核处理的row大小
    tilingDatafromBin->blockTail = 1;           // 尾核处理的row大小
    tilingDatafromBin->ubFormer = 1;            // ub整循环处理的row大小
    tilingDatafromBin->ubLoopOfFormerBlock = 1; // 整核处理的ub循环次数
    tilingDatafromBin->ubLoopOfTailBlock = 1;   // 尾核处理的ub循环次数
    tilingDatafromBin->ubTailOfFormerBlock = 1; // 整核ub尾循环处理的row大小
    tilingDatafromBin->ubTailOfTailBlock = 1;   // 尾核ub尾循环处理的row大小
    tilingDatafromBin->bFormer = 1;             // ubFormer借轴大小，ubFormer->16*bFormer
    tilingDatafromBin->dichotomizeAddDiffSize = 0;
    tilingDatafromBin->eps = 0.00001;
    tilingDatafromBin->coefficient = 0.5;
    tilingDatafromBin->nullptrGamma = 0;
    tilingDatafromBin->nullptrBeta = 0;

    ICPU_SET_TILING_KEY(200);
    ICPU_RUN_KF(
        layer_norm_v4, blockDim, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(210);
    ICPU_RUN_KF(
        layer_norm_v4, blockDim, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(211);
    ICPU_RUN_KF(
        layer_norm_v4, blockDim, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(220);
    ICPU_RUN_KF(
        layer_norm_v4, blockDim, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(222);
    ICPU_RUN_KF(
        layer_norm_v4, blockDim, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}