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
#include "layer_norm_v4_regbase_tiling.h"
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

TEST_F(layer_norm_v4_test, test_case_0003)
{
    size_t xByteSize = 64 * 2 * sizeof(float);
    size_t gammaByteSize = 2 * sizeof(float);
    size_t betaByteSize = 2 * sizeof(float);
    size_t yByteSize = 64 * 2 * sizeof(float);
    size_t meanByteSize = 64 * sizeof(float);
    size_t rstdByteSize = 64 * sizeof(float);
    size_t tiling_data_size = sizeof(LayerNormV4TilingDataWelford);
    uint32_t numBlocks = 64;

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

    LayerNormV4TilingDataWelford* tilingDatafromBin = reinterpret_cast<LayerNormV4TilingDataWelford*>(tiling);

    tilingDatafromBin->M = 0;
    tilingDatafromBin->N = 0;
    tilingDatafromBin->rAlign = 0;
    tilingDatafromBin->numBlocks = 0;
    tilingDatafromBin->mainBlockCount = 0;
    tilingDatafromBin->mainBlockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 0;
    tilingDatafromBin->tileLength = 0;
    tilingDatafromBin->welfordUpdateTimes = 0;
    tilingDatafromBin->welfordUpdateTail = 0;
    tilingDatafromBin->nullptrGamma = 0;
    tilingDatafromBin->nullptrBeta = 0;
    tilingDatafromBin->epsilon = 0;
    tilingDatafromBin->apiTempBufferSize = 0;

    ICPU_SET_TILING_KEY(400);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(410);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(411);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(420);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));
    ICPU_SET_TILING_KEY(422);
    ICPU_RUN_KF(
        layer_norm_v4, numBlocks, x, nullptr, gamma, beta, y, mean, rstd, workspace, (uint8_t*)(tilingDatafromBin));

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
