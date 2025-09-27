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
 * \file test_quant_matmul_reduce_sum.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "quant_matmul_reduce_sum_tiling_def.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void quant_matmul_reduce_sum(GM_ADDR x, GM_ADDR w, GM_ADDR dims, GM_ADDR bias,
                                        GM_ADDR pertokenScale, GM_ADDR scale, GM_ADDR yScale,
                                        GM_ADDR x1Offset, GM_ADDR x2Offset, GM_ADDR yOffset, GM_ADDR x2Table,
                                        GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class quant_matmul_reduce_sum_test : public testing::Test {
    protected:

    static void SetUpTestCase() {
        cout << "quant_matmul_reduce_sum_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "quant_matmul_reduce_sum_test TearDown\n" << endl;
    }
};

TEST_F(quant_matmul_reduce_sum_test, general_test_01)
{
    size_t b = 8;
    size_t m = 32;
    size_t k = 64;
    size_t n = 32;

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    size_t shapeX1 = b * m * k * sizeof(int8_t);
    size_t shapeX2 = b * k * n * sizeof(int8_t);
    size_t shapeDims = 1 * sizeof(int64_t);
    size_t shapeX2Scale = n * sizeof(uint16_t);  // bf16
    size_t shapeX1Scale = b * m * sizeof(float);
    size_t shapeY = m * n * sizeof(uint16_t);  // bf16

    uint8_t* x1Gm = (uint8_t*)AscendC::GmAlloc(shapeX1);
    uint8_t* x2Gm = (uint8_t*)AscendC::GmAlloc(shapeX2);
    uint8_t* dimsGM = (uint8_t*)AscendC::GmAlloc(shapeDims);
    uint8_t* x2ScaleGm = (uint8_t*)AscendC::GmAlloc(shapeX2Scale);
    uint8_t* x1ScaleGm = (uint8_t*)AscendC::GmAlloc(shapeX1Scale);
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(shapeY);

    size_t allWorkspaceSize = 16793600;
    size_t tilingSize = sizeof(QuantMatmulReduceSumTilingData);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    memset(x1Gm, 0, shapeX1);
    memset(x2Gm, 0, shapeX2);
    memset(dimsGM, 0, shapeDims);
    memset(x2ScaleGm, 0, shapeX2Scale);
    memset(x1ScaleGm, 0, shapeX1Scale);
    memset(yGM, 0, shapeY);

    QuantMatmulReduceSumTilingData* tilingData = reinterpret_cast<QuantMatmulReduceSumTilingData*>(tiling);
    tilingData->qbmmReduceSumParams.batchNum = 8;
    tilingData->qbmmReduceSumParams.coreNum = 1;
    tilingData->qbmmReduceSumParams.ubBaseK = 64;
    tilingData->qbmmReduceSumParams.ubBaseN = 32;
    tilingData->qbmmReduceSumParams.ubRestBytes = 16384;
    tilingData->qbmmReduceSumParams.ubCalSize = 1024;
    tilingData->qbmmReduceSumParams.isPertoken = 1;
    tilingData->qbmmReduceSumParams.isDetermine = 0;
    tilingData->qbmmReduceSumParams.workspaceSize = 0;

    tilingData->matmulTiling.usedCoreNum = 1;
    tilingData->matmulTiling.M = 32;
    tilingData->matmulTiling.N = 32;
    tilingData->matmulTiling.Ka = 64;
    tilingData->matmulTiling.Kb = 64;
    tilingData->matmulTiling.singleCoreM = 32;
    tilingData->matmulTiling.singleCoreN = 32;
    tilingData->matmulTiling.singleCoreK = 64;
    tilingData->matmulTiling.baseM = 32;
    tilingData->matmulTiling.baseN = 32;
    tilingData->matmulTiling.baseK = 64;
    tilingData->matmulTiling.depthA1 = 1;
    tilingData->matmulTiling.depthB1 = 1;
    tilingData->matmulTiling.stepM = 1;
    tilingData->matmulTiling.stepN = 1;
    tilingData->matmulTiling.isBias = 0;
    tilingData->matmulTiling.transLength = 0;
    tilingData->matmulTiling.iterateOrder = 0;
    tilingData->matmulTiling.shareMode = 0;
    tilingData->matmulTiling.shareL1Size = 4096;
    tilingData->matmulTiling.shareL0CSize = 4096;
    tilingData->matmulTiling.shareUbSize = 0;
    tilingData->matmulTiling.batchM = 1;
    tilingData->matmulTiling.batchN = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.singleBatchM = 1;
    tilingData->matmulTiling.stepKa = 1;
    tilingData->matmulTiling.stepKb = 1;
    tilingData->matmulTiling.depthAL1CacheUB = 0;
    tilingData->matmulTiling.depthBL1CacheUB = 0;
    tilingData->matmulTiling.dbL0A = 2;
    tilingData->matmulTiling.dbL0B = 2;
    tilingData->matmulTiling.dbL0C = 1;
    tilingData->matmulTiling.ALayoutInfoB = 0;
    tilingData->matmulTiling.ALayoutInfoS = 0;
    tilingData->matmulTiling.ALayoutInfoN = 0;
    tilingData->matmulTiling.ALayoutInfoG = 0;
    tilingData->matmulTiling.ALayoutInfoD = 0;
    tilingData->matmulTiling.BLayoutInfoB = 0;
    tilingData->matmulTiling.BLayoutInfoS = 0;
    tilingData->matmulTiling.BLayoutInfoN = 0;
    tilingData->matmulTiling.BLayoutInfoG = 0;
    tilingData->matmulTiling.BLayoutInfoD = 0;
    tilingData->matmulTiling.CLayoutInfoB = 0;
    tilingData->matmulTiling.CLayoutInfoS1 = 0;
    tilingData->matmulTiling.CLayoutInfoN = 0;
    tilingData->matmulTiling.CLayoutInfoG = 0;
    tilingData->matmulTiling.CLayoutInfoS2 = 0;
    tilingData->matmulTiling.BatchNum = 0;

    uint32_t blockDim = 1;
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(quant_matmul_reduce_sum, blockDim,
                x1Gm, x2Gm, dimsGM, nullptr, x1ScaleGm, x2ScaleGm,  // null:bias
                nullptr, nullptr, nullptr, nullptr, nullptr, // null: yScale, x1Offset, x2Offset, yOffset, x2Table
                yGM, workspace, tiling);

    AscendC::GmFree((void*)x1Gm);
    AscendC::GmFree((void*)x2Gm);
    AscendC::GmFree((void*)dimsGM);
    AscendC::GmFree((void*)x2ScaleGm);
    AscendC::GmFree((void*)x1ScaleGm);
    AscendC::GmFree((void*)yGM);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}
