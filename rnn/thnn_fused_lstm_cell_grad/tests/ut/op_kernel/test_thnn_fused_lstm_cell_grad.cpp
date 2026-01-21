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
 * \file test_thnn_fused_lstm_cell_grad.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_thnn_fused_lstm_cell_grad_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" void thnn_fused_lstm_cell_grad(
    GM_ADDR dhy, GM_ADDR dc, GM_ADDR cx, GM_ADDR cy,
    GM_ADDR storage, GM_ADDR dgates, GM_ADDR dc_prev, GM_ADDR db, GM_ADDR workspace, GM_ADDR lstmCellGradTiling);

class ThnnFusedLstmCellGradKernel : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "ThnnFusedLstmCellGrad Kernel SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ThnnFusedLstmCellGrad Kernel TearDown\n" << endl;
    }
};

template <typename T>
void TestThnnFusedLstmCellGradKernel(
    uint64_t batchSize, uint64_t hiddenSize, uint64_t workspaceSize, uint64_t blockDim,
    uint64_t tilingKey)
{
    size_t inithBits= batchSize * hiddenSize * sizeof(T);
    size_t gatesBits= 4 * hiddenSize * batchSize * sizeof(T);
    size_t bBits= 4 * hiddenSize * sizeof(T);
    
    size_t tilingDataSize = sizeof(ThnnFusedLstmCellGradTilingDataTest);

    uint8_t* dhy = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* dc = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* cx = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* cy = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* storage = (uint8_t*)AscendC::GmAlloc(gatesBits);

    T one = 1.0;
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(dhy + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(dc + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(cx + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(cy + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < gatesBits; idx += sizeof(T)) {
        std::memcpy(storage + idx, &one, sizeof(T));
    }


    uint8_t* dgates = (uint8_t*)AscendC::GmAlloc(gatesBits);
    uint8_t* dcPrev = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* db = (uint8_t*)AscendC::GmAlloc(bBits);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    ThnnFusedLstmCellGradTilingDataTest* tilingDatafromBin = reinterpret_cast<ThnnFusedLstmCellGradTilingDataTest*>(tiling);

    tilingDatafromBin->ubSize = 196352;
    // lstm input tiling
    tilingDatafromBin->batch = 1;
    tilingDatafromBin->hiddenSize = 8;

    // vector block params
    tilingDatafromBin->singleCoreM = 1;
    tilingDatafromBin->singleCoreMTail = 1;
    tilingDatafromBin->singleCoreN = 8;
    tilingDatafromBin->singleCoreNTail = 8;
    tilingDatafromBin->baseN = 8;
    tilingDatafromBin->baseM = 339;
    tilingDatafromBin->mCnt = 1;
    tilingDatafromBin->nCnt = 1;

    // reduce block params
    tilingDatafromBin->singleCoreReduceN = 8;
    tilingDatafromBin->singleCoreReduceNTail = 8;
    tilingDatafromBin->baseReduceN = 8;
    tilingDatafromBin->nReduceCnt = 4;
    tilingDatafromBin->maxReduceNumOnce = 6104;
    tilingDatafromBin->reduceBlockSize = 1;
    // tilingDatafromBin->reduceLoop = 1;

    // lstm attr
    tilingDatafromBin->isBias = 1;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        thnn_fused_lstm_cell_grad, blockDim, dhy, dc, cx, cy, storage, dgates, dcPrev, db, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dhy);
    AscendC::GmFree(dc);
    AscendC::GmFree(cx);
    AscendC::GmFree(cy);
    AscendC::GmFree(storage);
    AscendC::GmFree(dgates);
    AscendC::GmFree(db);
    AscendC::GmFree(dcPrev);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ThnnFusedLstmCellGradKernel, thnn_fused_lstm_cell_grad_case_float_0)
{
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 0;
    TestThnnFusedLstmCellGradKernel<float>(
        1, 8, workspaceSize, blockDim, tilingKey);
}

