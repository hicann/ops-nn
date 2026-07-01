/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_thnn_fused_lstm_cell.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_thnn_fused_lstm_cell_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" void thnn_fused_lstm_cell(
    GM_ADDR inputGates, GM_ADDR hiddenGates, GM_ADDR cx, GM_ADDR inputBias, GM_ADDR hiddenBias,
    GM_ADDR hy, GM_ADDR cy, GM_ADDR storage, GM_ADDR workspace, GM_ADDR tiling);

class ThnnFusedLstmCellKernel : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "ThnnFusedLstmCell Kernel SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ThnnFusedLstmCell Kernel TearDown\n" << endl;
    }
};

template <typename T>
void TestThnnFusedLstmCellKernel(
    uint64_t batchSize, uint64_t hiddenSize, uint64_t workspaceSize, uint64_t blockDim,
    uint64_t tilingKey)
{
    size_t inithBits= batchSize * hiddenSize * sizeof(T);
    size_t gatesBits= 4 * hiddenSize * batchSize * sizeof(T);
    size_t bBits= 4 * hiddenSize * sizeof(T);
    
    size_t tilingDataSize = sizeof(ThnnFusedLstmCellTilingDataTest);

    uint8_t* inputGates = (uint8_t*)AscendC::GmAlloc(gatesBits);
    uint8_t* hiddenGates = (uint8_t*)AscendC::GmAlloc(gatesBits);
    uint8_t* cx = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* inputBias = (uint8_t*)AscendC::GmAlloc(bBits);
    uint8_t* hiddenBias = (uint8_t*)AscendC::GmAlloc(bBits);
    uint8_t* hy = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* cy = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* storage = (uint8_t*)AscendC::GmAlloc(gatesBits);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    T one = 1.0;
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(hy + idx, &one, sizeof(T));
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
    for (size_t idx = 0; idx < gatesBits; idx += sizeof(T)) {
        std::memcpy(inputGates + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < gatesBits; idx += sizeof(T)) {
        std::memcpy(hiddenGates + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < bBits; idx += sizeof(T)) {
        std::memcpy(inputBias + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < bBits; idx += sizeof(T)) {
        std::memcpy(hiddenBias + idx, &one, sizeof(T));
    }

    ThnnFusedLstmCellTilingDataTest* tilingDatafromBin = reinterpret_cast<ThnnFusedLstmCellTilingDataTest*>(tiling);

    tilingDatafromBin->B=3;
    tilingDatafromBin->H = 5;
    tilingDatafromBin->BH = 15;
    tilingDatafromBin->col = 1;
    tilingDatafromBin->taskTotal = 3;
    tilingDatafromBin->taskSingle = 1;
    tilingDatafromBin->taskSize = 1280;
    tilingDatafromBin->tailSize = 5;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        thnn_fused_lstm_cell, blockDim, inputGates, hiddenGates, cx, inputBias, hiddenBias, hy, cy, storage, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(inputGates);
    AscendC::GmFree(hiddenGates);
    AscendC::GmFree(cx);
    AscendC::GmFree(inputBias);
    AscendC::GmFree(hiddenBias);
    AscendC::GmFree(hy);
    AscendC::GmFree(cy);
    AscendC::GmFree(storage);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ThnnFusedLstmCellKernel, thnn_fused_lstm_cell_case_float_0)
{
    uint64_t workspaceSize = 0;
    uint64_t blockDim = 1;
    uint64_t tilingKey = 0;
    TestThnnFusedLstmCellKernel<float>(
        1, 8, workspaceSize, blockDim, tilingKey);
}