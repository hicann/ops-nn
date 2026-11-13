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
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_thnn_fused_lstm_cell_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" void thnn_fused_lstm_cell(GM_ADDR inputGates, GM_ADDR hiddenGates, GM_ADDR cx, GM_ADDR inputBias,
                                     GM_ADDR hiddenBias, GM_ADDR hy, GM_ADDR cy, GM_ADDR storage, GM_ADDR workspace,
                                     GM_ADDR tiling);

class ThnnFusedLstmCellKernel : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ThnnFusedLstmCell Kernel SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "ThnnFusedLstmCell Kernel TearDown\n" << endl; }
};

template <typename T>
void TestThnnFusedLstmCellKernel(uint64_t batchSize, uint64_t hiddenSize, uint64_t workspaceSize, uint64_t blockDim,
                                 uint64_t tilingKey)
{
    size_t inithBits = batchSize * hiddenSize * sizeof(T);
    size_t gatesBits = 4 * hiddenSize * batchSize * sizeof(T);
    size_t bBits = 4 * hiddenSize * sizeof(T);

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

    // tiling配置与模板分配形状(batchSize×hiddenSize)一致;taskSize=1280对齐生产UB切块
    const uint32_t taskSize = 1280;
    const uint32_t tailSize = (hiddenSize % taskSize != 0U) ? static_cast<uint32_t>(hiddenSize % taskSize) : taskSize;
    tilingDatafromBin->B = batchSize;
    tilingDatafromBin->H = hiddenSize;
    tilingDatafromBin->BH = batchSize * hiddenSize;
    tilingDatafromBin->col = (hiddenSize + taskSize - 1) / taskSize;
    tilingDatafromBin->taskTotal = batchSize * tilingDatafromBin->col;
    tilingDatafromBin->taskSingle = (tilingDatafromBin->taskTotal + blockDim - 1) / blockDim;
    tilingDatafromBin->taskSize = taskSize;
    tilingDatafromBin->tailSize = tailSize;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(thnn_fused_lstm_cell, blockDim, inputGates, hiddenGates, cx, inputBias, hiddenBias, hy, cy, storage,
                workspace, (uint8_t*)(tilingDatafromBin));

    // 输出数值断言: 全1.0输入下四门求和恒为4.0,按实现门序[i,f,g,o]解析期望
    const double iExp = 1.0 / (1.0 + std::exp(-4.0));
    const double gExp = std::tanh(4.0);
    const double cyExp = iExp * 1.0 + iExp * gExp;
    const double hyExp = iExp * std::tanh(cyExp);
    const T* hyOut = reinterpret_cast<const T*>(hy);
    const T* cyOut = reinterpret_cast<const T*>(cy);
    const T* storageOut = reinterpret_cast<const T*>(storage);
    for (uint64_t b = 0; b < batchSize; ++b) {
        for (uint64_t j = 0; j < hiddenSize; ++j) {
            EXPECT_NEAR(static_cast<double>(hyOut[b * hiddenSize + j]), hyExp, 1e-5);
            EXPECT_NEAR(static_cast<double>(cyOut[b * hiddenSize + j]), cyExp, 1e-5);
            EXPECT_NEAR(static_cast<double>(storageOut[(b * 4 + 0) * hiddenSize + j]), iExp, 1e-5); // i门
            EXPECT_NEAR(static_cast<double>(storageOut[(b * 4 + 1) * hiddenSize + j]), iExp, 1e-5); // f门
            EXPECT_NEAR(static_cast<double>(storageOut[(b * 4 + 2) * hiddenSize + j]), gExp, 1e-5); // g门
            EXPECT_NEAR(static_cast<double>(storageOut[(b * 4 + 3) * hiddenSize + j]), iExp, 1e-5); // o门
        }
    }

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
    TestThnnFusedLstmCellKernel<float>(1, 8, workspaceSize, blockDim, tilingKey);
}
