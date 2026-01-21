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
 * \file test_single_layer_lstm_grad.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_single_layer_lstm_grad_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" void single_layer_lstm_grad(
    GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, GM_ADDR init_h, GM_ADDR init_c, GM_ADDR h, GM_ADDR c, GM_ADDR dy,
    GM_ADDR dh, GM_ADDR dc, GM_ADDR i, GM_ADDR j, GM_ADDR f, GM_ADDR o, GM_ADDR tanhct, GM_ADDR seq_length,
    GM_ADDR dw, GM_ADDR db, GM_ADDR dx, GM_ADDR dh_prev,
    GM_ADDR dc_prev, GM_ADDR workspace, GM_ADDR rnnGradTiling);

class SingleLayerLstmGradKernel : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "SingleLayerLstmGrad Kernel SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "SingleLayerLstmGrad Kernel TearDown\n" << endl;
    }
};

template <typename T>
void TestSingleLayerLstmGradKernel(
    uint64_t batchSize, uint64_t timeStep, uint64_t inputSize, uint64_t hiddenSize, uint64_t workspaceSize, uint64_t blockDim,
    uint64_t tilingKey)
{
    size_t xBits = batchSize * inputSize * timeStep * sizeof(T);
    size_t hBits= batchSize * timeStep * hiddenSize * sizeof(T);
    size_t inithBits= batchSize * hiddenSize * sizeof(T);
    size_t wBits= 4 * hiddenSize * (hiddenSize + inputSize) * sizeof(T);
    size_t bBits= 4 * hiddenSize * sizeof(T);
    
    size_t tilingDataSize = sizeof(SingleLayerLstmGradTilingDataTest);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xBits);
    uint8_t* w = (uint8_t*)AscendC::GmAlloc(wBits);
    uint8_t* b = (uint8_t*)AscendC::GmAlloc(bBits);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* inith = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* initc = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* h = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* c = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* dh = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* dc = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* i = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* j = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* f = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* o = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* tanhc = (uint8_t*)AscendC::GmAlloc(hBits);
    T one = 1.0;
    for (size_t idx = 0; idx < xBits; idx += sizeof(T)) {
        std::memcpy(x + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < wBits; idx += sizeof(T)) {
        std::memcpy(w + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < bBits; idx += sizeof(T)) {
        std::memcpy(b + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(y + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(inith + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(initc + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(h + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(c + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(dy + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(dh + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(dc + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(i + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(j + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(f + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(o + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(tanhc + idx, &one, sizeof(T));
    }

    uint8_t* dw = (uint8_t*)AscendC::GmAlloc(wBits);
    uint8_t* db = (uint8_t*)AscendC::GmAlloc(bBits);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(xBits);
    uint8_t* dhPrev = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* dcPrev = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    SingleLayerLstmGradTilingDataTest* tilingDatafromBin = reinterpret_cast<SingleLayerLstmGradTilingDataTest*>(tiling);

    tilingDatafromBin->ubSize = 196352;
    // lstm input tiling
    tilingDatafromBin->timeStep = 1;
    tilingDatafromBin->batch = 20;
    tilingDatafromBin->inputSize = 8;
    tilingDatafromBin->hiddenSize = 8;
    tilingDatafromBin->isBias = 1;
    tilingDatafromBin->isSeqLength = 0;

    // vector block params
    tilingDatafromBin->singleCoreM = 1;
    tilingDatafromBin->singleCoreMTail = 1;
    tilingDatafromBin->singleCoreN = 16;
    tilingDatafromBin->singleCoreNTail = 8;
    tilingDatafromBin->baseN = 16;
    tilingDatafromBin->baseM = 160;
    tilingDatafromBin->mCnt = 20;
    tilingDatafromBin->nCnt = 1;

    // reduce block params
    tilingDatafromBin->singleCoreReduceN = 16;
    tilingDatafromBin->singleCoreReduceNTail = 16;
    tilingDatafromBin->baseReduceN = 16;
    tilingDatafromBin->nReduceCnt = 2;
    tilingDatafromBin->maxReduceNumOnce = 3052;
    tilingDatafromBin->reduceBlockSize = 20;

    // lstm attr
    tilingDatafromBin->gateOrder = 0;
    tilingDatafromBin->direction = 0;
    tilingDatafromBin->cellClip = 0;
    tilingDatafromBin->forgetBias = 0;

    // split and concat params
    tilingDatafromBin->inputSizeAligned = 16;
    tilingDatafromBin->hiddenSizeAligned = 16;
    tilingDatafromBin->oneLineAligned = 32;

    tilingDatafromBin->dxhInputTiling.taskNum = 20;
    tilingDatafromBin->dxhInputTiling.copyMLines = 1;
    tilingDatafromBin->dxhInputTiling.copyMLinesTail = 1;
    tilingDatafromBin->dxhInputTiling.nLoop = 1;
    tilingDatafromBin->dxhInputTiling.copyNLength = 48832;
    tilingDatafromBin->dxhInputTiling.copyNLengthTail = 16;
    tilingDatafromBin->dxhInputTiling.splitTaskPerCore = 0;
    tilingDatafromBin->dxhInputTiling.splitPreCore = 20;

    tilingDatafromBin->dxhHiddenTiling.taskNum = 20;
    tilingDatafromBin->dxhHiddenTiling.copyMLines = 1;
    tilingDatafromBin->dxhHiddenTiling.copyMLinesTail = 1;
    tilingDatafromBin->dxhHiddenTiling.nLoop = 1;
    tilingDatafromBin->dxhHiddenTiling.copyNLength = 48832;
    tilingDatafromBin->dxhHiddenTiling.copyNLengthTail = 16;
    tilingDatafromBin->dxhHiddenTiling.splitTaskPerCore = 0;
    tilingDatafromBin->dxhHiddenTiling.splitPreCore = 20;

    tilingDatafromBin->xhInputTiling.taskNum = 20;
    tilingDatafromBin->xhInputTiling.copyMLines = 1;
    tilingDatafromBin->xhInputTiling.copyMLinesTail = 1;
    tilingDatafromBin->xhInputTiling.nLoop = 1;
    tilingDatafromBin->xhInputTiling.copyNLength = 32560;
    tilingDatafromBin->xhInputTiling.copyNLengthTail = 16;
    tilingDatafromBin->xhInputTiling.splitTaskPerCore = 0;
    tilingDatafromBin->xhInputTiling.splitPreCore = 20;

    tilingDatafromBin->xhHiddenTiling.taskNum = 20;
    tilingDatafromBin->xhHiddenTiling.copyMLines = 1;
    tilingDatafromBin->xhHiddenTiling.copyMLinesTail = 1;
    tilingDatafromBin->xhHiddenTiling.nLoop = 1;
    tilingDatafromBin->xhHiddenTiling.copyNLength = 32560;
    tilingDatafromBin->xhHiddenTiling.copyNLengthTail = 16;
    tilingDatafromBin->xhHiddenTiling.splitTaskPerCore = 0;
    tilingDatafromBin->xhHiddenTiling.splitPreCore = 20;

    tilingDatafromBin->dgateMMParam.usedCoreNum = 1;
    tilingDatafromBin->dgateMMParam.M = 20;
    tilingDatafromBin->dgateMMParam.N = 16;
    tilingDatafromBin->dgateMMParam.Ka = 32;
    tilingDatafromBin->dgateMMParam.Kb = 32;
    tilingDatafromBin->dgateMMParam.singleCoreM = 32;
    tilingDatafromBin->dgateMMParam.singleCoreN = 16;
    tilingDatafromBin->dgateMMParam.singleCoreK = 32;
    tilingDatafromBin->dgateMMParam.baseM = 32;
    tilingDatafromBin->dgateMMParam.baseN = 16;
    tilingDatafromBin->dgateMMParam.baseK = 32;
    tilingDatafromBin->dgateMMParam.depthA1 = 1;
    tilingDatafromBin->dgateMMParam.depthB1 = 1;
    tilingDatafromBin->dgateMMParam.depthAL1CacheUB = 0;
    tilingDatafromBin->dgateMMParam.depthBL1CacheUB = 0;
    tilingDatafromBin->dgateMMParam.stepM = 1;
    tilingDatafromBin->dgateMMParam.stepN = 1;
    tilingDatafromBin->dgateMMParam.isBias = 0;
    tilingDatafromBin->dgateMMParam.transLength = 0;
    tilingDatafromBin->dgateMMParam.iterateOrder = 0;
    tilingDatafromBin->dgateMMParam.shareMode = 0;
    tilingDatafromBin->dgateMMParam.shareL1Size = 6144;
    tilingDatafromBin->dgateMMParam.shareL0CSize = 2048;
    tilingDatafromBin->dgateMMParam.shareUbSize = 0;
    tilingDatafromBin->dgateMMParam.batchM = 1;
    tilingDatafromBin->dgateMMParam.batchN = 1;
    tilingDatafromBin->dgateMMParam.singleBatchM = 1;
    tilingDatafromBin->dgateMMParam.singleBatchN = 1;
    tilingDatafromBin->dgateMMParam.stepKa = 1;
    tilingDatafromBin->dgateMMParam.stepKb = 1;
    tilingDatafromBin->dgateMMParam.dbL0A = 2;
    tilingDatafromBin->dgateMMParam.dbL0B = 2;
    tilingDatafromBin->dgateMMParam.dbL0C = 1;
    tilingDatafromBin->dgateMMParam.ALayoutInfoB = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoS = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoN = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoG = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoD = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoB = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoS = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoN = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoG = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoD = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoB = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoS1 = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoN = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoG = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoS2 = 0;
    tilingDatafromBin->dgateMMParam.BatchNum = 0;

    tilingDatafromBin->dwMMParam.usedCoreNum = 1;
    tilingDatafromBin->dwMMParam.M = 32;
    tilingDatafromBin->dwMMParam.N = 16;
    tilingDatafromBin->dwMMParam.Ka = 20;
    tilingDatafromBin->dwMMParam.Kb = 20;
    tilingDatafromBin->dwMMParam.singleCoreM = 32;
    tilingDatafromBin->dwMMParam.singleCoreN = 16;
    tilingDatafromBin->dwMMParam.singleCoreK = 20;
    tilingDatafromBin->dwMMParam.baseM = 32;
    tilingDatafromBin->dwMMParam.baseN = 16;
    tilingDatafromBin->dwMMParam.baseK = 24;
    tilingDatafromBin->dwMMParam.depthA1 = 1;
    tilingDatafromBin->dwMMParam.depthB1 = 1;
    tilingDatafromBin->dwMMParam.depthAL1CacheUB = 0;
    tilingDatafromBin->dwMMParam.depthBL1CacheUB = 0;
    tilingDatafromBin->dwMMParam.stepM = 1;
    tilingDatafromBin->dwMMParam.stepN = 1;
    tilingDatafromBin->dwMMParam.isBias = 0;
    tilingDatafromBin->dwMMParam.transLength = 0;
    tilingDatafromBin->dwMMParam.iterateOrder = 0;
    tilingDatafromBin->dwMMParam.shareMode = 0;
    tilingDatafromBin->dwMMParam.shareL1Size = 4608;
    tilingDatafromBin->dwMMParam.shareL0CSize = 2048;
    tilingDatafromBin->dwMMParam.shareUbSize = 0;
    tilingDatafromBin->dwMMParam.batchM = 1;
    tilingDatafromBin->dwMMParam.batchN = 1;
    tilingDatafromBin->dwMMParam.singleBatchM = 1;
    tilingDatafromBin->dwMMParam.singleBatchN = 1;
    tilingDatafromBin->dwMMParam.stepKa = 1;
    tilingDatafromBin->dwMMParam.stepKb = 1;
    tilingDatafromBin->dwMMParam.dbL0A = 2;
    tilingDatafromBin->dwMMParam.dbL0B = 2;
    tilingDatafromBin->dwMMParam.dbL0C = 1;
    tilingDatafromBin->dwMMParam.ALayoutInfoB = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoS = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoN = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoG = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoD = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoB = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoS = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoN = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoG = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoD = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoB = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoS1 = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoN = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoG = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoS2 = 0;
    tilingDatafromBin->dwMMParam.BatchNum = 0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        single_layer_lstm_grad, blockDim, x, w, b, y, inith, initc, h, c, dy, dh, dc, i, j, f, o, tanhc, nullptr, dw, db, dx,
        dhPrev, dcPrev, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(w);
    AscendC::GmFree(b);
    AscendC::GmFree(y);
    AscendC::GmFree(inith);
    AscendC::GmFree(initc);
    AscendC::GmFree(h);
    AscendC::GmFree(c);
    AscendC::GmFree(dy);
    AscendC::GmFree(dh);
    AscendC::GmFree(dc);
    AscendC::GmFree(i);
    AscendC::GmFree(j);
    AscendC::GmFree(f);
    AscendC::GmFree(o);
    AscendC::GmFree(tanhc);
    AscendC::GmFree(dw);
    AscendC::GmFree(db);
    AscendC::GmFree(dx);
    AscendC::GmFree(dhPrev);
    AscendC::GmFree(dcPrev);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

template <typename T>
void TestSingleLayerLstmGradKernelLarge(
    uint64_t batchSize, uint64_t timeStep, uint64_t inputSize, uint64_t hiddenSize, uint64_t workspaceSize, uint64_t blockDim,
    uint64_t tilingKey)
{
    size_t xBits = batchSize * inputSize * timeStep * sizeof(T);
    size_t hBits= batchSize * timeStep * hiddenSize * sizeof(T);
    size_t inithBits= batchSize * hiddenSize * sizeof(T);
    size_t wBits= 4 * hiddenSize * (hiddenSize + inputSize) * sizeof(T);
    size_t bBits= 4 * hiddenSize * sizeof(T);
    
    size_t tilingDataSize = sizeof(SingleLayerLstmGradTilingDataTest);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xBits);
    uint8_t* w = (uint8_t*)AscendC::GmAlloc(wBits);
    uint8_t* b = (uint8_t*)AscendC::GmAlloc(bBits);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* inith = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* initc = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* h = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* c = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* dh = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* dc = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* i = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* j = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* f = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* o = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* tanhc = (uint8_t*)AscendC::GmAlloc(hBits);
    T one = 1.0;
    for (size_t idx = 0; idx < xBits; idx += sizeof(T)) {
        std::memcpy(x + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < wBits; idx += sizeof(T)) {
        std::memcpy(w + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < bBits; idx += sizeof(T)) {
        std::memcpy(b + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(y + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(inith + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(initc + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(h + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(c + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(dy + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(dh + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(dc + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(i + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(j + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(f + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(o + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(tanhc + idx, &one, sizeof(T));
    }

    uint8_t* dw = (uint8_t*)AscendC::GmAlloc(wBits);
    uint8_t* db = (uint8_t*)AscendC::GmAlloc(bBits);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(xBits);
    uint8_t* dhPrev = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* dcPrev = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    SingleLayerLstmGradTilingDataTest* tilingDatafromBin = reinterpret_cast<SingleLayerLstmGradTilingDataTest*>(tiling);

    tilingDatafromBin->ubSize = 196352;
    // lstm input tiling
    tilingDatafromBin->timeStep = 1;
    tilingDatafromBin->batch = 20;
    tilingDatafromBin->inputSize = 8;
    tilingDatafromBin->hiddenSize = 8;
    tilingDatafromBin->isBias = 1;
    tilingDatafromBin->isSeqLength = 0;

    // vector block params
    tilingDatafromBin->singleCoreM = 1;
    tilingDatafromBin->singleCoreMTail = 1;
    tilingDatafromBin->singleCoreN = 8;
    tilingDatafromBin->singleCoreNTail = 8;
    tilingDatafromBin->baseN = 8;
    tilingDatafromBin->baseM = 321;
    tilingDatafromBin->mCnt = 20;
    tilingDatafromBin->nCnt = 1;

    // reduce block params
    tilingDatafromBin->singleCoreReduceN = 8;
    tilingDatafromBin->singleCoreReduceNTail = 8;
    tilingDatafromBin->baseReduceN = 8;
    tilingDatafromBin->nReduceCnt = 4;
    tilingDatafromBin->maxReduceNumOnce = 6104;
    tilingDatafromBin->reduceBlockSize = 20;

    // lstm attr
    tilingDatafromBin->gateOrder = 1;
    tilingDatafromBin->direction = 0;
    tilingDatafromBin->cellClip = 0;
    tilingDatafromBin->forgetBias = 0;

    // split and concat params
    tilingDatafromBin->inputSizeAligned = 8;
    tilingDatafromBin->hiddenSizeAligned = 8;
    tilingDatafromBin->oneLineAligned = 16;

    tilingDatafromBin->dxhInputTiling.taskNum = 0;
    tilingDatafromBin->dxhInputTiling.copyMLines = 0;
    tilingDatafromBin->dxhInputTiling.copyMLinesTail = 0;
    tilingDatafromBin->dxhInputTiling.nLoop = 0;
    tilingDatafromBin->dxhInputTiling.copyNLength = 0;
    tilingDatafromBin->dxhInputTiling.copyNLengthTail = 0;
    tilingDatafromBin->dxhInputTiling.splitTaskPerCore = 0;
    tilingDatafromBin->dxhInputTiling.splitPreCore = 0;

    tilingDatafromBin->dxhHiddenTiling.taskNum = 0;
    tilingDatafromBin->dxhHiddenTiling.copyMLines = 0;
    tilingDatafromBin->dxhHiddenTiling.copyMLinesTail = 0;
    tilingDatafromBin->dxhHiddenTiling.nLoop = 0;
    tilingDatafromBin->dxhHiddenTiling.copyNLength = 0;
    tilingDatafromBin->dxhHiddenTiling.copyNLengthTail = 0;
    tilingDatafromBin->dxhHiddenTiling.splitTaskPerCore = 0;
    tilingDatafromBin->dxhHiddenTiling.splitPreCore = 0;

    tilingDatafromBin->xhInputTiling.taskNum = 0;
    tilingDatafromBin->xhInputTiling.copyMLines = 0;
    tilingDatafromBin->xhInputTiling.copyMLinesTail = 0;
    tilingDatafromBin->xhInputTiling.nLoop = 0;
    tilingDatafromBin->xhInputTiling.copyNLength = 0;
    tilingDatafromBin->xhInputTiling.copyNLengthTail = 0;
    tilingDatafromBin->xhInputTiling.splitTaskPerCore = 0;
    tilingDatafromBin->xhInputTiling.splitPreCore = 0;

    tilingDatafromBin->xhHiddenTiling.taskNum = 0;
    tilingDatafromBin->xhHiddenTiling.copyMLines = 0;
    tilingDatafromBin->xhHiddenTiling.copyMLinesTail = 0;
    tilingDatafromBin->xhHiddenTiling.nLoop = 0;
    tilingDatafromBin->xhHiddenTiling.copyNLength = 0;
    tilingDatafromBin->xhHiddenTiling.copyNLengthTail = 0;
    tilingDatafromBin->xhHiddenTiling.splitTaskPerCore = 0;
    tilingDatafromBin->xhHiddenTiling.splitPreCore = 0;

    tilingDatafromBin->dgateMMParam.usedCoreNum = 1;
    tilingDatafromBin->dgateMMParam.M = 20;
    tilingDatafromBin->dgateMMParam.N = 16;
    tilingDatafromBin->dgateMMParam.Ka = 32;
    tilingDatafromBin->dgateMMParam.Kb = 32;
    tilingDatafromBin->dgateMMParam.singleCoreM = 32;
    tilingDatafromBin->dgateMMParam.singleCoreN = 16;
    tilingDatafromBin->dgateMMParam.singleCoreK = 32;
    tilingDatafromBin->dgateMMParam.baseM = 32;
    tilingDatafromBin->dgateMMParam.baseN = 16;
    tilingDatafromBin->dgateMMParam.baseK = 32;
    tilingDatafromBin->dgateMMParam.depthA1 = 1;
    tilingDatafromBin->dgateMMParam.depthB1 = 1;
    tilingDatafromBin->dgateMMParam.depthAL1CacheUB = 0;
    tilingDatafromBin->dgateMMParam.depthBL1CacheUB = 0;
    tilingDatafromBin->dgateMMParam.stepM = 1;
    tilingDatafromBin->dgateMMParam.stepN = 1;
    tilingDatafromBin->dgateMMParam.isBias = 0;
    tilingDatafromBin->dgateMMParam.transLength = 0;
    tilingDatafromBin->dgateMMParam.iterateOrder = 0;
    tilingDatafromBin->dgateMMParam.shareMode = 0;
    tilingDatafromBin->dgateMMParam.shareL1Size = 6144;
    tilingDatafromBin->dgateMMParam.shareL0CSize = 2048;
    tilingDatafromBin->dgateMMParam.shareUbSize = 0;
    tilingDatafromBin->dgateMMParam.batchM = 1;
    tilingDatafromBin->dgateMMParam.batchN = 1;
    tilingDatafromBin->dgateMMParam.singleBatchM = 1;
    tilingDatafromBin->dgateMMParam.singleBatchN = 1;
    tilingDatafromBin->dgateMMParam.stepKa = 1;
    tilingDatafromBin->dgateMMParam.stepKb = 1;
    tilingDatafromBin->dgateMMParam.dbL0A = 2;
    tilingDatafromBin->dgateMMParam.dbL0B = 2;
    tilingDatafromBin->dgateMMParam.dbL0C = 1;
    tilingDatafromBin->dgateMMParam.ALayoutInfoB = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoS = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoN = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoG = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoD = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoB = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoS = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoN = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoG = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoD = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoB = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoS1 = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoN = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoG = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoS2 = 0;
    tilingDatafromBin->dgateMMParam.BatchNum = 0;

    tilingDatafromBin->dwMMParam.usedCoreNum = 1;
    tilingDatafromBin->dwMMParam.M = 32;
    tilingDatafromBin->dwMMParam.N = 16;
    tilingDatafromBin->dwMMParam.Ka = 20;
    tilingDatafromBin->dwMMParam.Kb = 20;
    tilingDatafromBin->dwMMParam.singleCoreM = 32;
    tilingDatafromBin->dwMMParam.singleCoreN = 16;
    tilingDatafromBin->dwMMParam.singleCoreK = 20;
    tilingDatafromBin->dwMMParam.baseM = 32;
    tilingDatafromBin->dwMMParam.baseN = 16;
    tilingDatafromBin->dwMMParam.baseK = 24;
    tilingDatafromBin->dwMMParam.depthA1 = 1;
    tilingDatafromBin->dwMMParam.depthB1 = 1;
    tilingDatafromBin->dwMMParam.depthAL1CacheUB = 0;
    tilingDatafromBin->dwMMParam.depthBL1CacheUB = 0;
    tilingDatafromBin->dwMMParam.stepM = 1;
    tilingDatafromBin->dwMMParam.stepN = 1;
    tilingDatafromBin->dwMMParam.isBias = 0;
    tilingDatafromBin->dwMMParam.transLength = 0;
    tilingDatafromBin->dwMMParam.iterateOrder = 0;
    tilingDatafromBin->dwMMParam.shareMode = 0;
    tilingDatafromBin->dwMMParam.shareL1Size = 4608;
    tilingDatafromBin->dwMMParam.shareL0CSize = 2048;
    tilingDatafromBin->dwMMParam.shareUbSize = 0;
    tilingDatafromBin->dwMMParam.batchM = 1;
    tilingDatafromBin->dwMMParam.batchN = 1;
    tilingDatafromBin->dwMMParam.singleBatchM = 1;
    tilingDatafromBin->dwMMParam.singleBatchN = 1;
    tilingDatafromBin->dwMMParam.stepKa = 1;
    tilingDatafromBin->dwMMParam.stepKb = 1;
    tilingDatafromBin->dwMMParam.dbL0A = 2;
    tilingDatafromBin->dwMMParam.dbL0B = 2;
    tilingDatafromBin->dwMMParam.dbL0C = 1;
    tilingDatafromBin->dwMMParam.ALayoutInfoB = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoS = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoN = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoG = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoD = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoB = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoS = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoN = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoG = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoD = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoB = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoS1 = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoN = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoG = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoS2 = 0;
    tilingDatafromBin->dwMMParam.BatchNum = 0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        single_layer_lstm_grad, blockDim, x, w, b, y, inith, initc, h, c, dy, dh, dc, i, j, f, o, tanhc, nullptr, dw, db, dx,
        dhPrev, dcPrev, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(w);
    AscendC::GmFree(b);
    AscendC::GmFree(y);
    AscendC::GmFree(inith);
    AscendC::GmFree(initc);
    AscendC::GmFree(h);
    AscendC::GmFree(c);
    AscendC::GmFree(dy);
    AscendC::GmFree(dh);
    AscendC::GmFree(dc);
    AscendC::GmFree(i);
    AscendC::GmFree(j);
    AscendC::GmFree(f);
    AscendC::GmFree(o);
    AscendC::GmFree(tanhc);
    AscendC::GmFree(dw);
    AscendC::GmFree(db);
    AscendC::GmFree(dx);
    AscendC::GmFree(dhPrev);
    AscendC::GmFree(dcPrev);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

template <typename T>
void TestSingleLayerLstmGradKernelLargeSeq(
    uint64_t batchSize, uint64_t timeStep, uint64_t inputSize, uint64_t hiddenSize, uint64_t workspaceSize, uint64_t blockDim,
    uint64_t tilingKey, bool isSeqLength=false)
{
    size_t xBits = batchSize * inputSize * timeStep * sizeof(T);
    size_t hBits= batchSize * timeStep * hiddenSize * sizeof(T);
    size_t inithBits= batchSize * hiddenSize * sizeof(T);
    size_t wBits= 4 * hiddenSize * (hiddenSize + inputSize) * sizeof(T);
    size_t bBits= 4 * hiddenSize * sizeof(T);
    
    size_t tilingDataSize = sizeof(SingleLayerLstmGradTilingDataTest);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xBits);
    uint8_t* w = (uint8_t*)AscendC::GmAlloc(wBits);
    uint8_t* b = (uint8_t*)AscendC::GmAlloc(bBits);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* inith = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* initc = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* h = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* c = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* dh = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* dc = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* i = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* j = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* f = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* o = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* tanhc = (uint8_t*)AscendC::GmAlloc(hBits);
    uint8_t* seqLength = (uint8_t*)AscendC::GmAlloc(hBits);
    T one = 1.0;
    for (size_t idx = 0; idx < xBits; idx += sizeof(T)) {
        std::memcpy(x + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < wBits; idx += sizeof(T)) {
        std::memcpy(w + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < bBits; idx += sizeof(T)) {
        std::memcpy(b + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(y + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(inith + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(initc + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(h + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(c + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(dy + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(dh + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < inithBits; idx += sizeof(T)) {
        std::memcpy(dc + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(i + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(j + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(f + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(o + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(tanhc + idx, &one, sizeof(T));
    }
    for (size_t idx = 0; idx < hBits; idx += sizeof(T)) {
        std::memcpy(seqLength + idx, &one, sizeof(T));
    }

    uint8_t* dw = (uint8_t*)AscendC::GmAlloc(wBits);
    uint8_t* db = (uint8_t*)AscendC::GmAlloc(bBits);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(xBits);
    uint8_t* dhPrev = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* dcPrev = (uint8_t*)AscendC::GmAlloc(inithBits);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    SingleLayerLstmGradTilingDataTest* tilingDatafromBin = reinterpret_cast<SingleLayerLstmGradTilingDataTest*>(tiling);

    tilingDatafromBin->ubSize = 196352;
    // lstm input tiling
    tilingDatafromBin->timeStep = 1;
    tilingDatafromBin->batch = 20;
    tilingDatafromBin->inputSize = 8;
    tilingDatafromBin->hiddenSize = 8;
    tilingDatafromBin->isBias = 0;
    tilingDatafromBin->isSeqLength = 1;

    // vector block params
    tilingDatafromBin->singleCoreM = 1;
    tilingDatafromBin->singleCoreMTail = 1;
    tilingDatafromBin->singleCoreN = 8;
    tilingDatafromBin->singleCoreNTail = 8;
    tilingDatafromBin->baseN = 8;
    tilingDatafromBin->baseM = 305;
    tilingDatafromBin->mCnt = 20;
    tilingDatafromBin->nCnt = 1;

    // reduce block params
    tilingDatafromBin->singleCoreReduceN = 8;
    tilingDatafromBin->singleCoreReduceNTail = 8;
    tilingDatafromBin->baseReduceN = 8;
    tilingDatafromBin->nReduceCnt = 4;
    tilingDatafromBin->maxReduceNumOnce = 6104;
    tilingDatafromBin->reduceBlockSize = 20;

    // lstm attr
    tilingDatafromBin->gateOrder = 1;
    tilingDatafromBin->direction = 0;
    tilingDatafromBin->cellClip = 0;
    tilingDatafromBin->forgetBias = 0;

    // split and concat params
    tilingDatafromBin->inputSizeAligned = 8;
    tilingDatafromBin->hiddenSizeAligned = 8;
    tilingDatafromBin->oneLineAligned = 16;

    tilingDatafromBin->dxhInputTiling.taskNum = 20;
    tilingDatafromBin->dxhInputTiling.copyMLines = 1;
    tilingDatafromBin->dxhInputTiling.copyMLinesTail = 1;
    tilingDatafromBin->dxhInputTiling.nLoop = 1;
    tilingDatafromBin->dxhInputTiling.copyNLength = 48832;
    tilingDatafromBin->dxhInputTiling.copyNLengthTail = 8;
    tilingDatafromBin->dxhInputTiling.splitTaskPerCore = 0;
    tilingDatafromBin->dxhInputTiling.splitPreCore = 20;

    tilingDatafromBin->dxhHiddenTiling.taskNum = 20;
    tilingDatafromBin->dxhHiddenTiling.copyMLines = 1;
    tilingDatafromBin->dxhHiddenTiling.copyMLinesTail = 1;
    tilingDatafromBin->dxhHiddenTiling.nLoop = 1;
    tilingDatafromBin->dxhHiddenTiling.copyNLength = 16280;
    tilingDatafromBin->dxhHiddenTiling.copyNLengthTail = 8;
    tilingDatafromBin->dxhHiddenTiling.splitTaskPerCore = 0;
    tilingDatafromBin->dxhHiddenTiling.splitPreCore = 20;

    tilingDatafromBin->xhInputTiling.taskNum = 20;
    tilingDatafromBin->xhInputTiling.copyMLines = 1;
    tilingDatafromBin->xhInputTiling.copyMLinesTail = 1;
    tilingDatafromBin->xhInputTiling.nLoop = 1;
    tilingDatafromBin->xhInputTiling.copyNLength = 48832;
    tilingDatafromBin->xhInputTiling.copyNLengthTail = 16;
    tilingDatafromBin->xhInputTiling.splitTaskPerCore = 0;
    tilingDatafromBin->xhInputTiling.splitPreCore = 20;

    tilingDatafromBin->xhHiddenTiling.taskNum = 20;
    tilingDatafromBin->xhHiddenTiling.copyMLines = 1;
    tilingDatafromBin->xhHiddenTiling.copyMLinesTail = 1;
    tilingDatafromBin->xhHiddenTiling.nLoop = 1;
    tilingDatafromBin->xhHiddenTiling.copyNLength = 48832;
    tilingDatafromBin->xhHiddenTiling.copyNLengthTail = 16;
    tilingDatafromBin->xhHiddenTiling.splitTaskPerCore = 0;
    tilingDatafromBin->xhHiddenTiling.splitPreCore = 20;

    tilingDatafromBin->dgateMMParam.usedCoreNum = 1;
    tilingDatafromBin->dgateMMParam.M = 20;
    tilingDatafromBin->dgateMMParam.N = 16;
    tilingDatafromBin->dgateMMParam.Ka = 32;
    tilingDatafromBin->dgateMMParam.Kb = 32;
    tilingDatafromBin->dgateMMParam.singleCoreM = 32;
    tilingDatafromBin->dgateMMParam.singleCoreN = 16;
    tilingDatafromBin->dgateMMParam.singleCoreK = 32;
    tilingDatafromBin->dgateMMParam.baseM = 32;
    tilingDatafromBin->dgateMMParam.baseN = 16;
    tilingDatafromBin->dgateMMParam.baseK = 32;
    tilingDatafromBin->dgateMMParam.depthA1 = 1;
    tilingDatafromBin->dgateMMParam.depthB1 = 1;
    tilingDatafromBin->dgateMMParam.depthAL1CacheUB = 0;
    tilingDatafromBin->dgateMMParam.depthBL1CacheUB = 0;
    tilingDatafromBin->dgateMMParam.stepM = 1;
    tilingDatafromBin->dgateMMParam.stepN = 1;
    tilingDatafromBin->dgateMMParam.isBias = 0;
    tilingDatafromBin->dgateMMParam.transLength = 0;
    tilingDatafromBin->dgateMMParam.iterateOrder = 0;
    tilingDatafromBin->dgateMMParam.shareMode = 0;
    tilingDatafromBin->dgateMMParam.shareL1Size = 6144;
    tilingDatafromBin->dgateMMParam.shareL0CSize = 2048;
    tilingDatafromBin->dgateMMParam.shareUbSize = 0;
    tilingDatafromBin->dgateMMParam.batchM = 1;
    tilingDatafromBin->dgateMMParam.batchN = 1;
    tilingDatafromBin->dgateMMParam.singleBatchM = 1;
    tilingDatafromBin->dgateMMParam.singleBatchN = 1;
    tilingDatafromBin->dgateMMParam.stepKa = 1;
    tilingDatafromBin->dgateMMParam.stepKb = 1;
    tilingDatafromBin->dgateMMParam.dbL0A = 2;
    tilingDatafromBin->dgateMMParam.dbL0B = 2;
    tilingDatafromBin->dgateMMParam.dbL0C = 1;
    tilingDatafromBin->dgateMMParam.ALayoutInfoB = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoS = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoN = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoG = 0;
    tilingDatafromBin->dgateMMParam.ALayoutInfoD = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoB = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoS = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoN = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoG = 0;
    tilingDatafromBin->dgateMMParam.BLayoutInfoD = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoB = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoS1 = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoN = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoG = 0;
    tilingDatafromBin->dgateMMParam.CLayoutInfoS2 = 0;
    tilingDatafromBin->dgateMMParam.BatchNum = 0;

    tilingDatafromBin->dwMMParam.usedCoreNum = 1;
    tilingDatafromBin->dwMMParam.M = 32;
    tilingDatafromBin->dwMMParam.N = 16;
    tilingDatafromBin->dwMMParam.Ka = 20;
    tilingDatafromBin->dwMMParam.Kb = 20;
    tilingDatafromBin->dwMMParam.singleCoreM = 32;
    tilingDatafromBin->dwMMParam.singleCoreN = 16;
    tilingDatafromBin->dwMMParam.singleCoreK = 20;
    tilingDatafromBin->dwMMParam.baseM = 32;
    tilingDatafromBin->dwMMParam.baseN = 16;
    tilingDatafromBin->dwMMParam.baseK = 24;
    tilingDatafromBin->dwMMParam.depthA1 = 1;
    tilingDatafromBin->dwMMParam.depthB1 = 1;
    tilingDatafromBin->dwMMParam.depthAL1CacheUB = 0;
    tilingDatafromBin->dwMMParam.depthBL1CacheUB = 0;
    tilingDatafromBin->dwMMParam.stepM = 1;
    tilingDatafromBin->dwMMParam.stepN = 1;
    tilingDatafromBin->dwMMParam.isBias = 0;
    tilingDatafromBin->dwMMParam.transLength = 0;
    tilingDatafromBin->dwMMParam.iterateOrder = 0;
    tilingDatafromBin->dwMMParam.shareMode = 0;
    tilingDatafromBin->dwMMParam.shareL1Size = 4608;
    tilingDatafromBin->dwMMParam.shareL0CSize = 2048;
    tilingDatafromBin->dwMMParam.shareUbSize = 0;
    tilingDatafromBin->dwMMParam.batchM = 1;
    tilingDatafromBin->dwMMParam.batchN = 1;
    tilingDatafromBin->dwMMParam.singleBatchM = 1;
    tilingDatafromBin->dwMMParam.singleBatchN = 1;
    tilingDatafromBin->dwMMParam.stepKa = 1;
    tilingDatafromBin->dwMMParam.stepKb = 1;
    tilingDatafromBin->dwMMParam.dbL0A = 2;
    tilingDatafromBin->dwMMParam.dbL0B = 2;
    tilingDatafromBin->dwMMParam.dbL0C = 1;
    tilingDatafromBin->dwMMParam.ALayoutInfoB = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoS = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoN = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoG = 0;
    tilingDatafromBin->dwMMParam.ALayoutInfoD = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoB = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoS = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoN = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoG = 0;
    tilingDatafromBin->dwMMParam.BLayoutInfoD = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoB = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoS1 = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoN = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoG = 0;
    tilingDatafromBin->dwMMParam.CLayoutInfoS2 = 0;
    tilingDatafromBin->dwMMParam.BatchNum = 0;

    ICPU_SET_TILING_KEY(tilingKey);
    auto seqLengthTensor = isSeqLength ? seqLength : nullptr;
    ICPU_RUN_KF(
        single_layer_lstm_grad, blockDim, x, w, b, y, inith, initc, h, c, dy, dh, dc, i, j, f, o, tanhc,
            seqLengthTensor, dw, db, dx, dhPrev, dcPrev, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(w);
    AscendC::GmFree(b);
    AscendC::GmFree(y);
    AscendC::GmFree(inith);
    AscendC::GmFree(initc);
    AscendC::GmFree(h);
    AscendC::GmFree(c);
    AscendC::GmFree(dy);
    AscendC::GmFree(dh);
    AscendC::GmFree(dc);
    AscendC::GmFree(i);
    AscendC::GmFree(j);
    AscendC::GmFree(f);
    AscendC::GmFree(o);
    AscendC::GmFree(tanhc);
    AscendC::GmFree(seqLength);
    AscendC::GmFree(dw);
    AscendC::GmFree(db);
    AscendC::GmFree(dx);
    AscendC::GmFree(dhPrev);
    AscendC::GmFree(dcPrev);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SingleLayerLstmGradKernel, single_layer_lstm_grad_case_float_0)
{
    uint64_t workspaceSize = 5120;
    uint64_t blockDim = 20;
    uint64_t tilingKey = 0;
    TestSingleLayerLstmGradKernel<float>(
        20, 1, 8, 8, workspaceSize, blockDim, tilingKey);
}

TEST_F(SingleLayerLstmGradKernel, single_layer_lstm_grad_case_float_1100)
{
    uint64_t workspaceSize = 5120;
    uint64_t blockDim = 20;
    uint64_t tilingKey = 1100;
    TestSingleLayerLstmGradKernelLarge<float>(
        20, 1, 8, 8, workspaceSize, blockDim, tilingKey);
}

TEST_F(SingleLayerLstmGradKernel, single_layer_lstm_grad_case_float_1)
{
    uint64_t workspaceSize = 5120;
    uint64_t blockDim = 20;
    uint64_t tilingKey = 0;
    TestSingleLayerLstmGradKernelLargeSeq<float>(
        20, 1, 8, 8, workspaceSize, blockDim, tilingKey, true);
}
