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
 * \brief Single Layer LSTM Gradient Operator Kernel Test (Optimized)
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

extern "C" void single_layer_lstm_grad(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, GM_ADDR init_h, GM_ADDR init_c,
                                       GM_ADDR h, GM_ADDR c, GM_ADDR dy, GM_ADDR dh, GM_ADDR dc, GM_ADDR i, GM_ADDR j,
                                       GM_ADDR f, GM_ADDR o, GM_ADDR tanhct, GM_ADDR seq_length, GM_ADDR dw, GM_ADDR db,
                                       GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dc_prev, GM_ADDR workspace,
                                       GM_ADDR rnnGradTiling);

class SingleLayerLstmGradKernel : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SingleLayerLstmGrad Kernel SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "SingleLayerLstmGrad Kernel TearDown\n" << endl; }
};

struct LstmGradBuffers {
    uint8_t* x = nullptr;
    uint8_t* w = nullptr;
    uint8_t* b = nullptr;
    uint8_t* y = nullptr;
    uint8_t* inith = nullptr;
    uint8_t* initc = nullptr;
    uint8_t* h = nullptr;
    uint8_t* c = nullptr;
    uint8_t* dy = nullptr;
    uint8_t* dh = nullptr;
    uint8_t* dc = nullptr;
    uint8_t* i = nullptr;
    uint8_t* j = nullptr;
    uint8_t* f = nullptr;
    uint8_t* o = nullptr;
    uint8_t* tanhc = nullptr;
    uint8_t* seqLength = nullptr;
    uint8_t* dw = nullptr;
    uint8_t* db = nullptr;
    uint8_t* dx = nullptr;
    uint8_t* dhPrev = nullptr;
    uint8_t* dcPrev = nullptr;
    uint8_t* workspace = nullptr;
    uint8_t* tiling = nullptr;
};

template <typename T>
void FillBufferFast(uint8_t* buffer, size_t size, T value)
{
    if (buffer == nullptr || size == 0)
        return;
    T* ptr = reinterpret_cast<T*>(buffer);
    size_t count = size / sizeof(T);

    ptr[0] = value;
    size_t filled = 1;
    while (filled < count) {
        size_t copyCount = std::min(filled, count - filled);
        std::memcpy(ptr + filled, ptr, copyCount * sizeof(T));
        filled += copyCount;
    }
}

template <typename T>
void AllocateBuffers(LstmGradBuffers& buffers, uint64_t batchSize, uint64_t timeStep, uint64_t inputSize,
                     uint64_t hiddenSize, uint64_t workspaceSize, bool needSeqLength = false)
{
    size_t xBits = batchSize * inputSize * timeStep * sizeof(T);
    size_t hBits = batchSize * timeStep * hiddenSize * sizeof(T);
    size_t inithBits = batchSize * hiddenSize * sizeof(T);
    size_t wBits = 4 * hiddenSize * (hiddenSize + inputSize) * sizeof(T);
    size_t bBits = 4 * hiddenSize * sizeof(T);

    buffers.x = (uint8_t*)AscendC::GmAlloc(xBits);
    buffers.w = (uint8_t*)AscendC::GmAlloc(wBits);
    buffers.b = (uint8_t*)AscendC::GmAlloc(bBits);
    buffers.y = (uint8_t*)AscendC::GmAlloc(hBits);
    buffers.inith = (uint8_t*)AscendC::GmAlloc(inithBits);
    buffers.initc = (uint8_t*)AscendC::GmAlloc(inithBits);
    buffers.h = (uint8_t*)AscendC::GmAlloc(hBits);
    buffers.c = (uint8_t*)AscendC::GmAlloc(hBits);
    buffers.dy = (uint8_t*)AscendC::GmAlloc(hBits);
    buffers.dh = (uint8_t*)AscendC::GmAlloc(inithBits);
    buffers.dc = (uint8_t*)AscendC::GmAlloc(inithBits);
    buffers.i = (uint8_t*)AscendC::GmAlloc(hBits);
    buffers.j = (uint8_t*)AscendC::GmAlloc(hBits);
    buffers.f = (uint8_t*)AscendC::GmAlloc(hBits);
    buffers.o = (uint8_t*)AscendC::GmAlloc(hBits);
    buffers.tanhc = (uint8_t*)AscendC::GmAlloc(hBits);
    if (needSeqLength) {
        buffers.seqLength = (uint8_t*)AscendC::GmAlloc(hBits);
    }

    buffers.dw = (uint8_t*)AscendC::GmAlloc(wBits);
    buffers.db = (uint8_t*)AscendC::GmAlloc(bBits);
    buffers.dx = (uint8_t*)AscendC::GmAlloc(xBits);
    buffers.dhPrev = (uint8_t*)AscendC::GmAlloc(inithBits);
    buffers.dcPrev = (uint8_t*)AscendC::GmAlloc(inithBits);
    buffers.workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    buffers.tiling = (uint8_t*)AscendC::GmAlloc(sizeof(SingleLayerLstmGradTilingDataTest));

    T one = static_cast<T>(1.0);
    FillBufferFast<T>(buffers.x, xBits, one);
    FillBufferFast<T>(buffers.w, wBits, one);
    FillBufferFast<T>(buffers.b, bBits, one);
    FillBufferFast<T>(buffers.y, hBits, one);
    FillBufferFast<T>(buffers.inith, inithBits, one);
    FillBufferFast<T>(buffers.initc, inithBits, one);
    FillBufferFast<T>(buffers.h, hBits, one);
    FillBufferFast<T>(buffers.c, hBits, one);
    FillBufferFast<T>(buffers.dy, hBits, one);
    FillBufferFast<T>(buffers.dh, inithBits, one);
    FillBufferFast<T>(buffers.dc, inithBits, one);
    FillBufferFast<T>(buffers.i, hBits, one);
    FillBufferFast<T>(buffers.j, hBits, one);
    FillBufferFast<T>(buffers.f, hBits, one);
    FillBufferFast<T>(buffers.o, hBits, one);
    FillBufferFast<T>(buffers.tanhc, hBits, one);
    if (needSeqLength) {
        FillBufferFast<T>(buffers.seqLength, hBits, one);
    }
}

void FreeBuffers(LstmGradBuffers& buffers)
{
#define SAFE_GM_FREE(ptr)     \
    if (ptr) {                \
        AscendC::GmFree(ptr); \
        ptr = nullptr;        \
    }
    SAFE_GM_FREE(buffers.x);
    SAFE_GM_FREE(buffers.w);
    SAFE_GM_FREE(buffers.b);
    SAFE_GM_FREE(buffers.y);
    SAFE_GM_FREE(buffers.inith);
    SAFE_GM_FREE(buffers.initc);
    SAFE_GM_FREE(buffers.h);
    SAFE_GM_FREE(buffers.c);
    SAFE_GM_FREE(buffers.dy);
    SAFE_GM_FREE(buffers.dh);
    SAFE_GM_FREE(buffers.dc);
    SAFE_GM_FREE(buffers.i);
    SAFE_GM_FREE(buffers.j);
    SAFE_GM_FREE(buffers.f);
    SAFE_GM_FREE(buffers.o);
    SAFE_GM_FREE(buffers.tanhc);
    SAFE_GM_FREE(buffers.seqLength);
    SAFE_GM_FREE(buffers.dw);
    SAFE_GM_FREE(buffers.db);
    SAFE_GM_FREE(buffers.dx);
    SAFE_GM_FREE(buffers.dhPrev);
    SAFE_GM_FREE(buffers.dcPrev);
    SAFE_GM_FREE(buffers.workspace);
    SAFE_GM_FREE(buffers.tiling);
#undef SAFE_GM_FREE
}

void InitTCubeTiling(TCubeTiling* tiling, uint64_t batchSize)
{
    tiling->usedCoreNum = 1;
    tiling->M = batchSize;
    tiling->N = 16;
    tiling->Ka = 32;
    tiling->Kb = 32;
    tiling->singleCoreM = 32;
    tiling->singleCoreN = 16;
    tiling->singleCoreK = 32;
    tiling->baseM = 32;
    tiling->baseN = 16;
    tiling->baseK = 32;
    tiling->depthA1 = 1;
    tiling->depthB1 = 1;
    tiling->depthAL1CacheUB = 0;
    tiling->depthBL1CacheUB = 0;
    tiling->stepM = 1;
    tiling->stepN = 1;
    tiling->isBias = 0;
    tiling->transLength = 0;
    tiling->iterateOrder = 0;
    tiling->shareMode = 0;
    tiling->shareL1Size = 6144;
    tiling->shareL0CSize = 2048;
    tiling->shareUbSize = 0;
    tiling->batchM = 1;
    tiling->batchN = 1;
    tiling->singleBatchM = 1;
    tiling->singleBatchN = 1;
    tiling->stepKa = 1;
    tiling->stepKb = 1;
    tiling->dbL0A = 2;
    tiling->dbL0B = 2;
    tiling->dbL0C = 1;
    tiling->ALayoutInfoB = 0;
    tiling->ALayoutInfoS = 0;
    tiling->ALayoutInfoN = 0;
    tiling->ALayoutInfoG = 0;
    tiling->ALayoutInfoD = 0;
    tiling->BLayoutInfoB = 0;
    tiling->BLayoutInfoS = 0;
    tiling->BLayoutInfoN = 0;
    tiling->BLayoutInfoG = 0;
    tiling->BLayoutInfoD = 0;
    tiling->CLayoutInfoB = 0;
    tiling->CLayoutInfoS1 = 0;
    tiling->CLayoutInfoN = 0;
    tiling->CLayoutInfoG = 0;
    tiling->CLayoutInfoS2 = 0;
    tiling->BatchNum = 0;
}

void InitDefaultTiling(SingleLayerLstmGradTilingDataTest* tiling, uint64_t batchSize, uint64_t inputSize,
                       uint64_t hiddenSize)
{
    tiling->ubSize = 196352;
    tiling->timeStep = 1;
    tiling->batch = batchSize;
    tiling->inputSize = inputSize;
    tiling->hiddenSize = hiddenSize;
    tiling->isBias = 1;
    tiling->isSeqLength = 0;

    tiling->singleCoreM = 1;
    tiling->singleCoreMTail = 1;
    tiling->singleCoreN = 16;
    tiling->singleCoreNTail = 8;
    tiling->baseN = 16;
    tiling->baseM = 160;
    tiling->mCnt = batchSize;
    tiling->nCnt = 1;

    tiling->singleCoreReduceN = 16;
    tiling->singleCoreReduceNTail = 16;
    tiling->baseReduceN = 16;
    tiling->nReduceCnt = 2;
    tiling->maxReduceNumOnce = 3052;
    tiling->reduceBlockSize = batchSize;

    tiling->gateOrder = 0;
    tiling->direction = 0;
    tiling->cellClip = 0;
    tiling->forgetBias = 0;

    tiling->inputSizeAligned = 16;
    tiling->hiddenSizeAligned = 16;
    tiling->oneLineAligned = 32;

    auto* dxhInput = &tiling->dxhInputTiling;
    dxhInput->taskNum = batchSize;
    dxhInput->copyMLines = 1;
    dxhInput->copyMLinesTail = 1;
    dxhInput->nLoop = 1;
    dxhInput->copyNLength = 48832;
    dxhInput->copyNLengthTail = 16;
    dxhInput->splitTaskPerCore = 0;
    dxhInput->splitPreCore = batchSize;

    auto* dxhHidden = &tiling->dxhHiddenTiling;
    dxhHidden->taskNum = batchSize;
    dxhHidden->copyMLines = 1;
    dxhHidden->copyMLinesTail = 1;
    dxhHidden->nLoop = 1;
    dxhHidden->copyNLength = 48832;
    dxhHidden->copyNLengthTail = 16;
    dxhHidden->splitTaskPerCore = 0;
    dxhHidden->splitPreCore = batchSize;

    auto* xhInput = &tiling->xhInputTiling;
    xhInput->taskNum = batchSize;
    xhInput->copyMLines = 1;
    xhInput->copyMLinesTail = 1;
    xhInput->nLoop = 1;
    xhInput->copyNLength = 32560;
    xhInput->copyNLengthTail = 16;
    xhInput->splitTaskPerCore = 0;
    xhInput->splitPreCore = batchSize;

    auto* xhHidden = &tiling->xhHiddenTiling;
    xhHidden->taskNum = batchSize;
    xhHidden->copyMLines = 1;
    xhHidden->copyMLinesTail = 1;
    xhHidden->nLoop = 1;
    xhHidden->copyNLength = 32560;
    xhHidden->copyNLengthTail = 16;
    xhHidden->splitTaskPerCore = 0;
    xhHidden->splitPreCore = batchSize;

    InitTCubeTiling(&tiling->dgateMMParam, batchSize);

    TCubeTiling* dwMM = &tiling->dwMMParam;
    dwMM->usedCoreNum = 1;
    dwMM->M = 32;
    dwMM->N = 16;
    dwMM->Ka = batchSize;
    dwMM->Kb = batchSize;
    dwMM->singleCoreM = 32;
    dwMM->singleCoreN = 16;
    dwMM->singleCoreK = batchSize;
    dwMM->baseM = 32;
    dwMM->baseN = 16;
    dwMM->baseK = 24;
    dwMM->depthA1 = 1;
    dwMM->depthB1 = 1;
    dwMM->depthAL1CacheUB = 0;
    dwMM->depthBL1CacheUB = 0;
    dwMM->stepM = 1;
    dwMM->stepN = 1;
    dwMM->isBias = 0;
    dwMM->transLength = 0;
    dwMM->iterateOrder = 0;
    dwMM->shareMode = 0;
    dwMM->shareL1Size = 4608;
    dwMM->shareL0CSize = 2048;
    dwMM->shareUbSize = 0;
    dwMM->batchM = 1;
    dwMM->batchN = 1;
    dwMM->singleBatchM = 1;
    dwMM->singleBatchN = 1;
    dwMM->stepKa = 1;
    dwMM->stepKb = 1;
    dwMM->dbL0A = 2;
    dwMM->dbL0B = 2;
    dwMM->dbL0C = 1;
    dwMM->ALayoutInfoB = 0;
    dwMM->ALayoutInfoS = 0;
    dwMM->ALayoutInfoN = 0;
    dwMM->ALayoutInfoG = 0;
    dwMM->ALayoutInfoD = 0;
    dwMM->BLayoutInfoB = 0;
    dwMM->BLayoutInfoS = 0;
    dwMM->BLayoutInfoN = 0;
    dwMM->BLayoutInfoG = 0;
    dwMM->BLayoutInfoD = 0;
    dwMM->CLayoutInfoB = 0;
    dwMM->CLayoutInfoS1 = 0;
    dwMM->CLayoutInfoN = 0;
    dwMM->CLayoutInfoG = 0;
    dwMM->CLayoutInfoS2 = 0;
    dwMM->BatchNum = 0;
}

template <typename T>
void TestSingleLayerLstmGradKernel(uint64_t batchSize, uint64_t timeStep, uint64_t inputSize, uint64_t hiddenSize,
                                   uint64_t workspaceSize, uint64_t blockDim, uint64_t tilingKey)
{
    LstmGradBuffers buffers;
    AllocateBuffers<T>(buffers, batchSize, timeStep, inputSize, hiddenSize, workspaceSize);

    SingleLayerLstmGradTilingDataTest* tilingData = reinterpret_cast<SingleLayerLstmGradTilingDataTest*>(
        buffers.tiling);
    InitDefaultTiling(tilingData, batchSize, inputSize, hiddenSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(single_layer_lstm_grad, blockDim, buffers.x, buffers.w, buffers.b, buffers.y, buffers.inith,
                buffers.initc, buffers.h, buffers.c, buffers.dy, buffers.dh, buffers.dc, buffers.i, buffers.j,
                buffers.f, buffers.o, buffers.tanhc, nullptr, buffers.dw, buffers.db, buffers.dx, buffers.dhPrev,
                buffers.dcPrev, buffers.workspace, buffers.tiling);

    FreeBuffers(buffers);
}

template <typename T>
void TestSingleLayerLstmGradKernelLarge(uint64_t batchSize, uint64_t timeStep, uint64_t inputSize, uint64_t hiddenSize,
                                        uint64_t workspaceSize, uint64_t blockDim, uint64_t tilingKey)
{
    LstmGradBuffers buffers;
    AllocateBuffers<T>(buffers, batchSize, timeStep, inputSize, hiddenSize, workspaceSize);

    SingleLayerLstmGradTilingDataTest* tilingData = reinterpret_cast<SingleLayerLstmGradTilingDataTest*>(
        buffers.tiling);
    InitDefaultTiling(tilingData, batchSize, inputSize, hiddenSize);

    tilingData->singleCoreN = 8;
    tilingData->singleCoreNTail = 8;
    tilingData->baseN = 8;
    tilingData->baseM = 321;
    tilingData->singleCoreReduceN = 8;
    tilingData->singleCoreReduceNTail = 8;
    tilingData->baseReduceN = 8;
    tilingData->nReduceCnt = 4;
    tilingData->maxReduceNumOnce = 6104;
    tilingData->gateOrder = 1;
    tilingData->inputSizeAligned = 8;
    tilingData->hiddenSizeAligned = 8;
    tilingData->oneLineAligned = 16;

    auto* dxhInput = &tilingData->dxhInputTiling;
    dxhInput->taskNum = 0;
    dxhInput->copyMLines = 0;
    dxhInput->copyMLinesTail = 0;
    dxhInput->nLoop = 0;
    dxhInput->copyNLength = 0;
    dxhInput->copyNLengthTail = 0;
    dxhInput->splitTaskPerCore = 0;
    dxhInput->splitPreCore = 0;

    auto* dxhHidden = &tilingData->dxhHiddenTiling;
    dxhHidden->taskNum = 0;
    dxhHidden->copyMLines = 0;
    dxhHidden->copyMLinesTail = 0;
    dxhHidden->nLoop = 0;
    dxhHidden->copyNLength = 0;
    dxhHidden->copyNLengthTail = 0;
    dxhHidden->splitTaskPerCore = 0;
    dxhHidden->splitPreCore = 0;

    auto* xhInput = &tilingData->xhInputTiling;
    xhInput->taskNum = 0;
    xhInput->copyMLines = 0;
    xhInput->copyMLinesTail = 0;
    xhInput->nLoop = 0;
    xhInput->copyNLength = 0;
    xhInput->copyNLengthTail = 0;
    xhInput->splitTaskPerCore = 0;
    xhInput->splitPreCore = 0;

    auto* xhHidden = &tilingData->xhHiddenTiling;
    xhHidden->taskNum = 0;
    xhHidden->copyMLines = 0;
    xhHidden->copyMLinesTail = 0;
    xhHidden->nLoop = 0;
    xhHidden->copyNLength = 0;
    xhHidden->copyNLengthTail = 0;
    xhHidden->splitTaskPerCore = 0;
    xhHidden->splitPreCore = 0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(single_layer_lstm_grad, blockDim, buffers.x, buffers.w, buffers.b, buffers.y, buffers.inith,
                buffers.initc, buffers.h, buffers.c, buffers.dy, buffers.dh, buffers.dc, buffers.i, buffers.j,
                buffers.f, buffers.o, buffers.tanhc, nullptr, buffers.dw, buffers.db, buffers.dx, buffers.dhPrev,
                buffers.dcPrev, buffers.workspace, buffers.tiling);

    FreeBuffers(buffers);
}

template <typename T>
void TestSingleLayerLstmGradKernelLargeSeq(uint64_t batchSize, uint64_t timeStep, uint64_t inputSize,
                                           uint64_t hiddenSize, uint64_t workspaceSize, uint64_t blockDim,
                                           uint64_t tilingKey, bool isSeqLength)
{
    LstmGradBuffers buffers;
    AllocateBuffers<T>(buffers, batchSize, timeStep, inputSize, hiddenSize, workspaceSize, isSeqLength);

    SingleLayerLstmGradTilingDataTest* tilingData = reinterpret_cast<SingleLayerLstmGradTilingDataTest*>(
        buffers.tiling);
    InitDefaultTiling(tilingData, batchSize, inputSize, hiddenSize);

    tilingData->isBias = 0;
    tilingData->isSeqLength = 1;
    tilingData->singleCoreN = 8;
    tilingData->singleCoreNTail = 8;
    tilingData->baseN = 8;
    tilingData->baseM = 305;
    tilingData->singleCoreReduceN = 8;
    tilingData->singleCoreReduceNTail = 8;
    tilingData->baseReduceN = 8;
    tilingData->nReduceCnt = 4;
    tilingData->maxReduceNumOnce = 6104;
    tilingData->gateOrder = 1;
    tilingData->inputSizeAligned = 8;
    tilingData->hiddenSizeAligned = 8;
    tilingData->oneLineAligned = 16;

    tilingData->dxhInputTiling.copyNLengthTail = 8;
    tilingData->dxhHiddenTiling.copyNLength = 16280;
    tilingData->dxhHiddenTiling.copyNLengthTail = 8;

    ICPU_SET_TILING_KEY(tilingKey);
    auto seqLengthTensor = isSeqLength ? buffers.seqLength : nullptr;
    ICPU_RUN_KF(single_layer_lstm_grad, blockDim, buffers.x, buffers.w, buffers.b, buffers.y, buffers.inith,
                buffers.initc, buffers.h, buffers.c, buffers.dy, buffers.dh, buffers.dc, buffers.i, buffers.j,
                buffers.f, buffers.o, buffers.tanhc, seqLengthTensor, buffers.dw, buffers.db, buffers.dx,
                buffers.dhPrev, buffers.dcPrev, buffers.workspace, buffers.tiling);

    FreeBuffers(buffers);
}

TEST_F(SingleLayerLstmGradKernel, single_layer_lstm_grad_case_float_0)
{
    uint64_t workspaceSize = 5120;
    uint64_t blockDim = 20;
    uint64_t tilingKey = 0;
    TestSingleLayerLstmGradKernel<float>(20, 1, 8, 8, workspaceSize, blockDim, tilingKey);
}

TEST_F(SingleLayerLstmGradKernel, single_layer_lstm_grad_case_float_1100)
{
    uint64_t workspaceSize = 5120;
    uint64_t blockDim = 20;
    uint64_t tilingKey = 1100;
    TestSingleLayerLstmGradKernelLarge<float>(20, 1, 8, 8, workspaceSize, blockDim, tilingKey);
}

TEST_F(SingleLayerLstmGradKernel, single_layer_lstm_grad_case_float_1)
{
    uint64_t workspaceSize = 5120;
    uint64_t blockDim = 20;
    uint64_t tilingKey = 0;
    TestSingleLayerLstmGradKernelLargeSeq<float>(20, 1, 8, 8, workspaceSize, blockDim, tilingKey, true);
}
