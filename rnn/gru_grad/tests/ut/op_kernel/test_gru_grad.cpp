/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gru_grad.cpp
 * \brief GRU 反向算子 (gru_grad) kernel 单元测试 (CPU 仿真)
 *
 * 注意: 本用例在 CPU 仿真 (ICPU_RUN_KF) 下运行 gru_grad kernel。kernel 内部包含
 *       vector 阶段 + 4 个 matmul + reduce, tiling 由本文件 InitGruGradTiling 手工填入。
 *       更改 (T, B, I, H) 或 dtype 时, 需同步重算 vector/matmul tiling 以避免 GM 越界。
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_gru_grad_tiling_def.h"
#include "data_utils.h"

using namespace std;

constexpr int64_t GRU_GATE_NUM = 3;

extern "C" void gru_grad(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden, GM_ADDR init_h, GM_ADDR output_h,
                         GM_ADDR reset_gate, GM_ADDR update_gate, GM_ADDR new_gate, GM_ADDR h_n, GM_ADDR dy, GM_ADDR dh,
                         GM_ADDR batch_sizes, GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dw_input, GM_ADDR dw_hidden,
                         GM_ADDR db_input, GM_ADDR db_hidden, GM_ADDR workspace, GM_ADDR gruGradTiling);

class GruGradKernel : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "GruGradKernel SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "GruGradKernel TearDown\n" << endl; }
};

struct GruGradBuffers {
    uint8_t* x = nullptr;
    uint8_t* wInput = nullptr;
    uint8_t* wHidden = nullptr;
    uint8_t* initH = nullptr;
    uint8_t* outputH = nullptr;
    uint8_t* resetGate = nullptr;
    uint8_t* updateGate = nullptr;
    uint8_t* newGate = nullptr;
    uint8_t* hN = nullptr;
    uint8_t* dy = nullptr;
    uint8_t* dh = nullptr;
    uint8_t* dx = nullptr;
    uint8_t* dhPrev = nullptr;
    uint8_t* dwInput = nullptr;
    uint8_t* dwHidden = nullptr;
    uint8_t* dbInput = nullptr;
    uint8_t* dbHidden = nullptr;
    uint8_t* workspace = nullptr;
    uint8_t* tiling = nullptr;
};

template <typename T>
void FillBufferFast(uint8_t* buffer, size_t size, T value)
{
    if (buffer == nullptr || size == 0) {
        return;
    }
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
void AllocateBuffers(GruGradBuffers& b, uint64_t batchSize, uint64_t timeStep, uint64_t inputSize, uint64_t hiddenSize,
                     uint64_t workspaceSize)
{
    size_t xBits = timeStep * batchSize * inputSize * sizeof(T);
    size_t hBits = timeStep * batchSize * hiddenSize * sizeof(T);
    size_t inithBits = batchSize * hiddenSize * sizeof(T);
    size_t wiBits = GRU_GATE_NUM * hiddenSize * inputSize * sizeof(T);
    size_t whBits = GRU_GATE_NUM * hiddenSize * hiddenSize * sizeof(T);
    size_t biasBits = GRU_GATE_NUM * hiddenSize * sizeof(T);

    b.x = (uint8_t*)AscendC::GmAlloc(xBits);
    b.wInput = (uint8_t*)AscendC::GmAlloc(wiBits);
    b.wHidden = (uint8_t*)AscendC::GmAlloc(whBits);
    b.initH = (uint8_t*)AscendC::GmAlloc(inithBits);
    b.outputH = (uint8_t*)AscendC::GmAlloc(hBits);
    b.resetGate = (uint8_t*)AscendC::GmAlloc(hBits);
    b.updateGate = (uint8_t*)AscendC::GmAlloc(hBits);
    b.newGate = (uint8_t*)AscendC::GmAlloc(hBits);
    b.hN = (uint8_t*)AscendC::GmAlloc(hBits);
    b.dy = (uint8_t*)AscendC::GmAlloc(hBits);
    b.dh = (uint8_t*)AscendC::GmAlloc(inithBits);

    b.dx = (uint8_t*)AscendC::GmAlloc(xBits);
    b.dhPrev = (uint8_t*)AscendC::GmAlloc(inithBits);
    b.dwInput = (uint8_t*)AscendC::GmAlloc(wiBits);
    b.dwHidden = (uint8_t*)AscendC::GmAlloc(whBits);
    b.dbInput = (uint8_t*)AscendC::GmAlloc(biasBits);
    b.dbHidden = (uint8_t*)AscendC::GmAlloc(biasBits);
    b.workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    b.tiling = (uint8_t*)AscendC::GmAlloc(sizeof(GruGradTilingDataTest));

    T one = static_cast<T>(1.0);
    FillBufferFast<T>(b.x, xBits, one);
    FillBufferFast<T>(b.wInput, wiBits, one);
    FillBufferFast<T>(b.wHidden, whBits, one);
    FillBufferFast<T>(b.initH, inithBits, one);
    FillBufferFast<T>(b.outputH, hBits, one);
    FillBufferFast<T>(b.resetGate, hBits, static_cast<T>(0.5));
    FillBufferFast<T>(b.updateGate, hBits, static_cast<T>(0.5));
    FillBufferFast<T>(b.newGate, hBits, static_cast<T>(0.5));
    FillBufferFast<T>(b.hN, hBits, one);
    FillBufferFast<T>(b.dy, hBits, one);
    FillBufferFast<T>(b.dh, inithBits, one);
}

void FreeBuffers(GruGradBuffers& b)
{
#define SAFE_GM_FREE(ptr)     \
    if (ptr) {                \
        AscendC::GmFree(ptr); \
        ptr = nullptr;        \
    }
    SAFE_GM_FREE(b.x);
    SAFE_GM_FREE(b.wInput);
    SAFE_GM_FREE(b.wHidden);
    SAFE_GM_FREE(b.initH);
    SAFE_GM_FREE(b.outputH);
    SAFE_GM_FREE(b.resetGate);
    SAFE_GM_FREE(b.updateGate);
    SAFE_GM_FREE(b.newGate);
    SAFE_GM_FREE(b.hN);
    SAFE_GM_FREE(b.dy);
    SAFE_GM_FREE(b.dh);
    SAFE_GM_FREE(b.dx);
    SAFE_GM_FREE(b.dhPrev);
    SAFE_GM_FREE(b.dwInput);
    SAFE_GM_FREE(b.dwHidden);
    SAFE_GM_FREE(b.dbInput);
    SAFE_GM_FREE(b.dbHidden);
    SAFE_GM_FREE(b.workspace);
    SAFE_GM_FREE(b.tiling);
#undef SAFE_GM_FREE
}

// 单核跑完整 (M,N,K) 的 matmul tiling, 限制 usedCoreNum=1 使 matmul 仅在 0 号核执行
void InitTCubeTiling(TCubeTiling* t, int64_t m, int64_t n, int64_t k)
{
    t->usedCoreNum = 1;
    t->M = m;
    t->N = n;
    t->Ka = k;
    t->Kb = k;
    t->singleCoreM = m;
    t->singleCoreN = n;
    t->singleCoreK = k;
    t->baseM = m;
    t->baseN = n;
    t->baseK = k;
    t->depthA1 = 1;
    t->depthB1 = 1;
    t->depthAL1CacheUB = 0;
    t->depthBL1CacheUB = 0;
    t->stepM = 1;
    t->stepN = 1;
    t->isBias = 0;
    t->transLength = 0;
    t->iterateOrder = 0;
    t->shareMode = 0;
    t->shareL1Size = 6144;
    t->shareL0CSize = 2048;
    t->shareUbSize = 0;
    t->batchM = 1;
    t->batchN = 1;
    t->singleBatchM = 1;
    t->singleBatchN = 1;
    t->stepKa = 1;
    t->stepKb = 1;
    t->dbL0A = 2;
    t->dbL0B = 2;
    t->dbL0C = 1;
    t->ALayoutInfoB = 0;
    t->ALayoutInfoS = 0;
    t->ALayoutInfoN = 0;
    t->ALayoutInfoG = 0;
    t->ALayoutInfoD = 0;
    t->BLayoutInfoB = 0;
    t->BLayoutInfoS = 0;
    t->BLayoutInfoN = 0;
    t->BLayoutInfoG = 0;
    t->BLayoutInfoD = 0;
    t->CLayoutInfoB = 0;
    t->CLayoutInfoS1 = 0;
    t->CLayoutInfoN = 0;
    t->CLayoutInfoG = 0;
    t->CLayoutInfoS2 = 0;
    t->BatchNum = 0;
}

void InitGruGradTiling(GruGradTilingDataTest* t, uint64_t batchSize, uint64_t timeStep, uint64_t inputSize,
                       uint64_t hiddenSize)
{
    int64_t threeH = GRU_GATE_NUM * static_cast<int64_t>(hiddenSize);
    int64_t tb = static_cast<int64_t>(timeStep) * static_cast<int64_t>(batchSize);

    t->ubSize = 196352;
    t->timeStep = static_cast<int64_t>(timeStep);
    t->batch = static_cast<int64_t>(batchSize);
    t->inputSize = static_cast<int64_t>(inputSize);
    t->hiddenSize = static_cast<int64_t>(hiddenSize);
    t->isBias = 1;
    t->isSeqLength = 0;

    // vector 分块: 每核每次处理 1 行, 整个 hidden 维一次处理
    t->singleCoreM = 1;
    t->singleCoreMTail = 1;
    t->singleCoreN = static_cast<int64_t>(hiddenSize);
    t->singleCoreNTail = static_cast<int64_t>(hiddenSize);
    t->baseN = 4096;
    t->baseM = 0;
    t->mCnt = static_cast<int64_t>(batchSize);
    t->nCnt = 1;

    // reduce: 对 [TB, 3H] 沿 TB 归约得到 [3H] bias 梯度
    t->singleCoreReduceN = 8;
    t->singleCoreReduceNTail = threeH % 8 == 0 ? 8 : threeH % 8;
    t->baseReduceN = 8;
    t->nReduceCnt = (threeH + 7) / 8;
    t->maxReduceNumOnce = 4096;
    t->reduceBlockSize = tb;

    t->direction = 0;
    t->inputSizeAligned = static_cast<int64_t>(inputSize);
    t->hiddenSizeAligned = static_cast<int64_t>(hiddenSize);
    t->oneLineAligned = static_cast<int64_t>(inputSize) + static_cast<int64_t>(hiddenSize);

    // 4 个 matmul (顺序: dwIh, dwHh, dgate, dx)
    // dgateMM: grad_h_prev = d_gh @ w_hh   M=B, N=H, K=3H
    // dwIhMM : dw_ih = d_gi^T @ x           M=3H, N=I, K=T*B
    // dwHhMM : dw_hh = d_gh^T @ h_prev      M=3H, N=H, K=T*B
    // dxMM   : dx = d_gi @ w_ih^T           M=T*B, N=I, K=3H
    InitTCubeTiling(&t->dwIhMMParam, threeH, static_cast<int64_t>(inputSize), tb);
    InitTCubeTiling(&t->dwHhMMParam, threeH, static_cast<int64_t>(hiddenSize), tb);
    InitTCubeTiling(&t->dgateMMParam, static_cast<int64_t>(batchSize), static_cast<int64_t>(hiddenSize), threeH);
    InitTCubeTiling(&t->dxMMParam, tb, static_cast<int64_t>(inputSize), threeH);
}

template <typename T>
void TestGruGradKernel(uint64_t batchSize, uint64_t timeStep, uint64_t inputSize, uint64_t hiddenSize,
                       uint64_t workspaceSize, uint64_t blockDim, uint64_t tilingKey)
{
    GruGradBuffers buffers;
    AllocateBuffers<T>(buffers, batchSize, timeStep, inputSize, hiddenSize, workspaceSize);

    GruGradTilingDataTest* tilingData = reinterpret_cast<GruGradTilingDataTest*>(buffers.tiling);
    InitGruGradTiling(tilingData, batchSize, timeStep, inputSize, hiddenSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(gru_grad, blockDim, buffers.x, buffers.wInput, buffers.wHidden, buffers.initH, buffers.outputH,
                buffers.resetGate, buffers.updateGate, buffers.newGate, buffers.hN, buffers.dy, buffers.dh, nullptr,
                buffers.dx, buffers.dhPrev, buffers.dwInput, buffers.dwHidden, buffers.dbInput, buffers.dbHidden,
                buffers.workspace, buffers.tiling);

    FreeBuffers(buffers);
}

// FP32 单步小形状 (kernel 由 CMake 以 -DDTYPE_X=float 编译, 故仅 float 用例)
TEST_F(GruGradKernel, gru_grad_case_float_single_step)
{
    uint64_t workspaceSize = 4 * 1024 * 1024;
    uint64_t blockDim = 2;
    uint64_t tilingKey = 0;
    TestGruGradKernel<float>(8, 1, 8, 8, workspaceSize, blockDim, tilingKey);
}
