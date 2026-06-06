/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file silu_grad.h
 */
#ifndef SILU_GRAD_H
#define SILU_GRAD_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "silu_grad_tiling_data.h"
#include "silu_grad_tiling_key.h"

namespace NsSiluGrad {
using namespace AscendC;

template <typename T>
class KernelSiluGrad {
public:
    __aicore__ inline KernelSiluGrad() {}

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR dx, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t tileDataNum, uint64_t tailBlockNum, uint64_t schMode);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t processDataNum);
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t processDataNum);
    __aicore__ inline void Compute(uint32_t processDataNum);
    __aicore__ inline void ProcessSingleBuffer();
    __aicore__ inline void ProcessDoubleBuffer();

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inQueueDY_;
    TQue<QuePosition::VECIN, 2> inQueueX_;
    TQue<QuePosition::VECOUT, 2> outQueueDX_;

    TBuf<QuePosition::VECCALC> tmpBuf1_;
    TBuf<QuePosition::VECCALC> tmpBuf2_;

    TBuf<QuePosition::VECCALC> castDYBuf_;
    TBuf<QuePosition::VECCALC> castXBuf_;
    TBuf<QuePosition::VECCALC> castDXBuf_;

    GlobalTensor<T> dyGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> dxGm_;

    uint64_t coreDataNum;
    uint64_t tileDataNum;
    uint64_t schMode;
};

template <typename T>
__aicore__ inline void KernelSiluGrad<T>::Init(GM_ADDR dy, GM_ADDR x, GM_ADDR dx, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t tileDataNum, uint64_t tailBlockNum, uint64_t schMode)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = GetBlockIdx();
    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;
    this->schMode = schMode;
    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
    }
    dyGm_.SetGlobalBuffer((__gm__ T*)dy + globalBufferIndex, this->coreDataNum);
    xGm_.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
    dxGm_.SetGlobalBuffer((__gm__ T*)dx + globalBufferIndex, this->coreDataNum);

    if constexpr (!std::is_same_v<T, float>) {
        pipe_.InitBuffer(castDYBuf_, this->tileDataNum * sizeof(float));
        pipe_.InitBuffer(castXBuf_, this->tileDataNum * sizeof(float));
        pipe_.InitBuffer(castDXBuf_, this->tileDataNum * sizeof(float));
    }
    uint64_t bufferNum = (schMode == 0) ? 1 : 2;
    pipe_.InitBuffer(inQueueDY_, bufferNum, this->tileDataNum * sizeof(T));
    pipe_.InitBuffer(inQueueX_, bufferNum, this->tileDataNum * sizeof(T));
    pipe_.InitBuffer(outQueueDX_, bufferNum, this->tileDataNum * sizeof(T));
    pipe_.InitBuffer(tmpBuf1_, this->tileDataNum * sizeof(float));
    pipe_.InitBuffer(tmpBuf2_, this->tileDataNum * sizeof(float));
}

template <typename T>
__aicore__ inline void KernelSiluGrad<T>::CopyIn(uint64_t offset, uint32_t processDataNum)
{
    LocalTensor<T> dyLocal = inQueueDY_.AllocTensor<T>();
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = processDataNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(dyLocal, dyGm_[offset], copyParams, {false, 0, 0, 0});
    AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, {false, 0, 0, 0});
    inQueueDY_.EnQue(dyLocal);
    inQueueX_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void KernelSiluGrad<T>::CopyOut(uint64_t offset, uint32_t processDataNum)
{
    LocalTensor<T> dxLocal = outQueueDX_.DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = processDataNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(dxGm_[offset], dxLocal, copyParams);
    outQueueDX_.FreeTensor(dxLocal);
}

template <typename T>
__aicore__ inline void KernelSiluGrad<T>::Compute(uint32_t processDataNum)
{
    if constexpr (std::is_same_v<T, float>) {
        LocalTensor<float> dyLocal = inQueueDY_.DeQue<float>();
        LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
        LocalTensor<float> dxLocal = outQueueDX_.AllocTensor<float>();
        LocalTensor<float> tmp1 = tmpBuf1_.Get<float>();
        LocalTensor<float> tmp2 = tmpBuf2_.Get<float>();

        // SiLU 梯度公式: dx = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        Muls(tmp1, xLocal, -1.0f, processDataNum);
        Exp(tmp1, tmp1, processDataNum);
        Adds(tmp1, tmp1, 1.0f, processDataNum);
        Duplicate(dxLocal, 1.0f, processDataNum);
        Div(tmp2, dxLocal, tmp1, processDataNum);
        Sub(tmp1, dxLocal, tmp2, processDataNum);
        Mul(tmp1, xLocal, tmp1, processDataNum);
        Adds(tmp1, tmp1, 1.0f, processDataNum);
        Mul(tmp1, tmp2, tmp1, processDataNum);
        Mul(dxLocal, dyLocal, tmp1, processDataNum);

        outQueueDX_.EnQue<float>(dxLocal);
        inQueueDY_.FreeTensor(dyLocal);
        inQueueX_.FreeTensor(xLocal);
    } else {
        LocalTensor<T> dyLocal = inQueueDY_.DeQue<T>();
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> dxLocal = outQueueDX_.AllocTensor<T>();
        LocalTensor<float> dyCalc = castDYBuf_.Get<float>();
        LocalTensor<float> xCalc = castXBuf_.Get<float>();
        LocalTensor<float> dxCalc = castDXBuf_.Get<float>();
        LocalTensor<float> tmp1 = tmpBuf1_.Get<float>();
        LocalTensor<float> tmp2 = tmpBuf2_.Get<float>();

        Cast(dyCalc, dyLocal, RoundMode::CAST_NONE, processDataNum);
        Cast(xCalc, xLocal, RoundMode::CAST_NONE, processDataNum);

        // SiLU 梯度公式: dx = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        Muls(tmp1, xCalc, -1.0f, processDataNum);
        Exp(tmp1, tmp1, processDataNum);
        Adds(tmp1, tmp1, 1.0f, processDataNum);
        Duplicate(dxCalc, 1.0f, processDataNum);
        Div(tmp2, dxCalc, tmp1, processDataNum);
        Sub(tmp1, dxCalc, tmp2, processDataNum);
        Mul(tmp1, xCalc, tmp1, processDataNum);
        Adds(tmp1, tmp1, 1.0f, processDataNum);
        Mul(tmp1, tmp2, tmp1, processDataNum);
        Mul(dxCalc, dyCalc, tmp1, processDataNum);

        Cast(dxLocal, dxCalc, RoundMode::CAST_RINT, processDataNum);
        outQueueDX_.EnQue<T>(dxLocal);
        inQueueDY_.FreeTensor(dyLocal);
        inQueueX_.FreeTensor(xLocal);
    }
}

template <typename T>
__aicore__ inline void KernelSiluGrad<T>::Process()
{
    if (this->schMode == 0) {
        ProcessSingleBuffer();
    } else {
        ProcessDoubleBuffer();
    }
}

template <typename T>
__aicore__ inline void KernelSiluGrad<T>::ProcessSingleBuffer()
{
    // 单循环优化：只有 1 个 tile 时直接串行执行，跳过循环开销
    if (coreDataNum <= tileDataNum) {
        uint32_t processDataNum = static_cast<uint32_t>(coreDataNum);
        CopyIn(0, processDataNum);
        Compute(processDataNum);
        CopyOut(0, processDataNum);
        return;
    }
    for (uint64_t offset = 0; offset < coreDataNum; offset += tileDataNum) {
        uint32_t processDataNum = static_cast<uint32_t>(AscendC::Std::min(tileDataNum, coreDataNum - offset));
        CopyIn(offset, processDataNum);
        Compute(processDataNum);
        CopyOut(offset, processDataNum);
    }
}

template <typename T>
__aicore__ inline void KernelSiluGrad<T>::ProcessDoubleBuffer()
{
    // 单循环优化：只有 1 个 tile 时直接串行执行，不启用流水线
    if (coreDataNum <= tileDataNum) {
        uint32_t processDataNum = static_cast<uint32_t>(coreDataNum);
        CopyIn(0, processDataNum);
        Compute(processDataNum);
        CopyOut(0, processDataNum);
        return;
    }

    uint64_t offset = 0;
    uint32_t processDataNum = static_cast<uint32_t>(AscendC::Std::min(tileDataNum, coreDataNum - offset));
    // 预加载第一个 tile
    CopyIn(offset, processDataNum);
    offset += tileDataNum;

    uint64_t outOffset = 0;
    // 主循环：CopyIn 下一个 tile 与 Compute 当前 tile 重叠（DMA + Vector 并行）
    while (offset < coreDataNum) {
        uint32_t nextProcessDataNum = static_cast<uint32_t>(AscendC::Std::min(tileDataNum, coreDataNum - offset));
        CopyIn(offset, nextProcessDataNum);
        Compute(processDataNum);
        CopyOut(outOffset, processDataNum);
        processDataNum = nextProcessDataNum;
        offset += tileDataNum;
        outOffset += tileDataNum;
    }
    // 排空最后一个 tile
    Compute(processDataNum);
    CopyOut(outOffset, processDataNum);
}

} // namespace NsSiluGrad

#endif // SILU_GRAD_H
