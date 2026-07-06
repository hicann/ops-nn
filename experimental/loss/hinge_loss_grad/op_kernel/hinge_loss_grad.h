/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hinge_loss_grad.h
 * \brief HingeLossGrad算子Kernel实现（支持float/float16/bfloat16）
 *
 * 公式: margin = 1 - target * predict
 *       grad_input = (margin > 0) ? (-target * grad_output) : 0
 */
#ifndef HINGE_LOSS_GRAD_H
#define HINGE_LOSS_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hinge_loss_grad_tiling_data.h"

namespace NsHingeLossGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t BLOCK_SIZE = 32;

__aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

template<typename T>
class HingeLossGrad {
public:
    __aicore__ inline HingeLossGrad() {}

    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR target, GM_ADDR gradOutput, GM_ADDR gradInput,
                                 const HingeLossGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint64_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(uint64_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> predictQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> targetQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradOutputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gradInputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;
    TBuf<QuePosition::VECCALC> castTmpBuf0;
    TBuf<QuePosition::VECCALC> castTmpBuf1;
    TBuf<QuePosition::VECCALC> castTmpBuf2;
    TBuf<QuePosition::VECCALC> castTmpBuf3;

    GlobalTensor<T> predictGm;
    GlobalTensor<T> targetGm;
    GlobalTensor<T> gradOutputGm;
    GlobalTensor<T> gradInputGm;

    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t tileDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
};

template<typename T>
__aicore__ inline void HingeLossGrad<T>::Init(
    GM_ADDR predict, GM_ADDR target, GM_ADDR gradOutput, GM_ADDR gradInput,
    const HingeLossGradTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreIdx = AscendC::GetBlockIdx();
    uint64_t globalOffset = tilingData->bigCoreDataNum * coreIdx;
    this->tileDataNum = tilingData->tileDataNum;

    if (tilingData->tailBlockNum > 0) {
        if (coreIdx < tilingData->tailBlockNum) {
            this->coreDataNum = tilingData->bigCoreDataNum;
            this->tileNum = tilingData->finalBigTileNum;
            this->tailDataNum = tilingData->bigTailDataNum;
        } else {
            this->coreDataNum = tilingData->smallCoreDataNum;
            this->tileNum = tilingData->finalSmallTileNum;
            this->tailDataNum = tilingData->smallTailDataNum;
            globalOffset -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (coreIdx - tilingData->tailBlockNum);
        }
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalOffset = tilingData->smallCoreDataNum * coreIdx;
    }

    predictGm.SetGlobalBuffer((__gm__ T*)predict + globalOffset, this->coreDataNum);
    targetGm.SetGlobalBuffer((__gm__ T*)target + globalOffset, this->coreDataNum);
    gradOutputGm.SetGlobalBuffer((__gm__ T*)gradOutput + globalOffset, this->coreDataNum);
    gradInputGm.SetGlobalBuffer((__gm__ T*)gradInput + globalOffset, this->coreDataNum);

    uint64_t dataAlignNum = AlignUp(this->tileDataNum * sizeof(T), BLOCK_SIZE) / sizeof(T);
    uint64_t floatAlignNum = AlignUp(this->tileDataNum * sizeof(float), BLOCK_SIZE) / sizeof(float);
    pipe.InitBuffer(predictQueue, BUFFER_NUM, dataAlignNum * sizeof(T));
    pipe.InitBuffer(targetQueue, BUFFER_NUM, dataAlignNum * sizeof(T));
    pipe.InitBuffer(gradOutputQueue, BUFFER_NUM, dataAlignNum * sizeof(T));
    pipe.InitBuffer(gradInputQueue, BUFFER_NUM, dataAlignNum * sizeof(T));
    pipe.InitBuffer(tmpBuf0, floatAlignNum * sizeof(float));
    pipe.InitBuffer(tmpBuf1, floatAlignNum * sizeof(float));
    pipe.InitBuffer(tmpBuf2, floatAlignNum * sizeof(float));

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        pipe.InitBuffer(castTmpBuf0, floatAlignNum * sizeof(float));
        pipe.InitBuffer(castTmpBuf1, floatAlignNum * sizeof(float));
        pipe.InitBuffer(castTmpBuf2, floatAlignNum * sizeof(float));
        pipe.InitBuffer(castTmpBuf3, floatAlignNum * sizeof(float));
    }
}

template<typename T>
__aicore__ inline void HingeLossGrad<T>::Process()
{
    this->processDataNum = this->tileDataNum;
    for (uint64_t index = 0; index < this->tileNum; ++index) {
        if (index == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(index);
        Compute();
        CopyOut(index);
    }
}

template<typename T>
__aicore__ inline void HingeLossGrad<T>::CopyIn(uint64_t progress)
{
    LocalTensor<T> predictLocal = predictQueue.AllocTensor<T>();
    LocalTensor<T> targetLocal = targetQueue.AllocTensor<T>();
    LocalTensor<T> gradOutputLocal = gradOutputQueue.AllocTensor<T>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(this->processDataNum * sizeof(T)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {true, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(predictLocal, predictGm[progress * this->tileDataNum], copyParams, padParams);
    DataCopyPad(targetLocal, targetGm[progress * this->tileDataNum], copyParams, padParams);
    DataCopyPad(gradOutputLocal, gradOutputGm[progress * this->tileDataNum], copyParams, padParams);

    predictQueue.EnQue(predictLocal);
    targetQueue.EnQue(targetLocal);
    gradOutputQueue.EnQue(gradOutputLocal);
}

template<typename T>
__aicore__ inline void HingeLossGrad<T>::Compute()
{
    constexpr float eps = 1e-7f;
    LocalTensor<T> predictLocal = predictQueue.DeQue<T>();
    LocalTensor<T> targetLocal = targetQueue.DeQue<T>();
    LocalTensor<T> gradOutputLocal = gradOutputQueue.DeQue<T>();
    LocalTensor<T> gradInputLocal = gradInputQueue.AllocTensor<T>();

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> predictFloat = castTmpBuf0.AllocTensor<float>();
        LocalTensor<float> targetFloat = castTmpBuf1.AllocTensor<float>();
        LocalTensor<float> gradOutputFloat = castTmpBuf2.AllocTensor<float>();
        LocalTensor<float> resultFloat = castTmpBuf3.AllocTensor<float>();

        Cast(predictFloat, predictLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(targetFloat, targetLocal, RoundMode::CAST_NONE, this->processDataNum);
        Cast(gradOutputFloat, gradOutputLocal, RoundMode::CAST_NONE, this->processDataNum);

        Mul(resultFloat, targetFloat, predictFloat, this->processDataNum);
        Muls(resultFloat, resultFloat, -1.0f, this->processDataNum);
        Adds(resultFloat, resultFloat, 1.0f, this->processDataNum);

        Mul(predictFloat, targetFloat, gradOutputFloat, this->processDataNum);
        Muls(predictFloat, predictFloat, -1.0f, this->processDataNum);

        Abs(targetFloat, resultFloat, this->processDataNum);
        Add(resultFloat, resultFloat, targetFloat, this->processDataNum);
        Adds(targetFloat, targetFloat, eps, this->processDataNum);
        Muls(targetFloat, targetFloat, 2.0f, this->processDataNum);
        Div(resultFloat, resultFloat, targetFloat, this->processDataNum);
        Mul(resultFloat, resultFloat, predictFloat, this->processDataNum);

        Cast(gradInputLocal, resultFloat, RoundMode::CAST_RINT, this->processDataNum);

        castTmpBuf0.FreeTensor(predictFloat);
        castTmpBuf1.FreeTensor(targetFloat);
        castTmpBuf2.FreeTensor(gradOutputFloat);
        castTmpBuf3.FreeTensor(resultFloat);
    } else {
        LocalTensor<float> margin = tmpBuf0.Get<float>();
        LocalTensor<float> gradCandidate = tmpBuf1.Get<float>();
        LocalTensor<float> absMargin = tmpBuf2.Get<float>();

        Mul(margin, targetLocal, predictLocal, this->processDataNum);
        Muls(margin, margin, -1.0f, this->processDataNum);
        Adds(margin, margin, 1.0f, this->processDataNum);

        Mul(gradCandidate, targetLocal, gradOutputLocal, this->processDataNum);
        Muls(gradCandidate, gradCandidate, -1.0f, this->processDataNum);

        Abs(absMargin, margin, this->processDataNum);
        Add(margin, margin, absMargin, this->processDataNum);
        Adds(absMargin, absMargin, eps, this->processDataNum);
        Muls(absMargin, absMargin, 2.0f, this->processDataNum);
        Div(margin, margin, absMargin, this->processDataNum);
        Mul(gradInputLocal, margin, gradCandidate, this->processDataNum);
    }

    gradInputQueue.EnQue<T>(gradInputLocal);
    predictQueue.FreeTensor(predictLocal);
    targetQueue.FreeTensor(targetLocal);
    gradOutputQueue.FreeTensor(gradOutputLocal);
}

template<typename T>
__aicore__ inline void HingeLossGrad<T>::CopyOut(uint64_t progress)
{
    LocalTensor<T> gradInputLocal = gradInputQueue.DeQue<T>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(this->processDataNum * sizeof(T)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(gradInputGm[progress * this->tileDataNum], gradInputLocal, copyParams);
    gradInputQueue.FreeTensor(gradInputLocal);
}

} // namespace NsHingeLossGrad
#endif // HINGE_LOSS_GRAD_H
