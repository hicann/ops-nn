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
 * \file l1_loss_grad.h
 * \brief
 */

#ifndef L1_LOSS_GRAD_H
#define L1_LOSS_GRAD_H

#include <type_traits>
#include "l1_loss_grad_tiling_data.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MyL1LossGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelL1LossGrad {
public:
    __aicore__ inline KernelL1LossGrad() = default;

    // 使用本地 tiling 数据初始化 Kernel
    __aicore__ inline void Init(
        GM_ADDR grads, GM_ADDR predict, GM_ADDR label, GM_ADDR y, const L1LossGradTilingData& tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void SetProcessDataNum(int32_t index, int32_t loopCount);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuePredict, inQueueGrads, inQueueLabel;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufA;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufB;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufC;

    AscendC::GlobalTensor<T> predictGm;
    AscendC::GlobalTensor<T> gradsGm;
    AscendC::GlobalTensor<T> labelGm;
    AscendC::GlobalTensor<T> yGm;

    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    uint64_t copyInDataNum = 0;
    float scaleValue = 1.0f;
};

template <typename T>
__aicore__ inline void KernelL1LossGrad<T>::Init(
    GM_ADDR grads, GM_ADDR predict, GM_ADDR label, GM_ADDR y, const L1LossGradTilingData& tilingData)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint64_t coreId = AscendC::GetBlockIdx();

    // 直接使用 kernel 入口解析出的本地 tilingData
    uint64_t smallCoreDataNum = tilingData.smallCoreDataNum;
    uint64_t bigCoreDataNum = tilingData.bigCoreDataNum;
    uint64_t finalBigTileNum = tilingData.finalBigTileNum;
    uint64_t finalSmallTileNum = tilingData.finalSmallTileNum;
    uint64_t tileDataNum = tilingData.tileDataNum;
    uint64_t smallTailDataNum = tilingData.smallTailDataNum;
    uint64_t bigTailDataNum = tilingData.bigTailDataNum;
    uint64_t tailBlockNum = tilingData.tailBlockNum;
    this->scaleValue = tilingData.scaleValue;

    uint64_t globalBufferIndex = bigCoreDataNum * coreId;
    this->tileDataNum = tileDataNum;

    if (coreId < tailBlockNum) {
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = finalBigTileNum;
        this->tailDataNum = bigTailDataNum;
    } else {
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = finalSmallTileNum;
        this->tailDataNum = smallTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
    }

    gradsGm.SetGlobalBuffer((__gm__ T*)grads + globalBufferIndex, this->coreDataNum);
    predictGm.SetGlobalBuffer((__gm__ T*)predict + globalBufferIndex, this->coreDataNum);
    labelGm.SetGlobalBuffer((__gm__ T*)label + globalBufferIndex, this->coreDataNum);
    yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

    auto AlignUpElems = [](uint64_t x, uint64_t align) -> uint64_t { return (x + align - 1) / align * align; };

    uint64_t alignElemsT = 32 / sizeof(T);
    uint64_t tileDataNumAlign = AlignUpElems(this->tileDataNum, alignElemsT);
    uint64_t tileBytes = tileDataNumAlign * sizeof(T);

    pipe.InitBuffer(inQueuePredict, BUFFER_NUM, tileBytes);
    pipe.InitBuffer(inQueueGrads, BUFFER_NUM, tileBytes);
    pipe.InitBuffer(inQueueLabel, BUFFER_NUM, tileBytes);
    pipe.InitBuffer(outQueueY, BUFFER_NUM, tileBytes);

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        uint64_t alignElemsF = 32 / sizeof(float);
        uint64_t tileDataNumAlignF = AlignUpElems(this->tileDataNum, alignElemsF);
        size_t bufSizeBytes = tileDataNumAlignF * sizeof(float);
        pipe.InitBuffer(tmpBufA, bufSizeBytes);
        pipe.InitBuffer(tmpBufB, bufSizeBytes);
        pipe.InitBuffer(tmpBufC, bufSizeBytes);
    } else {
        size_t bufSizeBytes = tileDataNumAlign * sizeof(T);
        size_t tmpBSizeBytes = (32 / sizeof(float)) * sizeof(float);
        if (this->tileDataNum > 0) {
            uint64_t alignElemsF = 32 / sizeof(float);
            uint64_t tileDataNumAlignF = AlignUpElems(this->tileDataNum, alignElemsF);
            tmpBSizeBytes = tileDataNumAlignF * sizeof(float);
        }
        pipe.InitBuffer(tmpBufA, bufSizeBytes);
        pipe.InitBuffer(tmpBufB, tmpBSizeBytes);
        pipe.InitBuffer(tmpBufC, tmpBSizeBytes);
    }
}

template <typename T>
__aicore__ inline void KernelL1LossGrad<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> predictLocal = inQueuePredict.template AllocTensor<T>();
    AscendC::LocalTensor<T> gradsLocal = inQueueGrads.template AllocTensor<T>();
    AscendC::LocalTensor<T> labelLocal = inQueueLabel.template AllocTensor<T>();

    (void)progress;

    AscendC::DataCopyExtParams copyParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(this->processDataNum * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    AscendC::DataCopyPadExtParams<T> padParams = {
        true, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    AscendC::DataCopyPad(predictLocal, predictGm[progress * this->tileDataNum], copyParams, padParams);
    AscendC::DataCopyPad(gradsLocal, gradsGm[progress * this->tileDataNum], copyParams, padParams);
    AscendC::DataCopyPad(labelLocal, labelGm[progress * this->tileDataNum], copyParams, padParams);

    inQueuePredict.EnQue(predictLocal);
    inQueueGrads.EnQue(gradsLocal);
    inQueueLabel.EnQue(labelLocal);
}

template <typename T>
__aicore__ inline void KernelL1LossGrad<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> predictLocal = inQueuePredict.template DeQue<T>();
    AscendC::LocalTensor<T> gradsLocal = inQueueGrads.template DeQue<T>();
    AscendC::LocalTensor<T> labelLocal = inQueueLabel.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();

    uint64_t computeDataNum = this->processDataNum;

    (void)progress;

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        // 使用独立 float 临时区，避免向量流水线读写冲突。
        AscendC::LocalTensor<float> diff = tmpBufA.Get<float>();
        AscendC::LocalTensor<float> signBuf = tmpBufB.Get<float>();
        AscendC::LocalTensor<float> gradsBuf = tmpBufC.Get<float>();

        AscendC::Cast(diff, predictLocal, AscendC::RoundMode::CAST_NONE, computeDataNum);
        AscendC::Cast(signBuf, labelLocal, AscendC::RoundMode::CAST_NONE, computeDataNum);
        AscendC::Sub(diff, diff, signBuf, computeDataNum);

        AscendC::Sign(signBuf, diff, computeDataNum);

        AscendC::Cast(gradsBuf, gradsLocal, AscendC::RoundMode::CAST_NONE, computeDataNum);
        AscendC::Mul(gradsBuf, gradsBuf, signBuf, computeDataNum);
        if (this->scaleValue != 1.0f) {
            AscendC::Muls(gradsBuf, gradsBuf, this->scaleValue, computeDataNum);
        }
        AscendC::Cast(yLocal, gradsBuf, AscendC::RoundMode::CAST_ROUND, computeDataNum);
    } else {
        // float / half 计算
        if constexpr (std::is_same_v<T, float>) {
            AscendC::LocalTensor<float> diff = tmpBufA.Get<float>();
            AscendC::LocalTensor<float> signBuf = tmpBufB.Get<float>();
            AscendC::Sub(diff, predictLocal, labelLocal, computeDataNum); // diff
            AscendC::Sign(signBuf, diff, computeDataNum);                 // sign(diff)
            AscendC::Mul(yLocal, gradsLocal, signBuf, computeDataNum);    // grads * sign
            if (this->scaleValue != 1.0f) {
                AscendC::Muls(yLocal, yLocal, this->scaleValue, computeDataNum);
            }
        } else {
            AscendC::LocalTensor<T> diff = tmpBufA.template Get<T>();
            AscendC::LocalTensor<T> signBuf = tmpBufB.template Get<T>();
            AscendC::Sub(diff, predictLocal, labelLocal, computeDataNum); // diff
            AscendC::Sign(signBuf, diff, computeDataNum);                 // sign(diff)
            AscendC::Mul(yLocal, gradsLocal, signBuf, computeDataNum);    // grads * sign
            if (this->scaleValue != 1.0f) {
                AscendC::Muls(yLocal, yLocal, static_cast<T>(this->scaleValue), computeDataNum);
            }
        }
    }

    outQueueY.EnQue<T>(yLocal);
    inQueuePredict.FreeTensor(predictLocal);
    inQueueGrads.FreeTensor(gradsLocal);
    inQueueLabel.FreeTensor(labelLocal);
}

template <typename T>
__aicore__ inline void KernelL1LossGrad<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> yLocal = outQueueY.template DeQue<T>();

    (void)progress;

    AscendC::DataCopyExtParams outParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(this->processDataNum * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    AscendC::DataCopyPad(yGm[progress * this->tileDataNum], yLocal, outParams);

    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void KernelL1LossGrad<T>::SetProcessDataNum(int32_t index, int32_t loopCount)
{
    if (index == loopCount - 1) {
        this->processDataNum = this->tailDataNum;
    } else {
        this->processDataNum = this->tileDataNum;
    }
    this->copyInDataNum = this->processDataNum;
}

template <typename T>
__aicore__ inline void KernelL1LossGrad<T>::Process()
{
    int32_t loopCount = this->tileNum;
    if (loopCount <= 0) {
        return;
    }

    SetProcessDataNum(0, loopCount);
    CopyIn(0);

    for (int32_t i = 1; i < loopCount; ++i) {
        SetProcessDataNum(i, loopCount);
        CopyIn(i);

        SetProcessDataNum(i - 1, loopCount);
        Compute(i - 1);
        CopyOut(i - 1);
    }

    SetProcessDataNum(loopCount - 1, loopCount);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace MyL1LossGrad

#endif // L1_LOSS_GRAD_H