/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef __BINARY_CROSS_ENTROPY_GRAD_V2_MS_H__
#define __BINARY_CROSS_ENTROPY_GRAD_V2_MS_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "binary_cross_entropy_grad_v2_tiling_data.h"
#include "binary_cross_entropy_grad_v2_tiling_key.h"

namespace NsBinaryCrossEntropyGradV2 {

using namespace AscendC;

template <typename T>
class BinaryCrossEntropyGradV2MS {
public:
    __aicore__ inline BinaryCrossEntropyGradV2MS(){};

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR logits, GM_ADDR labels, GM_ADDR weight, 
    GM_ADDR outgrad, const BinaryCrossEntropyGradV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueLogits;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueLabels;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueGrad;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueWeight;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueOutGrad;

    AscendC::GlobalTensor<T> logitsGm;
    AscendC::GlobalTensor<T> labelsGm;
    AscendC::GlobalTensor<T> gradGm;
    AscendC::GlobalTensor<T> weightGm;
    AscendC::GlobalTensor<T> outgradGm;

    int64_t coreDataNum;
    int64_t tileNum;
    int64_t tileDataNum;
    int64_t tailDataNum;
    int64_t processDataNum;
    int64_t totalLength;
    int64_t reduction;
};

template <typename T>
__aicore__ inline void BinaryCrossEntropyGradV2MS<T>::Init(GM_ADDR grad, GM_ADDR logits, GM_ADDR labels, GM_ADDR weight, 
     GM_ADDR outgrad, const BinaryCrossEntropyGradV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    int64_t coreNum = AscendC::GetBlockIdx();
    int64_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->totalLength = tilingData->totalLength;
    this->reduction = tilingData->reduction;
    if (coreNum < tilingData->tailBlockNum) { 
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    }
    else { 
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }
    logitsGm.SetGlobalBuffer((__gm__ T*)logits + globalBufferIndex, this->coreDataNum);
    labelsGm.SetGlobalBuffer((__gm__ T*)labels + globalBufferIndex, this->coreDataNum);
    gradGm.SetGlobalBuffer((__gm__ T*)grad, 16);
    weightGm.SetGlobalBuffer((__gm__ T*)weight + globalBufferIndex, this->coreDataNum);
    outgradGm.SetGlobalBuffer((__gm__ T*)outgrad + globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(inQueueLogits, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inQueueLabels, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inQueueGrad, BUFFER_NUM, 16 * sizeof(T));
    pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileDataNum * sizeof(T));   
    pipe.InitBuffer(outQueueOutGrad, BUFFER_NUM, this->tileDataNum * sizeof(T));
}

template <typename T>
__aicore__ inline void BinaryCrossEntropyGradV2MS<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> logitsLocal = inQueueLogits.AllocTensor<T>();
    AscendC::LocalTensor<T> labelsLocal = inQueueLabels.AllocTensor<T>();
    AscendC::LocalTensor<T> gradLocal = inQueueGrad.AllocTensor<T>();
    AscendC::LocalTensor<T> weightLocal = inQueueWeight.AllocTensor<T>();

    AscendC::DataCopy(logitsLocal, logitsGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(labelsLocal, labelsGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(gradLocal, gradGm, 16);
    AscendC::DataCopy(weightLocal, weightGm[progress * this->tileDataNum], this->processDataNum);

    inQueueLogits.EnQue(logitsLocal);
    inQueueLabels.EnQue(labelsLocal);
    inQueueGrad.EnQue(gradLocal);
    inQueueWeight.EnQue(weightLocal);
}

template <typename T>
__aicore__ inline void BinaryCrossEntropyGradV2MS<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> outgradLocal = outQueueOutGrad.DeQue<T>();
    AscendC::DataCopy(outgradGm[progress * this->tileDataNum], outgradLocal, this->processDataNum);
    outQueueOutGrad.FreeTensor(outgradLocal);
}

template <typename T>
__aicore__ inline void BinaryCrossEntropyGradV2MS<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> logitsLocal = inQueueLogits.DeQue<T>();
    AscendC::LocalTensor<T> labelsLocal = inQueueLabels.DeQue<T>();
    AscendC::LocalTensor<T> gradLocal = inQueueGrad.DeQue<T>();
    AscendC::LocalTensor<T> weightLocal = inQueueWeight.DeQue<T>();
    AscendC::LocalTensor<T> outgradLocal = outQueueOutGrad.AllocTensor<T>();

    T reductionScalar = reduction;
    T scalar = gradLocal.GetValue(0);
    float scalarf = 0;
    float redcf = static_cast<float>(reductionScalar);
    if(redcf == 2)  scalarf = static_cast<float>(scalar)/this->totalLength;
        else scalarf = static_cast<float>(scalar);
    scalar = static_cast<T>(scalarf);

    AscendC::Sub(logitsLocal, logitsLocal, labelsLocal, this->processDataNum);
    AscendC::Muls(logitsLocal, logitsLocal, scalar, this->processDataNum);
    AscendC::Mul(outgradLocal, logitsLocal, weightLocal, this->processDataNum);

    outQueueOutGrad.EnQue<T>(outgradLocal);
    inQueueLogits.FreeTensor(logitsLocal);
    inQueueLabels.FreeTensor(labelsLocal);
    inQueueGrad.FreeTensor(gradLocal);
    inQueueWeight.FreeTensor(weightLocal);
}

template <typename T>
__aicore__ inline void BinaryCrossEntropyGradV2MS<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}
} // namespace NsBinaryCrossEntropyGradV2
#endif // BINARY_CROSS_ENTROPY_GRAD_V2_H