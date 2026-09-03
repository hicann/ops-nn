/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_simd_two_dim.h
 * \brief SIMD two-dim kernel template for Embedding operator (contiguous).
 *        Scenario: x is 2D [gatherDimSize, innerSize], indices is 1D [gatherSize],
 *        y is 2D [gatherSize, innerSize]. Equivalent to gather(axis=0).
 */
#ifndef EMBEDDING_SIMD_TWO_DIM_H
#define EMBEDDING_SIMD_TWO_DIM_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/platform_util.h"
#include "embedding_simd_base.h"

namespace Embedding {
using namespace AscendC;

template <typename INDICES_T>
class EmbeddingSimdTwoDim : public EmbeddingSimdBase<INDICES_T> {
public:
    __aicore__ inline EmbeddingSimdTwoDim(TPipe* pipe) { this->pipe_ = pipe; };
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, const EmbeddingTilingDataSimdTwoDim* tilingData);
    __aicore__ inline INDICES_T GetIndex(int64_t idx, int64_t endIdx);
    __aicore__ inline void NoSplitColProcess(int64_t colsAlign);
    __aicore__ inline void SplitColProcess(int64_t colsAlign);
    __aicore__ inline void Process();

private:
    const EmbeddingTilingDataSimdTwoDim* tilingData_ = nullptr;
};

template <typename INDICES_T>
__aicore__ inline void EmbeddingSimdTwoDim<INDICES_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y,
                                                            const EmbeddingTilingDataSimdTwoDim* tilingData)
{
    tilingData_ = tilingData;
    this->InitBaseBuffer(this->pipe_, tilingData_->maxElement, tilingData_->dtypeSize, tilingData_->indiceFactor, x,
                         indices, y);
    this->indicesOffsetBase_ = 0;
}

template <typename INDICES_T>
__aicore__ inline INDICES_T EmbeddingSimdTwoDim<INDICES_T>::GetIndex(int64_t idx, int64_t endIdx)
{
    if (idx >= this->indicesOffsetBase_ + this->curIndexSize_) {
        int64_t copyLen = (endIdx - idx) > tilingData_->indiceFactor ? tilingData_->indiceFactor : endIdx - idx;
        this->indicesOffsetBase_ = idx;
        this->curIndexSize_ = copyLen;
        LocalTensor<INDICES_T> tmpLocal = this->indexBuf_.template Get<INDICES_T>();
        this->CopyInContiguous(tmpLocal, this->indicesGm_, this->indicesOffsetBase_, 1, copyLen);

        event_t eventIdMTE2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
    }
    LocalTensor<INDICES_T> indexLocal = this->indexBuf_.template Get<INDICES_T>();
    return indexLocal.GetValue(idx - this->indicesOffsetBase_);
}

template <typename INDICES_T>
__aicore__ inline void EmbeddingSimdTwoDim<INDICES_T>::NoSplitColProcess(int64_t colsAlign)
{
    int64_t yStart = 0;
    int64_t yEnd = 0;
    this->GetYStartYEnd(yStart, yEnd, tilingData_->blockFactor, tilingData_->tailBlockFactor);
    int64_t currentCoreElements = yEnd - yStart;
    int64_t onceMaxRows = tilingData_->maxElement * tilingData_->dtypeSize / colsAlign;
    int64_t loopNum = Ops::Base::CeilDiv(currentCoreElements, onceMaxRows);

    int64_t indiceEndIdx = yEnd;
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t rows = (i == loopNum - 1) ? (currentCoreElements - i * onceMaxRows) : onceMaxRows;
        int64_t cols = tilingData_->innerSize;
        LocalTensor<int8_t> xLocal = this->inQueue_.template AllocTensor<int8_t>();
        event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
        int64_t preIdx0 = -1;
        int64_t preIdx1 = -1;
        int64_t preIdx2 = -1;
        int32_t backOffset0 = 0;
        int32_t backOffset1 = 0;
        int32_t backOffset2 = 0;
        for (int64_t j = 0; j < rows; j++) {
            int64_t yIdx = yStart + i * onceMaxRows + j;

            INDICES_T index = GetIndex(yIdx, indiceEndIdx);
            int64_t xIndex = static_cast<int64_t>(index) * tilingData_->innerSize;
            int64_t offset = xIndex * tilingData_->dtypeSize;
            if (likely(index >= 0 && index < tilingData_->gatherDimSize)) {
                if (unlikely(xIndex == preIdx0 || xIndex == preIdx1 || xIndex == preIdx2)) {
                    event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
                    int32_t backStep = (xIndex == preIdx0) ? backOffset0 :
                                       (xIndex == preIdx1) ? backOffset1 :
                                                             backOffset2;
                    Copy(xLocal[j * colsAlign], xLocal[backStep], tilingData_->innerSize * tilingData_->dtypeSize);
                } else {
                    this->CopyInContiguous(xLocal[j * colsAlign], this->xGm_, offset, 1,
                                           tilingData_->innerSize * tilingData_->dtypeSize);
                    preIdx2 = preIdx1;
                    preIdx1 = preIdx0;
                    preIdx0 = xIndex;
                    backOffset2 = backOffset1;
                    backOffset1 = backOffset0;
                    backOffset0 = j * colsAlign;
                }
            } else {
                Duplicate<int8_t>(xLocal[j * colsAlign], 0, tilingData_->innerSize * tilingData_->dtypeSize);
            }
        }
        this->inQueue_.template EnQue<int8_t>(xLocal);
        int64_t yOffset = (yStart + i * onceMaxRows) * tilingData_->innerSize * tilingData_->dtypeSize;
        this->CopyOut(yOffset, rows, cols * tilingData_->dtypeSize);
    }
}

template <typename INDICES_T>
__aicore__ inline void EmbeddingSimdTwoDim<INDICES_T>::SplitColProcess(int64_t colsAlign)
{
    int64_t yStart = 0;
    int64_t yEnd = 0;
    this->GetYStartYEnd(yStart, yEnd, tilingData_->blockFactor, tilingData_->tailBlockFactor);
    int64_t indiceEndIdx = yEnd;
    int64_t loopSize = Ops::Base::CeilDiv(tilingData_->innerSize, tilingData_->maxElement);
    for (int64_t i = yStart; i < yEnd; i++) {
        INDICES_T index = GetIndex(i, indiceEndIdx);
        int64_t xIndex = static_cast<int64_t>(index) * tilingData_->innerSize;
        int64_t yIndex = i * tilingData_->innerSize;
        for (int64_t j = 0; j < loopSize; j++) {
            int64_t cols = (j == loopSize - 1) ? (tilingData_->innerSize - j * tilingData_->maxElement) :
                                                 tilingData_->maxElement;
            LocalTensor<int8_t> xLocal = this->inQueue_.template AllocTensor<int8_t>();
            int64_t offset = (xIndex + j * tilingData_->maxElement) * tilingData_->dtypeSize;
            if (likely(index >= 0 && index < tilingData_->gatherDimSize)) {
                this->CopyInContiguous(xLocal[0], this->xGm_, offset, 1, cols * tilingData_->dtypeSize);
            } else {
                event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
                WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
                Duplicate<int8_t>(xLocal[0], 0, cols * tilingData_->dtypeSize);
            }
            this->inQueue_.template EnQue<int8_t>(xLocal);
            int64_t yOffset = (yIndex + j * tilingData_->maxElement) * tilingData_->dtypeSize;
            this->CopyOut(yOffset, 1, cols * tilingData_->dtypeSize);
        }
    }
}

template <typename INDICES_T>
__aicore__ inline void EmbeddingSimdTwoDim<INDICES_T>::Process()
{
    if (static_cast<int32_t>(GetBlockIdx()) >= tilingData_->needCoreNum) {
        return;
    }
    if (tilingData_->gatherSize == 0 || tilingData_->innerSize == 0) {
        return;
    }
    int64_t colsAlign = this->GetColsAlign(tilingData_->innerSize, tilingData_->dtypeSize);

    if (colsAlign <= tilingData_->maxElement * tilingData_->dtypeSize) {
        NoSplitColProcess(colsAlign);
    } else {
        SplitColProcess(colsAlign);
    }
}
} // namespace Embedding
#endif // EMBEDDING_SIMD_TWO_DIM_H
