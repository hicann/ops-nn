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
 * \file embedding_simd_no_contiguous.h
 * \brief SIMD kernel template for Embedding operator with non-contiguous x/indices.
 *        Scenario: x and indices are both 2D with arbitrary strides.
 *        y = indices.shape + [innerSize], y is contiguous.
 */
#ifndef EMBEDDING_SIMD_NO_CONTIGUOUS_H
#define EMBEDDING_SIMD_NO_CONTIGUOUS_H

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
class EmbeddingSimdNoContiguous : public EmbeddingSimdBase<INDICES_T> {
public:
    __aicore__ inline EmbeddingSimdNoContiguous(TPipe* pipe) { this->pipe_ = pipe; };
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y,
                                const EmbeddingNoContiguousSimdTilingData* tilingData);
    __aicore__ inline INDICES_T GetIndex(int64_t gatherIdx, int64_t endIdx);
    __aicore__ inline void NoSplitColProcess(int64_t colsAlign);
    __aicore__ inline void SplitColProcess(int64_t colsAlign);
    __aicore__ inline void Process();

private:
    const EmbeddingNoContiguousSimdTilingData* tilingData_ = nullptr;
};

template <typename INDICES_T>
__aicore__ inline void EmbeddingSimdNoContiguous<INDICES_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y,
                                                                  const EmbeddingNoContiguousSimdTilingData* tilingData)
{
    tilingData_ = tilingData;
    this->InitBaseBuffer(this->pipe_, tilingData_->maxElement, tilingData_->dtypeSize, tilingData_->indiceFactor, x,
                         indices, y);
}

template <typename INDICES_T>
__aicore__ inline INDICES_T EmbeddingSimdNoContiguous<INDICES_T>::GetIndex(int64_t gatherIdx, int64_t endIdx)
{
    if (this->indicesOffsetBase_ < 0 || gatherIdx >= this->indicesOffsetBase_ + this->curIndexSize_) {
        int64_t copyLen = (endIdx - gatherIdx) > tilingData_->indiceFactor ? tilingData_->indiceFactor :
                                                                             endIdx - gatherIdx;
        this->indicesOffsetBase_ = gatherIdx;
        this->curIndexSize_ = copyLen;
        LocalTensor<INDICES_T> tmpLocal = this->indexBuf_.template Get<INDICES_T>();
        if (tilingData_->indicesDim0Stride == tilingData_->indicesDim1Size && tilingData_->indicesDim1Stride == 1) {
            this->CopyInContiguous(tmpLocal, this->indicesGm_, this->indicesOffsetBase_, 1, copyLen);
        } else if (tilingData_->indicesDim1Stride == 1) {
            int64_t dim0Idx = gatherIdx / tilingData_->indicesDim1Size;
            int64_t dim1Idx = gatherIdx - dim0Idx * tilingData_->indicesDim1Size;
            int64_t startOffset = dim0Idx * tilingData_->indicesDim0Stride + dim1Idx;
            int64_t remainingInRow = tilingData_->indicesDim1Size - dim1Idx;
            int64_t firstRowLen = copyLen > remainingInRow ? remainingInRow : copyLen;
            this->CopyInContiguous(tmpLocal, this->indicesGm_, startOffset, 1, firstRowLen);
            int64_t copied = firstRowLen;
            int64_t curDim0 = dim0Idx + 1;
            while (copied < copyLen) {
                int64_t chunk = copyLen - copied > tilingData_->indicesDim1Size ? tilingData_->indicesDim1Size :
                                                                                  copyLen - copied;
                this->CopyInContiguous(tmpLocal[copied], this->indicesGm_, curDim0 * tilingData_->indicesDim0Stride, 1,
                                       chunk);
                copied += chunk;
                curDim0++;
            }
        } else {
            for (int64_t i = 0; i < copyLen; i++) {
                int64_t dim0Idx = (gatherIdx + i) / tilingData_->indicesDim1Size;
                int64_t dim1Idx = (gatherIdx + i) - dim0Idx * tilingData_->indicesDim1Size;
                int64_t indicesIdx = dim0Idx * tilingData_->indicesDim0Stride +
                                     dim1Idx * tilingData_->indicesDim1Stride;
                tmpLocal.SetValue(i, this->indicesGm_.GetValue(indicesIdx));
            }
        }
        event_t eventIdMTE2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
    }
    LocalTensor<INDICES_T> indexLocal = this->indexBuf_.template Get<INDICES_T>();
    return indexLocal.GetValue(gatherIdx - this->indicesOffsetBase_);
}

template <typename INDICES_T>
__aicore__ inline void EmbeddingSimdNoContiguous<INDICES_T>::NoSplitColProcess(int64_t colsAlign)
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
        for (int64_t j = 0; j < rows; j++) {
            int64_t gatherIdx = yStart + i * onceMaxRows + j;
            INDICES_T index = GetIndex(gatherIdx, indiceEndIdx);
            int64_t xBaseOffset = static_cast<int64_t>(index) * tilingData_->xDim0Stride * tilingData_->dtypeSize;
            if (likely(index >= 0 && index < tilingData_->gatherDimSize)) {
                if (tilingData_->xDim1Stride == 1) {
                    this->CopyInContiguous(xLocal[j * colsAlign], this->xGm_, xBaseOffset, 1,
                                           tilingData_->innerSize * tilingData_->dtypeSize);
                } else {
                    this->template CopyInNoContiguous<int8_t>(xLocal[j * colsAlign], this->xGm_, xBaseOffset,
                                                              tilingData_->innerSize, tilingData_->dtypeSize,
                                                              (tilingData_->xDim1Stride - 1) * tilingData_->dtypeSize);
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
__aicore__ inline void EmbeddingSimdNoContiguous<INDICES_T>::SplitColProcess(int64_t colsAlign)
{
    int64_t yStart = 0;
    int64_t yEnd = 0;
    this->GetYStartYEnd(yStart, yEnd, tilingData_->blockFactor, tilingData_->tailBlockFactor);
    int64_t loopSize = Ops::Base::CeilDiv(tilingData_->innerSize, tilingData_->maxElement);
    int64_t indiceEndIdx = yEnd;
    for (int64_t i = yStart; i < yEnd; i++) {
        INDICES_T index = GetIndex(i, indiceEndIdx);
        int64_t xBaseOffset = static_cast<int64_t>(index) * tilingData_->xDim0Stride * tilingData_->dtypeSize;
        int64_t yIndex = i * tilingData_->innerSize;
        for (int64_t j = 0; j < loopSize; j++) {
            int64_t cols = (j == loopSize - 1) ? (tilingData_->innerSize - j * tilingData_->maxElement) :
                                                 tilingData_->maxElement;
            LocalTensor<int8_t> xLocal = this->inQueue_.template AllocTensor<int8_t>();
            if (likely(index >= 0 && index < tilingData_->gatherDimSize)) {
                if (tilingData_->xDim1Stride == 1) {
                    int64_t offset = xBaseOffset + j * tilingData_->maxElement * tilingData_->dtypeSize;
                    this->CopyInContiguous(xLocal[0], this->xGm_, offset, 1, cols * tilingData_->dtypeSize);
                } else {
                    int64_t elemOffset = xBaseOffset + j * tilingData_->maxElement * tilingData_->xDim1Stride *
                                                           tilingData_->dtypeSize;
                    this->template CopyInNoContiguous<int8_t>(xLocal[0], this->xGm_, elemOffset, cols,
                                                              tilingData_->dtypeSize,
                                                              (tilingData_->xDim1Stride - 1) * tilingData_->dtypeSize);
                }
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
__aicore__ inline void EmbeddingSimdNoContiguous<INDICES_T>::Process()
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
#endif // EMBEDDING_SIMD_NO_CONTIGUOUS_H
