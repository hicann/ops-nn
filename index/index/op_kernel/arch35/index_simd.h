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
 * \file index_simd.h
 * \brief
 */
#ifndef INDEX_SIMD_H
#define INDEX_SIMD_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Index {
using namespace AscendC;

constexpr int32_t BUFFER_NUM_SIMD = 2;

template <typename INDICES_T>
class IndexSimd {
public:
    __aicore__ inline IndexSimd(){};
    __aicore__ inline void Init(
        GM_ADDR output, GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
        const IndexSimdTilingData* tilingData, GM_ADDR value = nullptr);
    __aicore__ inline void Process();
    template <typename T>
    __aicore__ inline void CopyIn(
        LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst, uint32_t copyLen);
    __aicore__ inline void GetIndices(uint64_t idx, uint64_t endIdx);
    __aicore__ inline INDICES_T GetIndexForI(int64_t indexOffset, int64_t idx);
    __aicore__ inline int64_t GetSelfIndex(uint64_t (&shapeI)[8], uint64_t endIdx);
    __aicore__ inline void SetShapeI(uint64_t (&shapeI)[8], int64_t yIdx);
    __aicore__ inline uint64_t GetIndicesEndIdx(uint64_t (&shapeI)[8]);
    __aicore__ inline void CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen);
    __aicore__ inline void NoSplitColProcess(int64_t colsAlign);
    __aicore__ inline void SplitColProcess(int64_t colsAlign);

private:
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<int8_t> yGm_;
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM_SIMD> inQueue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    const IndexSimdTilingData* tilingData_;

    int32_t needCoreNum_;
    int64_t yOffsetBase_ = 0;
    uint64_t indicesOffsetBase_ = 0;
    int64_t maxIndex_ = 0;
    int64_t curIndexSize_ = 0;
    int64_t currentCoreElements_;
    int32_t blockIdx_;
    bool isIndicesValid_ = true;
    uint64_t indicesGmOffset_ = 0;
    uint64_t indicesBufOffset_ = 0;
    ListTensorDesc inputList_;
};

template <typename INDICES_T>
__aicore__ inline void IndexSimd<INDICES_T>::Init(
    GM_ADDR output, GM_ADDR inputX, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
    const IndexSimdTilingData* tilingData, GM_ADDR value)
{
    tilingData_ = tilingData;
    blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    needCoreNum_ = static_cast<int32_t>(tilingData_->needCoreNum);
    indicesGmOffset_ = tilingData_->indexSize * sizeof(INDICES_T);
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(indices));

    xGm_.SetGlobalBuffer((__gm__ int8_t*)inputX);

    if (GetBlockIdx() < tilingData_->lastCoreElements) {
        yOffsetBase_ = (tilingData_->perCoreElements + 1) * GetBlockIdx();
        currentCoreElements_ = tilingData_->perCoreElements + 1;
    } else {
        yOffsetBase_ = tilingData_->perCoreElements * GetBlockIdx() + tilingData_->lastCoreElements;
        currentCoreElements_ = tilingData_->perCoreElements;
    }

    yGm_.SetGlobalBuffer((__gm__ int8_t*)output);
    pipe_.InitBuffer(inQueue_, BUFFER_NUM_SIMD, tilingData_->maxElement * tilingData_->inputDtypeSize);
    pipe_.InitBuffer(indexBuf_, tilingData_->indiceUbSize);
    maxIndex_ = tilingData_->indiceUbSize / tilingData_->indexedDimNum / sizeof(INDICES_T);
    indicesBufOffset_ = maxIndex_;
}

template <typename INDICES_T>
template <typename T>
__aicore__ inline void IndexSimd<INDICES_T>::CopyIn(
    LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst, uint32_t copyLen)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename INDICES_T>
__aicore__ inline void IndexSimd<INDICES_T>::CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen;
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    LocalTensor<int8_t> yLocal = inQueue_.DeQue<int8_t>();
    event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
    DataCopyPad(yGm_[offset], yLocal, dataCoptExtParams);
    inQueue_.FreeTensor(yLocal);
}

template <typename INDICES_T>
__aicore__ inline void IndexSimd<INDICES_T>::GetIndices(uint64_t idx, uint64_t endIdx)
{
    if (idx >= indicesOffsetBase_ + curIndexSize_ || idx < indicesOffsetBase_) {
        int64_t copyLen = 0;
        if (maxIndex_ >= (endIdx - idx + 1)) {
            copyLen = endIdx - idx + 1;
        } else {
            copyLen = maxIndex_;
        }
        indicesOffsetBase_ = idx;
        curIndexSize_ = copyLen;
        for (uint32_t i = 0; i < tilingData_->indexedDimNum; ++i) {
            indicesGm_.SetGlobalBuffer(inputList_.GetDataPtr<INDICES_T>(i));
            LocalTensor<INDICES_T> tmpLocal = indexBuf_.Get<INDICES_T>();
            CopyIn(tmpLocal[i * indicesBufOffset_], indicesGm_, indicesOffsetBase_, 1, copyLen);
        }
        event_t eventIdMTE2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
    }
}

template <typename INDICES_T>
__aicore__ inline INDICES_T IndexSimd<INDICES_T>::GetIndexForI(int64_t indexOffset, int64_t idx)
{
    LocalTensor<INDICES_T> indexLocal = indexBuf_.Get<INDICES_T>();
    return indexLocal[indexOffset * indicesBufOffset_].GetValue(idx - indicesOffsetBase_);
}

template <typename INDICES_T>
__aicore__ inline int64_t IndexSimd<INDICES_T>::GetSelfIndex(uint64_t (&shapeI)[8], uint64_t endIdx)
{
    uint64_t indexToInputDim[8];
    for (int32_t i = 0; i < tilingData_->mergeOutputShapeDim - 1; i++) {
        if (tilingData_->mergeOutToInput[i] == -1) {
            GetIndices(shapeI[i], endIdx);
            for (int32_t j = 0; j < tilingData_->indexedDimNum; j++) {
                INDICES_T index = GetIndexForI(j, shapeI[i]);
                index = index < 0 ? index + tilingData_->mergeInputShape[tilingData_->indicesToInput[j]] : index;
                indexToInputDim[tilingData_->indicesToInput[j]] = index;
                if (index < 0 || index > tilingData_->mergeInputShape[tilingData_->indicesToInput[j]]) {
                    isIndicesValid_ = false;
                }
            }
        } else {
            indexToInputDim[tilingData_->mergeOutToInput[i]] = shapeI[i];
        }
    }
    int64_t selfIndex = 0;
    for (int32_t i = 0; i < tilingData_->mergeInputShapeDim - 1; i++) {
        selfIndex = (selfIndex + indexToInputDim[i]) * tilingData_->mergeInputShape[i + 1];
    }
    return selfIndex;
}

template <typename INDICES_T>
__aicore__ inline void IndexSimd<INDICES_T>::SetShapeI(uint64_t (&shapeI)[8], int64_t yIdx)
{
    shapeI[0] = yIdx;
    for (int64_t idx = 1; idx < tilingData_->mergeOutputShapeDim - 1; idx++) {
        if (idx == 1) {
            shapeI[0] = yIdx / tilingData_->mergeOutputShape[1];
        }
        yIdx %= tilingData_->mergeOutputShape[idx];
        shapeI[idx] = yIdx / tilingData_->mergeOutputShape[idx + 1];
        if (idx == tilingData_->mergeOutputShapeDim - 2) {
            shapeI[idx] = yIdx;
        }
    }
}

template <typename INDICES_T>
__aicore__ inline uint64_t IndexSimd<INDICES_T>::GetIndicesEndIdx(uint64_t (&shapeI)[8])
{
    if (tilingData_->isZeroOneZero && shapeI[0] > 0) {
        return tilingData_->indexSize;
    } else {
        for (int32_t i = 0; i < tilingData_->mergeOutputShapeDim - 1; i++) {
            if (tilingData_->mergeOutToInput[i] == -1) {
                return shapeI[i];
            }
        }
    }
    return tilingData_->indexSize;
}

template <typename INDICES_T>
__aicore__ inline void IndexSimd<INDICES_T>::NoSplitColProcess(int64_t colsAlign)
{
    int64_t onceMaxRows = tilingData_->maxElement * tilingData_->inputDtypeSize / colsAlign;
    int64_t loopNum = (currentCoreElements_ + onceMaxRows - 1) / onceMaxRows;

    int64_t yStart = yOffsetBase_;
    int64_t yEnd = yOffsetBase_ + currentCoreElements_;
    uint64_t shapeI[8]{};
    SetShapeI(shapeI, yEnd);
    uint64_t indiceEndIdx = GetIndicesEndIdx(shapeI);
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t rows = (i == loopNum - 1) ? (currentCoreElements_ - i * onceMaxRows) : onceMaxRows;
        uint64_t cols = tilingData_->mergeOutputShape[tilingData_->mergeOutputShapeDim - 1];
        LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
        int64_t preIdx0 = -1;
        int64_t preIdx1 = -1;
        int64_t preIdx2 = -1;
        int32_t backOffset0 = 0;
        int32_t backOffset1 = 0;
        int32_t backOffset2 = 0;
        for (int64_t j = 0; j < rows; j++) {
            int64_t yIdx = yStart + i * onceMaxRows + j;
            for (int32_t id = 0; id < 8; id++) {
                shapeI[id] = 0;
            }
            SetShapeI(shapeI, yIdx);

            int64_t xIndex = GetSelfIndex(shapeI, indiceEndIdx);
            int64_t offset = xIndex * tilingData_->inputDtypeSize;
            if (likely(isIndicesValid_)) {
                if (unlikely(xIndex == preIdx0 || xIndex == preIdx1 || xIndex == preIdx2)) {
                    event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
                    int32_t backStep = (xIndex == preIdx0) ? backOffset0 :
                                       (xIndex == preIdx1) ? backOffset1 :
                                                             backOffset2;
                    Copy(
                        xLocal[j * colsAlign], xLocal[backStep],
                        tilingData_->mergeOutputShape[tilingData_->mergeOutputShapeDim - 1] *
                            tilingData_->inputDtypeSize);
                } else {
                    CopyIn(
                        xLocal[j * colsAlign], xGm_, offset, 1,
                        tilingData_->mergeOutputShape[tilingData_->mergeOutputShapeDim - 1] *
                            tilingData_->inputDtypeSize);
                    preIdx2 = preIdx1;
                    preIdx1 = preIdx0;
                    preIdx0 = xIndex;
                    backOffset2 = backOffset1;
                    backOffset1 = backOffset0;
                    backOffset0 = j * colsAlign;
                }
            } else {
                Duplicate<int8_t>(
                    xLocal[j * colsAlign], 0,
                    tilingData_->mergeOutputShape[tilingData_->mergeOutputShapeDim - 1] * tilingData_->inputDtypeSize);
            }
        }
        inQueue_.EnQue<int8_t>(xLocal);
        int64_t yOffset = (yStart + i * onceMaxRows) *
                          tilingData_->mergeOutputShape[tilingData_->mergeOutputShapeDim - 1] *
                          tilingData_->inputDtypeSize;
        CopyOut(yOffset, rows, cols * tilingData_->inputDtypeSize);
    }
}

template <typename INDICES_T>
__aicore__ inline void IndexSimd<INDICES_T>::SplitColProcess(int64_t colsAlign)
{
    int64_t yStart = yOffsetBase_;
    int64_t yEnd = yOffsetBase_ + currentCoreElements_;
    uint64_t shapeI[8]{};
    SetShapeI(shapeI, yEnd);
    uint64_t indiceEndIdx = GetIndicesEndIdx(shapeI);
    int64_t loopSize =
        (tilingData_->mergeOutputShape[tilingData_->mergeOutputShapeDim - 1] + tilingData_->maxElement - 1) /
        tilingData_->maxElement;
    for (int64_t i = 0; i < currentCoreElements_; i++) {
        int64_t yIdx = yStart + i;
        for (int32_t id = 0; id < 8; id++) {
            shapeI[id] = 0;
        }
        SetShapeI(shapeI, yIdx);

        int64_t xIndex = GetSelfIndex(shapeI, indiceEndIdx);
        int64_t yIndex = (yStart + i) * tilingData_->mergeOutputShape[tilingData_->mergeOutputShapeDim - 1];
        for (int64_t j = 0; j < loopSize; j++) {
            int64_t cols = (j == loopSize - 1) ? (tilingData_->mergeOutputShape[tilingData_->mergeOutputShapeDim - 1] -
                                                  j * tilingData_->maxElement) :
                                                 tilingData_->maxElement;
            LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
            int64_t offset = (xIndex + j * tilingData_->maxElement) * tilingData_->inputDtypeSize;
            if (likely(isIndicesValid_)) {
                CopyIn(xLocal[0], xGm_, offset, 1, cols * tilingData_->inputDtypeSize);
            } else {
                event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
                WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
                Duplicate<int8_t>(xLocal[0], 0, cols * tilingData_->inputDtypeSize);
            }
            inQueue_.EnQue<int8_t>(xLocal);
            int64_t yOffset = (yIndex + j * tilingData_->maxElement) * tilingData_->inputDtypeSize;
            CopyOut(yOffset, 1, cols * tilingData_->inputDtypeSize);
        }
    }
}

template <typename INDICES_T>
__aicore__ inline void IndexSimd<INDICES_T>::Process()
{
    if (blockIdx_ >= tilingData_->needCoreNum) {
        return;
    }
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize());
    int64_t colsAlign =
        (tilingData_->mergeOutputShape[tilingData_->mergeOutputShapeDim - 1] * tilingData_->inputDtypeSize +
         ubBlockSize - 1) /
        ubBlockSize * ubBlockSize;
    if (colsAlign <= tilingData_->maxElement * tilingData_->inputDtypeSize) {
        NoSplitColProcess(colsAlign);
    } else {
        SplitColProcess(colsAlign);
    }
}
} // namespace Index
#endif // INDEX_SIMD_H
