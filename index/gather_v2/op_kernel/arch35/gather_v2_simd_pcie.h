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
 * \file gather_v2_simd_pcie.h
 * \brief
 */
#ifndef GATHER_V2_SIMD_PCIE_H
#define GATHER_V2_SIMD_PCIE_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/platform_util.h"

namespace gatherv2 {
using namespace AscendC;

constexpr int32_t BUFFER_NUM_SIMD_PCIE = 2;
constexpr int64_t INDICES_SIZE = 8192;

template <typename INDICES_T>
class Gatherv2SimdPcie {
public:
    __aicore__ inline Gatherv2SimdPcie(TPipe* pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y,
                                const GatherV2TilingData* tilingData);
    __aicore__ inline void Process();
    template <typename T>
    __aicore__ inline void CopyIn(LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst,
                                  uint32_t copyLen);
    __aicore__ inline INDICES_T GetIndex(int64_t idx, int64_t endIdx);
    __aicore__ inline void CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen);
    __aicore__ inline void NoSplitColProcess(int64_t colsAlign);
    __aicore__ inline void SplitColProcess(int64_t colsAlign);

private:
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<int8_t> yGm_;
    TPipe* pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM_SIMD_PCIE> inQueue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    const GatherV2TilingData* tilingData_;

    int32_t needCoreNum_ = 1;

    int64_t batchSize_ = 1;
    int64_t outerSize_ = 1;
    int64_t gatherDimSize_ = 1;
    int64_t gatherSize_ = 1;
    int64_t innerSize_ = 1;

    int64_t yOffsetBase_ = 0;
    int64_t indicesOffsetBase_ = 0;
    int64_t maxIndex_ = 0;
    int64_t curIndexSize_ = 0;
    bool negativeIndexSupport_ = false;

    int64_t currentCoreElements_ = 0;
    int32_t blockIdx_ = 0;
    int64_t maxElement_ = 1;
};


template <typename INDICES_T>
__aicore__ inline void Gatherv2SimdPcie<INDICES_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y,
                                                     const GatherV2TilingData* tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = static_cast<int32_t>(GetBlockIdx());

    int64_t aivNum = GetBlockNum();
    int64_t dcacheSize = 131072;
    uint32_t ubSize = GetUBSizeInBytes() - dcacheSize;

    batchSize_ = static_cast<int64_t>(tilingData_->batchSize);
    outerSize_ = static_cast<int64_t>(tilingData_->outerSize);
    gatherDimSize_ = static_cast<int64_t>(tilingData_->gatherDimSize);
    gatherSize_ = static_cast<int64_t>(tilingData_->gatherSize);
    innerSize_ = static_cast<int64_t>(tilingData_->innerSize);

    negativeIndexSupport_ = static_cast<bool>(tilingData_->negativeIndexSupport);

    int64_t blockFactor = batchSize_ * outerSize_ * gatherSize_ / aivNum;
    int64_t tailBlockFactor = batchSize_ * outerSize_ * gatherSize_ - blockFactor * aivNum;
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize());
    int64_t ubAviable = (ubSize - INDICES_SIZE) / ubBlockSize * ubBlockSize / tilingData_->dtypeSize / BUFFER_NUM_SIMD_PCIE;
    needCoreNum_ = tailBlockFactor;
    if (blockFactor > 0) {
        needCoreNum_ = aivNum;
    }
    maxElement_ = ubAviable;

    xGm_.SetGlobalBuffer((__gm__ int8_t*)x);

    if (GetBlockIdx() < tailBlockFactor) {
        yOffsetBase_ = (blockFactor + 1) * GetBlockIdx();
        currentCoreElements_ = blockFactor + 1;
    } else {
        yOffsetBase_ = blockFactor * GetBlockIdx() + tailBlockFactor;
        currentCoreElements_ = blockFactor;
    }

    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    yGm_.SetGlobalBuffer((__gm__ int8_t*)y);
    pipe_->InitBuffer(inQueue_, BUFFER_NUM_SIMD_PCIE, maxElement_ * tilingData_->dtypeSize);
    pipe_->InitBuffer(indexBuf_, INDICES_SIZE);
    maxIndex_ = INDICES_SIZE / sizeof(INDICES_T);
}

template <typename INDICES_T>
template <typename T>
__aicore__ inline void Gatherv2SimdPcie<INDICES_T>::CopyIn(LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset,
                                                       uint32_t nBurst, uint32_t copyLen)
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
__aicore__ inline void Gatherv2SimdPcie<INDICES_T>::CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen)
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
__aicore__ inline INDICES_T Gatherv2SimdPcie<INDICES_T>::GetIndex(int64_t idx, int64_t endIdx)
{
    if (idx >= indicesOffsetBase_ + curIndexSize_ || idx < indicesOffsetBase_) {
        int64_t startBIdx = idx / gatherSize_;
        int64_t startGIdx = idx % gatherSize_;
        int64_t endBIdx = endIdx / gatherSize_;
        int64_t endGIdx = endIdx % gatherSize_;
        int64_t copyLen = 0;
        if (maxIndex_ >= (endBIdx - startBIdx + 1) * gatherSize_) {
            copyLen = (endBIdx - startBIdx + 1) * gatherSize_;
            indicesOffsetBase_ = startBIdx * gatherSize_;
        } else if (maxIndex_ >= gatherSize_) {
            int64_t maxBSize = maxIndex_ / gatherSize_;
            copyLen = startBIdx + maxBSize > endBIdx ? (endBIdx - startBIdx + 1) * gatherSize_ : maxBSize * gatherSize_;
            indicesOffsetBase_ = startBIdx * gatherSize_;
        } else {
            copyLen = startGIdx + maxIndex_ >= (endBIdx + 1) * gatherSize_ ? (endBIdx + 1) * gatherSize_ - idx :
                                                                             maxIndex_;
            indicesOffsetBase_ = idx;
        }
        curIndexSize_ = copyLen;
        LocalTensor<INDICES_T> tmpLocal = indexBuf_.Get<INDICES_T>();
        CopyIn(tmpLocal, indicesGm_, indicesOffsetBase_, 1, copyLen);

        event_t eventIdMTE2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
    }
    LocalTensor<INDICES_T> indexLocal = indexBuf_.Get<INDICES_T>();
    return indexLocal.GetValue(idx - indicesOffsetBase_);
}
template <typename INDICES_T>
__aicore__ inline void Gatherv2SimdPcie<INDICES_T>::NoSplitColProcess(int64_t colsAlign)
{
    int64_t onceMaxRows = maxElement_ * tilingData_->dtypeSize / colsAlign;
    int64_t loopNum = (currentCoreElements_ + onceMaxRows - 1) / onceMaxRows;

    // indice -> b, g
    // output -> b, o, g, i
    int64_t yStart = yOffsetBase_;
    int64_t yEnd = yOffsetBase_ + currentCoreElements_ - 1;
    int64_t endBatch = yEnd / (outerSize_ * gatherSize_);
    int64_t endGIdx = yEnd % gatherSize_;
    int64_t indiceEndIdx = endBatch * gatherSize_ + endGIdx;
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t rows = (i == loopNum - 1) ? (currentCoreElements_ - i * onceMaxRows) : onceMaxRows;
        int64_t cols = tilingData_->innerSize;
        LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
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
            int64_t batchI = yIdx / (outerSize_ * gatherSize_);
            int64_t outerI = yIdx % (outerSize_ * gatherSize_) / gatherSize_;
            int64_t gatherI = yIdx % gatherSize_;
            int64_t indiceOffset = batchI * gatherSize_ + gatherI;
            INDICES_T index = GetIndex(indiceOffset, indiceEndIdx);
            if (negativeIndexSupport_) {
                index = index < 0 ? index + gatherDimSize_ : index;
            }
            int64_t xIndex = ((batchI * outerSize_ + outerI) * gatherDimSize_ + index) * innerSize_;
            int64_t offset = xIndex * tilingData_->dtypeSize;
            if (likely(index >= 0 && index < gatherDimSize_)) {
                if (unlikely(xIndex == preIdx0 || xIndex == preIdx1 || xIndex == preIdx2)) {
                    event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
                    int32_t backStep = (xIndex == preIdx0) ? backOffset0 :
                                       (xIndex == preIdx1) ? backOffset1 :
                                                             backOffset2;
                    Copy(xLocal[j * colsAlign], xLocal[backStep], tilingData_->innerSize * tilingData_->dtypeSize);
                } else {
                    CopyIn(xLocal[j * colsAlign], xGm_, offset, 1, tilingData_->innerSize * tilingData_->dtypeSize);
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
        inQueue_.EnQue<int8_t>(xLocal);
        int64_t yOffset = (yStart + i * onceMaxRows) * tilingData_->innerSize * tilingData_->dtypeSize;
        CopyOut(yOffset, rows, cols * tilingData_->dtypeSize);
    }
}

template <typename INDICES_T>
__aicore__ inline void Gatherv2SimdPcie<INDICES_T>::SplitColProcess(int64_t colsAlign)
{
    // indice -> b, g
    // output -> b, o, g, i
    int64_t yStart = yOffsetBase_;
    int64_t yEnd = yOffsetBase_ + currentCoreElements_ - 1;
    int64_t endBatch = yEnd / (outerSize_ * gatherSize_);
    int64_t endGIdx = yEnd % gatherSize_;
    int64_t indiceEndIdx = endBatch * gatherSize_ + endGIdx;
    int64_t loopSize = (tilingData_->innerSize + maxElement_ - 1) / maxElement_;
    for (int64_t i = 0; i < currentCoreElements_; i++) {
        int64_t yIdx = yStart + i;
        int64_t batchI = yIdx / (outerSize_ * gatherSize_);
        int64_t outerI = yIdx % (outerSize_ * gatherSize_) / gatherSize_;
        int64_t gatherI = yIdx % gatherSize_;
        int64_t indiceOffset = batchI * gatherSize_ + gatherI;
        INDICES_T index = GetIndex(indiceOffset, indiceEndIdx);
        if (negativeIndexSupport_) {
            index = index < 0 ? index + gatherDimSize_ : index;
        }
        int64_t xIndex = ((batchI * outerSize_ + outerI) * gatherDimSize_ + index) * innerSize_;
        int64_t yIndex = (yStart + i) * tilingData_->innerSize;
        for (int64_t j = 0; j < loopSize; j++) {
            int64_t cols = (j == loopSize - 1) ? (tilingData_->innerSize - j * maxElement_) :
                                                 maxElement_;
            LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
            int64_t offset = (xIndex + j * maxElement_) * tilingData_->dtypeSize;
            if (likely(index >= 0 && index < gatherDimSize_)) {
                CopyIn(xLocal[0], xGm_, offset, 1, cols * tilingData_->dtypeSize);
            } else {
                event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
                WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
                Duplicate<int8_t>(xLocal[0], 0, cols * tilingData_->dtypeSize);
            }
            inQueue_.EnQue<int8_t>(xLocal);
            int64_t yOffset = (yIndex + j * maxElement_) * tilingData_->dtypeSize;
            CopyOut(yOffset, 1, cols * tilingData_->dtypeSize);
        }
    }
}

template <typename INDICES_T>
__aicore__ inline void Gatherv2SimdPcie<INDICES_T>::Process()
{
    if (blockIdx_ >= needCoreNum_) {
        return;
    }
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize());
    int64_t colsAlign = (tilingData_->innerSize * tilingData_->dtypeSize + ubBlockSize - 1) / ubBlockSize * ubBlockSize;
    if (colsAlign <= maxElement_ * tilingData_->dtypeSize) {
        NoSplitColProcess(colsAlign);
    } else {
        SplitColProcess(colsAlign);
    }
}
} // namespace gatherv2
#endif // GATHER_V2_SIMD_PCIE_H
