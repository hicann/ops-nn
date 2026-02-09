/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_ND_simd.h
 * \brief
 */
#ifndef GATHER_ND_SIMD_H
#define GATHER_ND_SIMD_H

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace GatherNd {
using namespace AscendC;

constexpr int32_t BUFFER_NUM_SIMD = 2;

template <typename INDICES_T>
class GatherNdSimd {
 public:
  __aicore__ inline GatherNdSimd(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, const GatherNdTilingData* tilingData, TPipe* pipe);
  __aicore__ inline void Process();
  template <typename T>
  __aicore__ inline void CopyIn(LocalTensor<T> xLocal,
    GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst, uint32_t copyLen);
  __aicore__ inline INDICES_T GetIndex(int64_t idx, int64_t endIdx);
  __aicore__ inline void CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen);
  __aicore__ inline void NoSplitColProcess(int64_t colsAlign);
  __aicore__ inline void SplitColProcess(int64_t colsAlign);
 private:
  GlobalTensor<int8_t> xGm_;
  GlobalTensor<INDICES_T> indicesGm_;
  GlobalTensor<int8_t> yGm_;
  TPipe *pipe_;
  TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM_SIMD> inQueue_;
  TBuf<QuePosition::VECCALC> indexBuf_;
  const GatherNdTilingData* tilingData_;

  int32_t needCoreNum_;
  int64_t gatherSize_;
  int64_t indicesNum_;
  int64_t yOffsetBase_ = 0;
  int64_t indicesOffsetBase_ = -1;
  int64_t maxIndex_ = 0;
  bool negativeIndexSupport_;

  int64_t currentCoreElements_;
  int32_t blockIdx_;
  int64_t xShape[8];
};

template <typename INDICES_T>
__aicore__ inline void GatherNdSimd<INDICES_T>::Init(GM_ADDR x, GM_ADDR indices, 
                                                                GM_ADDR y, const GatherNdTilingData* tilingData, TPipe* pipe) {
  pipe_ = pipe;
  tilingData_ = tilingData;
  blockIdx_ = static_cast<int32_t>(GetBlockIdx());
  needCoreNum_ = static_cast<int32_t>(tilingData_->needCoreNum);
  gatherSize_ = static_cast<int64_t>(tilingData_->gatherSize);

  indicesNum_ = static_cast<int64_t>(tilingData_->indicesNum);
  negativeIndexSupport_ = static_cast<bool>(tilingData_->negativeIndexSupport);
  for (uint32_t i = 0; i < 8; i++) {
    xShape[i] = tilingData_->xShape[i];
  }

  xGm_.SetGlobalBuffer((__gm__ int8_t*)x);

  if (GetBlockIdx() < tilingData_->tailBlockFactor) {
     yOffsetBase_ = (tilingData_->blockFactor + 1) * GetBlockIdx();
     currentCoreElements_ = tilingData_->blockFactor + 1;
  } else {
     yOffsetBase_ = tilingData_->blockFactor * GetBlockIdx() + tilingData_->tailBlockFactor;
     currentCoreElements_ = tilingData_->blockFactor;
  }

  indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
  yGm_.SetGlobalBuffer((__gm__ int8_t*)y);
  pipe_->InitBuffer(inQueue_, BUFFER_NUM_SIMD, tilingData_->maxElement * tilingData_->dtypeSize);
  pipe_->InitBuffer(indexBuf_, tilingData_->indicesUbSize);
  maxIndex_ = tilingData_->indicesUbSize / sizeof(INDICES_T);
}

template <typename INDICES_T>
template <typename T>
__aicore__ inline void GatherNdSimd<INDICES_T>::CopyIn(LocalTensor<T> xLocal,
  GlobalTensor<T> xGm, int64_t offset,
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
__aicore__ inline void GatherNdSimd<INDICES_T>::CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen)
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
__aicore__ inline INDICES_T GatherNdSimd<INDICES_T>::GetIndex(int64_t idx, int64_t endIdx)
{
   if (indicesOffsetBase_ < 0 || idx >= indicesOffsetBase_ + maxIndex_) {
      int64_t copyLen = idx + maxIndex_ > endIdx ? endIdx - idx : maxIndex_; // endIdx最后一个有效
      LocalTensor<INDICES_T> tmpLocal = indexBuf_.Get<INDICES_T>();
      CopyIn(tmpLocal, indicesGm_, idx, 1, copyLen);
      indicesOffsetBase_ = idx;
      event_t eventIdMTE2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
      SetFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
      WaitFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
   }
   LocalTensor<INDICES_T> indexLocal = indexBuf_.Get<INDICES_T>();
   return indexLocal.GetValue(idx - indicesOffsetBase_);
}
template <typename INDICES_T>
__aicore__ inline void GatherNdSimd<INDICES_T>::NoSplitColProcess(int64_t colsAlign) {
  int64_t onceMaxRows = tilingData_->maxElement * tilingData_->dtypeSize / colsAlign;
  int64_t loopNum = (currentCoreElements_ + onceMaxRows - 1) / onceMaxRows;
  // indice -> g
  int64_t yStart = yOffsetBase_;
  int64_t indiceEndIdx = (yOffsetBase_ + currentCoreElements_) * tilingData_->rank;
  for (int64_t i = 0; i < loopNum; i++) {
    int64_t rows = (i == loopNum - 1) ? (currentCoreElements_ - i * onceMaxRows) : onceMaxRows;
    int64_t cols = tilingData_->gatherSize;
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
        bool idxOutOfBound = false;
        int64_t indicesIdx = (yStart + i * onceMaxRows + j) * tilingData_->rank;
        int64_t srcOffset = 0;
        for (int k = 0; k < tilingData_->rank; k++) {
            INDICES_T index = GetIndex(indicesIdx + k, indiceEndIdx);
            if (negativeIndexSupport_ && index < 0) {
                index += xShape[k];
            }
            if (index >= xShape[k] || index < 0) { 
                idxOutOfBound = true;
                break;
            }
            srcOffset = srcOffset * xShape[k] + index;
        }
        int64_t xIndex = srcOffset * gatherSize_;
        srcOffset = xIndex * tilingData_->dtypeSize;
        if (!idxOutOfBound) {
          if (unlikely(xIndex == preIdx0 || xIndex == preIdx1 || xIndex == preIdx2)) {
            event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
            int32_t backStep = (xIndex == preIdx0) ? backOffset0 : (xIndex == preIdx1) ? backOffset1 : backOffset2;
            Copy(xLocal[j * colsAlign], xLocal[backStep], tilingData_->gatherSize * tilingData_->dtypeSize);
          } else {
            CopyIn(xLocal[j * colsAlign], xGm_, srcOffset, 1, tilingData_->gatherSize * tilingData_->dtypeSize);
            preIdx2 = preIdx1;
            preIdx1 = preIdx0;
            preIdx0 = xIndex;
            backOffset2 = backOffset1;
            backOffset1 = backOffset0;
            backOffset0 = j * colsAlign;
          }
        } else {
          Duplicate<int8_t>(xLocal[j * colsAlign], 0, tilingData_->gatherSize * tilingData_->dtypeSize);
        }
    }
    inQueue_.EnQue<int8_t>(xLocal);
    int64_t yOffset = (yStart + i * onceMaxRows) * tilingData_->gatherSize * tilingData_->dtypeSize;
    CopyOut(yOffset, rows, cols * tilingData_->dtypeSize);
  }
}

template <typename INDICES_T>
__aicore__ inline void GatherNdSimd<INDICES_T>::SplitColProcess(int64_t colsAlign) {

  // indice -> b, g
  // output -> b, o, g, i
  int64_t yStart = yOffsetBase_;
  int64_t indiceEndIdx = (yOffsetBase_ + currentCoreElements_) * tilingData_->rank;
  int64_t loopSize = (tilingData_->gatherSize + tilingData_->maxElement - 1) / tilingData_->maxElement;
  for (int64_t i = 0; i < currentCoreElements_; i++) {
    bool idxOutOfBound = false;
    int64_t indicesIdx = (yStart + i) * tilingData_->rank;
    int64_t srcOffset = 0;
    for (int k = 0; k < tilingData_->rank; k++) {
        INDICES_T index = GetIndex(indicesIdx + k, indiceEndIdx);
        if (negativeIndexSupport_ && index < 0) {
            index += xShape[k];
        }
        if (index >= xShape[k] || index < 0) { 
            idxOutOfBound = true;
            break;
        }
        srcOffset = srcOffset * xShape[k] + index;
    }
    int64_t xIndex = srcOffset * gatherSize_;
    int64_t offset = xIndex * tilingData_->dtypeSize;
    int64_t yIndex = (yStart + i) * tilingData_->gatherSize;
    for (int64_t j = 0; j < loopSize; j++) {
      int64_t cols = (j == loopSize - 1) ? (tilingData_->gatherSize - j * tilingData_->maxElement) : tilingData_->maxElement;
      LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
      int64_t offset = (xIndex + j * tilingData_->maxElement) * tilingData_->dtypeSize;
      if (!idxOutOfBound) {
        CopyIn(xLocal[0], xGm_, offset, 1, cols * tilingData_->dtypeSize);
      } else {
        event_t eventIdMTE3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3toV);
        Duplicate<int8_t>(xLocal[0], 0, cols * tilingData_->dtypeSize);
      }
      inQueue_.EnQue<int8_t>(xLocal);
      int64_t yOffset = (yIndex + j * tilingData_->maxElement) * tilingData_->dtypeSize;
      CopyOut(yOffset, 1, cols * tilingData_->dtypeSize);
    }
  }
}

template <typename INDICES_T>
__aicore__ inline void GatherNdSimd<INDICES_T>::Process() {
  if (blockIdx_ >= tilingData_->needCoreNum) {
     return;
  }
  int64_t ubBlockSize = static_cast<int64_t>(platform::GetUbBlockSize());
  int64_t colsAlign = (tilingData_->gatherSize * tilingData_->dtypeSize + ubBlockSize - 1) / ubBlockSize * ubBlockSize ;
  if (colsAlign <= tilingData_->maxElement * tilingData_->dtypeSize) {
     NoSplitColProcess(colsAlign);
  } else {
     SplitColProcess(colsAlign);
  }
}
}  // namespace GatherNd
#endif  // ASCENDC_GATHER_ND_GATHER_ND_SIMD_H_ 