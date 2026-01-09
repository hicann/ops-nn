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
 * \file gather_v2.h
 * \brief
 */
#ifndef GATHER_V2_H
#define GATHER_V2_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
#else
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 2048;
#endif

namespace gatherv2 {
using namespace AscendC;

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
class Gatherv2 {
 public:
  __aicore__ inline Gatherv2(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y, const GatherV2TilingData* tilingData);
  __aicore__ inline void Process();

 private:
  template <const bool NIS>
  static __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void GatherSimt(const INDEX_SIZE_T yIndexBase,
  INDEX_SIZE_T currentCoreElements, INDEX_SIZE_T m0, INDEX_SIZE_T shift0, INDEX_SIZE_T m1, INDEX_SIZE_T shift1, INDEX_SIZE_T m2,
  INDEX_SIZE_T shift2, INDEX_SIZE_T gatherSize, INDEX_SIZE_T outerSize, INDEX_SIZE_T innerSize,
  INDEX_SIZE_T gatherDimSize, __gm__ X_T* x, __gm__ INDICES_T* indices, __gm__ volatile X_T* y);

 private:
  GlobalTensor<X_T> xGm_;
  GlobalTensor<INDICES_T> indicesGm_;
  GlobalTensor<X_T> yGm_;

  const GatherV2TilingData* tilingData_ = nullptr;
};

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
template <const bool NIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void Gatherv2<X_T, INDICES_T, INDEX_SIZE_T>::GatherSimt(const INDEX_SIZE_T yIndexBase,
  INDEX_SIZE_T currentCoreElements, INDEX_SIZE_T m0, INDEX_SIZE_T shift0, INDEX_SIZE_T m1, INDEX_SIZE_T shift1, INDEX_SIZE_T m2,
  INDEX_SIZE_T shift2, INDEX_SIZE_T gatherSize, INDEX_SIZE_T outerSize, INDEX_SIZE_T innerSize,
  INDEX_SIZE_T gatherDimSize, __gm__ X_T* x, __gm__ INDICES_T* indices, __gm__ volatile X_T* y) {
  for (INDEX_SIZE_T index = static_cast<INDEX_SIZE_T>(Simt::GetThreadIdx()); index < currentCoreElements;
      index += static_cast<INDEX_SIZE_T>(Simt::GetThreadNum())) {
    INDEX_SIZE_T yIndex = yIndexBase + index;

    INDEX_SIZE_T tmp = Simt::UintDiv(yIndex, m0, shift0);
    INDEX_SIZE_T tmp1 = Simt::UintDiv(tmp, m1, shift1);
    INDEX_SIZE_T gatherI = tmp - tmp1 * gatherSize;
    INDEX_SIZE_T batchI = Simt::UintDiv(tmp1, m2, shift2);

    INDEX_SIZE_T outerI = tmp1 - batchI * outerSize;
    INDEX_SIZE_T innerI = yIndex  - tmp * innerSize;
    INDEX_SIZE_T indicesIndex = batchI * gatherSize + gatherI;

    INDICES_T indicesValue = indices[indicesIndex];
    if constexpr(NIS) {
      if (unlikely(indicesValue < 0)) {
        indicesValue += gatherDimSize;
      }
    }
    INDEX_SIZE_T indicesValueI = static_cast<INDEX_SIZE_T>(indicesValue);
    INDEX_SIZE_T xIndex = (tmp1 * gatherDimSize + indicesValueI) * innerSize + innerI;    
    bool idxOutOfBound = indicesValue < 0 || indicesValueI >= gatherDimSize;
    y[yIndex] = idxOutOfBound ? 0 : x[xIndex];
  }
}

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
__aicore__ inline void Gatherv2<X_T, INDICES_T, INDEX_SIZE_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis,
                                                                    GM_ADDR y, const GatherV2TilingData* tilingData) {
  tilingData_ = tilingData;
  xGm_.SetGlobalBuffer((__gm__ X_T*)x);
  indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
  yGm_.SetGlobalBuffer((__gm__ X_T*)y);
}

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
__aicore__ inline void Gatherv2<X_T, INDICES_T, INDEX_SIZE_T>::Process() {
  int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
  int32_t needCoreNum = static_cast<int32_t>(tilingData_->needCoreNum);
  uint32_t threadNum = static_cast<uint32_t>(tilingData_->threadNum);

  INDEX_SIZE_T outerSize = static_cast<INDEX_SIZE_T>(tilingData_->outerSize);
  INDEX_SIZE_T gatherDimSize = static_cast<INDEX_SIZE_T>(tilingData_->gatherDimSize);
  INDEX_SIZE_T gatherSize = static_cast<INDEX_SIZE_T>(tilingData_->gatherSize);
  INDEX_SIZE_T innerSize = static_cast<INDEX_SIZE_T>(tilingData_->innerSize);

  bool negativeIndexSupport = static_cast<bool>(tilingData_->negativeIndexSupport);
  INDEX_SIZE_T currentCoreElements = static_cast<INDEX_SIZE_T>(tilingData_->perCoreElements);
  if (blockIdx == tilingData_->needCoreNum - 1) {
    currentCoreElements = static_cast<INDEX_SIZE_T>(tilingData_->lastCoreElements);
  }
  INDEX_SIZE_T m0 {0};
  INDEX_SIZE_T m1 {0};
  INDEX_SIZE_T m2 {0};

  INDEX_SIZE_T shift0 {0};
  INDEX_SIZE_T shift1 {0};
  INDEX_SIZE_T shift2 {0};

  // fast division
  GetUintDivMagicAndShift(m0, shift0, innerSize);
  GetUintDivMagicAndShift(m1, shift1, gatherSize);
  GetUintDivMagicAndShift(m2, shift2, outerSize);

  if (blockIdx < needCoreNum) {
    INDEX_SIZE_T yIndexBase = blockIdx * tilingData_->perCoreElements;
    if (unlikely(negativeIndexSupport)) {
      AscendC::Simt::VF_CALL<GatherSimt<true>>(Simt::Dim3(threadNum), yIndexBase, currentCoreElements, m0, shift0, m1, shift1, m2, shift2,
                  gatherSize, outerSize, innerSize, gatherDimSize, (__gm__ X_T*) (xGm_.GetPhyAddr()),
                  (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()), (__gm__ volatile X_T*) (yGm_.GetPhyAddr()));
    } else {
      AscendC::Simt::VF_CALL<GatherSimt<false>>(Simt::Dim3(threadNum), yIndexBase, currentCoreElements, m0, shift0, m1, shift1, m2, shift2,
                  gatherSize, outerSize, innerSize, gatherDimSize, (__gm__ X_T*) (xGm_.GetPhyAddr()),
                  (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()), (__gm__ volatile X_T*) (yGm_.GetPhyAddr()));
    }
  }
}
}  // namespace gatherv2
#endif  // GATHER_V2_H
