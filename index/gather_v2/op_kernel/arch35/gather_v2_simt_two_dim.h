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
 * \file gather_v2_simt_two_dim.h
 * \brief
 */
#ifndef GATHER_V2_SIMT_TWO_DIM_H
#define GATHER_V2_SIMT_TWO_DIM_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#if ASC_DEVKIT_MAJOR >=9
#include "basic_api/kernel_vec_intf.h"
#endif

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND_TWO_DIM = 512;
#else
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND_TWO_DIM = 2048;
#endif

namespace gatherv2 {
using namespace AscendC;

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
class Gatherv2SimtTwoDim {
 public:
  __aicore__ inline Gatherv2SimtTwoDim(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y, const GatherV2TilingDataSimtTwoDim* tilingData);
  __aicore__ inline void Process();

 private:
  template <const bool NIS>
  static __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_TWO_DIM) inline void GatherSimt(const INDEX_SIZE_T yIndexBase,
  INDEX_SIZE_T currentCoreElements, INDEX_SIZE_T m0, INDEX_SIZE_T shift0, INDEX_SIZE_T innerSize,
  INDEX_SIZE_T gatherDimSize, __gm__ X_T* x, __gm__ INDICES_T* indices, __gm__ volatile X_T* y);

 private:
  GlobalTensor<X_T> xGm_;
  GlobalTensor<INDICES_T> indicesGm_;
  GlobalTensor<X_T> yGm_;

  const GatherV2TilingDataSimtTwoDim* tilingData_ = nullptr;
};

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
template <const bool NIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_TWO_DIM) inline void Gatherv2SimtTwoDim<X_T, INDICES_T, INDEX_SIZE_T>::GatherSimt(const INDEX_SIZE_T yIndexBase,
  INDEX_SIZE_T currentCoreElements, INDEX_SIZE_T m0, INDEX_SIZE_T shift0, INDEX_SIZE_T innerSize,
  INDEX_SIZE_T gatherDimSize, __gm__ X_T* x, __gm__ INDICES_T* indices, __gm__ volatile X_T* y) {
  for (INDEX_SIZE_T index = static_cast<INDEX_SIZE_T>(Simt::GetThreadIdx()); index < currentCoreElements;
      index += static_cast<INDEX_SIZE_T>(Simt::GetThreadNum())) {
    INDEX_SIZE_T yIndex = yIndexBase + index;
    INDEX_SIZE_T gatherI = Simt::UintDiv(yIndex, m0, shift0);
    INDEX_SIZE_T innerI = yIndex  - gatherI * innerSize;

    INDICES_T indicesValue = indices[gatherI];
    if constexpr(NIS) {
      if (unlikely(indicesValue < 0)) {
        indicesValue += gatherDimSize;
      }
    }
    INDEX_SIZE_T indicesValueI = static_cast<INDEX_SIZE_T>(indicesValue);
    INDEX_SIZE_T xIndex = indicesValueI * innerSize + innerI;    
    bool idxOutOfBound = indicesValue < 0 || indicesValueI >= gatherDimSize;
    y[yIndex] = idxOutOfBound ? 0 : x[xIndex];
  }
}

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
__aicore__ inline void Gatherv2SimtTwoDim<X_T, INDICES_T, INDEX_SIZE_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis,
                                                                    GM_ADDR y, const GatherV2TilingDataSimtTwoDim* tilingData) {
  tilingData_ = tilingData;
  xGm_.SetGlobalBuffer((__gm__ X_T*)x);
  indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
  yGm_.SetGlobalBuffer((__gm__ X_T*)y);
}

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
__aicore__ inline void Gatherv2SimtTwoDim<X_T, INDICES_T, INDEX_SIZE_T>::Process() {
  int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
  int32_t needCoreNum = static_cast<int32_t>(tilingData_->needCoreNum);
  uint32_t threadNum = static_cast<uint32_t>(tilingData_->threadNum);
  INDEX_SIZE_T gatherDimSize = static_cast<INDEX_SIZE_T>(tilingData_->gatherDimSize);
  INDEX_SIZE_T innerSize = static_cast<INDEX_SIZE_T>(tilingData_->innerSize);

  bool negativeIndexSupport = static_cast<bool>(tilingData_->negativeIndexSupport);
  INDEX_SIZE_T currentCoreElements = static_cast<INDEX_SIZE_T>(tilingData_->perCoreElements);
  if (blockIdx == tilingData_->needCoreNum - 1) {
    currentCoreElements = static_cast<INDEX_SIZE_T>(tilingData_->lastCoreElements);
  }
  INDEX_SIZE_T m0 {0};
  INDEX_SIZE_T shift0 {0};

  // fast division
  GetUintDivMagicAndShift(m0, shift0, innerSize);

  if (blockIdx < needCoreNum) {
    INDEX_SIZE_T yIndexBase = blockIdx * tilingData_->perCoreElements;
    if (unlikely(negativeIndexSupport)) {
      AscendC::Simt::VF_CALL<GatherSimt<true>>(Simt::Dim3(threadNum), yIndexBase, currentCoreElements, m0, shift0, 
                  innerSize, gatherDimSize, (__gm__ X_T*) (xGm_.GetPhyAddr()),
                  (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()), (__gm__ volatile X_T*) (yGm_.GetPhyAddr()));
    } else {
      AscendC::Simt::VF_CALL<GatherSimt<false>>(Simt::Dim3(threadNum), yIndexBase, currentCoreElements, m0, shift0,
                  innerSize, gatherDimSize, (__gm__ X_T*) (xGm_.GetPhyAddr()),
                  (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()), (__gm__ volatile X_T*) (yGm_.GetPhyAddr()));
    }
  }
}
}  // namespace gatherv2
#endif  // GATHER_V2_SIMT_TWO_DIM_H
