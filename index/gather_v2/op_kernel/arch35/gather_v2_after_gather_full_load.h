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
 * \file gather_v2_after_gather_full_load.h
 * \brief
 */
#ifndef GATHER_V2_AFTER_GATHER_FULL_LOAD
#define GATHER_V2_AFTER_GATHER_FULL_LOAD

#include "kernel_operator.h"
#include "basic_api/kernel_vec_intf.h"
#include "op_kernel/platform_util.h"

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_BOUND = 512;
#else
constexpr uint32_t THREAD_NUM_BOUND = 2048;
#endif

namespace gatherv2 {
using namespace AscendC;

template <typename T, typename INDICES_T, typename U>
class Gatherv2FullLoad {
 public:
  __aicore__ inline Gatherv2FullLoad(TPipe *pipe): pipe_(pipe){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y, const GatherV2TilingData* tilingData);
  __aicore__ inline void CopyIn(LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst, uint64_t copyLen);
  __aicore__ inline void Process();
  __aicore__ inline void LoopColProcess();
  __aicore__ inline void NoLoopProcess();

 private:
  template <const bool NIS>
  static __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_BOUND) inline void GatherSimt3D(const U yIndexBase,
  U currentCoreElements, U m0, U shift0, U m1, U shift1, int64_t pOffset, U gatherSize, U innerSize,
  U gatherDimSize, __ubuf__ T* srcTensor, __gm__ INDICES_T* indices, __gm__ volatile T* y);

  template <const bool NIS>
  static __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_BOUND) inline void GatherSimt2D(const U yIndexBase,
  U currentCoreElements, U m0, U shift0, U innerSize, U gatherDimSize, __ubuf__ T* srcTensor, 
  __gm__ INDICES_T* indices, __gm__ volatile T* y);

 private:
  GlobalTensor<T> xGm_;
  GlobalTensor<INDICES_T> indicesGm_;
  GlobalTensor<T> yGm_;
  
  TPipe *pipe_;
  TQue<QuePosition::VECIN, 1> inQueue_;
  const GatherV2TilingData* tilingData_;

  int32_t blockIdx_;
  int32_t needCoreNum_;
  int32_t threadNum_;
  bool negativeIndexSupport_;

  U outerSize_;
  U gatherDimSize_;
  U gatherSize_;
  U innerSize_;
  U batchSize_;
  U ubAviable_;
  U currentCoreElements_;
  U loopNum_;
  U perLoopCoreElements_;
};

// output = [1,p,g,a] 3D 方向计算index
template <typename T, typename INDICES_T, typename U>
template <const bool NIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_BOUND) inline void Gatherv2FullLoad<T, INDICES_T, U>::GatherSimt3D(const U yIndexBase,
  U currentCoreElements, U m0, U shift0, U m1, U shift1, int64_t pOffset, U gatherSize, U innerSize,
  U gatherDimSize, __ubuf__ T* srcTensor, __gm__ INDICES_T* indices, __gm__ volatile T* y) {
  for (U index = static_cast<U>(Simt::GetThreadIdx()); index < currentCoreElements;
      index += static_cast<U>(Simt::GetThreadNum())) {

    U yIndex = yIndexBase + index;
    U tmp = Simt::UintDiv(yIndex, m0, shift0);
    U tmp1 = Simt::UintDiv(tmp, m1, shift1);
    U gatherI = tmp - tmp1 * gatherSize;
    U innerI = yIndex  - tmp * innerSize;

    U indicesIndex = gatherI;
    INDICES_T indicesValue = indices[indicesIndex];
    if constexpr(NIS) {
      if (unlikely(indicesValue < 0)) {
        indicesValue += gatherDimSize;
      }
    }
    U indicesValueI = static_cast<U>(indicesValue);
    U xIndex = ((tmp1 - pOffset) * gatherDimSize + indicesValueI) * innerSize + innerI;
    bool idxOutOfBound = indicesValue < 0 || indicesValueI >= gatherDimSize;
    y[yIndex] = idxOutOfBound ? 0 : srcTensor[xIndex];
  }
}

// output = [1,1,g,a] 2D 方向计算index
template <typename T, typename INDICES_T, typename U>
template <const bool NIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_BOUND) inline void Gatherv2FullLoad<T, INDICES_T, U>::GatherSimt2D(const U yIndexBase,
  U currentCoreElements, U m0, U shift0, U innerSize, U gatherDimSize, __ubuf__ T* srcTensor, 
  __gm__ INDICES_T* indices, __gm__ volatile T* y) {
  for (U index = static_cast<U>(Simt::GetThreadIdx()); index < currentCoreElements;
      index += static_cast<U>(Simt::GetThreadNum())) {

    U yIndex = yIndexBase + index;
    U gatherI = Simt::UintDiv(yIndex, m0, shift0);
    U innerI = yIndex  - gatherI * innerSize;

    U indicesIndex = gatherI;
    INDICES_T indicesValue = indices[indicesIndex];
    if constexpr(NIS) {
      if (unlikely(indicesValue < 0)) {
        indicesValue += gatherDimSize;
      }
    }
    U indicesValueI = static_cast<U>(indicesValue);
    U xIndex = indicesValueI * innerSize + innerI;
    bool idxOutOfBound = indicesValue < 0 || indicesValueI >= gatherDimSize;
    y[yIndex] = idxOutOfBound ? 0 : srcTensor[xIndex];
  }
}

template <typename T, typename INDICES_T, typename U>
__aicore__ inline void Gatherv2FullLoad<T, INDICES_T, U>::CopyIn(LocalTensor<T> xLocal,
  GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst, uint64_t copyLen)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen;
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T, typename INDICES_T, typename U>
__aicore__ inline void Gatherv2FullLoad<T, INDICES_T, U>::LoopColProcess()
{
  int64_t gDimInerSize = gatherDimSize_ * innerSize_;
  int64_t gInerSize = gatherSize_ * innerSize_;

  int64_t maxGdimA = ubAviable_ / gDimInerSize;
  int64_t start = blockIdx_ * tilingData_->perCoreElements;
  int64_t curCoreData = (blockIdx_ == needCoreNum_ -1) ? tilingData_->lastCoreElements : tilingData_->perCoreElements;
  int64_t end = start + curCoreData - 1;

  int64_t startP = start / gatherSize_;
  int64_t startG = start % gatherSize_;
  int64_t endP = end / gatherSize_;
  int64_t startGOffset = startG * innerSize_;

  loopNum_ = (endP - startP + 1 + maxGdimA -  1) / maxGdimA;
  U yIndexBase = blockIdx_ * tilingData_->perCoreElements * innerSize_;
  int64_t pOffset =  startP;
  
  for (int64_t loop = 0; loop < loopNum_; ++loop) {
    int64_t copySize;
    int64_t startGroup = startP + loop * maxGdimA;
    int64_t endGroup = startGroup + maxGdimA;
    int64_t groupCount = endGroup - startGroup;
    int64_t startOffset = pOffset * gDimInerSize;

    if (loopNum_ > 1) {
      copySize = groupCount * gDimInerSize;
      if (loop == 0) {
        currentCoreElements_ = groupCount * gInerSize - startGOffset;
      } else if (loop == (loopNum_ -1)) {
        copySize = ((endP - startP + 1) % maxGdimA) * gDimInerSize;
        currentCoreElements_ = perLoopCoreElements_ - (loop * maxGdimA) * gInerSize + startGOffset;
      } else {
        currentCoreElements_ = groupCount * gInerSize;
      }
    } else {
      copySize = (endP - startP + 1) * gDimInerSize;
    }
    
    //从GM 搬运数据到UB
    LocalTensor<T> gaBuffer = inQueue_.AllocTensor<T>();
    CopyIn(gaBuffer, xGm_, startOffset, 1, copySize * tilingData_->dtypeSize);

    //simt gather 计算
    U m0 {0};
    U m1 {0};
    U shift0 {0};
    U shift1 {0};
    GetUintDivMagicAndShift(m0, shift0, innerSize_);
    GetUintDivMagicAndShift(m1, shift1, gatherSize_);

    inQueue_.EnQue(gaBuffer);
    inQueue_.DeQue<T>();
    if (unlikely(negativeIndexSupport_)) {
      AscendC::Simt::VF_CALL<GatherSimt3D<true>>(Simt::Dim3(threadNum_), yIndexBase, currentCoreElements_, m0, shift0, m1, shift1, pOffset,
                  gatherSize_, innerSize_, gatherDimSize_, (__ubuf__ T*)(gaBuffer.GetPhyAddr()),
                  (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()), (__gm__ volatile T*) (yGm_.GetPhyAddr()));
    } else {
      AscendC::Simt::VF_CALL<GatherSimt3D<false>>(Simt::Dim3(threadNum_), yIndexBase, currentCoreElements_, m0, shift0, m1, shift1, pOffset,
                  gatherSize_, innerSize_, gatherDimSize_, (__ubuf__ T*)(gaBuffer.GetPhyAddr()),
                  (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()), (__gm__ volatile T*) (yGm_.GetPhyAddr()));
    }
    //计算每次循环yIndexBase
    if (loopNum_ > 1) {
      yIndexBase += currentCoreElements_;
      pOffset = yIndexBase / gInerSize;
    }
    inQueue_.FreeTensor(gaBuffer);
  }
}

template <typename T, typename INDICES_T, typename U>
__aicore__ inline void Gatherv2FullLoad<T, INDICES_T, U>::NoLoopProcess()
{
  int64_t copy_size = gatherDimSize_ * innerSize_;
  LocalTensor<T> gaBuffer = inQueue_.AllocTensor<T>();
  CopyIn(gaBuffer, xGm_, 0, 1, copy_size * tilingData_->dtypeSize);

  //simt gather 计算
    U m0 {0};
    U shift0 {0};
    GetUintDivMagicAndShift(m0, shift0, innerSize_);

    inQueue_.EnQue(gaBuffer);
    inQueue_.DeQue<T>();
    U yIndexBase = blockIdx_ * tilingData_->perCoreElements * innerSize_;
    if (unlikely(negativeIndexSupport_)) {
      AscendC::Simt::VF_CALL<GatherSimt2D<true>>(Simt::Dim3(threadNum_), yIndexBase, currentCoreElements_, m0, shift0,
                  innerSize_, gatherDimSize_, (__ubuf__ T*)(gaBuffer.GetPhyAddr()),
                  (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()), (__gm__ volatile T*) (yGm_.GetPhyAddr()));
    } else {
      AscendC::Simt::VF_CALL<GatherSimt2D<false>>(Simt::Dim3(threadNum_), yIndexBase, currentCoreElements_, m0, shift0,
                  innerSize_, gatherDimSize_, (__ubuf__ T*)(gaBuffer.GetPhyAddr()),
                  (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()), (__gm__ volatile T*) (yGm_.GetPhyAddr()));
    }
    inQueue_.FreeTensor(gaBuffer);
}

template <typename T, typename INDICES_T, typename U>
__aicore__ inline void Gatherv2FullLoad<T, INDICES_T, U>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis,
                                                                    GM_ADDR y, const GatherV2TilingData* tilingData) {
  tilingData_ = tilingData;
  xGm_.SetGlobalBuffer((__gm__ T*)x);
  indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
  yGm_.SetGlobalBuffer((__gm__ T*)y);

  pipe_->InitBuffer(inQueue_, 1, (tilingData_->maxElement * tilingData_->dtypeSize));
}


template <typename T, typename INDICES_T, typename U>
__aicore__ inline void Gatherv2FullLoad<T, INDICES_T, U>::Process() {
  blockIdx_ = static_cast<int32_t>(GetBlockIdx());
  needCoreNum_ = static_cast<int32_t>(tilingData_->needCoreNum);
  threadNum_ = static_cast<uint32_t>(tilingData_->threadNum);
  
  batchSize_ = static_cast<U>(tilingData_->batchSize);
  outerSize_ = static_cast<U>(tilingData_->outerSize);
  gatherDimSize_ = static_cast<U>(tilingData_->gatherDimSize);
  gatherSize_ = static_cast<U>(tilingData_->gatherSize);
  innerSize_ = static_cast<U>(tilingData_->innerSize);
  ubAviable_ = static_cast<U>(tilingData_->maxElement);

  negativeIndexSupport_ = static_cast<bool>(tilingData_->negativeIndexSupport);
  currentCoreElements_ = static_cast<U>(tilingData_->perCoreElements * innerSize_);
  if (blockIdx_ == tilingData_->needCoreNum - 1) {
    currentCoreElements_ = static_cast<U>(tilingData_->lastCoreElements * innerSize_);
  }
  perLoopCoreElements_ = currentCoreElements_;

  if (batchSize_ == 1 && outerSize_ > 1) {
    LoopColProcess();
  } else {
    NoLoopProcess();
  }
}
}  // namespace gatherv2
#endif  // GATHER_V2_H
