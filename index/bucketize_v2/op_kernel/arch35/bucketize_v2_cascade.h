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
 * \file bucketize_v2_cascade.h
 * \brief
 */
#ifndef BUCKETIZE_V2_CASCADE_H
#define BUCKETIZE_V2_CASCADE_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "bucketize_v2_common.h"
#include "bucketize_v2_struct.h"
 
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"

namespace BucketizeV2 {
using namespace AscendC;

constexpr int32_t CASCADE_BUFFER_NUM = 2;
constexpr uint32_t THREAD_DIM_2048 = 2048;
constexpr uint32_t QUERY_THREAD_DIM = 1024;

template <typename T, uint32_t THREAD_NUM_LAUNCH_BOUND=2048>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM_LAUNCH_BOUND) inline void SampleGmToUb(__gm__ T* x, __ubuf__ T* y, int64_t totalSize, int64_t sampleRatio, int32_t sampleSize) {
   for (int32_t index = threadIdx.x; index < sampleSize; index += blockDim.x) {
        int64_t gmIdx = index * sampleRatio;
        if (gmIdx >= totalSize) {
            y[index] = x[totalSize - 1];
            break;
        }
        y[index] = x[gmIdx];
   }
 }

template <typename X_T, typename B_T, typename Y_T, bool RIGHT=false>
__simt_callee__ __aicore__ inline Y_T InnerBinaryQuery(X_T value, Y_T start, Y_T end, __gm__ B_T* bound, int64_t innerMaxIter) {
    Y_T left = start;
    Y_T right = end;
    for (int64_t i = 0; i < innerMaxIter; i++) {
      if (left >= right) {
          break;
      }
      Y_T mid = left + ((right - left) >> 1);
      B_T midValue = bound[mid];
      bool cond = false;
      if constexpr(RIGHT) {
          cond = !(midValue > value);
      } else {
          cond = !(midValue >= value);
      }
      left = cond ? mid + 1 : left;
      right = cond ? right : mid;
    }
    return left;
 }

 template <typename X_T, typename B_T, typename Y_T, bool RIGHT=false, uint32_t THREAD_NUM_LAUNCH_BOUND=1024>
 __simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM_LAUNCH_BOUND) inline void CascadeQuery(__ubuf__ Y_T* midOut, __ubuf__ X_T* value, __gm__ B_T* bound, __ubuf__ Y_T* out, int32_t dataLen, int32_t sampleSize, int64_t innerMaxIter, int64_t sampleRatio, int64_t boundSize) {
   for (int32_t idx = threadIdx.x; idx < dataLen; idx += blockDim.x) {
        if (midOut[idx] == 0) {
            out[idx] = 0;
            continue;
        }
        if (midOut[idx] >= sampleSize) {
            out[idx] = boundSize;
            continue;
        }
        Y_T start = (midOut[idx] - 1) * sampleRatio;
        Y_T end = min(midOut[idx] * sampleRatio, boundSize);
        out[idx] = InnerBinaryQuery<X_T, B_T, Y_T, RIGHT>(value[idx], start, end, bound, innerMaxIter);
   }
 }
 
 template <typename X_T, typename B_T, typename Y_T, bool RIGHT=false>
 class BucketizeV2Cascade {
  public:
   __aicore__ inline BucketizeV2Cascade(TPipe *pipe): pipe_(pipe){};
   __aicore__ inline void Init(GM_ADDR x, GM_ADDR boundaries, GM_ADDR y, const BucketizeV2CascadeTilingData* tilingData);
   __aicore__ inline void Process();
   __aicore__ inline void CopyInX(int64_t offset, uint32_t copyLen);
   __aicore__ inline void CopyInCoarseBound();
   __aicore__ inline void Compute(LocalTensor<B_T> &boundLocal, uint32_t copyLen);
   __aicore__ inline void CopyOut(int64_t offset, uint32_t copyLen);
  private:
   GlobalTensor<X_T> xGm_;
   GlobalTensor<B_T> boundGm_;
   GlobalTensor<Y_T> yGm_;
   TPipe *pipe_;
   TQue<QuePosition::VECIN, CASCADE_BUFFER_NUM> xQueue_;
   TQue<QuePosition::VECIN, 1> boundQueue_;
   TQue<QuePosition::VECOUT, CASCADE_BUFFER_NUM> yQueue_;
   TBuf<QuePosition::VECCALC> midOutBuf_;

   const BucketizeV2CascadeTilingData* tilingData_;
 
   int64_t loopNum_ = 0;
   uint32_t tailUbFactor_ = 0;
   int32_t blockIdx_ = 0;
 };
 
 template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
 __aicore__ inline void BucketizeV2Cascade<X_T, B_T, Y_T, RIGHT>::Init(GM_ADDR x, GM_ADDR boundaries,
                                                                     GM_ADDR y, const BucketizeV2CascadeTilingData* tilingData) {
   tilingData_ = tilingData;
   blockIdx_ = static_cast<int32_t>(GetBlockIdx());

   xGm_.SetGlobalBuffer((__gm__ X_T*)x + blockIdx_ * tilingData_->blockFactor);
   boundGm_.SetGlobalBuffer((__gm__ B_T*)boundaries);
   yGm_.SetGlobalBuffer((__gm__ Y_T*)y + blockIdx_ * tilingData_->blockFactor);
 
   pipe_->InitBuffer(xQueue_, CASCADE_BUFFER_NUM, tilingData_->inUbSize);
   pipe_->InitBuffer(boundQueue_, 1, tilingData_->boundBufSize);
   pipe_->InitBuffer(yQueue_, CASCADE_BUFFER_NUM, tilingData_->outUbSize);
   pipe_->InitBuffer(midOutBuf_, tilingData_->midOutUbSize);

   int64_t curBlockFactor = (blockIdx_ < tilingData_->usedCoreNum - 1) ? tilingData_->blockFactor : tilingData_->blockTail;
   loopNum_ = curBlockFactor / tilingData_->ubFactor;
   tailUbFactor_ = curBlockFactor - loopNum_ * tilingData_->ubFactor;
   if (tailUbFactor_ == 0) {
     tailUbFactor_ = tilingData_->ubFactor;
   } else {
     loopNum_ += 1;
   }
 }
 
 template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
 __aicore__ inline void BucketizeV2Cascade<X_T, B_T, Y_T, RIGHT>::CopyInCoarseBound()
 {
     LocalTensor<B_T> boundLocal = boundQueue_.AllocTensor<B_T>();
     asc_vf_call<SampleGmToUb<B_T, THREAD_DIM_2048>>(dim3{THREAD_DIM_2048}, (__gm__ B_T*)(boundGm_.GetPhyAddr()),
        (__ubuf__ B_T*)(boundLocal.GetPhyAddr()), tilingData_->boundSize, tilingData_->sampleRatio, tilingData_->sampleBoundSize);
     boundQueue_.EnQue(boundLocal);
 }
 
 template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
 __aicore__ inline void BucketizeV2Cascade<X_T, B_T, Y_T, RIGHT>::CopyInX(int64_t offset,
    uint32_t copyLen)
 {
     LocalTensor<X_T> xLocal = xQueue_.AllocTensor<B_T>();
     CopyGmToUb<X_T>(xLocal, xGm_, offset, 1, copyLen);
     xQueue_.EnQue(xLocal);
 }
 
 template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
 __aicore__ inline void BucketizeV2Cascade<X_T, B_T, Y_T, RIGHT>::CopyOut(int64_t offset, uint32_t copyLen)
 {
     LocalTensor<Y_T> yLocal = yQueue_.DeQue<Y_T>();
     CopyUbToGm(yGm_, yLocal, offset, 1, copyLen);
     yQueue_.FreeTensor(yLocal);
 }
 
 template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
 __aicore__ inline void BucketizeV2Cascade<X_T, B_T, Y_T, RIGHT>::Compute(LocalTensor<B_T> &boundLocal, uint32_t dataLen)
 {
     LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
     LocalTensor<Y_T> midOutputUb = midOutBuf_.Get<Y_T>();
     BinaryQuery<X_T, B_T, Y_T, RIGHT>(xLocal, boundLocal, midOutputUb, dataLen, tilingData_->outerMaxIter, tilingData_->sampleBoundSize);
     LocalTensor<Y_T> yLocal = yQueue_.AllocTensor<Y_T>();
     asc_vf_call<CascadeQuery<X_T, X_T, Y_T, RIGHT, QUERY_THREAD_DIM>>(dim3{QUERY_THREAD_DIM}, (__ubuf__ Y_T*)(midOutputUb.GetPhyAddr()),
        (__ubuf__ X_T*)(xLocal.GetPhyAddr()), (__gm__ B_T*)(boundGm_.GetPhyAddr()), (__ubuf__ Y_T*)(yLocal.GetPhyAddr()), dataLen, tilingData_->sampleBoundSize, 
        tilingData_->innerMaxIter, tilingData_->sampleRatio, tilingData_->boundSize);
     yQueue_.EnQue(yLocal);
     xQueue_.FreeTensor(xLocal);
 }
 
 template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
 __aicore__ inline void BucketizeV2Cascade<X_T, B_T, Y_T, RIGHT>::Process() {
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    
    CopyInCoarseBound();
    uint32_t dataLen = static_cast<uint32_t>(tilingData_->ubFactor);
    LocalTensor<B_T> boundLocal = boundQueue_.DeQue<B_T>();
    for (int64_t i = 0; i < loopNum_; i++) {
        if (i == loopNum_ - 1) {
            dataLen = tailUbFactor_;
        }
        int64_t offset = i * tilingData_->ubFactor;
        CopyInX(offset, dataLen);
        Compute(boundLocal, dataLen);
        CopyOut(offset, dataLen);
    }
    boundQueue_.FreeTensor(boundLocal);
}
}  // namespace BucketizeV2
#endif  // BUCKETIZE_V2_CASCADE_H
