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
 * \file bucketize_v2_full_load.h
 * \brief
 */
#ifndef BUCKETIZE_V2_FULL_LOAD_H
#define BUCKETIZE_V2_FULL_LOAD_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "bucketize_v2_common.h"
#include "bucketize_v2_struct.h"

#if ASC_DEVKIT_MAJOR >=9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/platform_util.h"

namespace BucketizeV2 {
using namespace AscendC;

constexpr int32_t FULL_LOAD_BUFFER_NUM = 2;

template <typename X_T, typename B_T, typename Y_T, bool RIGHT=false>
class BucketizeV2FullLoad {
 public:
  __aicore__ inline BucketizeV2FullLoad(TPipe *pipe): pipe_(pipe){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR boundaries, GM_ADDR y, const BucketizeV2FullLoadTilingData* tilingData);
  __aicore__ inline void Process();
  __aicore__ inline void CopyOut(int64_t offset, uint32_t copyLen);
  __aicore__ inline void CopyInBound();
  __aicore__ inline void CopyInX(int64_t offset, uint32_t copyLen);
  __aicore__ inline void Compute(LocalTensor<B_T> &boundLocal, uint32_t copyLen);
 private:
  GlobalTensor<X_T> xGm_;
  GlobalTensor<B_T> boundGm_;
  GlobalTensor<Y_T> yGm_;
  TPipe *pipe_;
  TQue<QuePosition::VECIN, FULL_LOAD_BUFFER_NUM> xQueue_;
  TQue<QuePosition::VECIN, 1> boundQueue_;
  TQue<QuePosition::VECOUT, FULL_LOAD_BUFFER_NUM> yQueue_;

  const BucketizeV2FullLoadTilingData* tilingData_;

  int64_t loopNum_ = 0;
  uint32_t tailUbFactor_ = 0;
  int32_t blockIdx_ = 0;
};

template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
__aicore__ inline void BucketizeV2FullLoad<X_T, B_T, Y_T, RIGHT>::Init(GM_ADDR x, GM_ADDR boundaries,
                                                                    GM_ADDR y, const BucketizeV2FullLoadTilingData* tilingData) {
  tilingData_ = tilingData;
  blockIdx_ = static_cast<int32_t>(GetBlockIdx());

  xGm_.SetGlobalBuffer((__gm__ X_T*)x + GetBlockIdx() * tilingData_->blockFactor);
  boundGm_.SetGlobalBuffer((__gm__ B_T*)boundaries);
  yGm_.SetGlobalBuffer((__gm__ Y_T*)y + GetBlockIdx() * tilingData_->blockFactor);

  pipe_->InitBuffer(xQueue_, FULL_LOAD_BUFFER_NUM, tilingData_->inUbSize);
  pipe_->InitBuffer(boundQueue_, 1, tilingData_->boundBufSize );
  pipe_->InitBuffer(yQueue_, FULL_LOAD_BUFFER_NUM, tilingData_->outUbSize);

  int64_t curBlockFactor = (GetBlockIdx() < tilingData_->usedCoreNum - 1) ? tilingData_->blockFactor : tilingData_->blockTail;
  loopNum_ = curBlockFactor / tilingData_->ubFactor;
  tailUbFactor_ = curBlockFactor - loopNum_ * tilingData_->ubFactor;
  if (tailUbFactor_ == 0) {
    tailUbFactor_ = tilingData_->ubFactor;
  } else {
    loopNum_ += 1;
  }
}

template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
__aicore__ inline void BucketizeV2FullLoad<X_T, B_T, Y_T, RIGHT>::CopyInBound()
{
    LocalTensor<B_T> boundLocal = boundQueue_.AllocTensor<B_T>();
    CopyGmToUb<B_T>(boundLocal, boundGm_, 0, 1, tilingData_->boundSize);
    boundQueue_.EnQue(boundLocal);
}

template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
__aicore__ inline void BucketizeV2FullLoad<X_T, B_T, Y_T, RIGHT>::CopyInX(int64_t offset,
   uint32_t copyLen)
{
    LocalTensor<X_T> xLocal = xQueue_.AllocTensor<B_T>();
    CopyGmToUb<X_T>(xLocal, xGm_, offset, 1, copyLen);
    xQueue_.EnQue(xLocal);
}

template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
__aicore__ inline void BucketizeV2FullLoad<X_T, B_T, Y_T, RIGHT>::CopyOut(int64_t offset, uint32_t copyLen)
{
    LocalTensor<Y_T> yLocal = yQueue_.DeQue<Y_T>();
    CopyUbToGm(yGm_, yLocal, offset, 1, copyLen);
    yQueue_.FreeTensor(yLocal);
}

template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
__aicore__ inline void BucketizeV2FullLoad<X_T, B_T, Y_T, RIGHT>::Compute(LocalTensor<B_T> &boundLocal, uint32_t dataLen)
{
    LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
    LocalTensor<Y_T> yLocal = yQueue_.AllocTensor<Y_T>();
    BinaryQuery<X_T, B_T, Y_T, RIGHT>(xLocal, boundLocal, yLocal, dataLen, tilingData_->maxIter, tilingData_->boundSize);
    yQueue_.EnQue(yLocal);
    xQueue_.FreeTensor(xLocal);
}

template <typename X_T, typename B_T, typename Y_T, bool RIGHT>
__aicore__ inline void BucketizeV2FullLoad<X_T, B_T, Y_T, RIGHT>::Process() {
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    
    CopyInBound();
    LocalTensor<B_T> boundLocal = boundQueue_.DeQue<B_T>();
    uint32_t dataLen = static_cast<uint32_t>(tilingData_->ubFactor);
    for (int64_t i = 0; i < loopNum_; i++) {
      if (i == loopNum_ - 1) {
        dataLen = tailUbFactor_;
      }
      CopyInX(i * tilingData_->ubFactor, dataLen);
      Compute(boundLocal, dataLen);
      CopyOut(i * tilingData_->ubFactor, dataLen);
    }
    boundQueue_.FreeTensor(boundLocal);
}
}  // namespace BucketizeV2
#endif  // BUCKETIZE_V2_FULL_LOAD_H
