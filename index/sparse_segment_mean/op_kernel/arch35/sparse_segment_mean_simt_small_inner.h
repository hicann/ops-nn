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
 * \file sparse_segment_mean_simt_small_inner.h
 * \brief
 */

#ifndef SPARSE_SEGMENT_MEAN_SIMT_SMALL_INNER_H
#define SPARSE_SEGMENT_MEAN_SIMT_SMALL_INNER_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "sparse_segment_mean_struct.h"
#include "sparse_segment_mean_simt_base.h"
#include "kernel_operator.h"

constexpr uint32_t TBUF_SIZE = 4096;

namespace SparseSegmentMeanNameSpace {
using namespace AscendC;


template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
class SparseSegmentMeanSimtSmallInner {
public:
  __aicore__ inline SparseSegmentMeanSimtSmallInner(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR y,
                              GM_ADDR workspace, const SparseSegmentMeanSimtTilingData* tilingData);
  __aicore__ inline void Process();

private:
    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<SEGMENTIDS_T> segmentIdsGm_;
    GlobalTensor<X_T> yGm_;
    GlobalTensor<uint32_t> workspaceSegmentOffset_;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    const SparseSegmentMeanSimtTilingData* tilingData_ = nullptr;

    int64_t segOffsetBase_ = 0;
    int64_t curCoreSegments_ = 0;
    uint32_t segmentNum_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockNums_ = 0;
    uint32_t threadNumZ_ = 0;
};


template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
__aicore__ inline void SparseSegmentMeanSimtSmallInner<X_T, INDICES_T, SEGMENTIDS_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids,GM_ADDR y,
                                                                                GM_ADDR workspace, const SparseSegmentMeanSimtTilingData* tilingData) 
{
    tilingData_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ X_T*)x);
    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    segmentIdsGm_.SetGlobalBuffer((__gm__ SEGMENTIDS_T*)segment_ids);
    yGm_.SetGlobalBuffer((__gm__ X_T*)y);
    workspaceSegmentOffset_.SetGlobalBuffer((__gm__ uint32_t*)workspace);

    pipe_.InitBuffer(tmpBuf_, TBUF_SIZE);

    segmentNum_ = static_cast<uint32_t>(tilingData_->segmentNum);
    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    blockNums_ = static_cast<uint32_t>(GetBlockNum());

    // cal core offset
    if (tilingData_->specialBlockTiling) {
        if (tilingData_->secondToLastCoreSegmentNum == 0) {
            segOffsetBase_ = tilingData_->normalCoreSegmentNum * blockIdx_;
            curCoreSegments_ = blockIdx_ == tilingData_->needCoreNum - 1
                ? tilingData_->lastCoreSegmentNum
                : tilingData_->normalCoreSegmentNum;
        } else {
            if (blockIdx_ == tilingData_->needCoreNum - 1) {
                segOffsetBase_ = tilingData_->normalCoreSegmentNum * (blockIdx_ - 1) + tilingData_->secondToLastCoreSegmentNum;
                curCoreSegments_ = tilingData_->lastCoreSegmentNum;
            } else {
                segOffsetBase_ = tilingData_->normalCoreSegmentNum * blockIdx_;
                curCoreSegments_ = blockIdx_ == tilingData_->needCoreNum - 2
                    ? tilingData_->secondToLastCoreSegmentNum
                    : tilingData_->normalCoreSegmentNum;
            }
        }
        threadNumZ_ = blockIdx_ == tilingData_->needCoreNum - 1 ? tilingData_->lastCoreSegmentNum : 16;
    } else {
        if (blockIdx_ < tilingData_->resSegmentNum) {
            segOffsetBase_ = (tilingData_->perCoreSegmentNum + 1) * blockIdx_;
            curCoreSegments_ = tilingData_->perCoreSegmentNum + 1;
        } else {
            segOffsetBase_ = tilingData_->perCoreSegmentNum * blockIdx_ + tilingData_->resSegmentNum;
            curCoreSegments_ = tilingData_->perCoreSegmentNum;
        }
        threadNumZ_ = curCoreSegments_;
    }
}


template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
__aicore__ inline void SparseSegmentMeanSimtSmallInner<X_T, INDICES_T, SEGMENTIDS_T>::Process()
{
    if (blockIdx_ >= blockNums_) {
        return;
    }

    LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
    Duplicate<float>(tmpLocal, 0, TBUF_SIZE / sizeof(float));

    int64_t outterSize = tilingData_->outterSize;
    uint32_t threadNumX = static_cast<uint32_t>(tilingData_->threadNumX);
    uint32_t threadNumY = static_cast<uint32_t>(tilingData_->threadNumY);
    uint32_t innerSize = static_cast<uint32_t>(tilingData_->innerSize);

    AscendC::Simt::VF_CALL<SimtGetSegmentOffset<SEGMENTIDS_T>>(Simt::Dim3(MAX_THREAD_NUM), blockIdx_, outterSize, blockNums_, segmentNum_,
                                                               (__gm__ uint32_t*) (workspaceSegmentOffset_.GetPhyAddr()), (__gm__ SEGMENTIDS_T*) (segmentIdsGm_.GetPhyAddr()));
    
    SyncAll();

    AscendC::Simt::VF_CALL<SimtSmallInnerComputer<X_T, INDICES_T>>(Simt::Dim3{threadNumX, threadNumY, threadNumZ_}, segOffsetBase_, curCoreSegments_, innerSize,
                                                                   (__local_mem__ float*) (tmpLocal.GetPhyAddr()), (__gm__ X_T*) (xGm_.GetPhyAddr()), (__gm__ volatile X_T*) (yGm_.GetPhyAddr()),
                                                                   (__gm__ uint32_t*) (workspaceSegmentOffset_.GetPhyAddr()), (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()));
}


}  // namespace SparseSegmentMeanNameSpace
#endif  // SPARSE_SEGMENT_MEAN_SIMT_SMALL_INNER_H