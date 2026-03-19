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
 * \file sparse_segment_mean_simt_large_inner.h
 * \brief
 */

#ifndef SPARSE_SEGMENT_MEAN_SIMT_LARGE_INNER_H
#define SPARSE_SEGMENT_MEAN_SIMT_LARGE_INNER_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "sparse_segment_mean_struct.h"
#include "sparse_segment_mean_simt_base.h"
#include "kernel_operator.h"

#ifdef __DAV_FPGA__
constexpr uint32_t MAX_THREAD_NUM = 512;
#else
constexpr uint32_t MAX_THREAD_NUM = 2048;
#endif

namespace SparseSegmentMeanNameSpace {
using namespace AscendC;


template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
class SparseSegmentMeanSimtLargeInner {
public:
  __aicore__ inline SparseSegmentMeanSimtLargeInner(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR y,
                              GM_ADDR workspace, const SparseSegmentMeanSimtTilingData* tilingData);
  __aicore__ inline void Process();

private:
    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<SEGMENTIDS_T> segmentIdsGm_;
    GlobalTensor<X_T> yGm_;
    GlobalTensor<uint32_t> workspaceSegmentOffset_;
    
    const SparseSegmentMeanSimtTilingData* tilingData_ = nullptr;

    uint32_t segOffsetBase_ = 0;
    uint32_t curCoreSegments_ = 0;
    uint32_t segmentNum_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockNums_ = 0;
};


template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
__aicore__ inline void SparseSegmentMeanSimtLargeInner<X_T, INDICES_T, SEGMENTIDS_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR y,
                                                                                GM_ADDR workspace, const SparseSegmentMeanSimtTilingData* tilingData) 
{
    tilingData_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ X_T*)x);
    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    segmentIdsGm_.SetGlobalBuffer((__gm__ SEGMENTIDS_T*)segment_ids);
    yGm_.SetGlobalBuffer((__gm__ X_T*)y);
    segmentNum_ = static_cast<uint32_t>(tilingData_->segmentNum);
    workspaceSegmentOffset_.SetGlobalBuffer((__gm__ uint32_t*)workspace);

    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    blockNums_ = static_cast<uint32_t>(GetBlockNum());

    // cal core offset
    if (blockIdx_ < tilingData_->resSegmentNum) {
        segOffsetBase_ = (tilingData_->perCoreSegmentNum + 1) * blockIdx_;
        curCoreSegments_ = tilingData_->perCoreSegmentNum + 1;
    } else {
        segOffsetBase_ = tilingData_->perCoreSegmentNum * blockIdx_ + tilingData_->resSegmentNum;
        curCoreSegments_ = tilingData_->perCoreSegmentNum;
    }
}


template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
__aicore__ inline void SparseSegmentMeanSimtLargeInner<X_T, INDICES_T, SEGMENTIDS_T>::Process()
{
    if (blockIdx_ >= blockNums_) {
        return;
    }

    int64_t outterSize = tilingData_->outterSize;
    uint32_t threadNumX = static_cast<uint32_t>(tilingData_->threadNumX);
    uint32_t threadNumY = static_cast<uint32_t>(tilingData_->threadNumY);
    int64_t xDim0 = static_cast<uint32_t>(tilingData_->gatherSize);
    uint32_t innerSize = static_cast<uint32_t>(tilingData_->innerSize);

    AscendC::Simt::VF_CALL<SimtGetSegmentOffset<SEGMENTIDS_T>>(Simt::Dim3(MAX_THREAD_NUM), blockIdx_, outterSize, blockNums_, segmentNum_,
                                                               (__gm__ uint32_t*) (workspaceSegmentOffset_.GetPhyAddr()), (__gm__ SEGMENTIDS_T*) (segmentIdsGm_.GetPhyAddr()));
    
    SyncAll();

    AscendC::Simt::VF_CALL<SimtLargeInnerComputer<X_T, INDICES_T>>(Simt::Dim3{threadNumX, threadNumY}, segOffsetBase_, curCoreSegments_, innerSize, xDim0,
                                                                   (__gm__ X_T*) (xGm_.GetPhyAddr()), (__gm__ volatile X_T*) (yGm_.GetPhyAddr()),
                                                                   (__gm__ uint32_t*) (workspaceSegmentOffset_.GetPhyAddr()), (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()));
}


}  // namespace SparseSegmentMeanNameSpace
#endif  // SPARSE_SEGMENT_MEAN_SIMT_LARGE_INNER_H