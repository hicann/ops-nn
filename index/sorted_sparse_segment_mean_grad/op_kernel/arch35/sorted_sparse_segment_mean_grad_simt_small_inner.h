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
 * \file sorted_sparse_segment_mean_grad_simt_small_inner.h
 * \brief
 */

#ifndef SORTED_SPARSE_SEGMENT_MEAN_GRAD_SIMT_SMALL_INNER_H
#define SORTED_SPARSE_SEGMENT_MEAN_GRAD_SIMT_SMALL_INNER_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "sorted_sparse_segment_mean_grad_struct.h"
#include "sorted_sparse_segment_mean_grad_simt_base.h"
#include "kernel_operator.h"

constexpr uint32_t TBUF_SIZE = 512;

namespace SparseSegmentMeanGradNameSpace {
using namespace AscendC;

template <typename X_T, typename INDICES_T, typename LOCATION_T, typename SEGMENTIDS_T, typename OUTTER_T, typename INNER_T>
class SortedSparseSegmentMeanGradSimtSmallInner {
public:
  __aicore__ inline SortedSparseSegmentMeanGradSimtSmallInner(){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR output_dim0,
                              GM_ADDR location, GM_ADDR y, GM_ADDR workspace,
                              const SortedSparseSegmentMeanGradSimtTilingData* tilingData);
  __aicore__ inline void Process();

private:
    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<SEGMENTIDS_T> segmentIdsGm_;
    GlobalTensor<LOCATION_T> locationGm_;
    GlobalTensor<X_T> yGm_;
    GlobalTensor<OUTTER_T> workspaceSegmentOffset_;
    GlobalTensor<float> workspaceWeight_;
    GlobalTensor<OUTTER_T> workspaceIndicesOffset_;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    const SortedSparseSegmentMeanGradSimtTilingData* tilingData_ = nullptr;

    uint32_t indicesOffsetBase_ = 0;
    uint32_t curCoreIndices_ = 0;
    SEGMENTIDS_T segmentNum_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockNums_ = 0;
};


template <typename X_T, typename INDICES_T, typename LOCATION_T, typename SEGMENTIDS_T, typename OUTTER_T, typename INNER_T>
__aicore__ inline void SortedSparseSegmentMeanGradSimtSmallInner<X_T, INDICES_T, LOCATION_T, SEGMENTIDS_T, OUTTER_T, INNER_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR output_dim0,
                                                                                                 GM_ADDR location, GM_ADDR y, GM_ADDR workspace,
                                                                                                 const SortedSparseSegmentMeanGradSimtTilingData* tilingData) 
{
    tilingData_ = tilingData;
    pipe_.InitBuffer(tmpBuf_, TBUF_SIZE);
    xGm_.SetGlobalBuffer((__gm__ X_T*)x);
    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    segmentIdsGm_.SetGlobalBuffer((__gm__ SEGMENTIDS_T*)segment_ids);
    locationGm_.SetGlobalBuffer((__gm__ LOCATION_T*)location);
    yGm_.SetGlobalBuffer((__gm__ X_T*)y);
    segmentNum_ = static_cast<SEGMENTIDS_T>(tilingData_->segmentNum);
    workspaceSegmentOffset_.SetGlobalBuffer((__gm__ OUTTER_T*)workspace);
    workspaceWeight_.SetGlobalBuffer((__gm__ float*)workspace);
    workspaceIndicesOffset_.SetGlobalBuffer((__gm__ OUTTER_T*)workspace);

    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    blockNums_ = static_cast<uint32_t>(GetBlockNum());

    // cal core offset
    if (blockIdx_ < static_cast<uint32_t>(tilingData_->resIndicesNum)) {
        indicesOffsetBase_ = (static_cast<uint32_t>(tilingData_->perCoreIndicesNum) + 1) * blockIdx_;
        curCoreIndices_ = static_cast<uint32_t>(tilingData_->perCoreIndicesNum) + 1;
    } else {
        indicesOffsetBase_ = static_cast<uint32_t>(tilingData_->perCoreIndicesNum) * blockIdx_ + static_cast<uint32_t>(tilingData_->resIndicesNum);
        curCoreIndices_ = static_cast<uint32_t>(tilingData_->perCoreIndicesNum);
    }
}


template <typename X_T, typename INDICES_T, typename LOCATION_T, typename SEGMENTIDS_T, typename OUTTER_T, typename INNER_T>
__aicore__ inline void SortedSparseSegmentMeanGradSimtSmallInner<X_T, INDICES_T, LOCATION_T, SEGMENTIDS_T, OUTTER_T, INNER_T>::Process()
{
    if (blockIdx_ >= blockNums_) {
        return;
    }

    LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
    Duplicate<float>(tmpLocal, 0, TBUF_SIZE / sizeof(float));

    OUTTER_T outterSize = static_cast<OUTTER_T>(tilingData_->outterSize);
    uint32_t threadNumX = static_cast<uint32_t>(tilingData_->threadNumX);
    uint32_t threadNumY = static_cast<uint32_t>(tilingData_->threadNumY);
    uint32_t segThreadNumX = static_cast<uint32_t>(tilingData_->segThreadNumX);
    uint32_t segThreadNumY = static_cast<uint32_t>(tilingData_->segThreadNumY);
    uint32_t indexThreadNumX = static_cast<uint32_t>(tilingData_->indexThreadNumX);
    uint32_t indexThreadNumY = static_cast<uint32_t>(tilingData_->indexThreadNumY);
    INNER_T innerSize = static_cast<INNER_T>(tilingData_->innerSize);

    AscendC::Simt::VF_CALL<SimtGetSegmentOffset<SEGMENTIDS_T, OUTTER_T>>(Simt::Dim3(segThreadNumY, segThreadNumX), blockIdx_, outterSize, blockNums_, segmentNum_, 
                                                               (__gm__ OUTTER_T*) (workspaceSegmentOffset_.GetPhyAddr()), (__gm__ SEGMENTIDS_T*) (segmentIdsGm_.GetPhyAddr()));
    SyncAll();
    AscendC::Simt::VF_CALL<SimtCalcWeight<SEGMENTIDS_T, OUTTER_T>>(Simt::Dim3(MAX_SIMPLE_THREAD_NUM), blockIdx_, blockNums_, segmentNum_,
                                                               (__gm__ OUTTER_T*) (workspaceSegmentOffset_.GetPhyAddr()), (__gm__ float*) (workspaceWeight_.GetPhyAddr(segmentNum_ + 1)));
    AscendC::Simt::VF_CALL<SimtGetSegmentOffset<INDICES_T, OUTTER_T>>(Simt::Dim3(indexThreadNumY, indexThreadNumX), blockIdx_, outterSize, blockNums_, static_cast<INDICES_T>(tilingData_->outputDim0),
                                                               (__gm__ OUTTER_T*) (workspaceIndicesOffset_.GetPhyAddr(2 * (segmentNum_ + 1))), (__gm__ INDICES_T*) (indicesGm_.GetPhyAddr()));
    SyncAll();

    AscendC::Simt::VF_CALL<SimtSmallInnerComputer<X_T, LOCATION_T, SEGMENTIDS_T, OUTTER_T, INNER_T>>(Simt::Dim3{threadNumX, threadNumY}, indicesOffsetBase_, curCoreIndices_, threadNumY, innerSize, segmentNum_, threadNumX,
                                                                   (__local_mem__ float*) (tmpLocal.GetPhyAddr()), (__gm__ X_T*) (xGm_.GetPhyAddr()), (__gm__ volatile X_T*) (yGm_.GetPhyAddr()),
                                                                   (__gm__ OUTTER_T*) (workspaceIndicesOffset_.GetPhyAddr(2 * (segmentNum_ + 1))), (__gm__ SEGMENTIDS_T*) (segmentIdsGm_.GetPhyAddr()),
                                                                   (__gm__ LOCATION_T*) (locationGm_.GetPhyAddr()), (__gm__ float*) (workspaceWeight_.GetPhyAddr(segmentNum_ + 1)),
                                                                   static_cast<uint32_t>(tilingData_->outputDim0));
}


}  // namespace SparseSegmentMeanGradNameSpace
#endif  // SORTED_SPARSE_SEGMENT_MEAN_GRAD_SIMT_SMALL_INNER_H