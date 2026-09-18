/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_SMALL_AXIS_INSERTION_H
#define KTH_VALUE_SMALL_AXIS_INSERTION_H

#include <type_traits>

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "kth_value_median_utils.h"
#include "kth_value_tiling_data.h"
#include "common/small_axis_insertion_base.h"

namespace KthValue {
using namespace AscendC;

template <typename T, typename CONVERT_TYPE, bool EnableMedian>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::INSERTION_THREAD_NUM) __aicore__
    void SimtStoreKthInsertionBatch(uint32_t validSegs, uint32_t segmentLen, uint32_t kthIndex, uint32_t medianMode,
                                    uint32_t valueRowElems, uint32_t indexRowElems, uint64_t outputStart,
                                    __ubuf__ CONVERT_TYPE* values, __ubuf__ uint32_t* indices,
                                    __gm__ volatile T* outputValue, __gm__ volatile int64_t* outputIndex)
{
    for (uint32_t seg = static_cast<uint32_t>(threadIdx.x); seg < validSegs;
         seg += SmallAxisCommon::INSERTION_THREAD_NUM) {
        uint32_t rowOffset = seg * valueRowElems;
        uint32_t selectedK = kthIndex;
        if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
            selectedK = ResolveMedianKFromSorted(values, rowOffset, segmentLen, kthIndex, medianMode);
        }
        uint32_t valueOffset = rowOffset + selectedK;
        uint32_t indexOffset = seg * indexRowElems + selectedK;
        outputValue[outputStart + seg] = static_cast<T>(values[valueOffset]);
        outputIndex[outputStart + seg] = static_cast<int64_t>(indices[indexOffset]);
    }
}

template <typename T, typename CONVERT_TYPE, bool EnableMedian = false>
class KthValueSmallAxisInsertion
    : public SmallAxisCommon::SmallAxisInsertionBase<KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>, T,
                                                     CONVERT_TYPE, uint32_t, false, IS_MEDIAN_FLOAT_TYPE<T>> {
    using Base = SmallAxisCommon::SmallAxisInsertionBase<KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>, T,
                                                         CONVERT_TYPE, uint32_t, false, IS_MEDIAN_FLOAT_TYPE<T>>;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const KthValueTilingData* tiling,
                                TPipe* pipe);
    __aicore__ inline void Process() { Base::Process(); }

    friend Base;

private:
    using Base::batchNum_;
    using Base::blockDim_;
    using Base::blockIdx_;
    using Base::indexRowStride_;
    using Base::indices_;
    using Base::pipe_;
    using Base::segmentLen_;
    using Base::segmentsPerBatch_;
    using Base::valueRowStride_;
    using Base::values_;

    __aicore__ inline bool IsProcessInvalid() const;
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline void ProcessBatch(uint32_t batchId, uint32_t validSegs);
    __aicore__ inline bool IsNonLastMode() const;
    __aicore__ inline int64_t GetOutputStart(uint32_t batchId) const;
    __aicore__ inline int64_t GetInputStart(uint32_t batchId) const;
    __aicore__ inline void LoadBatch(uint32_t batchId, uint32_t validSegs);
    __aicore__ inline void StoreKth(int64_t segStart, uint32_t validSegs);

    GlobalTensor<T> inputGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<int64_t> indexGm_;

    const KthValueTilingData* tiling_ = nullptr;

    uint32_t kthIndex_ = 0;
    uint32_t medianMode_ = 0;
    int64_t totalSegs_ = 0;
    int64_t innerSize_ = 1;
    uint32_t innerLoopNum_ = 0;
};

template <typename T, typename CONVERT_TYPE, bool EnableMedian>
__aicore__ inline void KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>::Init(GM_ADDR x, GM_ADDR values,
                                                                                       GM_ADDR indices,
                                                                                       const KthValueTilingData* tiling,
                                                                                       TPipe* pipe)
{
    if (tiling == nullptr || pipe == nullptr) {
        return;
    }
    tiling_ = tiling;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    segmentLen_ = tiling_->numTileDataSize;
    segmentsPerBatch_ = tiling_->keyParams0;
    batchNum_ = tiling_->keyParams1;
    kthIndex_ = tiling_->kthIndex;
    medianMode_ = tiling_->medianMode;
    totalSegs_ = tiling_->unsortedDimNum;
    innerSize_ = tiling_->innerSize <= 0 ? 1 : tiling_->innerSize;
    innerLoopNum_ = tiling_->innerLoopNum;

    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
    indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));

    if (segmentLen_ == 0U || segmentsPerBatch_ == 0U || segmentsPerBatch_ > SmallAxisCommon::MAX_DATACOPY_BLOCK_COUNT) {
        return;
    }

    Base::InitInsertionBuffers(pipe);
}

template <typename T, typename CONVERT_TYPE, bool EnableMedian>
__aicore__ inline bool KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>::IsProcessInvalid() const
{
    return blockIdx_ >= blockDim_ || segmentLen_ == 0U || segmentsPerBatch_ == 0U ||
           (IsNonLastMode() && innerSize_ <= 0);
}

template <typename T, typename CONVERT_TYPE, bool EnableMedian>
__aicore__ inline uint32_t KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>::ComputeValidSegs(
    uint32_t batchId) const
{
    if (IsNonLastMode()) {
        uint32_t innerTileId = batchId % innerLoopNum_;
        int64_t innerStart = static_cast<int64_t>(innerTileId) * static_cast<int64_t>(segmentsPerBatch_);
        int64_t remain = innerSize_ - innerStart;
        if (remain <= 0) {
            return 0;
        }
        return remain >= static_cast<int64_t>(segmentsPerBatch_) ? segmentsPerBatch_ : static_cast<uint32_t>(remain);
    }
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(segmentsPerBatch_);
    int64_t segRemain = totalSegs_ - segStart;
    if (segRemain <= 0) {
        return 0;
    }
    if (segRemain >= static_cast<int64_t>(segmentsPerBatch_)) {
        return segmentsPerBatch_;
    }
    return static_cast<uint32_t>(segRemain);
}

template <typename T, typename CONVERT_TYPE, bool EnableMedian>
__aicore__ inline void KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>::ProcessBatch(uint32_t batchId,
                                                                                               uint32_t validSegs)
{
    int64_t segStart = GetOutputStart(batchId);
    LoadBatch(batchId, validSegs);
    Base::SortBatch(validSegs);
    StoreKth(segStart, validSegs);
}

template <typename T, typename CONVERT_TYPE, bool EnableMedian>
__aicore__ inline bool KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>::IsNonLastMode() const
{
    return innerLoopNum_ != 0U;
}

template <typename T, typename CONVERT_TYPE, bool EnableMedian>
__aicore__ inline int64_t KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>::GetOutputStart(
    uint32_t batchId) const
{
    if (!IsNonLastMode()) {
        return static_cast<int64_t>(batchId) * static_cast<int64_t>(segmentsPerBatch_);
    }
    int64_t outerId = static_cast<int64_t>(batchId / innerLoopNum_);
    int64_t innerTileId = static_cast<int64_t>(batchId % innerLoopNum_);
    return outerId * innerSize_ + innerTileId * static_cast<int64_t>(segmentsPerBatch_);
}

template <typename T, typename CONVERT_TYPE, bool EnableMedian>
__aicore__ inline int64_t KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>::GetInputStart(
    uint32_t batchId) const
{
    if (!IsNonLastMode()) {
        return static_cast<int64_t>(batchId) * static_cast<int64_t>(segmentsPerBatch_) *
               static_cast<int64_t>(segmentLen_);
    }
    int64_t outerId = static_cast<int64_t>(batchId / innerLoopNum_);
    int64_t innerTileId = static_cast<int64_t>(batchId % innerLoopNum_);
    int64_t innerStart = innerTileId * static_cast<int64_t>(segmentsPerBatch_);
    return outerId * static_cast<int64_t>(segmentLen_) * innerSize_ + innerStart;
}

template <typename T, typename CONVERT_TYPE, bool EnableMedian>
__aicore__ inline void KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>::LoadBatch(uint32_t batchId,
                                                                                            uint32_t validSegs)
{
    if (IsNonLastMode()) {
        uint64_t outerId = static_cast<uint64_t>(batchId / innerLoopNum_);
        uint64_t innerTileId = static_cast<uint64_t>(batchId % innerLoopNum_);
        uint64_t innerStart = innerTileId * static_cast<uint64_t>(segmentsPerBatch_);
        uint64_t outerBaseOffset = outerId * static_cast<uint64_t>(segmentLen_) * static_cast<uint64_t>(innerSize_);
        Base::LoadNonLastBatch(inputGm_, outerBaseOffset, innerStart, static_cast<uint64_t>(innerSize_), validSegs);
    } else {
        Base::LoadContiguousBatch(inputGm_, GetInputStart(batchId), validSegs, true);
    }
}

template <typename T, typename CONVERT_TYPE, bool EnableMedian>
__aicore__ inline void KthValueSmallAxisInsertion<T, CONVERT_TYPE, EnableMedian>::StoreKth(int64_t segStart,
                                                                                           uint32_t validSegs)
{
    if (validSegs == 0U) {
        return;
    }
    asc_vf_call<SimtStoreKthInsertionBatch<T, CONVERT_TYPE, EnableMedian>>(
        dim3(SmallAxisCommon::INSERTION_THREAD_NUM), validSegs, segmentLen_, kthIndex_, medianMode_, valueRowStride_,
        indexRowStride_, static_cast<uint64_t>(segStart),
        reinterpret_cast<__ubuf__ CONVERT_TYPE*>(values_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(indices_.GetPhyAddr()), (__gm__ volatile T*)valueGm_.GetPhyAddr(),
        (__gm__ volatile int64_t*)indexGm_.GetPhyAddr());
    event_t eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

} // namespace KthValue

#endif
