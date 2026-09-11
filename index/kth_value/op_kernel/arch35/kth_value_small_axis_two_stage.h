/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_SMALL_AXIS_TWO_STAGE_H
#define KTH_VALUE_SMALL_AXIS_TWO_STAGE_H

#include <type_traits>

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "kth_value_median_utils.h"
#include "kth_value_tiling_data.h"
#include "common/small_axis_two_stage_base.h"

namespace KthValue {
using namespace AscendC;

// Pre-sort median hook: the two-stage base sorter has no NaN-aware comparator (no OrderNan),
// so canonicalize every value in the batch first to make all NaNs identical and sort as a suffix.
template <typename T>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::TWO_STAGE_THREAD_NUM) __aicore__
    void SimtCanonicalizeMedianSortValues(uint32_t totalElems, __ubuf__ T* values)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < totalElems;
         idx += SmallAxisCommon::TWO_STAGE_THREAD_NUM) {
        values[idx] = CanonicalizeMedianSortValue(values[idx]);
    }
}

// Post-sort median hook: resolves the effective k by binary search on the sorted row, then
// reads the selected value back from the raw GM tensor — the sort buffer was canonicalized,
// so tracing back preserves the original NaN payload (non-median takes the buffer value directly).
template <typename T, bool EnableMedian>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::TWO_STAGE_THREAD_NUM) __aicore__
    void SimtStoreKthTwoStageBatch(uint32_t validSegs, uint32_t segmentLen, uint32_t kthIndex, uint32_t medianMode,
                                   uint64_t outputStart, uint64_t inputStart, uint64_t innerSize, uint32_t nonLastMode,
                                   __ubuf__ T* finalValues, __ubuf__ uint32_t* finalIdx, __gm__ volatile T* outputValue,
                                   __gm__ volatile int64_t* outputIndex, __gm__ volatile T* input)
{
    for (uint32_t seg = static_cast<uint32_t>(threadIdx.x); seg < validSegs;
         seg += SmallAxisCommon::TWO_STAGE_THREAD_NUM) {
        uint32_t rowOffset = seg * segmentLen;
        uint32_t selectedK = kthIndex;
        if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
            selectedK = ResolveMedianKFromSorted(finalValues, rowOffset, segmentLen, kthIndex, medianMode);
        }
        uint32_t srcOffset = rowOffset + selectedK;
        uint32_t selectedIndex = finalIdx[srcOffset];
        uint64_t outputOffset = outputStart + seg;
        if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
            uint64_t inputOffset = inputStart + static_cast<uint64_t>(seg) * segmentLen + selectedIndex;
            if (nonLastMode != 0U) {
                inputOffset = inputStart + static_cast<uint64_t>(selectedIndex) * innerSize +
                              static_cast<uint64_t>(seg);
            }
            outputValue[outputOffset] = input[inputOffset];
        } else {
            outputValue[outputOffset] = finalValues[srcOffset];
        }
        outputIndex[outputOffset] = static_cast<int64_t>(selectedIndex);
    }
}

template <typename T, bool EnableMedian = false>
class KthValueSmallAxisTwoStage
    : public SmallAxisCommon::SmallAxisTwoStageBase<KthValueSmallAxisTwoStage<T, EnableMedian>, T, uint32_t, false> {
    using Base = SmallAxisCommon::SmallAxisTwoStageBase<KthValueSmallAxisTwoStage<T, EnableMedian>, T, uint32_t, false>;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const KthValueTilingData* tiling,
                                TPipe* pipe);

    friend Base;

private:
    using Base::batchNum_;
    using Base::batchSize_;
    using Base::blockDim_;
    using Base::blockIdx_;
    using Base::finalIdx_;
    using Base::finalValues_;
    using Base::inputValues_;
    using Base::maxFlatElems_;
    using Base::pipe_;
    using Base::segmentLen_;

    __aicore__ inline bool IsProcessInvalid() const;
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline void ProcessBatch(uint32_t batchId, uint32_t validSegs);
    __aicore__ inline bool IsNonLastMode() const;
    __aicore__ inline int64_t GetOutputStart(uint32_t batchId) const;
    __aicore__ inline int64_t GetInputStart(uint32_t batchId) const;
    __aicore__ inline void LoadBatch(uint32_t batchId, uint32_t validSegs, uint32_t totalElems);
    __aicore__ inline void StoreKth(uint32_t batchId, int64_t segStart, uint32_t validSegs);

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

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueSmallAxisTwoStage<T, EnableMedian>::Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                                                        const KthValueTilingData* tiling, TPipe* pipe)
{
    if (tiling == nullptr || pipe == nullptr) {
        return;
    }
    pipe_ = pipe;
    tiling_ = tiling;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    batchSize_ = tiling_->keyParams0;
    batchNum_ = tiling_->keyParams1;
    segmentLen_ = tiling_->numTileDataSize;
    maxFlatElems_ = batchSize_ * segmentLen_;
    kthIndex_ = tiling_->kthIndex;
    medianMode_ = tiling_->medianMode;
    totalSegs_ = tiling_->unsortedDimNum;
    innerSize_ = tiling_->innerSize <= 0 ? 1 : tiling_->innerSize;
    innerLoopNum_ = tiling_->innerLoopNum;

    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
    indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));

    if (batchSize_ == 0U || segmentLen_ == 0U || maxFlatElems_ == 0U) {
        return;
    }

    Base::InitSortBuffers(pipe, maxFlatElems_, tiling_->tmpUbSize, tiling_->keyParams2 != 0U, sizeof(uint32_t));
}

template <typename T, bool EnableMedian>
__aicore__ inline bool KthValueSmallAxisTwoStage<T, EnableMedian>::IsProcessInvalid() const
{
    return blockIdx_ >= blockDim_ || batchSize_ == 0U || segmentLen_ == 0U;
}

template <typename T, bool EnableMedian>
__aicore__ inline uint32_t KthValueSmallAxisTwoStage<T, EnableMedian>::ComputeValidSegs(uint32_t batchId) const
{
    if (IsNonLastMode()) {
        uint32_t innerTileId = batchId % innerLoopNum_;
        int64_t innerStart = static_cast<int64_t>(innerTileId) * static_cast<int64_t>(batchSize_);
        int64_t remain = innerSize_ - innerStart;
        if (remain <= 0) {
            return 0;
        }
        return remain >= static_cast<int64_t>(batchSize_) ? batchSize_ : static_cast<uint32_t>(remain);
    }
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    int64_t segRemain = totalSegs_ - segStart;
    if (segRemain <= 0) {
        return 0;
    }
    if (segRemain >= static_cast<int64_t>(batchSize_)) {
        return batchSize_;
    }
    return static_cast<uint32_t>(segRemain);
}

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueSmallAxisTwoStage<T, EnableMedian>::ProcessBatch(uint32_t batchId, uint32_t validSegs)
{
    uint32_t totalElems = validSegs * segmentLen_;
    int64_t segStart = GetOutputStart(batchId);
    LoadBatch(batchId, validSegs, totalElems);
    if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
        asc_vf_call<SimtCanonicalizeMedianSortValues<T>>(dim3(SmallAxisCommon::TWO_STAGE_THREAD_NUM), totalElems,
                                                         reinterpret_cast<__ubuf__ T*>(inputValues_.GetPhyAddr()));
    }
    Base::RunTwoStageSort(totalElems);
    StoreKth(batchId, segStart, validSegs);
}

template <typename T, bool EnableMedian>
__aicore__ inline bool KthValueSmallAxisTwoStage<T, EnableMedian>::IsNonLastMode() const
{
    return innerLoopNum_ != 0U;
}

template <typename T, bool EnableMedian>
__aicore__ inline int64_t KthValueSmallAxisTwoStage<T, EnableMedian>::GetOutputStart(uint32_t batchId) const
{
    if (!IsNonLastMode()) {
        return static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    }
    int64_t outerId = static_cast<int64_t>(batchId / innerLoopNum_);
    int64_t innerTileId = static_cast<int64_t>(batchId % innerLoopNum_);
    return outerId * innerSize_ + innerTileId * static_cast<int64_t>(batchSize_);
}

template <typename T, bool EnableMedian>
__aicore__ inline int64_t KthValueSmallAxisTwoStage<T, EnableMedian>::GetInputStart(uint32_t batchId) const
{
    if (!IsNonLastMode()) {
        return static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_) * static_cast<int64_t>(segmentLen_);
    }
    int64_t outerId = static_cast<int64_t>(batchId / innerLoopNum_);
    int64_t innerTileId = static_cast<int64_t>(batchId % innerLoopNum_);
    int64_t innerStart = innerTileId * static_cast<int64_t>(batchSize_);
    return outerId * static_cast<int64_t>(segmentLen_) * innerSize_ + innerStart;
}

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueSmallAxisTwoStage<T, EnableMedian>::LoadBatch(uint32_t batchId, uint32_t validSegs,
                                                                             uint32_t totalElems)
{
    if (IsNonLastMode()) {
        uint64_t outerId = static_cast<uint64_t>(batchId / innerLoopNum_);
        uint64_t innerTileId = static_cast<uint64_t>(batchId % innerLoopNum_);
        uint64_t innerStart = innerTileId * static_cast<uint64_t>(batchSize_);
        uint64_t outerBaseOffset = outerId * static_cast<uint64_t>(segmentLen_) * static_cast<uint64_t>(innerSize_);
        Base::LoadNonLastBatch(inputGm_, outerBaseOffset, innerStart, static_cast<uint64_t>(innerSize_), validSegs,
                               totalElems);
    } else {
        Base::LoadContiguousBatch(inputGm_, GetInputStart(batchId), totalElems);
    }
}

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueSmallAxisTwoStage<T, EnableMedian>::StoreKth(uint32_t batchId, int64_t segStart,
                                                                            uint32_t validSegs)
{
    event_t eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    uint32_t nonLastMode = IsNonLastMode() ? 1U : 0U;
    asc_vf_call<SimtStoreKthTwoStageBatch<T, EnableMedian>>(
        dim3(SmallAxisCommon::TWO_STAGE_THREAD_NUM), validSegs, segmentLen_, kthIndex_, medianMode_,
        static_cast<uint64_t>(segStart), static_cast<uint64_t>(GetInputStart(batchId)),
        static_cast<uint64_t>(innerSize_), nonLastMode, (__ubuf__ T*)finalValues_.GetPhyAddr(),
        (__ubuf__ uint32_t*)finalIdx_.GetPhyAddr(), (__gm__ volatile T*)valueGm_.GetPhyAddr(),
        (__gm__ volatile int64_t*)indexGm_.GetPhyAddr(), (__gm__ volatile T*)inputGm_.GetPhyAddr());
    eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

} // namespace KthValue

#endif
