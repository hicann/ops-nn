/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNIQUE_SORT_SMALL_AXIS_INSERTION_H
#define UNIQUE_SORT_SMALL_AXIS_INSERTION_H

#include "basic_api/kernel_vec_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "unique_sort_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "../../../kth_value/arch35/common/kth_value_util_type_simd.h"
#include "../../../kth_value/arch35/common/kth_value_small_axis_insertion_base.h"

namespace UniqueSort {
using namespace AscendC;

template <typename T, typename OUT_IDX_T>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::INSERTION_THREAD_NUM) __aicore__
    void SimtStoreNonLastInsertionBatch(uint32_t totalElems, uint32_t validSegs, uint32_t segmentLen,
                                        uint32_t valueRowElems, uint32_t indexRowElems, uint64_t outerBaseOffset,
                                        uint64_t innerStart, uint64_t innerSize, __ubuf__ T* inputValue,
                                        __ubuf__ OUT_IDX_T* inputIdx, __gm__ volatile T* outputValue,
                                        __gm__ volatile OUT_IDX_T* outputIdx)
{
    // Store sorted sort-major segments back to the original non-last-axis GM layout.
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < totalElems;
         idx += SmallAxisCommon::INSERTION_THREAD_NUM) {
        uint32_t axis = idx / validSegs;
        uint32_t seg = idx - axis * validSegs;
        uint64_t gmOffset = outerBaseOffset + static_cast<uint64_t>(axis) * innerSize + innerStart + seg;
        outputValue[gmOffset] = inputValue[seg * valueRowElems + axis];
        outputIdx[gmOffset] = inputIdx[seg * indexRowElems + axis];
    }
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
class SortSmallAxisInsertion
    : public SmallAxisCommon::SmallAxisInsertionBase<SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>, T,
                                                     CONVERT_TYPE, OUT_IDX_T, IsDescend, true> {
    using Base = SmallAxisCommon::SmallAxisInsertionBase<SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>,
                                                         T, CONVERT_TYPE, OUT_IDX_T, IsDescend, true>;

public:
    __aicore__ inline SortSmallAxisInsertion() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace,
                                const UniqueSortRegBaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process() { Base::Process(); }

    friend Base;

private:
    using Base::batchNum_;
    using Base::blockDim_;
    using Base::blockIdx_;
    using Base::castBuf_;
    using Base::castRowStride_;
    using Base::indexRowStride_;
    using Base::indices_;
    using Base::pipe_;
    using Base::segmentLen_;
    using Base::segmentsPerBatch_;
    using Base::valueRowStride_;
    using Base::values_;

    // Main pipeline functions (in call order)
    __aicore__ inline bool IsProcessInvalid() const;
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline void ProcessBatch(uint32_t batchId, uint32_t validSegs);
    __aicore__ inline void StoreBatch(int64_t segStart, uint32_t validSegs);
    __aicore__ inline void StoreNonLastBatch(uint64_t outerId, uint64_t innerStart, uint32_t validSegs);

    const UniqueSortRegBaseTilingData* tilingData_ = nullptr;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<OUT_IDX_T> outIdxGm_;

    int64_t totalSegs_ = 0;
    int64_t outerSize_ = 1;
    int64_t innerSize_ = 1;
    uint32_t innerLoopNum_ = 1;
    bool isNonLastAxis_ = false;
};

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace, const UniqueSortRegBaseTilingData* tilingData, TPipe* pipe)
{
    (void)workspace;
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    segmentLen_ = tilingData_->numTileDataSize;
    segmentsPerBatch_ = tilingData_->keyParams0;
    batchNum_ = tilingData_->keyParams1;
    totalSegs_ = tilingData_->unsortedDimNum;
    isNonLastAxis_ = tilingData_->keyParams3 != 0U;
    outerSize_ = tilingData_->outerSize;
    innerSize_ = tilingData_->innerSize;
    innerLoopNum_ = tilingData_->innerLoopNum;
    inputXGm_.SetGlobalBuffer((__gm__ T*)x);
    outValueGm_.SetGlobalBuffer((__gm__ T*)y);
    outIdxGm_.SetGlobalBuffer((__gm__ OUT_IDX_T*)idx);

    if (segmentLen_ == 0 || segmentsPerBatch_ == 0 || segmentsPerBatch_ > SmallAxisCommon::MAX_DATACOPY_BLOCK_COUNT) {
        return;
    }

    Base::InitInsertionBuffers(pipe);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline bool SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::IsProcessInvalid() const
{
    return blockIdx_ >= blockDim_ || segmentLen_ == 0 || segmentsPerBatch_ == 0 ||
           (isNonLastAxis_ && (innerLoopNum_ == 0 || outerSize_ <= 0 || innerSize_ <= 0));
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline uint32_t SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::ComputeValidSegs(
    uint32_t batchId) const
{
    if (isNonLastAxis_) {
        // Non-last batching iterates inner tiles inside each outer slice; the tail tile
        // may contain fewer segments than the configured batch size.
        uint32_t innerTileId = batchId % innerLoopNum_;
        int64_t innerStart = static_cast<int64_t>(innerTileId) * static_cast<int64_t>(segmentsPerBatch_);
        int64_t innerRemain = innerSize_ - innerStart;
        if (innerRemain <= 0) {
            return 0;
        }
        return innerRemain >= static_cast<int64_t>(segmentsPerBatch_) ? segmentsPerBatch_ :
                                                                        static_cast<uint32_t>(innerRemain);
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

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::ProcessBatch(uint32_t batchId,
                                                                                                   uint32_t validSegs)
{
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(segmentsPerBatch_);
    uint64_t outerId = 0;
    uint64_t innerStart = 0;
    if (isNonLastAxis_) {
        // batchId is linearized as [outerId, innerTileId] for non-last-axis work.
        outerId = static_cast<uint64_t>(batchId / innerLoopNum_);
        uint32_t innerTileId = batchId % innerLoopNum_;
        innerStart = static_cast<uint64_t>(innerTileId) * static_cast<uint64_t>(segmentsPerBatch_);
        uint64_t outerBaseOffset = outerId * static_cast<uint64_t>(segmentLen_) * static_cast<uint64_t>(innerSize_);
        Base::LoadNonLastBatch(inputXGm_, outerBaseOffset, innerStart, static_cast<uint64_t>(innerSize_), validSegs);
    } else {
        Base::LoadContiguousBatch(inputXGm_, segStart * static_cast<int64_t>(segmentLen_), validSegs, false);
    }
    Base::SortBatch(validSegs);
    if (isNonLastAxis_) {
        StoreNonLastBatch(outerId, innerStart, validSegs);
    } else {
        StoreBatch(segStart, validSegs);
    }
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::StoreNonLastBatch(
    uint64_t outerId, uint64_t innerStart, uint32_t validSegs)
{
    uint32_t totalElems = validSegs * segmentLen_;
    uint64_t outerBaseOffset = outerId * static_cast<uint64_t>(segmentLen_) * static_cast<uint64_t>(innerSize_);
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        // Convert widened sorted values back to the output storage dtype before SIMT GM stores.
        LocalTensor<T> storeBuf = castBuf_.template Get<T>();
        Cast(storeBuf, values_, RoundMode::CAST_RINT, validSegs * valueRowStride_);
        asc_vf_call<SimtStoreNonLastInsertionBatch<T, OUT_IDX_T>>(
            dim3(SmallAxisCommon::INSERTION_THREAD_NUM), totalElems, validSegs, segmentLen_, castRowStride_,
            indexRowStride_, outerBaseOffset, innerStart, static_cast<uint64_t>(innerSize_),
            (__ubuf__ T*)storeBuf.GetPhyAddr(), (__ubuf__ OUT_IDX_T*)indices_.GetPhyAddr(),
            (__gm__ volatile T*)outValueGm_.GetPhyAddr(), (__gm__ volatile OUT_IDX_T*)outIdxGm_.GetPhyAddr());
    } else {
        asc_vf_call<SimtStoreNonLastInsertionBatch<T, OUT_IDX_T>>(
            dim3(SmallAxisCommon::INSERTION_THREAD_NUM), totalElems, validSegs, segmentLen_, valueRowStride_,
            indexRowStride_, outerBaseOffset, innerStart, static_cast<uint64_t>(innerSize_),
            (__ubuf__ T*)values_.GetPhyAddr(), (__ubuf__ OUT_IDX_T*)indices_.GetPhyAddr(),
            (__gm__ volatile T*)outValueGm_.GetPhyAddr(), (__gm__ volatile OUT_IDX_T*)outIdxGm_.GetPhyAddr());
    }
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::StoreBatch(int64_t segStart,
                                                                                                 uint32_t validSegs)
{
    if (validSegs == 0) {
        return;
    }
    // SIMT/VF insertion sort writes values_/indices_; store path consumes those UB buffers.
    event_t e = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);

    int64_t gmOffset = segStart * segmentLen_;
    // For diff-type, valueRowStride_ is intentionally equal to castRowStride_, so the
    // GM-transfer T buffer and the CONVERT_TYPE sort buffer share the same row pitch.
    uint32_t valueSrcStrideBlocks = ((valueRowStride_ - segmentLen_) * sizeof(T)) / UB_BLOCK_SIZE;
    DataCopyExtParams valueCopyParam{static_cast<uint16_t>(validSegs), static_cast<uint32_t>(segmentLen_ * sizeof(T)),
                                     valueSrcStrideBlocks, 0, 0};
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        LocalTensor<T> storeBuf = castBuf_.template Get<T>();
        Cast(storeBuf, values_, RoundMode::CAST_RINT, validSegs * valueRowStride_);
        // Cast writes storeBuf on vector pipe; MTE3 DataCopyPad reads it for GM writeback.
        event_t e = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
        DataCopyPad(outValueGm_[gmOffset], storeBuf, valueCopyParam);
    } else {
        DataCopyPad(outValueGm_[gmOffset], values_, valueCopyParam);
    }

    uint32_t idxStrideBlocks = ((indexRowStride_ - segmentLen_) * sizeof(OUT_IDX_T)) / UB_BLOCK_SIZE;
    DataCopyExtParams idxCopyParam{static_cast<uint16_t>(validSegs),
                                   static_cast<uint32_t>(segmentLen_ * sizeof(OUT_IDX_T)), idxStrideBlocks, 0, 0};
    DataCopyPad(outIdxGm_[gmOffset], indices_, idxCopyParam);

    // The next batch is issued by scalar/control flow after these MTE3 writebacks complete.
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventId);
    WaitFlag<HardEvent::MTE3_S>(eventId);
}

} // namespace UniqueSort

#endif
