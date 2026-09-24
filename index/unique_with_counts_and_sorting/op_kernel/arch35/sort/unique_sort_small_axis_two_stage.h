/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNIQUE_SORT_SMALL_AXIS_TWO_STAGE_H
#define UNIQUE_SORT_SMALL_AXIS_TWO_STAGE_H

#include "basic_api/kernel_vec_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "unique_sort_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "../../../kth_value/arch35/common/kth_value_small_axis_two_stage_base.h"

namespace UniqueSort {
using namespace AscendC;
constexpr uint32_t NDDMA_DIM = 2;

template <typename T, typename OutIdxT>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::TWO_STAGE_THREAD_NUM) __aicore__
    void StoreNonLastBatchSimt(uint32_t totalElems, uint32_t segmentLen, uint32_t validSegs, uint64_t outerBaseOffset,
                               uint64_t innerStart, uint64_t innerSize, __ubuf__ T* inputValue,
                               __ubuf__ OutIdxT* inputIdx, __gm__ volatile T* outputValue,
                               __gm__ volatile OutIdxT* outputIdx)
{
    // Scatter final sorted rows from [inner segment, axis] order back to original GM offsets.
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < totalElems;
         idx += SmallAxisCommon::TWO_STAGE_THREAD_NUM) {
        uint32_t axis = idx / validSegs;
        uint32_t seg = idx - axis * validSegs;
        uint32_t ubOffset = seg * segmentLen + axis;
        uint64_t gmOffset = outerBaseOffset + static_cast<uint64_t>(axis) * innerSize + innerStart + seg;
        outputValue[gmOffset] = inputValue[ubOffset];
        outputIdx[gmOffset] = inputIdx[ubOffset];
    }
}

template <typename T, typename OutIdxT, bool IsDescend>
class SortSmallAxisTwoStage
    : public SmallAxisCommon::SmallAxisTwoStageBase<SortSmallAxisTwoStage<T, OutIdxT, IsDescend>, T, OutIdxT,
                                                    IsDescend> {
    using Base = SmallAxisCommon::SmallAxisTwoStageBase<SortSmallAxisTwoStage<T, OutIdxT, IsDescend>, T, OutIdxT,
                                                        IsDescend>;

public:
    __aicore__ inline SortSmallAxisTwoStage() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace,
                                const UniqueSortRegBaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

    friend Base;

private:
    using Base::batchNum_;
    using Base::batchSize_;
    using Base::blockDim_;
    using Base::blockIdx_;
    using Base::finalIdx_;
    using Base::finalValues_;
    using Base::maxFlatElems_;
    using Base::segmentLen_;

    __aicore__ inline bool IsProcessInvalid() const;
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline void ProcessBatch(uint32_t batchId, uint32_t validSegs);
    __aicore__ inline void LoadNonLastBatchWithNddma(uint64_t outerBaseOffset, uint64_t innerStart, uint32_t validSegs);
    __aicore__ inline void StoreBatch(int64_t segStart, uint32_t totalElems);
    __aicore__ inline void StoreNonLastBatch(uint64_t outerId, uint64_t innerStart, uint32_t validSegs,
                                             uint32_t totalElems);

    const UniqueSortRegBaseTilingData* tilingData_ = nullptr;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<OutIdxT> outIdxGm_;

    int64_t totalSegs_ = 0;
    int64_t outerSize_ = 1;
    int64_t innerSize_ = 1;
    uint32_t innerLoopNum_ = 1;
    bool isNonLastAxis_ = false;
};

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx,
                                                                          GM_ADDR workspace,
                                                                          const UniqueSortRegBaseTilingData* tilingData,
                                                                          TPipe* pipe)
{
    (void)workspace;
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    batchSize_ = tilingData_->keyParams0;
    batchNum_ = tilingData_->keyParams1;
    segmentLen_ = tilingData_->numTileDataSize;
    maxFlatElems_ = batchSize_ * segmentLen_;
    totalSegs_ = tilingData_->unsortedDimNum;
    isNonLastAxis_ = tilingData_->keyParams3 != 0U;
    outerSize_ = tilingData_->outerSize;
    innerSize_ = tilingData_->innerSize;
    innerLoopNum_ = tilingData_->innerLoopNum;

    inputXGm_.SetGlobalBuffer((__gm__ T*)x);
    outValueGm_.SetGlobalBuffer((__gm__ T*)y);
    outIdxGm_.SetGlobalBuffer((__gm__ OutIdxT*)idx);

    if (batchSize_ == 0 || segmentLen_ == 0 || maxFlatElems_ == 0) {
        return;
    }

    constexpr uint32_t kAliasElemBytes = static_cast<uint32_t>(sizeof(OutIdxT));
    Base::InitSortBuffers(pipe, maxFlatElems_, tilingData_->tmpUbSize, tilingData_->keyParams2 != 0U, kAliasElemBytes);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline bool SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::IsProcessInvalid() const
{
    return blockIdx_ >= blockDim_ || batchSize_ == 0 || segmentLen_ == 0 ||
           (isNonLastAxis_ && (innerLoopNum_ == 0 || outerSize_ <= 0 || innerSize_ <= 0));
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::Process()
{
    // Keep the generic non-last-axis mapping in the common base, but avoid carrying its batch mapping branches
    // through the hot loop for contiguous last-axis sorting.
    if (isNonLastAxis_) {
        Base::Process();
        return;
    }
    if (blockIdx_ >= blockDim_ || batchSize_ == 0 || segmentLen_ == 0) {
        return;
    }
    for (uint32_t batchId = blockIdx_; batchId < batchNum_; batchId += blockDim_) {
        uint32_t validSegs = ComputeValidSegs(batchId);
        if (validSegs == 0U) {
            continue;
        }
        uint32_t totalElems = validSegs * segmentLen_;
        int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
        Base::LoadContiguousBatch(inputXGm_, segStart * static_cast<int64_t>(segmentLen_), totalElems);
        Base::RunTwoStageSort(totalElems);
        StoreBatch(segStart, totalElems);
    }
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline uint32_t SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::ComputeValidSegs(uint32_t batchId) const
{
    if (isNonLastAxis_) {
        // Non-last batches are grouped by outer slice, then by inner tile.
        // The last inner tile can be shorter than batchSize_.
        uint32_t tileInOuter = batchId % innerLoopNum_;
        int64_t tileStart = static_cast<int64_t>(tileInOuter) * static_cast<int64_t>(batchSize_);
        int64_t remainingInTile = innerSize_ - tileStart;
        if (remainingInTile <= 0) {
            return 0;
        }
        uint32_t validInnerSegs = remainingInTile >= static_cast<int64_t>(batchSize_) ?
                                      batchSize_ :
                                      static_cast<uint32_t>(remainingInTile);
        return validInnerSegs;
    }
    int64_t batchStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    int64_t remainingSegs = totalSegs_ - batchStart;
    if (remainingSegs <= 0) {
        return 0;
    }
    if (remainingSegs >= static_cast<int64_t>(batchSize_)) {
        return batchSize_;
    }
    return static_cast<uint32_t>(remainingSegs);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::LoadNonLastBatchWithNddma(uint64_t outerBaseOffset,
                                                                                               uint64_t innerStart,
                                                                                               uint32_t validSegs)
{
    // Gather the strided GM [axis, inner] tile directly into dense UB [inner, axis]. This changes only the load
    // engine: inputValues_ keeps its existing size and lifetime, with no new buffer or alias.
    NdDmaLoopInfo<NDDMA_DIM> loopInfo{
        {1, static_cast<uint64_t>(innerSize_)}, {segmentLen_, 1}, {validSegs, segmentLen_}, {0, 0}, {0, 0}};
    NdDmaParams<T, NDDMA_DIM> params{loopInfo, static_cast<T>(0)};
    NdDmaDci();
    static constexpr NdDmaConfig config;
    DataCopy<T, NDDMA_DIM, config>(this->inputValues_, inputXGm_[outerBaseOffset + innerStart], params);
    // NDDMA/MTE2 produces inputValues_; the following RunTwoStageSort consumes it on the vector pipeline.
    event_t eventIdMte2ToV = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::ProcessBatch(uint32_t batchId, uint32_t validSegs)
{
    uint32_t totalElems = validSegs * segmentLen_;
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    // Pipeline: load -> value sort -> restore per-segment order -> store.
    uint64_t outerId = 0;
    uint64_t innerStart = 0;
    if (isNonLastAxis_) {
        // batchId is linearized as outerId * innerLoopNum_ + innerTileId.
        outerId = static_cast<uint64_t>(batchId / innerLoopNum_);
        uint32_t innerTileId = batchId % innerLoopNum_;
        innerStart = static_cast<uint64_t>(innerTileId) * static_cast<uint64_t>(batchSize_);
        uint64_t outerBaseOffset = outerId * static_cast<uint64_t>(segmentLen_) * static_cast<uint64_t>(innerSize_);
        LoadNonLastBatchWithNddma(outerBaseOffset, innerStart, validSegs);
    } else {
        Base::LoadContiguousBatch(inputXGm_, segStart * static_cast<int64_t>(segmentLen_), totalElems);
    }
    Base::RunTwoStageSort(totalElems);
    if (isNonLastAxis_) {
        StoreNonLastBatch(outerId, innerStart, validSegs, totalElems);
    } else {
        StoreBatch(segStart, totalElems);
    }
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::StoreNonLastBatch(uint64_t outerId,
                                                                                       uint64_t innerStart,
                                                                                       uint32_t validSegs,
                                                                                       uint32_t totalElems)
{
    uint64_t outerBaseOffset = outerId * static_cast<uint64_t>(segmentLen_) * static_cast<uint64_t>(innerSize_);
    // finalValues_/finalIdx_ are already in per-segment order when this store runs.
    asc_vf_call<StoreNonLastBatchSimt<T, OutIdxT>>(
        dim3(SmallAxisCommon::TWO_STAGE_THREAD_NUM), totalElems, segmentLen_, validSegs, outerBaseOffset, innerStart,
        static_cast<uint64_t>(innerSize_), (__ubuf__ T*)finalValues_.GetPhyAddr(),
        (__ubuf__ OutIdxT*)finalIdx_.GetPhyAddr(), (__gm__ volatile T*)outValueGm_.GetPhyAddr(),
        (__gm__ volatile OutIdxT*)outIdxGm_.GetPhyAddr());
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::StoreBatch(int64_t segStart, uint32_t totalElems)
{
    int64_t gmOffset = segStart * static_cast<int64_t>(segmentLen_);
    DataCopyExtParams valueCopyParam{1, static_cast<uint32_t>(totalElems * sizeof(T)), 0, 0, 0};
    DataCopyExtParams idxCopyParam{1, static_cast<uint32_t>(totalElems * sizeof(OutIdxT)), 0, 0, 0};
    // Rank-inverse writes final buffers via SIMT VF; BuildOutputs writes them via vector APIs.
    // Wait for the producing VF/vector work before GM writeback.
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(outValueGm_[gmOffset], finalValues_, valueCopyParam);
    DataCopyPad(outIdxGm_[gmOffset], finalIdx_, idxCopyParam);

    // finalValues_ aliases inputValues_; next LoadBatch uses MTE2 to overwrite that UB.
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    // The next loop iteration is issued by scalar/control flow, so wait on MTE3_S, not MTE3_V.
    event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
}

} // namespace UniqueSort

#endif
