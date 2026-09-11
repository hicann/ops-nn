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
 * \file merge_sort_simd.h
 * \brief merge_sort kernel entry
 */
#ifndef KTH_VALUE_MERGE_SORT_MORE_CORE_H
#define KTH_VALUE_MERGE_SORT_MORE_CORE_H
#include <cmath>
#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "kth_value_median_utils.h"
#include "kth_value_tiling_data.h"
#include "common/merge_sort_constants.h"
#include "common/merge_more_core_base.h"

namespace KthValue {
using namespace AscendC;

// Import shared constants from MergeSortConstants namespace
using MergeSortConstants::DEALING_CONCAT_NUM_ONCE;
using MergeSortConstants::DEALING_EXTRACT_NUM_ONCE;
using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::MERGE_LIST_MAX_NUM;
using MergeSortConstants::MERGE_MORE_BUFFER_NUM;
using MergeSortConstants::MERGE_SORT_WORKSPACE_PARAM;
using MergeSortConstants::UB_BLOCK_BYTES;
using MergeSortConstants::XOR_OP_VALUE_FP;
using MergeSortConstants::XOR_OP_VALUE_HALF;

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool EnableMedian = false>
struct KthValueMergeSortMoreCore : public MergeMoreCoreCommon::MergeMoreCoreBase<
                                       KthValueMergeSortMoreCore<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, EnableMedian>,
                                       T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE> {
    using Base = MergeMoreCoreCommon::MergeMoreCoreBase<
        KthValueMergeSortMoreCore<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, EnableMedian>, T, CONVERT_TYPE, IS_DESCEND,
        INDEX_TYPE>;
    friend Base;

    __aicore__ inline KthValueMergeSortMoreCore() {}
    __aicore__ inline void Init(GM_ADDR inputValue, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace,
                                const KthValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void InitMergeBuffers();
    __aicore__ inline void ExtractAndCopyOut();
    __aicore__ inline void OnInputLoaded(LocalTensor<T> inputLocal, uint32_t tileNum);
    __aicore__ inline void PrepareInputForSort(LocalTensor<T> inputLocal, uint32_t alignedCount);
    __aicore__ inline uint32_t ResolveRowK();

    // KthValue-specific member
    uint32_t kthIndex_ = 0;
    uint32_t medianMode_ = 0;
    GlobalTensor<uint32_t> medianCountGm_;
};

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool EnableMedian>
__aicore__ inline void KthValueMergeSortMoreCore<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, EnableMedian>::Init(
    GM_ADDR inputValue, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace, const KthValueTilingData* tilingData,
    TPipe* pipe)
{
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    this->blockIdx_ = GetBlockIdx();
    this->pipe_ = pipe;
    this->outputLastDimValue_ = tilingData->lastDimTileNum;
    this->numTileData_ = tilingData->numTileDataSize;
    this->frontCoreNum_ = tilingData->lastDimNeedCore;
    kthIndex_ = tilingData->kthIndex;
    medianMode_ = tilingData->medianMode;
    if (this->frontCoreNum_ == 0U) {
        return;
    }
    uint32_t sortBufferSize = 8;
    this->rowIdx_ = this->blockIdx_ / this->frontCoreNum_;
    this->rowCoreIdx_ = this->blockIdx_ % this->frontCoreNum_;
    this->rowDataOffset_ = static_cast<int64_t>(this->rowIdx_) * static_cast<int64_t>(this->outputLastDimValue_);
    // Per-row workspace stores Sort API sort-struct data. This capacity uses sortBufferSize bytes per
    // original element and UB-block byte alignment; it must cover later GetSortLen-based accesses.
    uint64_t rowWorkspaceBytes = ROUND_UP_AGLIN_UINT64(static_cast<uint64_t>(this->outputLastDimValue_) *
                                                       sortBufferSize);
    uint64_t rowWorkspaceElements = rowWorkspaceBytes / sizeof(CONVERT_TYPE);
    this->rowWorkspaceOffset_ = static_cast<int64_t>(this->rowIdx_) * static_cast<int64_t>(rowWorkspaceElements) * 2;
    this->onceMaxElements_ = tilingData->keyParams0 / DEALING_SORT_NUM_ONCE * DEALING_SORT_NUM_ONCE;

    this->inputValueGm_.SetGlobalBuffer((__gm__ T*)(inputValue));
    this->outValueGm_.SetGlobalBuffer((__gm__ T*)(value));
    this->outIndexGm_.SetGlobalBuffer((__gm__ INDEX_TYPE*)(indices));
    this->workspaceGm_[0].SetGlobalBuffer((__gm__ CONVERT_TYPE*)(workSpace) + this->rowWorkspaceOffset_,
                                          rowWorkspaceElements);
    this->workspaceGm_[1].SetGlobalBuffer(
        (__gm__ CONVERT_TYPE*)(workSpace) + this->rowWorkspaceOffset_ + rowWorkspaceElements, rowWorkspaceElements);
    uint64_t medianCountOffset = ROUND_UP_AGLIN_UINT64(static_cast<uint64_t>(tilingData->lastAxisNum) *
                                                       static_cast<uint64_t>(tilingData->unsortedDimNum) *
                                                       MERGE_SORT_WORKSPACE_PARAM * sizeof(uint32_t));
    medianCountGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workSpace + medianCountOffset),
                                   tilingData->lastDimNeedCore * tilingData->unsortedDimNum);

    uint32_t tailNum = this->outputLastDimValue_ - (this->frontCoreNum_ - 1) * this->numTileData_;
    uint32_t alignTile = ROUND_UP_AGLIN(tailNum);
    this->pipe_->InitBuffer(this->inputQueue_, MERGE_MORE_BUFFER_NUM, alignTile * sizeof(T));

    this->pipe_->InitBuffer(this->sortedValueUb_, alignTile * sortBufferSize);
    this->pipe_->InitBuffer(this->sortedValueIndexUb_, alignTile * sizeof(uint32_t));
    this->pipe_->InitBuffer(this->concatTempBuf_, alignTile * sortBufferSize);
    this->pipe_->InitBuffer(this->sortTempBuf_, alignTile * sortBufferSize);
    this->pipe_->InitBuffer(this->sortedValueLocalCastTbuf_, alignTile * sortBufferSize);
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool EnableMedian>
__aicore__ inline void KthValueMergeSortMoreCore<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, EnableMedian>::OnInputLoaded(
    LocalTensor<T> inputLocal, uint32_t tileNum)
{
    if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
        LocalTensor<float> scratch = this->concatTempBuf_.template Get<float>();
        uint32_t nonNanCount = CountNonNan(inputLocal, tileNum, scratch, this->pipe_);
        LocalTensor<uint32_t> countLocal = scratch.template ReinterpretCast<uint32_t>();
        countLocal.SetValue(0, nonNanCount);
        event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventId);
        WaitFlag<HardEvent::S_MTE3>(eventId);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(uint32_t)), 0, 0, 0};
        DataCopyPad(medianCountGm_[this->blockIdx_], countLocal, copyParams);
    }
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool EnableMedian>
__aicore__ inline void KthValueMergeSortMoreCore<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE,
                                                 EnableMedian>::PrepareInputForSort(LocalTensor<T> inputLocal,
                                                                                    uint32_t alignedCount)
{
    if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
        if (medianMode_ != MEDIAN_MODE_STATIC) {
            CanonicalizeNanValues(inputLocal, alignedCount);
        }
    }
}

// Resolves the effective k for the current row: gathers the per-core non-NaN counts
// published to the median count workspace by all frontCoreNum_ cores of this row, sums
// them up, then maps the row total to a sorted index via ResolveMedianK.
template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool EnableMedian>
__aicore__ inline uint32_t
KthValueMergeSortMoreCore<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, EnableMedian>::ResolveRowK()
{
    LocalTensor<uint32_t> countLocal = this->concatTempBuf_.template Get<uint32_t>();
    uint32_t countOffset = this->rowIdx_ * this->frontCoreNum_;
    DataCopyExtParams copyParams{1, this->frontCoreNum_ * static_cast<uint32_t>(sizeof(uint32_t)), 0, 0, 0};
    DataCopyPad(countLocal, medianCountGm_[countOffset], copyParams, {false, 0, 0, 0});
    event_t eventId = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
    uint32_t nonNanCount = 0U;
    for (uint32_t core = 0U; core < this->frontCoreNum_; ++core) {
        nonNanCount += countLocal.GetValue(core);
    }
    return ResolveMedianK(kthIndex_, this->outputLastDimValue_, nonNanCount, medianMode_);
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool EnableMedian>
__aicore__ inline void
KthValueMergeSortMoreCore<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, EnableMedian>::InitMergeBuffers()
{
    uint32_t sortBufferSize = 8;
    this->pipe_->InitBuffer(this->sortedQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sortBufferSize);
    this->pipe_->InitBuffer(this->copyInQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sortBufferSize);
    this->pipe_->InitBuffer(this->castValueQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sizeof(CONVERT_TYPE));
    this->pipe_->InitBuffer(this->castIndexQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sizeof(uint32_t));
    this->pipe_->InitBuffer(this->outValueQueue_, 1, UB_BLOCK_BYTES);
    this->pipe_->InitBuffer(this->outIndexQueue_, 1, UB_BLOCK_BYTES);
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE, bool EnableMedian>
__aicore__ inline void
KthValueMergeSortMoreCore<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE, EnableMedian>::ExtractAndCopyOut()
{
    if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
        if (this->outOffset_ == 0) {
            kthIndex_ = ResolveRowK();
        }
    }
    LocalTensor<CONVERT_TYPE> sortTempBuffer = this->sortedQueue_.template DeQue<CONVERT_TYPE>();
    LocalTensor<CONVERT_TYPE> castValue = this->castValueQueue_.template AllocTensor<CONVERT_TYPE>();
    LocalTensor<uint32_t> castIndex = this->castIndexQueue_.template AllocTensor<uint32_t>();
    AscendC::Extract(castValue, castIndex, sortTempBuffer,
                     ((this->curLoopSortedNum_ + DEALING_EXTRACT_NUM_ONCE - 1) / DEALING_EXTRACT_NUM_ONCE));
    if constexpr (!IS_DESCEND) {
        this->FlipSignBit(castValue, ROUND_UP_AGLIN(this->curLoopSortedNum_));
    }
    this->castValueQueue_.EnQue(castValue);
    this->castIndexQueue_.EnQue(castIndex);
    castValue = this->castValueQueue_.template DeQue<CONVERT_TYPE>();
    castIndex = this->castIndexQueue_.template DeQue<uint32_t>();

    uint32_t sortedBase = static_cast<uint32_t>(this->outOffset_);
    uint32_t sortedEnd = sortedBase + static_cast<uint32_t>(this->curLoopSortedNum_);
    // Extract chunks are scanned in sorted order. Only the chunk covering kthIndex_ writes GM output;
    // other chunks are merge intermediates and are discarded for KthValue.
    if (kthIndex_ >= sortedBase && kthIndex_ < sortedEnd) {
        uint32_t localK = kthIndex_ - sortedBase;
        event_t eventIdVToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        LocalTensor<T> outValue = this->outValueQueue_.template AllocTensor<T>();
        LocalTensor<INDEX_TYPE> outIndex = this->outIndexQueue_.template AllocTensor<INDEX_TYPE>();
        outValue.SetValue(0, castValue.GetValue(localK));
        outIndex.SetValue(0, static_cast<INDEX_TYPE>(castIndex.GetValue(localK)));
        this->outValueQueue_.EnQue(outValue);
        this->outIndexQueue_.EnQue(outIndex);

        outValue = this->outValueQueue_.template DeQue<T>();
        outIndex = this->outIndexQueue_.template DeQue<INDEX_TYPE>();
        event_t eventIdSToMte3 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        DataCopyExtParams valueCopyParam{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(this->outValueGm_[this->rowIdx_], outValue, valueCopyParam);
        DataCopyExtParams indexCopyParam{1, static_cast<uint32_t>(sizeof(INDEX_TYPE)), 0, 0, 0};
        DataCopyPad(this->outIndexGm_[this->rowIdx_], outIndex, indexCopyParam);
        event_t eventIdMte3ToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
        this->outValueQueue_.FreeTensor(outValue);
        this->outIndexQueue_.FreeTensor(outIndex);
        this->allRemainElements_ = 0;
    }
    this->castIndexQueue_.FreeTensor(castIndex);
    this->castValueQueue_.FreeTensor(castValue);
    this->sortedQueue_.FreeTensor(sortTempBuffer);
    this->outOffset_ += this->curLoopSortedNum_;
}
} // namespace KthValue
#endif // KTH_VALUE_MERGE_SORT_MORE_CORE_H
