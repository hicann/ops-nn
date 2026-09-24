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
 * \file sort_merge_intra_core.h
 * \brief Intra-core block merge sort for fp32
 * \details Each core independently sorts blocks and merges them via 4-way MrgSort.
 *          Two or three blocks remain in UB through the final merge and output.
 *          Larger rows use workspace without cross-core merge synchronization.
 *          Handles fp32, N > 4096, blocksPerRow <= 256.
 */

#ifndef UNIQUE_SORT_MERGE_INTRA_CORE_H
#define UNIQUE_SORT_MERGE_INTRA_CORE_H

#include <cmath>
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "unique_sort_tiling_data.h"
#include "../../../kth_value/arch35/common/kth_value_merge_sort_constants.h"
#include "../../../kth_value/arch35/common/kth_value_merge_intra_core_base.h"

namespace UniqueSort {
using namespace AscendC;

// Import shared constants from MergeSortConstants namespace
using MergeSortConstants::DEALING_EXTRACT_NUM_ONCE;
using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::MERGE_INTRA_BUFFER_NUM;
using MergeSortConstants::MERGE_LIST_MAX_NUM;

/**
 * @brief Intra-core block merge sort: independent per-core sort+merge without inter-core coordination
 * @tparam ValueType Input data type (float)
 * @tparam IndexType Index data type (int32_t or int64_t)
 * @tparam IsDescend Sort order: true for descending, false for ascending
 */
template <typename ValueType, typename IndexType, bool IsDescend>
class SortMergeIntraCore
    : public MergeIntraCoreCommon::MergeIntraCoreBase<SortMergeIntraCore<ValueType, IndexType, IsDescend>, ValueType,
                                                      IndexType, IsDescend> {
    using Base = MergeIntraCoreCommon::MergeIntraCoreBase<SortMergeIntraCore<ValueType, IndexType, IsDescend>,
                                                          ValueType, IndexType, IsDescend>;
    friend Base;

public:
    using MergeListContext = MergeIntraCoreCommon::MergeListContext;

    __aicore__ inline SortMergeIntraCore() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR indices, GM_ADDR workspace,
                                const UniqueSortRegBaseTilingData* tilingData, TPipe* pipe);

    __aicore__ inline void Process()
    {
        if (this->blockSortSize_ == 0U) {
            return;
        }
        if (this->blocksPerRow_ < SORT_RESIDENT_MERGE_MIN_BLOCKS ||
            this->blocksPerRow_ > SORT_RESIDENT_MERGE_MAX_BLOCKS) {
            Base::Process();
            return;
        }
        int64_t first = static_cast<int64_t>(this->blockIdx_) * this->batchPerCore_;
        int64_t end = first + this->batchPerCore_ < this->batchNum_ ? first + this->batchPerCore_ : this->batchNum_;
        for (int64_t row = first; row < end; ++row) {
            ProcessResidentBatch(row);
        }
    }

protected:
    __aicore__ inline void InitPhase1Buffers();
    __aicore__ inline void InitPhase2Buffers();
    __aicore__ inline void InitPhase3Buffers();
    __aicore__ inline void PrepareInputForSort(LocalTensor<ValueType>, uint32_t) {}
    __aicore__ inline void ExtractAndCopyChunk(int64_t cacheBatchOffset, uint32_t cacheOffset, int64_t outputOffset,
                                               uint32_t elemProcessed, uint32_t elemCount);

private:
    __aicore__ inline void ProcessResidentBatch(int64_t row);
    __aicore__ inline void CopyResidentOutputs(int64_t row, LocalTensor<ValueType> proposal);
    // Phase 3 only
    __aicore__ inline void ExtractAndCopyOut(int64_t batchIdx, uint32_t resultRegion);
};

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeIntraCore<ValueType, IndexType, IsDescend>::Init(
    GM_ADDR x, GM_ADDR value, GM_ADDR indices, GM_ADDR workspace, const UniqueSortRegBaseTilingData* tilingData,
    TPipe* pipe)
{
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    this->blockIdx_ = GetBlockIdx();
    this->pipe_ = pipe;

    // Parse tiling data
    this->batchNum_ = tilingData->unsortedDimNum;
    this->sortAxisNum_ = tilingData->lastAxisNum;
    this->batchPerCore_ = tilingData->keyParams0;
    this->blockSortSize_ = tilingData->numTileDataSize;
    this->extractChunkSize_ = tilingData->keyParams4;
    this->blocksPerRow_ = tilingData->lastDimTileNum;
    this->maxCoreNum_ = tilingData->lastDimNeedCore;
    this->alignNum_ = tilingData->keyParams3;
    this->maxMergeIterations_ = tilingData->keyParams5;

    if (this->blockSortSize_ == 0 || this->extractChunkSize_ == 0 || this->blocksPerRow_ == 0 ||
        this->sortAxisNum_ <= 0) {
        return;
    }

    // Precompute constants
    this->blockSortLen_ = AscendC::GetSortLen<ValueType>(this->blockSortSize_);
    this->batchSortLen_ = AscendC::GetSortLen<ValueType>(this->alignNum_);
    this->sortBufferSize_ = this->blockSortLen_ * sizeof(ValueType);
    this->sortRepeatTimes_ = this->blockSortSize_ / DEALING_SORT_NUM_ONCE;
    this->lastBlockSize_ = static_cast<uint32_t>(this->sortAxisNum_ -
                                                 static_cast<int64_t>(this->blocksPerRow_ - 1) * this->blockSortSize_);

    // Set GM buffers
    this->inputXGm_.SetGlobalBuffer((__gm__ ValueType*)x);
    this->outValueGm_.SetGlobalBuffer((__gm__ ValueType*)value);
    this->outIdxGm_.SetGlobalBuffer((__gm__ IndexType*)indices);

    // Cache stores sort struct data (8 bytes per element: index + value)
    // Each core has its own cache region, reused across batches
    // perBatchCacheLen: sort struct length for one batch (with ping-pong, in ValueType units)
    int64_t perCoreCacheLen = static_cast<int64_t>(this->batchSortLen_) * 2; // ping-pong, reused per batch

    this->cacheGm_.SetGlobalBuffer((__gm__ ValueType*)workspace +
                                   static_cast<int64_t>(this->blockIdx_) * perCoreCacheLen);

    // Note: Queue/Buffer initialization is deferred to base Process() per phase
    // to optimize UB usage. Each phase initializes only what it needs.
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeIntraCore<ValueType, IndexType, IsDescend>::InitPhase1Buffers()
{
    this->pipe_->InitBuffer(this->inQueueX_, MERGE_INTRA_BUFFER_NUM, this->blockSortSize_ * sizeof(ValueType));
    this->pipe_->InitBuffer(this->sortTmpBuf_, this->sortBufferSize_);
    this->pipe_->InitBuffer(this->sortedOutQueue_, MERGE_INTRA_BUFFER_NUM, this->sortBufferSize_);
    this->pipe_->InitBuffer(this->indexTmpBuf_, this->blockSortSize_ * sizeof(uint32_t));
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeIntraCore<ValueType, IndexType, IsDescend>::InitPhase2Buffers()
{
    uint32_t mergeBufferSize = MERGE_LIST_MAX_NUM * this->sortBufferSize_;
    this->pipe_->InitBuffer(this->mergeInQueue_, 1, mergeBufferSize);
    this->pipe_->InitBuffer(this->mergeOutQueue_, 1, mergeBufferSize);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeIntraCore<ValueType, IndexType, IsDescend>::InitPhase3Buffers()
{
    uint32_t extractInSize = AscendC::GetSortLen<ValueType>(this->extractChunkSize_) * sizeof(ValueType);
    this->pipe_->InitBuffer(this->extractInQueue_, MERGE_INTRA_BUFFER_NUM, extractInSize);
    this->pipe_->InitBuffer(this->outValueQueue_, MERGE_INTRA_BUFFER_NUM, this->extractChunkSize_ * sizeof(ValueType));
    this->pipe_->InitBuffer(this->outIdxQueue_, MERGE_INTRA_BUFFER_NUM, this->extractChunkSize_ * sizeof(uint32_t));
    if constexpr (IsSameType<int64_t, IndexType>::value) {
        this->pipe_->InitBuffer(this->outIdxInt64Queue_, MERGE_INTRA_BUFFER_NUM,
                                this->extractChunkSize_ * sizeof(int64_t));
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeIntraCore<ValueType, IndexType, IsDescend>::ExtractAndCopyOut(int64_t batchIdx,
                                                                                              uint32_t resultRegion)
{
    int64_t outputOffset = batchIdx * this->sortAxisNum_;

    // resultRegion: 0 = Ping (offset 0), 1 = Pong (offset batchSortLen_)
    int64_t cacheBatchOffset = (resultRegion == 1) ? this->batchSortLen_ : 0;

    uint32_t elemProcessed = 0;
    uint32_t cacheOffset = 0;

    while (elemProcessed < this->sortAxisNum_) {
        uint32_t elemCount = (elemProcessed + this->extractChunkSize_ <= this->sortAxisNum_) ?
                                 this->extractChunkSize_ :
                                 (this->sortAxisNum_ - elemProcessed);
        if (elemCount == 0)
            break;

        ExtractAndCopyChunk(cacheBatchOffset, cacheOffset, outputOffset, elemProcessed, elemCount);

        elemProcessed += elemCount;
        cacheOffset += AscendC::GetSortLen<ValueType>(elemCount);
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeIntraCore<ValueType, IndexType, IsDescend>::ExtractAndCopyChunk(
    int64_t cacheBatchOffset, uint32_t cacheOffset, int64_t outputOffset, uint32_t elemProcessed, uint32_t elemCount)
{
    LocalTensor<ValueType> cacheLocal = this->extractInQueue_.template AllocTensor<ValueType>();
    DataCopyExtParams loadParams{
        1, static_cast<uint32_t>(AscendC::GetSortLen<ValueType>(elemCount) * sizeof(ValueType)), 0, 0, 0};
    DataCopyPad(cacheLocal, this->cacheGm_[cacheBatchOffset + cacheOffset], loadParams, {false, 0, 0, 0});
    this->extractInQueue_.EnQue(cacheLocal);

    cacheLocal = this->extractInQueue_.template DeQue<ValueType>();
    LocalTensor<ValueType> valueLocal = this->outValueQueue_.template AllocTensor<ValueType>();
    LocalTensor<uint32_t> indexLocal = this->outIdxQueue_.template AllocTensor<uint32_t>();
    Extract(valueLocal, indexLocal, cacheLocal, Ops::Base::CeilDiv(elemCount, DEALING_EXTRACT_NUM_ONCE));

    // Flip back sign bit for ascending order (was flipped in SortBlockToStruct)
    if constexpr (!IsDescend) {
        Adds(valueLocal.template ReinterpretCast<int32_t>(), valueLocal.template ReinterpretCast<int32_t>(), 0x80000000,
             elemCount);
    }

    this->outValueQueue_.EnQue(valueLocal);
    this->outIdxQueue_.EnQue(indexLocal);
    this->extractInQueue_.FreeTensor(cacheLocal);
    valueLocal = this->outValueQueue_.template DeQue<ValueType>();
    indexLocal = this->outIdxQueue_.template DeQue<uint32_t>();

    DataCopyExtParams outParams{1, static_cast<uint32_t>(elemCount * sizeof(ValueType)), 0, 0, 0};
    DataCopyPad(this->outValueGm_[outputOffset + elemProcessed], valueLocal, outParams);

    LocalTensor<int32_t> indexInt32 = indexLocal.template ReinterpretCast<int32_t>();
    if constexpr (IsSameType<int64_t, IndexType>::value) {
        LocalTensor<int64_t> indexInt64 = this->outIdxInt64Queue_.template AllocTensor<int64_t>();
        Cast(indexInt64, indexInt32, RoundMode::CAST_NONE, Ops::Base::CeilAlign(elemCount, 4u));
        this->outIdxInt64Queue_.EnQue(indexInt64);
        indexInt64 = this->outIdxInt64Queue_.template DeQue<int64_t>();
        outParams.blockLen = static_cast<uint32_t>(elemCount * sizeof(int64_t));
        DataCopyPad(this->outIdxGm_[outputOffset + elemProcessed], indexInt64, outParams);
        this->outIdxInt64Queue_.FreeTensor(indexInt64);
    } else {
        outParams.blockLen = static_cast<uint32_t>(elemCount * sizeof(int32_t));
        DataCopyPad(this->outIdxGm_[outputOffset + elemProcessed], indexInt32, outParams);
    }
    this->outIdxQueue_.FreeTensor(indexLocal);
    this->outValueQueue_.FreeTensor(valueLocal);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeIntraCore<ValueType, IndexType, IsDescend>::ProcessResidentBatch(int64_t row)
{
    // Two proposal arrays plus one block each of input, temporary proposals and source indices
    // use (16 * blocksPerRow + 16) bytes per block element, at most the existing 64-byte budget.
    this->pipe_->InitBuffer(this->inQueueX_, 1, this->blockSortSize_ * sizeof(ValueType));
    this->pipe_->InitBuffer(this->sortTmpBuf_, this->sortBufferSize_);
    this->pipe_->InitBuffer(this->indexTmpBuf_, this->blockSortSize_ * sizeof(uint32_t));
    this->pipe_->InitBuffer(this->sortedOutQueue_, 1, this->blocksPerRow_ * this->sortBufferSize_);
    this->pipe_->InitBuffer(this->mergeOutQueue_, 1, this->blocksPerRow_ * this->sortBufferSize_);
    LocalTensor<ValueType> sortedBlocks = this->sortedOutQueue_.template AllocTensor<ValueType>();
    uint16_t counts[MERGE_LIST_MAX_NUM] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < this->blocksPerRow_; ++i) {
        counts[i] = i + 1U == this->blocksPerRow_ ? this->lastBlockSize_ : this->blockSortSize_;
        LocalTensor<ValueType> block = this->inQueueX_.template AllocTensor<ValueType>();
        this->CopyInBlock(this->inputXGm_[row * this->sortAxisNum_], block, i * this->blockSortSize_, counts[i]);
        this->inQueueX_.EnQue(block);
        block = this->inQueueX_.template DeQue<ValueType>();
        this->SortBlockToStruct(block, sortedBlocks[i * this->blockSortLen_], counts[i], i * this->blockSortSize_);
        this->inQueueX_.FreeTensor(block);
    }
    LocalTensor<ValueType> proposal = this->mergeOutQueue_.template AllocTensor<ValueType>();
    LocalTensor<ValueType> lists[MERGE_LIST_MAX_NUM];
    for (uint32_t i = 0; i < MERGE_LIST_MAX_NUM; ++i) {
        lists[i] = sortedBlocks[i < this->blocksPerRow_ ? i * this->blockSortLen_ : 0U];
    }
    MrgSortSrcList<ValueType> sources(lists[0], lists[1], lists[2], lists[3]);
    uint32_t consumed[MERGE_LIST_MAX_NUM];
    MrgSort<ValueType, false>(proposal, sources, counts, consumed,
                              static_cast<uint16_t>((1U << this->blocksPerRow_) - 1U), 1);
    CopyResidentOutputs(row, proposal);
    this->sortedOutQueue_.FreeTensor(sortedBlocks);
    this->mergeOutQueue_.FreeTensor(proposal);
    event_t flushed = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(flushed);
    WaitFlag<HardEvent::MTE3_MTE2>(flushed);
    this->pipe_->Reset();
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void SortMergeIntraCore<ValueType, IndexType, IsDescend>::CopyResidentOutputs(
    int64_t row, LocalTensor<ValueType> proposal)
{
    // Sorting is complete: reuse input, source-index and temporary-proposal buffers for outputs.
    LocalTensor<ValueType> values = this->inQueueX_.template AllocTensor<ValueType>();
    LocalTensor<uint32_t> indices = this->indexTmpBuf_.template Get<uint32_t>();
    LocalTensor<int64_t> wideIndices = this->sortTmpBuf_.template Get<int64_t>();
    for (uint32_t offset = 0; offset < this->sortAxisNum_; offset += this->blockSortSize_) {
        uint32_t count = this->sortAxisNum_ - offset < this->blockSortSize_ ? this->sortAxisNum_ - offset :
                                                                              this->blockSortSize_;
        Extract(values, indices, proposal[GetSortLen<ValueType>(offset)],
                Ops::Base::CeilDiv(count, DEALING_EXTRACT_NUM_ONCE));
        if constexpr (!IsDescend) {
            Adds(values.template ReinterpretCast<int32_t>(), values.template ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(MergeSortConstants::XOR_OP_VALUE_FP), count);
        }
        if constexpr (IsSameType<int64_t, IndexType>::value) {
            Cast(wideIndices, indices.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
                 Ops::Base::CeilAlign(count, static_cast<uint32_t>(Ops::Base::GetUbBlockSize() / sizeof(int64_t))));
        }
        event_t ready = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(ValueType)), 0, 0, 0};
        DataCopyPad(this->outValueGm_[row * this->sortAxisNum_ + offset], values, copy);
        copy.blockLen = count * sizeof(IndexType);
        if constexpr (IsSameType<int64_t, IndexType>::value) {
            DataCopyPad(this->outIdxGm_[row * this->sortAxisNum_ + offset], wideIndices, copy);
        } else {
            DataCopyPad(this->outIdxGm_[row * this->sortAxisNum_ + offset], indices.template ReinterpretCast<int32_t>(),
                        copy);
        }
        event_t reused = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(reused);
        WaitFlag<HardEvent::MTE3_V>(reused);
    }
    this->inQueueX_.FreeTensor(values);
}

} // namespace UniqueSort

#endif // SORT_MERGE_INTRA_CORE_H
