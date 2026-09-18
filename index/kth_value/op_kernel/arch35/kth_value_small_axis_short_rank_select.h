/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_SMALL_AXIS_SHORT_RANK_SELECT_H
#define KTH_VALUE_SMALL_AXIS_SHORT_RANK_SELECT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "kth_value_tiling_data.h"

// Top-K partial selection for short axis (<=32) kth_value.
// Maintains a sorted candidate array of size K, scans once per row via SIMT.
namespace KthValue {
using namespace AscendC;

constexpr uint32_t SHORT_RANK_SELECT_THREAD_NUM = 1024;
constexpr uint32_t SHORT_RANK_SELECT_MAX_CANDIDATES = 8;

// Returns true if (value, index) is strictly better than (baseValue, baseIndex).
template <typename T, bool SelectLargest>
__simt_callee__ __aicore__ inline bool IsShortRankSelectBetter(T value, uint32_t index, T baseValue, uint32_t baseIndex)
{
    if constexpr (SelectLargest) {
        return (value > baseValue) || ((value == baseValue) && (index > baseIndex));
    } else {
        return (value < baseValue) || ((value == baseValue) && (index < baseIndex));
    }
}

// Bubble-up: swap candidate at pos with pos-1 if it is better (insertion-sort step).
template <typename T, bool SelectLargest>
__simt_callee__ __aicore__ inline void PromoteShortRankSelectCandidate(T* candidateValues, uint32_t* candidateIndices,
                                                                       uint32_t pos)
{
    T rightValue = candidateValues[pos];
    uint32_t rightIndex = candidateIndices[pos];
    if (!IsShortRankSelectBetter<T, SelectLargest>(rightValue, rightIndex, candidateValues[pos - 1U],
                                                   candidateIndices[pos - 1U])) {
        return;
    }
    candidateValues[pos] = candidateValues[pos - 1U];
    candidateIndices[pos] = candidateIndices[pos - 1U];
    candidateValues[pos - 1U] = rightValue;
    candidateIndices[pos - 1U] = rightIndex;
}

// SIMT kernel: each thread does top-K selection on one row. CandidateNum is compile-time constant.
template <typename T, uint32_t CandidateNum, bool SelectLargest>
__simt_vf__ LAUNCH_BOUND(SHORT_RANK_SELECT_THREAD_NUM) __aicore__
    void SimtSelectShortRankFromUbFixed(uint32_t validSegs, uint32_t segmentLen, __ubuf__ T* input,
                                        __ubuf__ T* outputValue, __ubuf__ int64_t* outputIndex)
{
    // Grid-stride loop: each thread processes rows with stride = thread count
    for (uint32_t row = static_cast<uint32_t>(threadIdx.x); row < validSegs; row += SHORT_RANK_SELECT_THREAD_NUM) {
        T candidateValues[SHORT_RANK_SELECT_MAX_CANDIDATES];
        uint32_t candidateIndices[SHORT_RANK_SELECT_MAX_CANDIDATES];
        uint32_t rowOffset = row * segmentLen;

        // Phase 1: Initialize candidate array with first K elements
        for (uint32_t col = 0U; col < CandidateNum; ++col) {
            candidateValues[col] = input[rowOffset + col];
            candidateIndices[col] = col;
        }
        // Phase 2: Sort initial candidates using insertion sort (candidateValues[0] = best, [K-1] = worst)
        for (uint32_t pass = 1U; pass < CandidateNum; ++pass) {
            for (uint32_t pos = CandidateNum - 1U; pos >= pass; --pos) {
                PromoteShortRankSelectCandidate<T, SelectLargest>(candidateValues, candidateIndices, pos);
            }
        }

        // Phase 3: Scan remaining elements, maintain top-K heap
        for (uint32_t col = CandidateNum; col < segmentLen; ++col) {
            T value = input[rowOffset + col];
            // Skip if not better than the worst candidate (threshold)
            if (!IsShortRankSelectBetter<T, SelectLargest>(value, col, candidateValues[CandidateNum - 1U],
                                                           candidateIndices[CandidateNum - 1U])) {
                continue;
            }

            // Replace worst candidate and bubble up to correct position
            candidateValues[CandidateNum - 1U] = value;
            candidateIndices[CandidateNum - 1U] = col;
            for (uint32_t pos = CandidateNum - 1U; pos > 0U; --pos) {
                PromoteShortRankSelectCandidate<T, SelectLargest>(candidateValues, candidateIndices, pos);
            }
        }

        // Output the K-th element (worst among top-K = the answer)
        outputValue[row] = candidateValues[CandidateNum - 1U];
        outputIndex[row] = static_cast<int64_t>(candidateIndices[CandidateNum - 1U]);
    }
}

// Recursive template dispatch: maps runtime candidateNum to compile-time CandidateNum for loop unrolling.
template <typename T, bool SelectLargest, uint32_t CandidateNum = 1U>
__aicore__ inline void DispatchShortRankSelectFixed(uint32_t candidateNum, uint32_t validSegs, uint32_t segmentLen,
                                                    __ubuf__ T* input, __ubuf__ T* outputValue,
                                                    __ubuf__ int64_t* outputIndex)
{
    if (candidateNum == CandidateNum) {
        asc_vf_call<SimtSelectShortRankFromUbFixed<T, CandidateNum, SelectLargest>>(
            dim3(SHORT_RANK_SELECT_THREAD_NUM), validSegs, segmentLen, input, outputValue, outputIndex);
        return;
    }
    if constexpr (CandidateNum < SHORT_RANK_SELECT_MAX_CANDIDATES) {
        DispatchShortRankSelectFixed<T, SelectLargest, CandidateNum + 1U>(candidateNum, validSegs, segmentLen, input,
                                                                          outputValue, outputIndex);
    }
}

template <typename T>
class KthValueSmallAxisShortRankSelect {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const KthValueTilingData* tiling,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline bool IsProcessInvalid() const;
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline int64_t GetInputStart(uint32_t batchId) const;
    __aicore__ inline void CopyIn(uint32_t batchId, uint32_t validSegs);
    __aicore__ inline void SelectShortRank(uint32_t validSegs, LocalTensor<T>& input, LocalTensor<T>& outputValue,
                                           LocalTensor<int64_t>& outputIndex);
    __aicore__ inline void CopyOut(uint32_t batchId, uint32_t validSegs, LocalTensor<T>& outputValue,
                                   LocalTensor<int64_t>& outputIndex);

    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputValueQueue_;
    TQue<QuePosition::VECOUT, 1> outputIndexQueue_;

    GlobalTensor<T> inputGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<int64_t> indexGm_;
    const KthValueTilingData* tiling_ = nullptr;
    TPipe* pipe_ = nullptr;
    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t batchNum_ = 0;
    uint32_t segmentLen_ = 0;
    uint32_t kthIndex_ = 0;
    int64_t totalSegs_ = 0;
};

template <typename T>
__aicore__ inline void KthValueSmallAxisShortRankSelect<T>::Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                                                 const KthValueTilingData* tiling, TPipe* pipe)
{
    if (tiling == nullptr || pipe == nullptr) {
        return;
    }
    tiling_ = tiling;
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    batchSize_ = tiling_->keyParams0;
    batchNum_ = tiling_->keyParams1;
    segmentLen_ = tiling_->numTileDataSize;
    kthIndex_ = static_cast<uint32_t>(tiling_->kthIndex);
    totalSegs_ = tiling_->unsortedDimNum;

    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
    indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));

    if (batchSize_ == 0U || segmentLen_ == 0U) {
        return;
    }

    pipe_->InitBuffer(inputQueue_, 1, ROUND_UP_AGLIN(batchSize_ * segmentLen_ * sizeof(T)));
    pipe_->InitBuffer(outputValueQueue_, 1, ROUND_UP_AGLIN(batchSize_ * sizeof(T)));
    pipe_->InitBuffer(outputIndexQueue_, 1, ROUND_UP_AGLIN(batchSize_ * sizeof(int64_t)));
}

template <typename T>
__aicore__ inline bool KthValueSmallAxisShortRankSelect<T>::IsProcessInvalid() const
{
    if (tiling_ == nullptr || pipe_ == nullptr || blockIdx_ >= blockDim_ || blockDim_ == 0U || batchSize_ == 0U ||
        batchNum_ == 0U || segmentLen_ == 0U || totalSegs_ <= 0 || segmentLen_ > 32U || kthIndex_ >= segmentLen_) {
        return true;
    }
    uint32_t rankFromHead = kthIndex_ + 1U;
    uint32_t rankFromTail = segmentLen_ - kthIndex_;
    uint32_t candidateNum = rankFromTail < rankFromHead ? rankFromTail : rankFromHead;
    return candidateNum == 0U || candidateNum > SHORT_RANK_SELECT_MAX_CANDIDATES;
}

template <typename T>
__aicore__ inline uint32_t KthValueSmallAxisShortRankSelect<T>::ComputeValidSegs(uint32_t batchId) const
{
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    int64_t remain = totalSegs_ - segStart;
    if (remain <= 0) {
        return 0;
    }
    return remain >= static_cast<int64_t>(batchSize_) ? batchSize_ : static_cast<uint32_t>(remain);
}

template <typename T>
__aicore__ inline int64_t KthValueSmallAxisShortRankSelect<T>::GetInputStart(uint32_t batchId) const
{
    return static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_) * static_cast<int64_t>(segmentLen_);
}

template <typename T>
__aicore__ inline void KthValueSmallAxisShortRankSelect<T>::CopyIn(uint32_t batchId, uint32_t validSegs)
{
    LocalTensor<T> input = inputQueue_.AllocTensor<T>();
    uint32_t totalElems = validSegs * segmentLen_;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyParam{1, static_cast<uint32_t>(totalElems * sizeof(T)), 0, 0, 0};
    DataCopyPad(input, inputGm_[GetInputStart(batchId)], copyParam, padParams);
    inputQueue_.EnQue<T>(input);
}

template <typename T>
__aicore__ inline void KthValueSmallAxisShortRankSelect<T>::SelectShortRank(uint32_t validSegs, LocalTensor<T>& input,
                                                                            LocalTensor<T>& outputValue,
                                                                            LocalTensor<int64_t>& outputIndex)
{
    if (segmentLen_ == 0U || segmentLen_ > 32U || kthIndex_ >= segmentLen_) {
        return;
    }
    // Pick the shorter direction: find K-th smallest from head or K-th largest from tail.
    uint32_t rankFromHead = kthIndex_ + 1U;
    uint32_t rankFromTail = segmentLen_ - kthIndex_;
    bool selectLargest = rankFromTail < rankFromHead;
    uint32_t candidateNum = selectLargest ? rankFromTail : rankFromHead;
    if (candidateNum == 0U || candidateNum > SHORT_RANK_SELECT_MAX_CANDIDATES) {
        return;
    }

    __ubuf__ T* inputPtr = (__ubuf__ T*)input.GetPhyAddr();
    __ubuf__ T* outputValuePtr = (__ubuf__ T*)outputValue.GetPhyAddr();
    __ubuf__ int64_t* outputIndexPtr = (__ubuf__ int64_t*)outputIndex.GetPhyAddr();
    if (selectLargest) {
        DispatchShortRankSelectFixed<T, true>(candidateNum, validSegs, segmentLen_, inputPtr, outputValuePtr,
                                              outputIndexPtr);
    } else {
        DispatchShortRankSelectFixed<T, false>(candidateNum, validSegs, segmentLen_, inputPtr, outputValuePtr,
                                               outputIndexPtr);
    }
}

template <typename T>
__aicore__ inline void KthValueSmallAxisShortRankSelect<T>::CopyOut(uint32_t batchId, uint32_t validSegs,
                                                                    LocalTensor<T>& outputValue,
                                                                    LocalTensor<int64_t>& outputIndex)
{
    int64_t outputStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    DataCopyExtParams valueCopyParam{1, static_cast<uint32_t>(validSegs * sizeof(T)), 0, 0, 0};
    DataCopyExtParams indexCopyParam{1, static_cast<uint32_t>(validSegs * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(valueGm_[outputStart], outputValue, valueCopyParam);
    DataCopyPad(indexGm_[outputStart], outputIndex, indexCopyParam);
}

template <typename T>
__aicore__ inline void KthValueSmallAxisShortRankSelect<T>::Process()
{
    if (IsProcessInvalid()) {
        return;
    }
    // Grid-stride loop: each core processes ceil(batchNum/blockDim) batches.
    for (uint32_t batchId = blockIdx_; batchId < batchNum_; batchId += blockDim_) {
        uint32_t validSegs = ComputeValidSegs(batchId);
        if (validSegs == 0U) {
            continue;
        }
        CopyIn(batchId, validSegs);
        LocalTensor<T> input = inputQueue_.DeQue<T>();
        LocalTensor<T> outputValue = outputValueQueue_.AllocTensor<T>();
        LocalTensor<int64_t> outputIndex = outputIndexQueue_.AllocTensor<int64_t>();
        SelectShortRank(validSegs, input, outputValue, outputIndex);
        inputQueue_.FreeTensor(input);
        outputValueQueue_.EnQue<T>(outputValue);
        outputIndexQueue_.EnQue<int64_t>(outputIndex);
        outputValue = outputValueQueue_.DeQue<T>();
        outputIndex = outputIndexQueue_.DeQue<int64_t>();
        CopyOut(batchId, validSegs, outputValue, outputIndex);
        outputValueQueue_.FreeTensor(outputValue);
        outputIndexQueue_.FreeTensor(outputIndex);
    }
}

} // namespace KthValue

#endif
