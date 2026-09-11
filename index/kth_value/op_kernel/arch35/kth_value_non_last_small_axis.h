/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_NON_LAST_SMALL_AXIS_H
#define KTH_VALUE_NON_LAST_SMALL_AXIS_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "kth_value_median_utils.h"
#include "kth_value_tiling_data.h"
#include "common/non_last_small_axis_base.h"

namespace KthValue {
using namespace AscendC;

// Median hook (pre-sort): canonicalizes NaN/signed-zero in place over the transposed
// sort-major layout, which is strided (valueAxisElems), hence a scalar SIMT pass.
template <typename T>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM) __aicore__
    void SimtCanonicalizeNonLastMedianRows(uint32_t validElems, uint32_t axisLen, uint32_t valueAxisElems,
                                           __ubuf__ T* sortInput)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < validElems;
         idx += SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM) {
        uint32_t inner = idx / axisLen;
        uint32_t axis = idx - inner * axisLen;
        uint32_t offset = inner * valueAxisElems + axis;
        sortInput[offset] = CanonicalizeMedianSortValue(sortInput[offset]);
    }
}

// Median hook (post-sort, per inner lane): binary-searches the NaN boundary on the sorted
// row, resolves the effective k per medianMode, then traces back the selected element to the
// original input tile (raw T layout), preserving precision and the original NaN payload.
template <typename T, typename SortT>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM) __aicore__
    void SimtStoreNonLastMedian(uint32_t curInnerChunk, uint32_t axisLen, uint32_t kthIndex, uint32_t medianMode,
                                uint32_t inputRowElems, uint32_t valueAxisElems, uint32_t indexAxisElems,
                                __ubuf__ T* inputTile, __ubuf__ SortT* sortedValue, __ubuf__ uint32_t* sortedIndex,
                                __ubuf__ T* compactValue, __ubuf__ int64_t* compactIndex)
{
    for (uint32_t inner = static_cast<uint32_t>(threadIdx.x); inner < curInnerChunk;
         inner += SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM) {
        uint32_t rowOffset = inner * valueAxisElems;
        uint32_t left = 0U;
        uint32_t right = axisLen;
        while (left < right) {
            uint32_t middle = left + (right - left) / 2U;
            if (IsNanValue(sortedValue[rowOffset + middle])) {
                right = middle;
            } else {
                left = middle + 1U;
            }
        }
        uint32_t selectedK = kthIndex;
        if (medianMode == MEDIAN_MODE_PROPAGATE_NAN && left < axisLen) {
            selectedK = left;
        } else if (medianMode == MEDIAN_MODE_IGNORE_NAN) {
            selectedK = left == 0U ? 0U : (left - 1U) / 2U;
        }
        uint32_t selectedIndex = sortedIndex[inner * indexAxisElems + selectedK];
        compactValue[inner] = inputTile[selectedIndex * inputRowElems + inner];
        compactIndex[inner] = static_cast<int64_t>(selectedIndex);
    }
}

template <typename T, bool IsDescend, bool UseMergeSort, bool EnableMedian = false>
class KthValueNonLastSmallAxis
    : public SmallAxisCommon::NonLastSmallAxisBase<
          KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort, EnableMedian>, T,
          std::conditional_t<UseMergeSort && std::is_same_v<T, bfloat16_t>, float, T>,
          std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>,
          std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>,
          std::conditional_t<sizeof(T) == 1, std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>,
          IsDescend, UseMergeSort, UseMergeSort && std::is_same_v<T, bfloat16_t>,
          !UseMergeSort && SignedZeroSortCommon::IS_FLOATING_POINT_V<T>> {
    using Base = SmallAxisCommon::NonLastSmallAxisBase<
        KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort, EnableMedian>, T,
        std::conditional_t<UseMergeSort && std::is_same_v<T, bfloat16_t>, float, T>,
        std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>,
        std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>,
        std::conditional_t<sizeof(T) == 1, std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>,
        IsDescend, UseMergeSort, UseMergeSort && std::is_same_v<T, bfloat16_t>,
        !UseMergeSort && SignedZeroSortCommon::IS_FLOATING_POINT_V<T>>;

    using SortT = std::conditional_t<UseMergeSort && std::is_same_v<T, bfloat16_t>, float, T>;
    static constexpr bool IS_BF16_MERGE = UseMergeSort && std::is_same_v<T, bfloat16_t>;
    static constexpr bool NORMALIZE_SIGNED_ZERO = !UseMergeSort && SignedZeroSortCommon::IS_FLOATING_POINT_V<T>;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                const KthValueTilingData* tilingData, TPipe* pipe);

    __aicore__ inline void Process();
    __aicore__ inline void StoreTile(int64_t inputOffset, int64_t outputOffset, uint32_t curInnerChunk);

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void PrepareSortRows(uint32_t curInnerChunk);
    __aicore__ inline void CopyKthToOutput(uint32_t curInnerChunk, int64_t outputOffset);

    const KthValueTilingData* tilingData_ = nullptr;
    GlobalTensor<T> valueGm_;
    GlobalTensor<int64_t> indexGm_;

    TBuf<TPosition::VECCALC> compactValueBuf_;
    TBuf<TPosition::VECCALC> compactCastValueBuf_;
    TBuf<TPosition::VECCALC> compactIndexBuf_;
    TBuf<TPosition::VECCALC> sourceIndexBuf_;

    LocalTensor<T> compactValue_;
    LocalTensor<SortT> compactCastValue_;
    LocalTensor<int64_t> compactIndex_;

    uint32_t kthIndex_ = 0;
    uint32_t medianMode_ = MEDIAN_MODE_STATIC;
};

template <typename T, bool IsDescend, bool UseMergeSort, bool EnableMedian>
__aicore__ inline void KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort, EnableMedian>::Init(
    GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace, const KthValueTilingData* tilingData, TPipe* pipe)
{
    (void)workspace;
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    this->pipe_ = pipe;
    tilingData_ = tilingData;
    this->blockIdx_ = GetBlockIdx();
    this->blockDim_ = GetBlockNum();
    ParseTilingData();

    this->inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
    indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));

    if (this->axisLen_ == 0 || this->innerChunk_ == 0 || this->innerLoopNum_ == 0 || this->outerSize_ <= 0 ||
        this->innerSize_ <= 0) {
        return;
    }
    this->pipe_->InitBuffer(this->inputTileBuf_, this->axisLen_ * this->inputRowBytes_);
    if constexpr (IS_BF16_MERGE) {
        this->pipe_->InitBuffer(this->inputCastBuf_, this->innerChunk_ * this->inputValueAxisBytes_);
        this->inputCast_ = this->inputCastBuf_.template Get<T>();
    }
    this->pipe_->InitBuffer(this->sortInputBuf_, this->innerChunk_ * this->valueAxisBytes_);
    this->pipe_->InitBuffer(this->sortedValueBuf_, this->innerChunk_ * this->valueAxisBytes_);
    this->pipe_->InitBuffer(this->sortedIndexBuf_, this->innerChunk_ * this->indexAxisBytes_);
    this->pipe_->InitBuffer(compactValueBuf_, ROUND_UP_AGLIN(this->innerChunk_ * sizeof(T)));
    if constexpr (IS_BF16_MERGE) {
        this->pipe_->InitBuffer(compactCastValueBuf_, ROUND_UP_AGLIN(this->innerChunk_ * sizeof(SortT)));
        compactCastValue_ = compactCastValueBuf_.template Get<SortT>();
    }
    this->pipe_->InitBuffer(compactIndexBuf_, ROUND_UP_AGLIN(this->innerChunk_ * sizeof(int64_t)));
    if constexpr (NORMALIZE_SIGNED_ZERO) {
        this->pipe_->InitBuffer(sourceIndexBuf_, this->indexAxisBytes_);
    }
    this->pipe_->InitBuffer(this->tmpBuf_, this->tmpUbSize_);

    this->inputTile_ = this->inputTileBuf_.template Get<T>();
    this->sortInput_ = this->sortInputBuf_.template Get<SortT>();
    this->sortedValue_ = this->sortedValueBuf_.template Get<SortT>();
    this->sortedIndex_ = this->sortedIndexBuf_.template Get<uint32_t>();
    compactValue_ = compactValueBuf_.template Get<T>();
    compactIndex_ = compactIndexBuf_.template Get<int64_t>();
    this->tmp_ = this->tmpBuf_.template Get<uint8_t>();
    if constexpr (NORMALIZE_SIGNED_ZERO) {
        this->sourceIndex_ = sourceIndexBuf_.template Get<uint32_t>();
    }
}

template <typename T, bool IsDescend, bool UseMergeSort, bool EnableMedian>
__aicore__ inline void KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort, EnableMedian>::Process()
{
    if (this->blockIdx_ >= this->blockDim_ || this->axisLen_ == 0 || this->innerChunk_ == 0 ||
        this->innerLoopNum_ == 0) {
        return;
    }
    uint64_t tileCount = static_cast<uint64_t>(this->outerSize_) * static_cast<uint64_t>(this->innerLoopNum_);
    uint64_t tilesPerCore = Ops::Base::CeilDiv(tileCount, static_cast<uint64_t>(this->blockDim_));
    uint64_t startTile = static_cast<uint64_t>(this->blockIdx_) * tilesPerCore;
    uint64_t endTile = startTile + tilesPerCore;
    if (endTile > tileCount) {
        endTile = tileCount;
    }
    for (uint64_t tileId = startTile; tileId < endTile; ++tileId) {
        uint64_t outerId = tileId / this->innerLoopNum_;
        uint32_t innerTileId = static_cast<uint32_t>(tileId - outerId * static_cast<uint64_t>(this->innerLoopNum_));
        uint32_t curInnerChunk = this->GetCurrentInnerChunk(innerTileId);
        if (curInnerChunk == 0U) {
            continue;
        }
        int64_t innerStart = static_cast<int64_t>(innerTileId) * static_cast<int64_t>(this->innerChunk_);
        int64_t inputOffset = static_cast<int64_t>(outerId) * static_cast<int64_t>(this->axisLen_) * this->innerSize_ +
                              innerStart;
        int64_t outputOffset = static_cast<int64_t>(outerId) * this->innerSize_ + innerStart;
        this->LoadTile(inputOffset, curInnerChunk);
        this->TransposeToSortMajor(curInnerChunk);
        PrepareSortRows(curInnerChunk);
        this->SortRows(curInnerChunk);
        StoreTile(inputOffset, outputOffset, curInnerChunk);
    }
}

template <typename T, bool IsDescend, bool UseMergeSort, bool EnableMedian>
__aicore__ inline void KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort, EnableMedian>::ParseTilingData()
{
    this->axisLen_ = static_cast<uint32_t>(tilingData_->lastAxisNum);
    kthIndex_ = static_cast<uint32_t>(tilingData_->kthIndex);
    if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
        medianMode_ = tilingData_->medianMode;
    }
    this->outerSize_ = tilingData_->outerSize;
    this->innerSize_ = tilingData_->innerSize;
    this->innerLoopNum_ = tilingData_->innerLoopNum;
    this->innerChunk_ = tilingData_->innerChunk;
    this->inputRowBytes_ = tilingData_->inputRowBytes;
    this->valueAxisBytes_ = tilingData_->valueAxisBytes;
    this->indexAxisBytes_ = tilingData_->indexAxisBytes;
    this->inputRowElems_ = this->inputRowBytes_ / sizeof(T);
    this->valueAxisElems_ = this->valueAxisBytes_ / sizeof(SortT);
    this->indexAxisElems_ = this->indexAxisBytes_ / sizeof(uint32_t);
    this->sortCount_ = tilingData_->keyParams0 == 0U ? this->axisLen_ : tilingData_->keyParams0;
    if constexpr (IS_BF16_MERGE) {
        this->inputValueAxisBytes_ = tilingData_->keyParams1;
        this->inputValueAxisElems_ = this->inputValueAxisBytes_ / sizeof(T);
    }
    this->tmpUbSize_ = tilingData_->tmpUbSize;
}

// Median hook point: runs after transpose, before SortRows.
template <typename T, bool IsDescend, bool UseMergeSort, bool EnableMedian>
__aicore__ inline void KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort, EnableMedian>::PrepareSortRows(
    uint32_t curInnerChunk)
{
    if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
        if (medianMode_ != MEDIAN_MODE_STATIC) {
            asc_vf_call<SimtCanonicalizeNonLastMedianRows<SortT>>(
                dim3(SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM), curInnerChunk * this->axisLen_, this->axisLen_,
                this->valueAxisElems_, reinterpret_cast<__ubuf__ SortT*>(this->sortInput_.GetPhyAddr()));
            event_t eventIdVToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);
        }
    }
}

template <typename T, bool IsDescend, bool UseMergeSort, bool EnableMedian>
__aicore__ inline void KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort, EnableMedian>::CopyKthToOutput(
    uint32_t curInnerChunk, int64_t outputOffset)
{
    event_t eventIdVToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    uint32_t compactCastElems = 0U;
    if constexpr (IS_BF16_MERGE && !EnableMedian) {
        compactCastElems = ROUND_UP_AGLIN(curInnerChunk * sizeof(SortT)) / sizeof(SortT);
        Duplicate(compactCastValue_, static_cast<SortT>(0), compactCastElems);
        event_t eventIdVToSForInit = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToSForInit);
        WaitFlag<HardEvent::V_S>(eventIdVToSForInit);
    }
    if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
        // Median hook point: k resolution fused with value extraction, tracing back to the raw input tile.
        asc_vf_call<SimtStoreNonLastMedian<T, SortT>>(
            dim3(SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM), curInnerChunk, this->axisLen_, kthIndex_, medianMode_,
            this->inputRowElems_, this->valueAxisElems_, this->indexAxisElems_,
            reinterpret_cast<__ubuf__ T*>(this->inputTile_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ SortT*>(this->sortedValue_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(this->sortedIndex_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ T*>(compactValue_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int64_t*>(compactIndex_.GetPhyAddr()));
        eventIdVToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
    } else {
        // Compact one kth element per inner lane. The GM output is [outer, 1, inner], so outputOffset
        // points to a contiguous inner chunk.
        for (uint32_t inner = 0; inner < curInnerChunk; ++inner) {
            uint32_t srcOffset = inner * this->valueAxisElems_ + kthIndex_;
            uint32_t idxOffset = inner * this->indexAxisElems_ + kthIndex_;
            if constexpr (IS_BF16_MERGE) {
                compactCastValue_.SetValue(inner, this->sortedValue_.GetValue(srcOffset));
            } else {
                compactValue_.SetValue(inner, this->sortedValue_.GetValue(srcOffset));
            }
            compactIndex_.SetValue(inner, static_cast<int64_t>(this->sortedIndex_.GetValue(idxOffset)));
        }
    }
    if constexpr (IS_BF16_MERGE && !EnableMedian) {
        event_t eventIdSToV = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Cast(compactValue_, compactCastValue_, RoundMode::CAST_RINT, compactCastElems);
        event_t eventIdVToMte3 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    }
    event_t eventIdSToMte3 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    DataCopyExtParams valueCopyParam{1, static_cast<uint32_t>(curInnerChunk * sizeof(T)), 0, 0, 0};
    DataCopyPad(valueGm_[outputOffset], compactValue_, valueCopyParam);
    DataCopyExtParams indexCopyParam{1, static_cast<uint32_t>(curInnerChunk * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(indexGm_[outputOffset], compactIndex_, indexCopyParam);
    event_t eventIdMte3ToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
}

template <typename T, bool IsDescend, bool UseMergeSort, bool EnableMedian>
__aicore__ inline void KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort, EnableMedian>::StoreTile(
    int64_t inputOffset, int64_t outputOffset, uint32_t curInnerChunk)
{
    (void)inputOffset;
    CopyKthToOutput(curInnerChunk, outputOffset);
}

} // namespace KthValue

#endif
