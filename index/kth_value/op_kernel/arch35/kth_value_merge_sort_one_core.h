/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_MERGE_SORT_ONE_CORE_H
#define KTH_VALUE_MERGE_SORT_ONE_CORE_H

#include <cmath>
#include <limits>
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "kth_value_median_utils.h"
#include "kth_value_tiling_data.h"
#include "common/merge_sort_constants.h"
#include "common/ping_pong_merge_sort.h"
#include "common/util_type_simd.h"

namespace KthValue {
using namespace AscendC;

using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::XOR_OP_VALUE_FP;
using MergeSortConstants::XOR_OP_VALUE_HALF;

constexpr uint32_t UB_BLOCK_SIZE = Ops::Base::GetUbBlockSize();

// Gather one fixed-rank element per row and widen its nonnegative index.
// Rebase each value block so its relative gather offsets fit in uint16_t.
__aicore__ inline void GatherKthInt16(LocalTensor<int16_t>& values, LocalTensor<uint32_t>& indices,
                                      LocalTensor<int16_t>& outValues, LocalTensor<int64_t>& outIndices, uint32_t rows,
                                      uint32_t stride, uint32_t kth)
{
    constexpr uint32_t lanes = Ops::Base::GetVRegSize() / sizeof(uint32_t);
    constexpr uint32_t wideLanes = Ops::Base::GetVRegSize() / sizeof(int64_t);
    __ubuf__ int16_t* valuePtr = reinterpret_cast<__ubuf__ int16_t*>(values.GetPhyAddr());
    __ubuf__ uint32_t* indexPtr = reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr());
    __ubuf__ int16_t* valueOut = reinterpret_cast<__ubuf__ int16_t*>(outValues.GetPhyAddr());
    __ubuf__ int64_t* indexOut = reinterpret_cast<__ubuf__ int64_t*>(outIndices.GetPhyAddr());
    uint16_t repeats = static_cast<uint16_t>(CeilDivision(rows, lanes));
    uint32_t remain = rows;
    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t> arange;
        Reg::RegTensor<uint32_t> offsets, selected, zero;
        Reg::RegTensor<int64_t> low, high;
        Reg::RegTensor<int16_t> result;
        Reg::RegTensor<uint16_t> narrowOffsets, highWords;
        Reg::MaskReg full = Reg::CreateMask<uint32_t>();
        Reg::Arange(arange, 0);
        Reg::Muls(arange, arange, static_cast<int32_t>(stride), full);
        Reg::Adds(arange, arange, static_cast<int32_t>(kth), full);
        offsets = (Reg::RegTensor<uint32_t>&)arange;
        Reg::Duplicate(zero, 0U);
        Reg::DeInterleave(narrowOffsets, highWords, (Reg::RegTensor<uint16_t>&)offsets,
                          (Reg::RegTensor<uint16_t>&)offsets);
        for (uint16_t rep = 0; rep < repeats; ++rep) {
            uint32_t count = remain < lanes ? remain : lanes;
            uint32_t valueCount = count;
            uint32_t lowCount = count < wideLanes ? count : wideLanes;
            uint32_t highCount = count > wideLanes ? count - wideLanes : 0;
            Reg::MaskReg valid = Reg::UpdateMask<uint32_t>(remain);
            Reg::MaskReg valueMask = Reg::UpdateMask<int16_t>(valueCount);
            Reg::MaskReg lowMask = Reg::UpdateMask<int64_t>(lowCount);
            Reg::MaskReg highMask = Reg::UpdateMask<int64_t>(highCount);
            Reg::Gather(result, valuePtr + rep * lanes * stride, narrowOffsets, valueMask);
            Reg::Gather(selected, indexPtr, offsets, valid);
            Reg::Interleave((Reg::RegTensor<uint32_t>&)low, (Reg::RegTensor<uint32_t>&)high, selected, zero);
            Reg::StoreAlign(valueOut + rep * lanes, result, valueMask);
            Reg::StoreAlign(indexOut + rep * lanes, low, lowMask);
            if (count > wideLanes) {
                Reg::StoreAlign(indexOut + rep * lanes + wideLanes, high, highMask);
            }
            Reg::Adds(offsets, offsets, stride * lanes, full);
        }
    }
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis = 0, bool EnableMedian = false>
class KthValueMergeSortOneCore {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                const KthValueMergeOneCoreTilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void InitIndexLocal();
    __aicore__ inline void CopyDataIn(uint64_t tileOffset, uint32_t currTileSize, uint32_t rowNum);
    __aicore__ inline void FlipSignBit(LocalTensor<CONVERT_TYPE> xLocal, uint32_t offset, uint32_t count);
    __aicore__ inline void SortRows(LocalTensor<T> xLocal, LocalTensor<T> sortedValueLocal,
                                    LocalTensor<uint32_t> sortedIndexLocal, uint32_t rowNum);
    __aicore__ inline void PadConvertedInt16(LocalTensor<CONVERT_TYPE> xSortLocal, uint32_t rowNum);
    __aicore__ inline void SortCompactBatches(LocalTensor<CONVERT_TYPE> xSortLocal,
                                              LocalTensor<CONVERT_TYPE> sortedValueCast,
                                              LocalTensor<uint32_t> sortedIndexLocal, uint32_t rowNum,
                                              uint32_t outputStride);
    __aicore__ inline void SelectScalarRows(LocalTensor<T> sortedValueLocal, LocalTensor<uint32_t> sortedIndexLocal,
                                            LocalTensor<T> compactValue, LocalTensor<int64_t> compactIndex,
                                            uint32_t rowNum, uint32_t outputStride, uint32_t outputK);
    __aicore__ inline void CopyKthToGm(uint64_t outputOffset, uint32_t rowNum);
    __aicore__ inline void ProcessSingleRound(uint32_t round);

    static constexpr uint32_t SORT_STRUCT_BYTES = 8;

    uint32_t mergeBatchRows_ = 1U;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> valuesGm_;
    GlobalTensor<int64_t> indicesGm_;

    TPipe* pipe_{nullptr};
    const KthValueMergeOneCoreTilingData* tiling_{nullptr};
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outValueQueue_;
    TQue<QuePosition::VECOUT, 1> outIndexQueue_;
    TQue<QuePosition::VECOUT, 1> compactValueQueue_;
    TQue<QuePosition::VECOUT, 1> compactIndexQueue_;
    TBuf<TPosition::VECCALC> sortTmpBuf_;
    TBuf<TPosition::VECCALC> sortedLocalBuf_;
    TBuf<TPosition::VECCALC> xCastBuf_;
    TBuf<TPosition::VECCALC> indexLocalBuf_;
    LocalTensor<uint32_t> indexLocal_;

    uint32_t blockIdx_ = 0;
    uint32_t oneCoreRowNum_ = 0;
    uint32_t numTileData_ = 0;
    uint32_t alignSize_ = 0;
    uint32_t sortLoopTimes_ = 0;
    uint32_t unsortedDimParallel_ = 0;
    uint32_t kthIndex_ = 0;
    uint32_t medianMode_ = 0;
    int64_t unsortedDimNum_ = 0;
};

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::Init(
    GM_ADDR x, GM_ADDR values, GM_ADDR indices, const KthValueMergeOneCoreTilingData* tiling, TPipe* pipe)
{
    if (tiling == nullptr || pipe == nullptr) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    tiling_ = tiling;
    ParseTilingData();

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    valuesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
    indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));

    uint32_t bufferNum = tiling_->keyParams4 == 0U ? 1U : tiling_->keyParams4;
    pipe_->InitBuffer(inQueueX_, bufferNum, tiling_->keyParams1);
    constexpr bool compact = !EnableMedian && IsSameType<T, int16_t>::value;
    if constexpr (compact) {
        if (alignSize_ >= KTH_INT16_MERGE_BATCH_MIN_AXIS && alignSize_ <= KTH_INT16_MERGE_BATCH_MAX_AXIS &&
            oneCoreRowNum_ >= KTH_INT16_MERGE_BATCH_ROWS) {
            mergeBatchRows_ = KTH_INT16_MERGE_BATCH_ROWS;
        }
    }
    uint32_t outputElements = DEALING_SORT_NUM_ONCE * oneCoreRowNum_;
    pipe_->InitBuffer(outValueQueue_, bufferNum, compact ? outputElements * sizeof(T) : tiling_->keyParams1);
    pipe_->InitBuffer(outIndexQueue_, bufferNum, compact ? outputElements * sizeof(uint32_t) : tiling_->keyParams2);
    pipe_->InitBuffer(compactValueQueue_, bufferNum, ROUND_UP_AGLIN(oneCoreRowNum_ * sizeof(T)));
    pipe_->InitBuffer(compactIndexQueue_, bufferNum, ROUND_UP_AGLIN(oneCoreRowNum_ * sizeof(int64_t)));

    uint32_t sortStructBytes = alignSize_ * SORT_STRUCT_BYTES * mergeBatchRows_;
    pipe_->InitBuffer(indexLocalBuf_, alignSize_ * sizeof(uint32_t) * mergeBatchRows_);
    pipe_->InitBuffer(sortTmpBuf_, sortStructBytes);
    pipe_->InitBuffer(sortedLocalBuf_, sortStructBytes);
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        pipe_->InitBuffer(xCastBuf_, alignSize_ * oneCoreRowNum_ * sizeof(CONVERT_TYPE));
    }
    indexLocal_ = indexLocalBuf_.AllocTensor<uint32_t>();
    InitIndexLocal();
    // Each row keeps its own zero-based indices even when Sort32 processes multiple rows.
    if (mergeBatchRows_ > 1U) {
        for (uint32_t row = 1; row < mergeBatchRows_; ++row) {
            DataCopy(indexLocal_[row * alignSize_], indexLocal_, alignSize_);
        }
    }
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::ParseTilingData()
{
    oneCoreRowNum_ = tiling_->keyParams0;
    numTileData_ = tiling_->numTileDataSize;
    alignSize_ = tiling_->keyParams3;
    sortLoopTimes_ = tiling_->sortLoopTimes;
    unsortedDimParallel_ = tiling_->unsortedDimParallel;
    kthIndex_ = tiling_->kthIndex;
    medianMode_ = tiling_->medianMode;
    unsortedDimNum_ = tiling_->unsortedDimNum;
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::InitIndexLocal()
{
    __ubuf__ int32_t* indexValuePtr = reinterpret_cast<__ubuf__ int32_t*>(indexLocal_.GetPhyAddr());
    uint32_t vfLenB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint16_t repeatTime = Ops::Base::CeilDiv(alignSize_, vfLenB32);
    uint32_t alignSizeCopy = alignSize_;
    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t> vciTensor;
        Reg::RegTensor<int32_t> indexTensor;
        Reg::Arange(vciTensor, 0);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            Reg::MaskReg mask = Reg::UpdateMask<uint32_t>(alignSizeCopy);
            Reg::Adds(indexTensor, vciTensor, i * vfLenB32, mask);
            Reg::StoreAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(indexValuePtr, indexTensor, vfLenB32, mask);
        }
    }
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::CopyDataIn(
    uint64_t tileOffset, uint32_t currTileSize, uint32_t rowNum)
{
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    T padding = 0;
    if constexpr (!IsSameType<T, int16_t>::value) {
        padding = static_cast<T>(NAN);
    }
    Duplicate(xLocal, padding, alignSize_ * rowNum);
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);

    uint32_t currTileSizeAlign = (currTileSize * sizeof(T) + UB_BLOCK_SIZE - 1U) / UB_BLOCK_SIZE * UB_BLOCK_SIZE /
                                 sizeof(T);
    uint32_t dstStride = ((alignSize_ - currTileSizeAlign) * sizeof(T)) / UB_BLOCK_SIZE;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(currTileSizeAlign - currTileSize), padding};
    DataCopyExtParams copyParam{static_cast<uint16_t>(rowNum), static_cast<uint32_t>(currTileSize * sizeof(T)), 0,
                                dstStride, 0};
    DataCopyPad(xLocal, xGm_[tileOffset], copyParam, padParams);
    inQueueX_.EnQue<T>(xLocal);
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::FlipSignBit(
    LocalTensor<CONVERT_TYPE> xLocal, uint32_t offset, uint32_t count)
{
    if constexpr (IsSameType<float, CONVERT_TYPE>::value) {
        LocalTensor<int32_t> castTensor = xLocal[offset].template ReinterpretCast<int32_t>();
        Adds(castTensor, castTensor, XOR_OP_VALUE_FP, count);
    } else if constexpr (IsSameType<half, CONVERT_TYPE>::value) {
        LocalTensor<int16_t> castTensor = xLocal[offset].template ReinterpretCast<int16_t>();
        Adds(castTensor, castTensor, XOR_OP_VALUE_HALF, count);
    }
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::PadConvertedInt16(
    LocalTensor<CONVERT_TYPE> xSortLocal, uint32_t rowNum)
{
    // Integer padding cannot represent infinity. Mark it after the exact
    // conversion so it sorts after every real value, including INT16_MAX.
    constexpr uint32_t vectorElems = Ops::Base::GetVRegSize() / sizeof(float);
    uint32_t tailStart = numTileData_ / vectorElems * vectorElems;
    uint32_t firstPadLane = numTileData_ - tailStart;
    uint32_t tailElems = alignSize_ - tailStart;
    uint32_t rowStride = alignSize_;
    __ubuf__ float* castPtr = reinterpret_cast<__ubuf__ float*>(xSortLocal.GetPhyAddr());
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> infinity;
        Reg::RegTensor<int32_t> lanes;
        Reg::MaskReg valid = Reg::UpdateMask<float>(tailElems);
        Reg::MaskReg paddingMask;
        Reg::Arange(lanes, 0);
        Reg::Compares<int32_t, CMPMODE::GE>(paddingMask, lanes, static_cast<int32_t>(firstPadLane), valid);
        Reg::Duplicate(infinity, static_cast<float>(INFINITY));
        for (uint16_t row = 0; row < static_cast<uint16_t>(rowNum); ++row) {
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(castPtr + row * rowStride + tailStart, infinity,
                                                                  paddingMask);
        }
    }
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::SortCompactBatches(
    LocalTensor<CONVERT_TYPE> xSortLocal, LocalTensor<CONVERT_TYPE> sortedValueCast,
    LocalTensor<uint32_t> sortedIndexLocal, uint32_t rowNum, uint32_t outputStride)
{
    uint32_t sortRepeatTimes = alignSize_ / DEALING_SORT_NUM_ONCE;
    LocalTensor<CONVERT_TYPE> sortedLocal = sortedLocalBuf_.Get<CONVERT_TYPE>();
    LocalTensor<CONVERT_TYPE> sortTmp = sortTmpBuf_.Get<CONVERT_TYPE>();
    for (uint32_t first = 0; first < rowNum; first += mergeBatchRows_) {
        uint32_t rows = rowNum - first < mergeBatchRows_ ? rowNum - first : mergeBatchRows_;
        Sort32(sortedLocal, xSortLocal[first * alignSize_], indexLocal_, sortRepeatTimes * rows);
        bool inPing = true;
        uint32_t runCount = sortRepeatTimes;
        for (uint32_t run = DEALING_SORT_NUM_ONCE; runCount > 1U; run *= MergeSortConstants::MERGE_LIST_MAX_NUM) {
            for (uint32_t row = 0; row < rows; ++row) {
                uint32_t offset = GetSortOffset<CONVERT_TYPE>(row * alignSize_);
                if (inPing) {
                    PingPongMergeSortCommon::MergeStage(sortTmp[offset], sortedLocal[offset], alignSize_, run,
                                                        runCount);
                } else {
                    PingPongMergeSortCommon::MergeStage(sortedLocal[offset], sortTmp[offset], alignSize_, run,
                                                        runCount);
                }
            }
            inPing = !inPing;
            runCount = (runCount + MergeSortConstants::MERGE_LIST_MAX_NUM - 1U) /
                       MergeSortConstants::MERGE_LIST_MAX_NUM;
        }
        LocalTensor<CONVERT_TYPE> proposal = inPing ? sortedLocal : sortTmp;
        __ubuf__ uint32_t* packed = (__ubuf__ uint32_t*)proposal.GetPhyAddr();
        __ubuf__ uint32_t* valueOut = (__ubuf__ uint32_t*)sortedValueCast.GetPhyAddr() + first * outputStride;
        __ubuf__ uint32_t* indexOut = (__ubuf__ uint32_t*)sortedIndexLocal.GetPhyAddr() + first * outputStride;
        // Each FP32 proposal stores the value followed by its original B32 index.
        // Gather only k, keeping one result per row instead of extracting a Sort32 block.
        constexpr uint32_t proposalWords = SORT_STRUCT_BYTES / sizeof(uint32_t);
        constexpr uint32_t proposalIndexWord = 1U;
        static_assert(KTH_INT16_MERGE_BATCH_ROWS <= Ops::Base::GetVRegSize() / sizeof(uint32_t));
        uint32_t proposalStride = alignSize_ * proposalWords;
        uint32_t selected = kthIndex_ * proposalWords;
        __VEC_SCOPE__
        {
            Reg::RegTensor<uint32_t> lanes, srcIdx, values, indices;
            Reg::MaskReg mask = Reg::UpdateMask<uint32_t>(rows);
            Reg::Arange((Reg::RegTensor<int32_t>&)lanes, 0);
            Reg::Muls(srcIdx, lanes, proposalStride, mask);
            Reg::Adds(srcIdx, srcIdx, selected, mask);
            Reg::Gather(values, packed, srcIdx, mask);
            Reg::Adds(srcIdx, srcIdx, proposalIndexWord, mask);
            Reg::Gather(indices, packed, srcIdx, mask);
            Reg::StoreAlign(valueOut, values, mask);
            Reg::StoreAlign(indexOut, indices, mask);
        }
    }
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::SortRows(
    LocalTensor<T> xLocal, LocalTensor<T> sortedValueLocal, LocalTensor<uint32_t> sortedIndexLocal, uint32_t rowNum)
{
    uint32_t sortRepeatTimes = alignSize_ / DEALING_SORT_NUM_ONCE;
    LocalTensor<CONVERT_TYPE> sortedLocal = sortedLocalBuf_.Get<CONVERT_TYPE>();
    LocalTensor<CONVERT_TYPE> sortTmp = sortTmpBuf_.Get<CONVERT_TYPE>();
    LocalTensor<CONVERT_TYPE> xSortLocal;
    LocalTensor<CONVERT_TYPE> sortedValueCast;
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        xSortLocal = xCastBuf_.Get<CONVERT_TYPE>();
        // SortToProposal consumes each source row before Extract, so the cast input can hold the result.
        sortedValueCast = xSortLocal;
        Cast(xSortLocal, xLocal, RoundMode::CAST_NONE, alignSize_ * rowNum);
        if constexpr (IsSameType<T, int16_t>::value) {
            PadConvertedInt16(xSortLocal, rowNum);
        }
    } else {
        xSortLocal = xLocal;
        sortedValueCast = sortedValueLocal;
    }
    if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<CONVERT_TYPE>) {
        if (medianMode_ != MEDIAN_MODE_STATIC) {
            for (uint32_t row = 0, offset = 0; row < rowNum; ++row, offset += alignSize_) {
                CanonicalizeNanValues(xSortLocal[offset], alignSize_);
            }
        }
    }
    // MergeSort sorts sign-flipped ascending keys. Flip before Sort/Sort32 and flip back after Extract.
    FlipSignBit(xSortLocal, 0, alignSize_ * rowNum);
    // Batched int16 selects k directly; the single-row fallback extracts its Sort32 block.
    constexpr bool compactExtract = !EnableMedian && IsSameType<T, int16_t>::value;
    uint32_t outputStride = compactExtract ? (mergeBatchRows_ > 1U ? 1U : DEALING_SORT_NUM_ONCE) : alignSize_;
    // Batch Sort32 and process rows at each merge level; each MergeStage remains confined to one row.
    if (compactExtract && mergeBatchRows_ > 1U) {
        SortCompactBatches(xSortLocal, sortedValueCast, sortedIndexLocal, rowNum, outputStride);
    } else {
        for (uint32_t row = 0, offset = 0; row < rowNum; ++row, offset += alignSize_) {
            bool resultInPing = PingPongMergeSortCommon::SortToProposal(sortedLocal, sortTmp, xSortLocal[offset],
                                                                        indexLocal_, sortRepeatTimes);
            // Packed results overwrite only rows whose input has already been consumed.
            LocalTensor<CONVERT_TYPE> proposal = resultInPing ? sortedLocal : sortTmp;
            uint32_t proposalOffset = compactExtract ? GetSortOffset<CONVERT_TYPE>(kthIndex_ / DEALING_SORT_NUM_ONCE *
                                                                                   DEALING_SORT_NUM_ONCE) :
                                                       0U;
            Extract(sortedValueCast[row * outputStride], sortedIndexLocal[row * outputStride], proposal[proposalOffset],
                    compactExtract || isSort32SmallAxis == 1 ? 1 : sortRepeatTimes);
        }
    }
    FlipSignBit(sortedValueCast, 0, outputStride * rowNum);
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        Cast(sortedValueLocal, sortedValueCast, RoundMode::CAST_RINT, outputStride * rowNum);
    }
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::SelectScalarRows(
    LocalTensor<T> sortedValueLocal, LocalTensor<uint32_t> sortedIndexLocal, LocalTensor<T> compactValue,
    LocalTensor<int64_t> compactIndex, uint32_t rowNum, uint32_t outputStride, uint32_t outputK)
{
    for (uint32_t row = 0; row < rowNum; ++row) {
        uint32_t rowOffset = row * outputStride;
        uint32_t selectedK = outputK;
        if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
            if constexpr (KTH_VALUE_ENABLE_STATIC_MEDIAN_FAST_PATH) {
                if (medianMode_ != MEDIAN_MODE_STATIC) {
                    uint32_t nonNanCount = CountNonNan(sortedValueLocal[rowOffset], numTileData_,
                                                       sortTmpBuf_.Get<float>(), pipe_);
                    selectedK = ResolveMedianK(kthIndex_, numTileData_, nonNanCount, medianMode_);
                }
            } else {
                uint32_t nonNanCount = CountNonNan(sortedValueLocal[rowOffset], numTileData_, sortTmpBuf_.Get<float>(),
                                                   pipe_);
                selectedK = ResolveMedianK(kthIndex_, numTileData_, nonNanCount, medianMode_);
            }
        }
        uint32_t srcOffset = rowOffset + selectedK;
        compactValue.SetValue(row, sortedValueLocal.GetValue(srcOffset));
        compactIndex.SetValue(row, static_cast<int64_t>(sortedIndexLocal.GetValue(srcOffset)));
    }
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::CopyKthToGm(
    uint64_t outputOffset, uint32_t rowNum)
{
    LocalTensor<T> sortedValueLocal = outValueQueue_.DeQue<T>();
    LocalTensor<uint32_t> sortedIndexLocal = outIndexQueue_.DeQue<uint32_t>();

    constexpr bool compactExtract = !EnableMedian && IsSameType<T, int16_t>::value;
    uint32_t outputStride = compactExtract ? (mergeBatchRows_ > 1U ? 1U : DEALING_SORT_NUM_ONCE) : alignSize_;
    uint32_t outputK = compactExtract ? (mergeBatchRows_ > 1U ? 0U : kthIndex_ % DEALING_SORT_NUM_ONCE) : kthIndex_;
    constexpr uint32_t gatherLanes = Ops::Base::GetVRegSize() / sizeof(uint32_t);
    constexpr uint32_t minGatherRows = 8U;
    // With k < stride, the largest B16 gather offset is 63*stride+k
    // <= 64*stride-1 <= 65471 for stride <= floor(65535/64) = 1023.
    constexpr uint32_t maxGatherStride = std::numeric_limits<uint16_t>::max() / gatherLanes;
    const bool useGather = !EnableMedian && IsSameType<T, int16_t>::value && rowNum >= minGatherRows &&
                           alignSize_ <= maxGatherStride;
    if (!useGather) {
        event_t eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
    }
    LocalTensor<T> compactValue = compactValueQueue_.AllocTensor<T>();
    LocalTensor<int64_t> compactIndex = compactIndexQueue_.AllocTensor<int64_t>();
    if (useGather) {
        LocalTensor<int16_t> sourceValues = sortedValueLocal.template ReinterpretCast<int16_t>();
        LocalTensor<int16_t> targetValues = compactValue.template ReinterpretCast<int16_t>();
        GatherKthInt16(sourceValues, sortedIndexLocal, targetValues, compactIndex, rowNum, outputStride, outputK);
    } else {
        SelectScalarRows(sortedValueLocal, sortedIndexLocal, compactValue, compactIndex, rowNum, outputStride, outputK);
    }
    compactValueQueue_.EnQue<T>(compactValue);
    compactIndexQueue_.EnQue<int64_t>(compactIndex);
    compactValue = compactValueQueue_.DeQue<T>();
    compactIndex = compactIndexQueue_.DeQue<int64_t>();
    if (useGather) {
        event_t ready = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
    } else {
        event_t eventIdSToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    }
    DataCopyExtParams valueCopyParam{1, static_cast<uint32_t>(rowNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(valuesGm_[outputOffset], compactValue, valueCopyParam);
    DataCopyExtParams indexCopyParam{1, static_cast<uint32_t>(rowNum * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(indicesGm_[outputOffset], compactIndex, indexCopyParam);
    compactValueQueue_.FreeTensor(compactValue);
    compactIndexQueue_.FreeTensor(compactIndex);
    outValueQueue_.FreeTensor(sortedValueLocal);
    outIndexQueue_.FreeTensor(sortedIndexLocal);
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::ProcessSingleRound(
    uint32_t round)
{
    int64_t rowStart = (static_cast<int64_t>(blockIdx_) + static_cast<int64_t>(round) * unsortedDimParallel_) *
                       oneCoreRowNum_;
    if (rowStart >= unsortedDimNum_) {
        return;
    }
    uint32_t rowNum = oneCoreRowNum_;
    int64_t remain = unsortedDimNum_ - rowStart;
    if (remain < static_cast<int64_t>(rowNum)) {
        rowNum = static_cast<uint32_t>(remain);
    }
    uint64_t inputOffset = static_cast<uint64_t>(rowStart) * numTileData_;
    CopyDataIn(inputOffset, numTileData_, rowNum);
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    LocalTensor<T> sortedValueLocal = outValueQueue_.AllocTensor<T>();
    LocalTensor<uint32_t> sortedIndexLocal = outIndexQueue_.AllocTensor<uint32_t>();
    SortRows(xLocal, sortedValueLocal, sortedIndexLocal, rowNum);
    outValueQueue_.EnQue<T>(sortedValueLocal);
    outIndexQueue_.EnQue<uint32_t>(sortedIndexLocal);
    inQueueX_.FreeTensor(xLocal);
    CopyKthToGm(static_cast<uint64_t>(rowStart), rowNum);
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }
    for (uint32_t round = 0; round < sortLoopTimes_; ++round) {
        ProcessSingleRound(round);
    }
}
} // namespace KthValue

#endif
