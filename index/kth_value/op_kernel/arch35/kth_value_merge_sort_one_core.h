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
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "kth_value_median_utils.h"
#include "kth_value_tiling_data.h"
#include "common/merge_sort_constants.h"
#include "common/util_type_simd.h"

namespace KthValue {
using namespace AscendC;

using MergeSortConstants::DEALING_CONCAT_NUM_ONCE;
using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::XOR_OP_VALUE_FP;
using MergeSortConstants::XOR_OP_VALUE_HALF;

constexpr uint32_t UB_BLOCK_SIZE = Ops::Base::GetUbBlockSize();

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis = 0, bool EnableMedian = false>
class KthValueMergeSortOneCore {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const KthValueTilingData* tiling,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void InitIndexLocal();
    __aicore__ inline void CopyDataIn(uint64_t tileOffset, uint32_t currTileSize, uint32_t rowNum);
    __aicore__ inline void FlipSignBit(LocalTensor<CONVERT_TYPE> xLocal, uint32_t offset, uint32_t count);
    __aicore__ inline void SortRows(LocalTensor<T> xLocal, LocalTensor<T> sortedValueLocal,
                                    LocalTensor<uint32_t> sortedIndexLocal, uint32_t rowNum);
    __aicore__ inline void CopyKthToGm(uint64_t outputOffset, uint32_t rowNum);
    __aicore__ inline void ProcessSingleRound(uint32_t round);

    static constexpr uint32_t SORT_STRUCT_BYTES = 8;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> valuesGm_;
    GlobalTensor<int64_t> indicesGm_;

    TPipe* pipe_{nullptr};
    const KthValueTilingData* tiling_{nullptr};
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outValueQueue_;
    TQue<QuePosition::VECOUT, 1> outIndexQueue_;
    TQue<QuePosition::VECOUT, 1> compactValueQueue_;
    TQue<QuePosition::VECOUT, 1> compactIndexQueue_;
    TBuf<TPosition::VECCALC> concatTmpBuf_;
    TBuf<TPosition::VECCALC> sortTmpBuf_;
    TBuf<TPosition::VECCALC> sortedLocalBuf_;
    TBuf<TPosition::VECCALC> xCastBuf_;
    TBuf<TPosition::VECCALC> sortedValueCastBuf_;
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
    GM_ADDR x, GM_ADDR values, GM_ADDR indices, const KthValueTilingData* tiling, TPipe* pipe)
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
    pipe_->InitBuffer(outValueQueue_, bufferNum, tiling_->keyParams1);
    pipe_->InitBuffer(outIndexQueue_, bufferNum, tiling_->keyParams2);
    pipe_->InitBuffer(compactValueQueue_, bufferNum, ROUND_UP_AGLIN(oneCoreRowNum_ * sizeof(T)));
    pipe_->InitBuffer(compactIndexQueue_, bufferNum, ROUND_UP_AGLIN(oneCoreRowNum_ * sizeof(int64_t)));

    uint32_t sortStructElems = alignSize_ * SORT_STRUCT_BYTES;
    pipe_->InitBuffer(indexLocalBuf_, alignSize_ * sizeof(uint32_t));
    pipe_->InitBuffer(concatTmpBuf_, tiling_->tmpUbSize);
    pipe_->InitBuffer(sortTmpBuf_, sortStructElems * sizeof(CONVERT_TYPE));
    pipe_->InitBuffer(sortedLocalBuf_, sortStructElems * sizeof(CONVERT_TYPE));
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        pipe_->InitBuffer(xCastBuf_, alignSize_ * oneCoreRowNum_ * sizeof(CONVERT_TYPE));
        pipe_->InitBuffer(sortedValueCastBuf_, alignSize_ * oneCoreRowNum_ * sizeof(CONVERT_TYPE));
    }
    indexLocal_ = indexLocalBuf_.AllocTensor<uint32_t>();
    InitIndexLocal();
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
    Duplicate(xLocal, static_cast<T>(NAN), alignSize_ * rowNum);
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);

    uint32_t currTileSizeAlign = (currTileSize * sizeof(T) + UB_BLOCK_SIZE - 1U) / UB_BLOCK_SIZE * UB_BLOCK_SIZE /
                                 sizeof(T);
    uint32_t dstStride = ((alignSize_ - currTileSizeAlign) * sizeof(T)) / UB_BLOCK_SIZE;
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(currTileSizeAlign - currTileSize),
                                      static_cast<T>(NAN)};
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
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::SortRows(
    LocalTensor<T> xLocal, LocalTensor<T> sortedValueLocal, LocalTensor<uint32_t> sortedIndexLocal, uint32_t rowNum)
{
    uint32_t sortRepeatTimes = alignSize_ / DEALING_SORT_NUM_ONCE;
    uint32_t concatRepeatTimes = alignSize_ / DEALING_CONCAT_NUM_ONCE;
    LocalTensor<CONVERT_TYPE> concatTmp = concatTmpBuf_.Get<CONVERT_TYPE>();
    LocalTensor<CONVERT_TYPE> sortedLocal = sortedLocalBuf_.Get<CONVERT_TYPE>();
    LocalTensor<CONVERT_TYPE> sortTmp = sortTmpBuf_.Get<CONVERT_TYPE>();
    LocalTensor<CONVERT_TYPE> xSortLocal;
    LocalTensor<CONVERT_TYPE> sortedValueCast;
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        xSortLocal = xCastBuf_.Get<CONVERT_TYPE>();
        sortedValueCast = sortedValueCastBuf_.Get<CONVERT_TYPE>();
        Cast(xSortLocal, xLocal, RoundMode::CAST_NONE, alignSize_ * rowNum);
    } else {
        xSortLocal = xLocal;
        sortedValueCast = sortedValueLocal;
    }
    for (uint32_t row = 0; row < rowNum; ++row) {
        uint32_t offset = row * alignSize_;
        if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<CONVERT_TYPE>) {
            if (medianMode_ != MEDIAN_MODE_STATIC) {
                CanonicalizeNanValues(xSortLocal[offset], alignSize_);
            }
        }
        // MergeSort sorts sign-flipped ascending keys. Flip before Sort/Sort32 and flip back after Extract.
        FlipSignBit(xSortLocal, offset, alignSize_);
        if constexpr (isSort32SmallAxis == 1) {
            Sort32<CONVERT_TYPE>(sortedLocal, xSortLocal[offset], indexLocal_, 1);
        } else {
            LocalTensor<CONVERT_TYPE> concatLocal;
            Concat(concatLocal, xSortLocal[offset], concatTmp, concatRepeatTimes);
            AscendC::Sort<CONVERT_TYPE, true>(sortedLocal, concatLocal, indexLocal_, sortTmp, sortRepeatTimes);
        }
        Extract(sortedValueCast[offset], sortedIndexLocal[offset], sortedLocal,
                isSort32SmallAxis == 1 ? 1 : sortRepeatTimes);
        FlipSignBit(sortedValueCast, offset, alignSize_);
    }
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        Cast(sortedValueLocal, sortedValueCast, RoundMode::CAST_RINT, alignSize_ * rowNum);
    }
}

template <typename T, typename CONVERT_TYPE, uint64_t isSort32SmallAxis, bool EnableMedian>
__aicore__ inline void KthValueMergeSortOneCore<T, CONVERT_TYPE, isSort32SmallAxis, EnableMedian>::CopyKthToGm(
    uint64_t outputOffset, uint32_t rowNum)
{
    LocalTensor<T> sortedValueLocal = outValueQueue_.DeQue<T>();
    LocalTensor<uint32_t> sortedIndexLocal = outIndexQueue_.DeQue<uint32_t>();

    event_t eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    LocalTensor<T> compactValue = compactValueQueue_.AllocTensor<T>();
    LocalTensor<int64_t> compactIndex = compactIndexQueue_.AllocTensor<int64_t>();
    for (uint32_t row = 0; row < rowNum; ++row) {
        uint32_t rowOffset = row * alignSize_;
        uint32_t selectedK = kthIndex_;
        if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
            uint32_t nonNanCount = CountNonNan(sortedValueLocal[rowOffset], numTileData_, concatTmpBuf_.Get<float>(),
                                               pipe_);
            selectedK = ResolveMedianK(kthIndex_, numTileData_, nonNanCount, medianMode_);
        }
        uint32_t srcOffset = rowOffset + selectedK;
        compactValue.SetValue(row, sortedValueLocal.GetValue(srcOffset));
        compactIndex.SetValue(row, static_cast<int64_t>(sortedIndexLocal.GetValue(srcOffset)));
    }
    compactValueQueue_.EnQue<T>(compactValue);
    compactIndexQueue_.EnQue<int64_t>(compactIndex);
    compactValue = compactValueQueue_.DeQue<T>();
    compactIndex = compactIndexQueue_.DeQue<int64_t>();
    event_t eventIdSToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
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
