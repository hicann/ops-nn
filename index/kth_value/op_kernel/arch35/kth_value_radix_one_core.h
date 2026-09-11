/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_RADIX_ONE_CORE_H
#define KTH_VALUE_RADIX_ONE_CORE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "kth_value_median_utils.h"
#include "kth_value_tiling_data.h"
#include "common/signed_zero_sort_utils.h"
#include "common/util_type_simd.h"

namespace KthValue {
using namespace AscendC;

template <typename T, bool EnableMedian = false>
class KthValueRadixOneCore {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const KthValueTilingData* tiling,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void CopyInputToUb(LocalTensor<T>& xLocal, int64_t row);
    __aicore__ inline void ProcessOneRow(int64_t row, uint32_t localOffset, LocalTensor<T>& compactValue,
                                         LocalTensor<int64_t>& compactIndex);
    __aicore__ inline void CopyOutputToGm(int64_t rowStart, uint32_t rowCount, LocalTensor<T>& compactValue,
                                          LocalTensor<int64_t>& compactIndex);
    __aicore__ inline void ProcessRows(int64_t rowStart, uint32_t rowCount);

    static constexpr SortConfig SORT_CONFIG{SortType::RADIX_SORT, false};
    static constexpr uint32_t UB_BLOCK_SIZE = Ops::Base::GetUbBlockSize();

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
    TBuf<TPosition::VECCALC> tmpUb_;
    TBuf<TPosition::VECCALC> sourceOrderBuf_;
    LocalTensor<uint32_t> sourceOrder_;

    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t numTileData_ = 0;
    uint32_t kthIndex_ = 0;
    uint32_t medianMode_ = 0;
    uint32_t unsortedDimParallel_ = 0;
    uint32_t xUbSize_ = 0;
    uint32_t idxUbSize_ = 0;
    uint32_t bufferNum_ = 1;
    uint32_t tmpUbSize_ = 0;
    uint32_t outputRowsPerLoop_ = 1;
    int64_t lastAxisNum_ = 0;
    int64_t unsortedDimNum_ = 0;
};

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueRadixOneCore<T, EnableMedian>::Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                                                   const KthValueTilingData* tiling, TPipe* pipe)
{
    if (tiling == nullptr || pipe == nullptr) {
        return;
    }
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    pipe_ = pipe;
    tiling_ = tiling;
    ParseTilingData();

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    valuesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
    indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));

    pipe_->InitBuffer(inQueueX_, bufferNum_, xUbSize_);
    pipe_->InitBuffer(outValueQueue_, bufferNum_, xUbSize_);
    pipe_->InitBuffer(outIndexQueue_, bufferNum_, idxUbSize_);
    pipe_->InitBuffer(compactValueQueue_, bufferNum_, ROUND_UP_AGLIN(outputRowsPerLoop_ * sizeof(T)));
    pipe_->InitBuffer(compactIndexQueue_, bufferNum_, ROUND_UP_AGLIN(outputRowsPerLoop_ * sizeof(int64_t)));
    pipe_->InitBuffer(tmpUb_, tmpUbSize_);
    if constexpr (SignedZeroSortCommon::IS_FLOATING_POINT_V<T>) {
        pipe_->InitBuffer(sourceOrderBuf_, idxUbSize_);
        sourceOrder_ = sourceOrderBuf_.Get<uint32_t>();
    }
}

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueRadixOneCore<T, EnableMedian>::ParseTilingData()
{
    numTileData_ = tiling_->numTileDataSize;
    kthIndex_ = tiling_->kthIndex;
    medianMode_ = tiling_->medianMode;
    unsortedDimParallel_ = tiling_->unsortedDimParallel;
    xUbSize_ = tiling_->keyParams0;
    idxUbSize_ = tiling_->keyParams1;
    bufferNum_ = tiling_->keyParams3 == 2U ? 2U : 1U;
    outputRowsPerLoop_ = tiling_->keyParams4 == 0U ? 1U : tiling_->keyParams4;
    tmpUbSize_ = tiling_->tmpUbSize;
    lastAxisNum_ = tiling_->lastAxisNum;
    unsortedDimNum_ = tiling_->unsortedDimNum;
}

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueRadixOneCore<T, EnableMedian>::CopyInputToUb(LocalTensor<T>& xLocal, int64_t row)
{
    int64_t inputOffset = row * lastAxisNum_;
    uint32_t copyBytes = static_cast<uint32_t>(numTileData_ * sizeof(T));

    DataCopyExtParams inputCopyParam{1, copyBytes, 0, 0, 0};
    DataCopyPadExtParams<T> inputPadParam{true, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[inputOffset], inputCopyParam, inputPadParam);
}

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueRadixOneCore<T, EnableMedian>::ProcessOneRow(int64_t row, uint32_t localOffset,
                                                                            LocalTensor<T>& compactValue,
                                                                            LocalTensor<int64_t>& compactIndex)
{
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    CopyInputToUb(xLocal, row);
    inQueueX_.EnQue<T>(xLocal);
    xLocal = inQueueX_.DeQue<T>();
    LocalTensor<T> sortedValue = outValueQueue_.AllocTensor<T>();
    LocalTensor<uint32_t> sortedIndex = outIndexQueue_.AllocTensor<uint32_t>();
    LocalTensor<uint8_t> tmpUb = tmpUb_.Get<uint8_t>();
    uint32_t selectedK = kthIndex_;
    if constexpr (EnableMedian && IS_MEDIAN_FLOAT_TYPE<T>) {
        if (medianMode_ != MEDIAN_MODE_STATIC) {
            CanonicalizeNanValues(xLocal, numTileData_);
        }
        uint32_t nonNanCount = CountNonNan(xLocal, numTileData_, tmpUb_.Get<float>(), pipe_);
        selectedK = ResolveMedianK(kthIndex_, numTileData_, nonNanCount, medianMode_);
    }

    bool hasNegativeZero = false;
    if constexpr (SignedZeroSortCommon::IS_FLOATING_POINT_V<T>) {
        hasNegativeZero = SignedZeroSortCommon::HasNegativeZeroVec(xLocal, sourceOrder_, numTileData_);
        if (hasNegativeZero) {
            SignedZeroSortCommon::PrepareSignedZeroKeysVec(xLocal, sourceOrder_, numTileData_);
        }
        AscendC::Sort<T, false, SORT_CONFIG>(sortedValue, sortedIndex, xLocal, tmpUb, numTileData_);
        if (hasNegativeZero) {
            SignedZeroSortCommon::RestoreSignedZeroValuesByIndexVec(sortedValue, sortedIndex, sourceOrder_,
                                                                    numTileData_);
        }
    } else {
        AscendC::Sort<T, false, SORT_CONFIG>(sortedValue, sortedIndex, xLocal, tmpUb, numTileData_);
    }
    inQueueX_.FreeTensor(xLocal);
    outValueQueue_.EnQue<T>(sortedValue);
    outIndexQueue_.EnQue<uint32_t>(sortedIndex);
    sortedValue = outValueQueue_.DeQue<T>();
    sortedIndex = outIndexQueue_.DeQue<uint32_t>();
    event_t eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    compactValue.SetValue(localOffset, sortedValue.GetValue(selectedK));
    compactIndex.SetValue(localOffset, static_cast<int64_t>(sortedIndex.GetValue(selectedK)));

    outValueQueue_.FreeTensor(sortedValue);
    outIndexQueue_.FreeTensor(sortedIndex);
}

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueRadixOneCore<T, EnableMedian>::CopyOutputToGm(int64_t rowStart, uint32_t rowCount,
                                                                             LocalTensor<T>& compactValue,
                                                                             LocalTensor<int64_t>& compactIndex)
{
    event_t eventIdSToMte3 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);

    DataCopyExtParams valueCopyParam{1, static_cast<uint32_t>(rowCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(valuesGm_[rowStart], compactValue, valueCopyParam);
    DataCopyExtParams indexCopyParam{1, static_cast<uint32_t>(rowCount * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(indicesGm_[rowStart], compactIndex, indexCopyParam);
}

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueRadixOneCore<T, EnableMedian>::ProcessRows(int64_t rowStart, uint32_t rowCount)
{
    LocalTensor<T> compactValue = compactValueQueue_.AllocTensor<T>();
    LocalTensor<int64_t> compactIndex = compactIndexQueue_.AllocTensor<int64_t>();
    for (uint32_t i = 0; i < rowCount; ++i) {
        ProcessOneRow(rowStart + i, i, compactValue, compactIndex);
    }
    compactValueQueue_.EnQue<T>(compactValue);
    compactIndexQueue_.EnQue<int64_t>(compactIndex);
    compactValue = compactValueQueue_.DeQue<T>();
    compactIndex = compactIndexQueue_.DeQue<int64_t>();
    CopyOutputToGm(rowStart, rowCount, compactValue, compactIndex);
    compactValueQueue_.FreeTensor(compactValue);
    compactIndexQueue_.FreeTensor(compactIndex);
}

template <typename T, bool EnableMedian>
__aicore__ inline void KthValueRadixOneCore<T, EnableMedian>::Process()
{
    if (blockIdx_ >= blockNum_ || blockIdx_ >= unsortedDimParallel_) {
        return;
    }
    int64_t rowsPerCore = unsortedDimNum_ / static_cast<int64_t>(unsortedDimParallel_);
    uint32_t tailRows = static_cast<uint32_t>(unsortedDimNum_ % static_cast<int64_t>(unsortedDimParallel_));
    int64_t blockRows = rowsPerCore + (blockIdx_ < tailRows ? 1 : 0);
    uint32_t priorTailRows = blockIdx_ < tailRows ? blockIdx_ : tailRows;
    int64_t blockOffset = static_cast<int64_t>(blockIdx_) * rowsPerCore + static_cast<int64_t>(priorTailRows);

    uint64_t blockRows64 = static_cast<uint64_t>(blockRows);
    uint64_t outputRowsPerLoop64 = static_cast<uint64_t>(outputRowsPerLoop_);
    uint64_t loopTimes = Ops::Base::CeilDiv(blockRows64, outputRowsPerLoop64);
    for (uint64_t loopIdx = 0; loopIdx < loopTimes; ++loopIdx) {
        uint64_t localOffset = loopIdx * outputRowsPerLoop64;
        uint64_t remainRows = blockRows64 - localOffset;
        uint32_t rowCount = remainRows < outputRowsPerLoop64 ? static_cast<uint32_t>(remainRows) : outputRowsPerLoop_;
        ProcessRows(blockOffset + static_cast<int64_t>(localOffset), rowCount);
    }
}
} // namespace KthValue

#endif
